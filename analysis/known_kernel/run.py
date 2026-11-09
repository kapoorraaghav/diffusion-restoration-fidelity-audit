"""Score a development-frozen, true-kernel Tikhonov inverse on matched DIV2K."""
from pathlib import Path
from contextlib import nullcontext
import hashlib
import io
import json
import math
import sys
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.three_pipeline import extra_blur
from experiments.stage_attribution import fidelity_measures
from experiments.metrics import psnr
from experiments.revision_analysis import nondominated

HERE=Path(__file__).resolve().parent
CACHE=ROOT/'runs/div2k_known_kernel'
OUTPUT=ROOT/'outputs/Paper2_Known_Kernel_Baseline'
METRICS=['psnr','ssim','lpips','dists']
SIGN=np.array([1,1,-1,-1])


def sha(path):
    with path.open('rb') as f:
        digest=hashlib.sha256()
        for block in iter(lambda:f.read(1<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def source_archive(part):
    path=CACHE/f'DIV2K_{part}_HR.zip'
    return zipfile.ZipFile(path) if path.is_file() else nullcontext(CACHE/part)


def tensor(zip_file, row, part, device, previews=None):
    name=f'{int(row.original_id):04d}.png'
    raw=(zip_file/name).read_bytes() if isinstance(zip_file,Path) else zip_file.read(f'DIV2K_{part}_HR/{name}')
    assert hashlib.sha256(raw).hexdigest()==row.original_sha256
    with Image.open(io.BytesIO(raw)) as source:
        w,h=source.size
        side=min(w,h)
        x,y=(w-side)//2,(h-side)//2
        crop=(x,y,x+side,y+side)
        assert crop==tuple(json.loads(row.crop_box)) and [w,h]==json.loads(row.original_size)
        image=source.convert('RGB').crop(crop).resize((512,512),Image.Resampling.LANCZOS)
    pixels=np.array(image)
    if previews is not None:
        member=f'images/{row.image_id}/reference.png'
        if member in previews.namelist():
            blob=previews.read(member)
            assert hashlib.sha256(blob).hexdigest()==row.sha256
            with Image.open(io.BytesIO(blob)) as archived:
                assert np.array_equal(np.asarray(archived.convert('RGB')),pixels)
    return torch.from_numpy(pixels).to(device).permute(2,0,1)[None].float()/255


class ReflectInverse:
    def __init__(self,sigma,device):
        self.radius=math.ceil(3*sigma)
        offsets=torch.arange(-self.radius,self.radius+1,device=device,dtype=torch.float32)
        kernel=torch.exp(-offsets.square()/(2*sigma*sigma))
        kernel/=kernel.sum()
        pos=torch.arange(512,device=device)[:,None]+offsets.long()[None,:]
        pos=torch.where(pos<0,-pos,pos)
        pos=torch.where(pos>=512,1022-pos,pos)
        a=torch.zeros(512,512,device=device,dtype=torch.float64)
        a.scatter_add_(1,pos,kernel.double().expand(512,-1))
        self.a=a
        self.u,s,vh=torch.linalg.svd(a)
        self.v=vh.T.contiguous()
        self.s=s[:,None]*s[None,:]
        generator=torch.Generator(device=device).manual_seed(170)
        x=torch.rand((1,3,512,512),generator=generator,device=device)
        actual=extra_blur(x,sigma)
        predicted=(a@x.double()@a.T).float()
        self.operator_error=float((actual-predicted).abs().max())
        assert self.operator_error<3e-7
        const=torch.full_like(x,.5)
        assert float((self.solve(const,.00001)-const).abs().max())<2e-6

    def solve(self,y,lam):
        y=y.double()
        residual=y-self.a@y@self.a.T
        weights=(self.u.T@residual@self.u)*self.s/(self.s.square()+lam)
        return (y+self.v@weights@self.v.T).float()


def init_metrics(device,protocol):
    import lpips
    sys.path.insert(0,str(ROOT/'followup_v9/reviewer_extension'))
    from DISTS_pt import DISTS
    from torchvision.models import VGG16_Weights,AlexNet_Weights
    model_dir=CACHE/'torch_hub/checkpoints'
    model_dir.mkdir(parents=True,exist_ok=True)
    torch.hub.set_dir(str(model_dir.parent))
    import shutil
    for w in [VGG16_Weights.IMAGENET1K_V1,AlexNet_Weights.IMAGENET1K_V1]:
        name=Path(w.url).name
        source=CACHE/name
        assert sha(source)==protocol['metric_weights'][name]
        destination=model_dir/name
        if not destination.exists():
            shutil.copyfile(source,destination)
        assert sha(destination)==protocol['metric_weights'][name]
    lp=lpips.LPIPS(net='alex',version='0.1').eval().to(device)
    assert sha(Path(lpips.__file__).parent/'weights/v0.1/alex.pth')==protocol['metric_weights']['alex.pth']
    dists_path=ROOT/'followup_v9/reviewer_extension/weights.pt'
    assert sha(dists_path)==protocol['dists_weights_sha256']
    assert sha(ROOT/'followup_v9/reviewer_extension/DISTS_pt.py')==protocol['dists_source_sha256']
    dists=DISTS(load_weights=False).eval()
    weights=torch.load(dists_path,map_location='cpu',weights_only=True)
    dists.alpha.data.copy_(weights['alpha'])
    dists.beta.data.copy_(weights['beta'])
    return lp,dists.to(device)


def compute_scores(output,ref,lp,dists):
    result=fidelity_measures(output,ref,lp)
    return {**{m:result[m] for m in METRICS[:3]},'dists':float(dists(output,ref).item())}


@torch.inference_mode()
def main():
    protocol=json.loads((HERE/'protocol.json').read_text())
    OUTPUT.mkdir(parents=True,exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    with zipfile.ZipFile(ROOT/'followup_v9/input_audit/Paper2_Focus_Input_Tables.zip') as z:
        dev=pd.read_csv(z.open('inventories/development.csv')).query('dataset=="div2k"').sort_values('image_id')
        test=pd.read_csv(z.open('inventories/div2k.csv')).sort_values('image_id')
    assert len(dev)==50 and len(test)==99 and set(dev.image_id).isdisjoint(test.image_id)
    prior=ROOT/'outputs/Paper2_MultiRobust_Submission_v34'
    old=pd.concat([pd.read_csv(prior/'severity_metrics.csv'),pd.read_csv(prior/'supplementary/sigma3/metrics.csv')])
    inverses={s:ReflectInverse(s,device) for s in protocol['sigma']}
    print('Reflection and constant-field checks passed',flush=True)
    frozen_path=OUTPUT/'frozen_selection.json'
    signature=dict(protocol_sha256=sha(HERE/'protocol.json'),implementation_sha256=sha(Path(__file__)),
                   development_ids=dev.image_id.tolist(),evaluation_ids=test.image_id.tolist())
    if frozen_path.exists():
        frozen=json.loads(frozen_path.read_text())
        assert all(frozen[k]==v for k,v in signature.items())
        selected={float(k):v for k,v in frozen['lambda_by_sigma'].items()}
    else:
        development=[]
        archive=Path('<LOCAL_HOME>/Downloads/p2w_development_full_v2_CROSS_FINAL.zip')
        with source_archive('train') as z, zipfile.ZipFile(archive) as previews:
            assert sum(f'images/{i}/reference.png' in previews.namelist() for i in dev.image_id)==20
            for index,row in enumerate(dev.itertuples(),1):
                ref=tensor(z,row,'train',device,previews)
                for sigma,inverse in inverses.items():
                    y=extra_blur(ref,sigma)
                    for lam in protocol['regularization_grid']:
                        out=inverse.solve(y,lam).clamp(0,1)
                        development.append(dict(source_id=row.image_id,sigma=sigma,regularization=lam,psnr=psnr(out,ref)))
                if index%10==0:print(f'Development {index}/50',flush=True)
        rows=pd.DataFrame(development)
        rows.to_csv(OUTPUT/'development_scores.csv',index=False)
        aggregate=rows.groupby(['sigma','regularization'],as_index=False).psnr.mean()
        best=aggregate.sort_values(['psnr','regularization'],ascending=False).groupby('sigma').head(1)
        selected=dict(zip(best.sigma,best.regularization))
        frozen={**signature,'lambda_by_sigma':selected,'development_scores_sha256':sha(OUTPUT/'development_scores.csv')}
        frozen_path.write_text(json.dumps(frozen,indent=2))
    print('Frozen development choices',selected,flush=True)
    lp,dists=init_metrics(device,json.loads((prior/'severity_protocol.json').read_text()))
    expected=old.query('candidate=="input"').set_index(['source_id','sigma'])
    evaluation=[]
    parity=[]
    with source_archive('valid') as z:
        for index,row in enumerate(test.itertuples(),1):
            ref=tensor(z,row,'valid',device)
            for sigma,inverse in inverses.items():
                y=extra_blur(ref,sigma)
                input_scores=compute_scores(y,ref,lp,dists)
                errors={m:abs(input_scores[m]-expected.loc[(row.image_id,sigma),m]) for m in METRICS}
                assert all(errors[m]<=protocol['parity_tolerance'][m] for m in METRICS),(row.image_id,sigma,errors)
                parity.append(dict(source_id=row.image_id,sigma=sigma,**errors))
                raw=inverse.solve(y,selected[sigma])
                output=raw.clamp(0,1)
                border=torch.ones((512,512),device=device,dtype=torch.bool)
                r=inverse.radius
                border[r:-r,r:-r]=False
                squared=(output-ref).square()
                input_squared=(y-ref).square()
                evaluation.append(dict(source_id=row.image_id,sigma=sigma,regularization=selected[sigma],
                    **compute_scores(output,ref,lp,dists),clip_fraction=float(((raw<0)|(raw>1)).float().mean()),
                    border_mse=float(squared[...,border].mean()),interior_mse=float(squared[...,~border].mean()),
                    input_border_mse=float(input_squared[...,border].mean()),input_interior_mse=float(input_squared[...,~border].mean())))
            if index%10==0:print(f'Evaluation {index}/99',flush=True)
    e=pd.DataFrame(evaluation)
    assert len(e)==297 and len(parity)==297
    e.to_csv(OUTPUT/'evaluation_scores.csv',index=False)
    pd.DataFrame(parity).to_csv(OUTPUT/'input_metric_parity.csv',index=False)
    means=old.groupby(['source_id','sigma','candidate'],as_index=False)[METRICS].mean()
    effects=[]
    fronts=[]
    for sigma,g in e.groupby('sigma'):
        a=g.set_index('source_id')[METRICS].sort_index()
        baseline=means[means.sigma==sigma]
        weights=np.random.default_rng(20261102).multinomial(99,np.full(99,1/99),size=2000)/99
        for candidate,b in baseline.groupby('candidate'):
            b=b.set_index('source_id')[METRICS].reindex(a.index)
            delta=(a-b).to_numpy()*SIGN
            assert np.isfinite(delta).all()
            lo,hi=np.quantile(weights@delta,[.025,.975],axis=0)
            for i,metric in enumerate(METRICS):
                effects.append(dict(sigma=sigma,comparator=candidate,metric=metric,n=99,
                    mean_improvement=delta[:,i].mean(),ci95_low=lo[i],ci95_high=hi[i],
                    wins=int((delta[:,i]>0).sum()),ties=int((delta[:,i]==0).sum()),losses=int((delta[:,i]<0).sum())))
        augmented=baseline.groupby('candidate')[METRICS].mean()
        augmented.loc['known_kernel']=a.mean()
        for k in [3,4]:
            mask=nondominated(-augmented.to_numpy()[:,:k]*SIGN[:k])
            fronts.append(dict(sigma=sigma,metrics=k,frontier=';'.join(augmented.index[mask])))
    pd.DataFrame(effects).to_csv(OUTPUT/'paired_effects.csv',index=False)
    pd.DataFrame(fronts).to_csv(OUTPUT/'augmented_frontiers.csv',index=False)
    (OUTPUT/'verification.json').write_text(json.dumps(dict(sources=99,development=50,device=device,torch=torch.__version__,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,max_operator_errors={s:a.operator_error for s,a in inverses.items()},
        max_input_parity=pd.DataFrame(parity)[METRICS].max().to_dict()),indent=2))
    print(e.groupby('sigma')[METRICS].mean().to_string(),flush=True)


if __name__=='__main__':
    main()
