"""Development selection and archived-score evaluation of frozen blur policies."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from followup_v9.div2k_known_kernel.run import CACHE,init_metrics,source_archive,tensor
from experiments.three_pipeline import extra_blur
from experiments.stage_attribution import fidelity_measures

OUT=ROOT/'outputs/Paper2_Observed_Blur_Policy'
PAPER=ROOT/'outputs/Paper2_MultiRobust_Submission_v38'
SIGMAS=[.6,1.2,3.]
METRICS=['psnr','ssim','lpips','dists']


def blur_proxy(y):
    gray=y.double().mean(dim=1)
    first=[gray.diff(dim=k) for k in [-1,-2]]
    return float(sum(d.diff(dim=k).square().mean() for d,k in zip(first,[-1,-2]))/
                 (sum(d.square().mean() for d in first)+1e-12))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision(frame,kind,threshold):
    if threshold=='pass':
        return np.zeros(len(frame),bool)
    if threshold=='clean':
        return np.ones(len(frame),bool)
    return (frame.proxy.to_numpy()<threshold) if kind=='image_only' else (frame.sigma.to_numpy()>=threshold)


def freeze(dev):
    rows=[]
    thresholds=['pass',*np.quantile(dev.proxy,np.arange(1,20)/20).tolist(),'clean']
    for kind,grid in [('image_only',thresholds),('privileged_sigma',['pass',3.,1.2,'clean'])]:
        for order,threshold in enumerate(grid):
            use=decision(dev,kind,threshold)
            psnr=np.where(use,dev.cleaner_psnr-dev.input_psnr,0.)
            lpips=np.where(use,dev.input_lpips-dev.cleaner_lpips,0.)
            rows.append(dict(kind=kind,threshold=threshold,order=order,cleaned=int(use.sum()),
                serious_harm=int((psnr<-1).sum()),mean_lpips_improvement=float(lpips.mean())))
    candidates=pd.DataFrame(rows)
    candidates['eligible']=candidates.serious_harm/len(dev)<=.05
    candidates.to_csv(OUT/'development_grid.csv',index=False)
    rules={}
    for kind,group in candidates[candidates.eligible].groupby('kind'):
        winner=group.sort_values(['mean_lpips_improvement','serious_harm','cleaned','order'],
            ascending=[False,True,True,True]).iloc[0]
        rules[kind]=winner.threshold
    return rules


@torch.inference_mode()
def development():
    paths=list((ROOT/'runs/input_audit_local/cache').glob('cnn_prior_focus_*'))
    assert len(paths)==1
    sys.path.insert(0,str(paths[0]))
    from experiments import __path__ as experiment_paths
    experiment_paths.append(str(paths[0]/'experiments'))
    from experiments import focus_worker as f
    from omegaconf import OmegaConf
    sys.path.insert(0,str(f.source_path('DiffBIR')))
    from diffbir.utils.common import instantiate_from_config
    checkpoint=CACHE/'realesrgan_s4_swinir_100k.pth'
    protocol=json.loads((PAPER/'severity_protocol.json').read_text())
    assert sha(checkpoint)==protocol['models'][checkpoint.name]
    model=instantiate_from_config(OmegaConf.load(f.source_path('DiffBIR')/'configs/inference/swinir.yaml'))
    f.load_cleaner_weights(model,torch.load(checkpoint,map_location='cpu',weights_only=True))
    model=model.eval().cuda()
    lpips,_=init_metrics('cuda',protocol)
    with zipfile.ZipFile(ROOT/'followup_v9/input_audit/Paper2_Focus_Input_Tables.zip') as archive:
        inv=pd.read_csv(archive.open('inventories/development.csv')).query('dataset=="div2k"').sort_values('image_id')
    assert len(inv)==50
    rows=[]
    with source_archive('train') as z, zipfile.ZipFile('<LOCAL_HOME>/Downloads/p2w_development_full_v2_CROSS_FINAL.zip') as previews:
        assert sum(f'images/{i}/reference.png' in previews.namelist() for i in inv.image_id)==20
        for index,row in enumerate(inv.itertuples(),1):
            ref=tensor(z,row,'train','cuda',previews)
            for sigma in SIGMAS:
                y=extra_blur(ref,sigma)
                c=model(y).float().clamp(0,1)
                a,b=fidelity_measures(y,ref,lpips),fidelity_measures(c,ref,lpips)
                rows.append(dict(source_id=row.image_id,sigma=sigma,proxy=blur_proxy(y),
                    input_psnr=a['psnr'],cleaner_psnr=b['psnr'],input_lpips=a['lpips'],cleaner_lpips=b['lpips']))
            if index%10==0:print(f'Development {index}/50',flush=True)
    dev=pd.DataFrame(rows)
    dev.to_csv(OUT/'development_scores.csv',index=False)
    return dev


@torch.inference_mode()
def div2k_features():
    inv=pd.read_csv(PAPER/'supplementary/sigma3/source_inventory.csv')
    rows=[]
    with source_archive('valid') as z:
        for row in inv.itertuples():
            ref=tensor(z,row,'valid','cuda')
            for sigma in SIGMAS:
                rows.append(dict(source_id=row.image_id,sigma=sigma,proxy=blur_proxy(extra_blur(ref,sigma)),
                                 reference_sha256=row.sha256))
    return pd.DataFrame(rows)


def evaluation(features,domain,rules):
    if domain=='div2k':
        metrics=pd.concat([pd.read_csv(PAPER/'severity_metrics.csv'),pd.read_csv(PAPER/'supplementary/sigma3/metrics.csv')])
        inv=pd.read_csv(PAPER/'supplementary/sigma3/source_inventory.csv')
    else:
        metrics=pd.read_csv(PAPER/'supplementary/imagenet100/metrics.csv')
        inv=pd.read_csv(PAPER/'supplementary/imagenet100/source_inventory.csv')
    assert set(features.source_id)==set(inv.image_id)
    assert features.reference_sha256.eq(features.source_id.map(inv.set_index('image_id').sha256)).all()
    assert not features.duplicated(['source_id','sigma']).any() and len(features)==3*len(inv)
    assert set(features.sigma)==set(SIGMAS) and np.isfinite(features.proxy).all()
    pair=metrics.query('candidate in ["input","cleaner"]').pivot(index=['source_id','sigma'],columns='candidate',values=METRICS)
    results=[]
    for name,kind,threshold in [('always_pass','image_only','pass'),('always_clean','image_only','clean'),
                               *[(k,k,t) for k,t in rules.items()]]:
        frame=features.sort_values(['source_id','sigma']).copy()
        use=decision(frame,kind,threshold)
        frame['cleaned']=use
        match=pair.reindex(pd.MultiIndex.from_frame(frame[['source_id','sigma']]))
        assert np.isfinite(match.to_numpy()).all()
        for m in METRICS:
            change=(match[m].cleaner-match[m].input).to_numpy()
            frame[m]=np.where(use,change if m in ['psnr','ssim'] else -change,0.)
        frame['policy']=name
        results.append(frame)
    scores=pd.concat(results)
    scores.to_csv(OUT/f'{domain}_policy_records.csv',index=False)
    rows=[]
    for name,group in scores.groupby('policy'):
        for sigma,g in [('all',group),*group.groupby('sigma')]:
            r=dict(domain=domain,sigma=sigma,policy=name,n=len(g),cleaned=int(g.cleaned.sum()),
                   psnr_harm=int((g.psnr<0).sum()),serious_harm=int((g.psnr<-1).sum()))
            r.update(psnr_harm_rate=r['psnr_harm']/len(g),serious_harm_rate=r['serious_harm']/len(g))
            r.update({m:g[m].mean() for m in METRICS})
            rows.append(r)
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/f'{domain}_policy_summary.csv',index=False)
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=True
    frozen=OUT/'frozen_policy.json'
    if not frozen.exists():
        dev=development()
        rules=freeze(dev)
        frozen.write_text(json.dumps(dict(rules=rules,protocol_sha256=sha(HERE/'protocol.json'),
            development_scores_sha256=sha(OUT/'development_scores.csv'),source_count=50,score_count=150),indent=2))
    else:
        artifact=json.loads(frozen.read_text())
        assert artifact['protocol_sha256']==sha(HERE/'protocol.json')
        assert artifact['development_scores_sha256']==sha(OUT/'development_scores.csv')
        rules=artifact['rules']
    print('Frozen policies',rules,flush=True)
    if len(sys.argv)>1:
        evaluation(pd.read_csv(sys.argv[1]),'imagenet',rules)
    else:
        features=div2k_features()
        features.to_csv(OUT/'div2k_features.csv',index=False)
        evaluation(features,'div2k',rules)
