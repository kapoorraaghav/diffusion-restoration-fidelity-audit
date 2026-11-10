"""Resumable review extension. Real pretrained inference only, never fake results."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import torch

from .revision_design import (VERSION, CFGS, SEEDS, ALPHAS, BETAS, PROFILES,
                              candidates, compose, degradation, digest, frozen_protocol, strategy, execution_candidates)
from .revision_analysis import analyze_run, freeze_preferences, load_preferences, fronts_and_hv
from .workshop_data import prepare as old_prepare
from .cross_domain_data import sha
from .cross_domain_worker import EXPECTED_MODELS, FACE_HASH, GENERAL_HASH, atomic_json, atomic_csv
from .opensource import load_diffbir, source_path, save_image, SOURCES
from .conditioning_audit import load_cleaner_weights
from .paper2_worker import paper2_pipeline, sample
from .paper2_design import BASELINE
from .stage_attribution import fidelity_measures

MUSIQ_REVISION='c753dca46d76714c9bb7849eadc2bd5854716320'
MUSIQ_SHA256='e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa'


def collect(run):
    complete=sorted(p for p in (run/'records').glob('*/*') if (p/'complete.json').is_file())
    if not complete:return
    for name in ['metrics.csv','checks.csv','raw_hashes.csv']:
        atomic_csv(run/('results.csv' if name=='metrics.csv' else name),pd.concat([pd.read_csv(p/name) for p in complete],ignore_index=True))


def export(run,job,complete=False):
    """One complete ZIP per run, rebuilt atomically; hashes and CRC verified."""
    run,job=Path(run),Path(job)
    dest=job.parent.parent/'paper2_review_downloads';dest.mkdir(parents=True,exist_ok=True)
    table=dest/(job.name+'_tables.zip');temp=table.with_suffix('.partial')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for p in sorted(run.iterdir()):
            if p.is_file() and p.suffix in ['.json','.csv','.txt']:
                z.write(p,'experiment/'+p.name)
    with zipfile.ZipFile(temp) as z:
        if z.testzip():raise ValueError('Tables CRC validation failed')
    temp.replace(table)
    if (run/'frozen_preferences.json').exists():
        (dest/'frozen_preferences.json').write_bytes((run/'frozen_preferences.json').read_bytes())
    if not complete:return str(table)
    final=dest/(job.name+'_FINAL.zip');temp=final.with_suffix('.partial')
    files=sorted(p for p in run.rglob('*') if p.is_file())
    manifest=[dict(path=p.relative_to(run).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for p in files:z.write(p,'experiment/'+p.relative_to(run).as_posix())
        z.writestr('backup_manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(temp) as z:
        if z.testzip():raise ValueError('Complete ZIP CRC validation failed')
    temp.replace(final)
    atomic_json(dest/(job.name+'_FINAL.sha256.json'),dict(file=final.name,bytes=final.stat().st_size,sha256=sha(final),files=len(files)))
    return str(final)


def prepare(settings):
    source=dict(settings,stage='development' if settings['stage']=='development' else 'final',
                smoke=bool(settings['smoke']) if settings['stage']=='development' else False)
    if settings['dataset']=='imagenet' and Path(settings.get('imagenet_source','')).is_dir():
        # Accept the known authorized HF filename variant without inventing validation IDs.
        import re
        import shutil
        supplied=Path(settings['imagenet_source'])
        wanted=set(frozen_protocol()['final']['imagenet'])
        aliases=Path(settings['data_root'])/'imagenet'/'canonical_aliases';aliases.mkdir(parents=True,exist_ok=True)
        files={}
        for p in supplied.rglob('ILSVRC2012_val_*.JPEG'):
            match=re.fullmatch(r'ILSVRC2012_val_(\d{8})(?:_n\d{8})?\.JPEG',p.name)
            if not match or int(match[1]) not in wanted:continue
            ident=int(match[1])
            if ident in files and sha(p)!=sha(files[ident]):raise ValueError('Conflicting ImageNet sources for one validation ID')
            files[ident]=p
        if set(files)!=wanted:raise ValueError('Authorized ImageNet directory lacks fixed validation IDs; no substitution')
        for ident,p in files.items():
            dest=aliases/f'ILSVRC2012_val_{ident:08d}.JPEG'
            if not dest.exists() or sha(dest)!=sha(p):shutil.copy2(p,dest)
        source['imagenet_source']=str(aliases)
    records=old_prepare(source)
    for r in records:
        # Correct metadata label, without pretending this establishes pretraining exclusion.
        r['selection_seed']=20261031
        r['selection_population']='research-defined FFHQ60000-69999 pool' if r['dataset']=='ffhq' else 'original frozen dataset inventory'
    allowed=None
    if settings['stage']=='severity':
        allowed=set(frozen_protocol()['severity'][settings['dataset']])
    records=[r for r in records if allowed is None or int(r['original_id']) in allowed]
    if settings['smoke']:
        if settings['stage']=='development':
            records=[next(r for r in records if r['dataset']==d) for d in ['ffhq','div2k']]
        else:records=records[:1]
    return records


def main(settings,job):
    import lpips
    import pyiqa
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    if not torch.cuda.is_available():raise RuntimeError('Select a CUDA GPU in Molab before starting')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    job=Path(job);run=job/'experiment';(run/'records').mkdir(parents=True,exist_ok=True)
    design=frozen_protocol()
    preference=load_preferences(settings['selection_file']) if settings['stage']!='development' else None
    specifications=execution_candidates(settings['stage'],preference)
    cfgs=sorted({c['cfg'] for c in specifications})
    records=prepare(settings)
    if not records:raise ValueError('No dataset records')
    expected=design['development'] if settings['stage']=='development' else {
        settings['dataset']:design['severity' if settings['stage']=='severity' else 'final'][settings['dataset']]}
    if not settings['smoke']:
        if {(r['dataset'],int(r['original_id'])) for r in records}!={(d,i) for d,ids in expected.items() for i in ids}:
            raise ValueError('Inventory differs from frozen source IDs')
    models,cache=Path(settings['model_root']),Path(settings['cache_root'])
    def checkpoint(name):
        return next(p for p in list(models.rglob(name))+[cache/'weights'/name] if p.is_file())
    hashes={n:sha(checkpoint(n)) for n in EXPECTED_MODELS}
    if hashes!=EXPECTED_MODELS:raise ValueError('DiffBIR/SD checkpoint mismatch')
    assets={}
    for domain in sorted({r['dataset'] for r in records}):
        face=domain=='ffhq';name='face_swinir_v1.ckpt' if face else 'realesrgan_s4_swinir_100k.pth'
        path=Path(hf_hub_download('lxq007/DiffBIR' if face else 'lxq007/DiffBIR-v2',name,cache_dir=str(cache/'huggingface')))
        expected_hash=FACE_HASH if face else GENERAL_HASH
        if sha(path)!=expected_hash:raise ValueError('Cleaner checkpoint mismatch')
        assets[domain]=dict(path=str(path),sha256=expected_hash)
    quality_path=Path(hf_hub_download('chaofengc/IQA-PyTorch-Weights','musiq_koniq_ckpt-e95806b9.pth',
                       revision=MUSIQ_REVISION,cache_dir=str(cache/'huggingface')))
    if sha(quality_path)!=MUSIQ_SHA256:raise ValueError('MUSIQ checkpoint hash mismatch')
    sys.path.insert(0,str(source_path('DiffBIR')))
    from diffbir.utils.common import instantiate_from_config,wavelet_reconstruction
    fixed=dict(version=VERSION,stage=settings['stage'],dataset=settings['dataset'],smoke=settings['smoke'],
        design_sha256=design['sha256'],preference_sha256=preference['sha256'] if preference else None,
        code_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},execution_candidates=specifications,
        source_revisions=SOURCES,checkpoints=hashes,cleaners={d:a['sha256'] for d,a in assets.items()},
        musiq=dict(repo_revision=MUSIQ_REVISION,sha256=sha(quality_path),data='koniq10k'),
        sources=[dict(dataset=r['dataset'],image_id=r['image_id'],sha256=r['sha256']) for r in records],
        torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version(),gpu=torch.cuda.get_device_name(),
        packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','Pillow','torchvision','lpips','pyiqa','torchsde']},
        precision='float16 diffusion; float32 VAE, wavelet and metrics; no autocast/TF32',
        evidence_status='review-driven extension on previously inspected final source pool; no untouched confirmation claim')
    if preference and preference.get('source_hashes')!=fixed['code_hashes']:
        raise ValueError('Development/evaluation source differs; use the same notebook release')
    runtime={k:fixed[k] for k in ['torch','cuda','python','packages','checkpoints','cleaners','musiq','precision']}
    # Development has both cleaner checkpoints; compare only those needed in this domain.
    if preference:
        previous=dict(preference.get('runtime_fingerprint',{}))
        previous['cleaners']={d:h for d,h in previous.get('cleaners',{}).items() if d in fixed['cleaners']}
        # Natural-domain cleaners are identical across DIV2K/Urban100/ImageNet.
        if any(d not in previous['cleaners'] for d in fixed['cleaners']):
            source=preference.get('runtime_fingerprint',{}).get('cleaners',{})
            previous['cleaners']={d:source.get('ffhq' if d=='ffhq' else 'div2k') for d in fixed['cleaners']}
        if previous!=runtime:raise ValueError('Development/evaluation runtime or weights differ; use the same package environment')
    signature=digest(fixed)
    if (run/'protocol.json').exists() and json.loads((run/'protocol.json').read_text())['protocol_sha256']!=signature:
        raise ValueError('Resume protocol/data/weights/software changed; keep run and use new label')
    atomic_json(run/'protocol.json',dict(**fixed,protocol_sha256=signature))
    atomic_json(run/'design.json',design)
    if preference:atomic_json(run/'frozen_preferences.json',preference)
    atomic_csv(run/'dataset_inventory.csv',pd.DataFrame(records))
    model,sampler=load_diffbir(checkpoint('sd2.1-base-zsnr-laionaes5.ckpt'),checkpoint('DiffBIR_v2.1.pt'),'cuda',version='v2.1')
    pipe=paper2_pipeline(model,sampler)
    if next(model.vae.parameters()).dtype!=torch.float32:
        raise ValueError('VAE dtype differs from registered float32 path')
    cleaners={}
    for domain,asset in assets.items():
        cleaner=instantiate_from_config(OmegaConf.load(source_path('DiffBIR')/'configs/inference/swinir.yaml'))
        load_cleaner_weights(cleaner,torch.load(asset['path'],map_location='cpu',weights_only=True))
        cleaners[domain]=cleaner.eval().cuda()
    torch.hub.set_dir(str(cache/'torch_hub'))
    perceptual=lpips.LPIPS(net='alex',version='0.1').eval().cuda()
    quality=pyiqa.create_metric('musiq',pretrained_model_path=str(quality_path),device='cuda').eval()
    # Record auxiliary metric cache files; checkpoint revision and content hash already fixed above.
    profiles=PROFILES if settings['stage']=='severity' and not settings['smoke'] else PROFILES[:1]
    review=set()
    for domain in sorted(cleaners):
        ids=[r['image_id'] for r in records if r['dataset']==domain]
        review.update(sorted(ids,key=lambda i:hashlib.sha256((VERSION+i).encode()).hexdigest())[:5])
    atomic_json(run/'visual_selection.json',dict(rule='SHA256 ordering, independent of metrics',ids=sorted(review),
        exact_float_scope='all CFG/seed raw arrays for these five images per domain; reference/y/z once per profile',
        full_rerender_scope='other images need inference; raw hashes alone do not reconstruct arrays'))
    calls=0;done_count=0;start_all=time.perf_counter()
    for record in records:
        ident,domain=record['image_id'],record['dataset']
        with Image.open(record['path']) as im:array=np.array(im.convert('RGB'),copy=True)
        target=torch.from_numpy(array).permute(2,0,1)[None].float().cuda()/255
        if target.shape!=(1,3,512,512):raise ValueError('Unexpected reference shape')
        pipe.cleaner=cleaners[domain]
        for profile in profiles:
            folder=run/'records'/ident/profile['name'];folder.mkdir(parents=True,exist_ok=True)
            marker=folder/'complete.json'
            if marker.exists():
                saved=json.loads(marker.read_text())
                if saved['protocol_sha256']!=signature or any(not (run/f).is_file() or sha(run/f)!=h for f,h in saved['files'].items()):
                    raise ValueError('Completed checkpoint failed integrity check; do not silently overwrite')
                calls+=saved['sampling_calls'];done_count+=1;print('Resume verified',ident,profile['name'],flush=True);continue
            rows=[];checks=[];raw_hashes=[]
            images=run/'images'/ident/profile['name']
            if ident in review:images.mkdir(parents=True,exist_ok=True)
            with torch.inference_mode():
                y=degradation(target,profile,ident)
                z=pipe.apply_cleaner(y,False,512,256).float()
                if not torch.isfinite(z).all():raise ValueError('Nonfinite cleaner output')
                def metric(name,value,seed=-1,**extra):
                    value=value.float().clamp(0,1)
                    values=fidelity_measures(value,target,perceptual)
                    values['musiq']=float(quality(value).mean())
                    if not np.isfinite(list(values.values())).all():raise ValueError('Nonfinite metric')
                    return dict(dataset=domain,image_id=ident,condition='restored',profile=profile['name'],
                                strategy=name,seed=seed,**values,**extra)
                for name,value in [('input_y',y),('cleaner',z)]:rows.append(metric(name,value))
                for label,value in [('x_oracle',target),('y',y),('z',z)]:
                    latent=model.vae_encode(value*2-1,sample=False)
                    alternate=model.prepare_condition(value,[''])['c_img']
                    error=float((latent-alternate).abs().max())
                    checks.append(dict(image_id=ident,profile=profile['name'],input=label,max_abs=error,passed=error<=1e-5))
                    if error>1e-5:raise ValueError('VAE entry parity failed')
                    reconstruction=(model.vae_decode(latent)+1)/2
                    if reconstruction.shape!=target.shape:raise ValueError('VAE reconstruction shape mismatch')
                    rows.append(metric('vae_'+label,reconstruction))
                if ident in review:
                    for label,value in [('reference',target),('input_y',y),('cleaner',z)]:
                        np.save(images/(label+'.npy'),value.cpu().numpy(),allow_pickle=False)
                        save_image(value.clamp(0,1),images/(label+'.png'))
                for cfg in cfgs:
                    for seed in SEEDS:
                        config=dict(BASELINE,cfg=cfg)
                        torch.cuda.synchronize();started=time.perf_counter()
                        raw=sample(pipe,z,seed,config).float()
                        torch.cuda.synchronize();seconds=time.perf_counter()-started
                        if not torch.isfinite(raw).all():raise ValueError('Nonfinite raw output')
                        checksum=hashlib.sha256(raw.cpu().contiguous().numpy().tobytes()).hexdigest()
                        full=wavelet_reconstruction(raw,z).float()
                        raw_hashes.append(dict(image_id=ident,profile=profile['name'],cfg=cfg,seed=seed,sha256=checksum,seconds=seconds))
                        for spec in [c for c in specifications if c['cfg']==cfg]:
                            alpha,beta,name=spec['alpha'],spec['beta'],spec['strategy']
                            output=compose(raw,full,z,alpha,beta)
                            rows.append(metric(name,output,seed,raw_sha256=checksum,cfg=cfg,alpha=alpha,beta=beta))
                            if ident in review and seed==11 and name in ['cfg1_alpha1','cfg6_alpha1','blend_cfg1_beta0.5']:
                                save_image(output,images/(name+'.png'))
                        if ident in review:np.save(images/f'cfg{cfg:g}_seed{seed}_raw.npy',raw.cpu().numpy(),allow_pickle=False)
                        calls+=1
                if len(rows)!=5+len(specifications)*3:raise ValueError('Image-profile candidate coverage mismatch')
                atomic_csv(folder/'metrics.csv',pd.DataFrame(rows));atomic_csv(folder/'checks.csv',pd.DataFrame(checks))
                atomic_csv(folder/'raw_hashes.csv',pd.DataFrame(raw_hashes))
                files=list(folder.glob('*.csv'))+(list(images.glob('*')) if images.exists() else [])
                atomic_json(marker,dict(protocol_sha256=signature,sampling_calls=len(cfgs)*3,files={p.relative_to(run).as_posix():sha(p) for p in files}))
            done_count+=1
            atomic_json(run/'status.json',dict(completed=False,stage=settings['stage'],completed_image_profiles=done_count,
                total_image_profiles=len(records)*len(profiles),sampling_calls=calls,elapsed_seconds=time.perf_counter()-start_all))
            # Per-profile files commit every time. Avoid quadratic full-table rewrites on 1,000 images.
            if settings['smoke'] or done_count%25==0:
                collect(run);export(run,job)
            print(f'Completed {done_count}/{len(records)*len(profiles)} image-profiles; {ident}/{profile["name"]}; calls={calls}',flush=True)
            torch.cuda.empty_cache()
        del target
    collect(run)
    collected=pd.read_csv(run/'results.csv')
    expected_names={c['strategy'] for c in specifications}|{'input_y','cleaner','vae_x_oracle','vae_y','vae_z'}
    keys=['dataset','image_id','profile','strategy','seed']
    if collected.duplicated(keys).any() or set(collected.strategy)!=expected_names:
        raise ValueError('Duplicate or missing candidate/seed keys')
    coverage=collected.groupby(['image_id','profile','strategy']).seed.nunique()
    expected_replicates=np.array([1 if k[2] in ['input_y','cleaner','vae_x_oracle','vae_y','vae_z'] else 3 for k in coverage.index])
    if not np.array_equal(coverage.to_numpy(),expected_replicates):raise ValueError('Incomplete matched seed coverage')
    if calls!=len(records)*len(profiles)*len(cfgs)*3:raise ValueError('Sampling call count differs from the frozen budget')
    means=analyze_run(run,settings['stage'],signature)
    if settings['stage']=='development' and not settings['smoke']:
        preference=freeze_preferences(means,signature,run/'frozen_preferences.json')
        preference['source_hashes']=fixed['code_hashes']
        preference['runtime_fingerprint']=runtime
        preference['sha256']=digest({k:v for k,v in preference.items() if k!='sha256'})
        atomic_json(run/'frozen_preferences.json',preference)
    if preference:
        for profile,g in means.groupby('profile'):
            front,hv=fronts_and_hv(g,preference['normalization'])
            front.to_csv(run/f'{profile}_fronts.csv',index=False);hv.to_csv(run/f'{profile}_hypervolume.csv',index=False)
        # These are frozen global preference choices, never per-domain test winners.
        choices=pd.DataFrame([dict(preference=k,strategy=v) for k,v in preference['choices'].items()])
        means.merge(choices,on='strategy').groupby(['dataset','profile','preference','strategy'])[['psnr','ssim','lpips','mse','musiq']].mean().reset_index().to_csv(run/'frozen_preference_outcomes.csv',index=False)
    expected=len(records)*len(profiles)*(5+len(specifications)*3)
    if len(pd.read_csv(run/'results.csv'))!=expected:raise ValueError('Final coverage mismatch')
    atomic_json(run/'status.json',dict(completed=True,stage=settings['stage'],smoke=settings['smoke'],completed_images=len(records),
        completed_image_profiles=done_count,sampling_calls=calls,expected_sampling_calls=len(records)*len(profiles)*len(cfgs)*3,
        protocol_sha256=signature,elapsed_seconds=time.perf_counter()-start_all))
    print('Complete; verify and download:',export(run,job,True),flush=True)
