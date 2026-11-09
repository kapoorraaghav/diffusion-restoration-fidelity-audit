"""Resumable, per-image cross-domain CFG/conditioning/postprocessing experiment."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import torch

from .cross_domain_data import prepare, sha
from .paper2_design import BASELINE, candidate, digest
from .paper2_worker import paper2_pipeline, sample
from .opensource import load_diffbir, source_path, save_image, SOURCES
from .conditioning_audit import load_cleaner_weights
from .stage_attribution import fidelity_measures
from .three_pipeline import extra_blur
from .metrics import bootstrap_ci

EXPECTED_MODELS = {
    'sd2.1-base-zsnr-laionaes5.ckpt': '30bdda1968c2dae28ff23b2b44b0f2a87058d76eea5fc4eadfa3448e40e85cf9',
    'DiffBIR_v2.1.pt': '49477de737be9028ac2d9352cf45633324a0090afde57a7aa358321dd6a1de53',
}
FACE_HASH = '62fb6605702e6ad35712299a23cc19ea7cd2fb4bbd50a8080fcefa8ad2620c71'
GENERAL_HASH = '577b8f84815d324e5ee17ef3c85d8bc9b7ebd05b20a6d61a5452b66e7e24dcaf'


def atomic_json(path, value):
    temp = Path(str(path)+'.partial')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8'); temp.replace(path)


def atomic_csv(path, frame):
    temp = Path(str(path)+'.partial'); frame.to_csv(temp,index=False); temp.replace(path)


def export_tables(run, job):
    destination = Path(job).parent.parent/'cross_domain_downloads'
    destination.mkdir(parents=True,exist_ok=True)
    path = destination/(Path(job).name+'_tables.zip')
    temp = path.with_suffix('.partial')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(run.iterdir()):
            if p.is_file() and p.suffix in ('.csv','.json','.txt'):
                archive.write(p,'experiment/'+p.name)
        for name in ('settings.json','job.json','exit.json','worker.py','launcher.py','worker.log'):
            p=Path(job)/name
            if p.is_file(): archive.write(p,'job_logs/'+name)
    temp.replace(path)
    return str(path)


def collect(run):
    folders = sorted(p for p in run.iterdir() if p.is_dir() and (p/'image_complete.json').exists())
    if not folders: return None
    results=pd.concat([pd.read_csv(p/'metrics.csv') for p in folders],ignore_index=True)
    checks=pd.concat([pd.read_csv(p/'checks.csv') for p in folders],ignore_index=True)
    atomic_csv(run/'results.csv',results);atomic_csv(run/'checks.csv',checks)
    return results


def summarize(run,frame):
    fields=['psnr','ssim','lpips','mse']
    means=frame.groupby(['image_id','condition','configuration','postprocessing'])[fields].mean().reset_index()
    atomic_csv(run/'image_means.csv',means)
    atomic_csv(run/'summary.csv',means.groupby(['condition','configuration','postprocessing'])[fields].mean().reset_index())
    b=means[means.configuration=='baseline'];v=means[means.configuration=='intervention']
    paired=b.merge(v,on=['image_id','condition','postprocessing'],suffixes=('_baseline','_intervention'),validate='one_to_one')
    for metric in fields:paired['delta_'+metric]=paired[metric+'_intervention']-paired[metric+'_baseline']
    atomic_csv(run/'paired_effects.csv',paired)
    rows=[]
    for keys,g in paired.groupby(['condition','postprocessing']):
        for metric in fields:rows.append(dict(condition=keys[0],postprocessing=keys[1],metric=metric,**bootstrap_ci(g['delta_'+metric])))
    atomic_csv(run/'paired_bootstrap.csv',pd.DataFrame(rows))
    # Conditioning contrasts use the same image and seed-averaged configuration/postprocessing.
    contrasts=[]
    for left,right in [('degraded','restored'),('oracle','restored'),('degraded','oracle')]:
        if left not in set(means.condition) or right not in set(means.condition):continue
        pair=means[means.condition==left].merge(means[means.condition==right],on=['image_id','configuration','postprocessing'],suffixes=('_left','_right'),validate='one_to_one')
        for metric in fields:pair['delta_'+metric]=pair[metric+'_right']-pair[metric+'_left']
        pair['contrast']=right+' minus '+left;contrasts.append(pair)
    if contrasts:
        pairs=pd.concat(contrasts,ignore_index=True);atomic_csv(run/'conditioning_effects.csv',pairs)
        rows=[]
        for keys,g in pairs.groupby(['contrast','configuration','postprocessing']):
            for metric in fields:rows.append(dict(contrast=keys[0],configuration=keys[1],postprocessing=keys[2],metric=metric,**bootstrap_ci(g['delta_'+metric])))
        atomic_csv(run/'conditioning_bootstrap.csv',pd.DataFrame(rows))


def main(settings,job):
    import lpips
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    sys.path.insert(0,str(source_path('DiffBIR')))
    from diffbir.utils.common import instantiate_from_config,wavelet_reconstruction
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    run=Path(job)/'experiment';run.mkdir(exist_ok=True)
    records=prepare(settings)
    # Smoke runs are separate labels and never a substitute for full evaluation.
    if settings['smoke']:records=records[:1]
    configs=dict(baseline=BASELINE,intervention=candidate('cfg1'))
    conditions=['degraded'] if settings['dataset']=='ffhq' else ['oracle','restored','degraded']
    cache=Path(settings['cache_root']);models=Path(settings['model_root'])
    def checkpoint(name):return next(p for p in list(models.rglob(name))+[cache/'weights'/name] if p.is_file())
    hashes={name:sha(checkpoint(name)) for name in EXPECTED_MODELS}
    if hashes!=EXPECTED_MODELS:raise ValueError('Diffusion weights differ from completed FFHQ run')
    dataset=settings['dataset']
    cleaner_file='face_swinir_v1.ckpt' if dataset=='ffhq' else 'realesrgan_s4_swinir_100k.pth'
    cleaner_repo='lxq007/DiffBIR' if dataset=='ffhq' else 'lxq007/DiffBIR-v2'
    cleaner_path=hf_hub_download(cleaner_repo,cleaner_file,cache_dir=str(cache/'huggingface'))
    cleaner_hash=sha(cleaner_path)
    if dataset=='ffhq' and cleaner_hash!=FACE_HASH:raise ValueError('FFHQ face-cleaner hash changed')
    if dataset!='ffhq' and cleaner_hash!=GENERAL_HASH:raise ValueError('General cleaner hash changed')
    original_ffhq = None
    if dataset=='ffhq':
        with zipfile.ZipFile(settings['ffhq_tables']) as archive:
            old_protocol=json.loads(archive.read('experiment/protocol.json'))
            old_status=json.loads(archive.read('experiment/status.json'))
            original_ffhq=pd.read_csv(archive.open('experiment/dataset_inventory.csv'))
        if not old_status.get('completed') or old_status.get('n_images')!=2000:raise ValueError('Upload the completed FFHQ2000 tables ZIP')
        if old_protocol['configs']!=configs or old_protocol['checkpoints']!=hashes or old_protocol['face_cleaner_sha256']!=FACE_HASH:
            raise ValueError('FFHQ add-on differs from historical protocol')
        if old_protocol['seeds']!=[11,22,33] or old_protocol['sigma']!=.6 or old_protocol['size']!=512 or old_protocol['source_revisions']!=SOURCES:
            raise ValueError('Historical FFHQ sampling/data protocol differs')
        old_refs=original_ffhq.set_index('image_id')['sha256'].to_dict()
        if any(old_refs.get(r['image_id'])!=r['sha256'] for r in records):raise ValueError('FFHQ reference pixels differ from completed run')
    code_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    version_names=['torch','torchvision','numpy','pandas','pillow','lpips','timm','omegaconf','einops','ftfy','regex','torchsde','huggingface-hub']
    core_versions={name:importlib.metadata.version(name) for name in version_names}
    fixed=dict(version=1,dataset=dataset,smoke=settings['smoke'],n_images=len(records),conditions=conditions,
        configs=configs,seeds=[11,22,33],sigma=.6,size=512,source_revisions=SOURCES,checkpoints=hashes,
        cleaner=dict(repo=cleaner_repo,file=cleaner_file,sha256=cleaner_hash),code_hashes=code_hashes,
        selection=[dict(image_id=r['image_id'],sha256=r['sha256']) for r in records],
        preprocessing=records[0]['preprocessing'],torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        python=platform.python_version(),cuda=torch.version.cuda,core_package_versions=core_versions,
        primary_postprocessing='wavelet',primary_metrics=['psnr','lpips'],supporting_metrics=['ssim','mse'],
        notes=['Fixed Gaussian0.6 at512, not standard published SR protocol',
               'Oracle is a diagnostic, not identity enforcement','Marginal95% image-bootstrap intervals; no familywise claim',
               'Natural-image center crop differs from historical FFHQ full frame; domain is not sole causal difference',
               'General cleaner trained for restoration/SR, not a matched Gaussian deblurring specialist',
               'Degraded branch on FFHQ is an additive experiment; old branches retained separately'])
    signature=digest(fixed)
    protocol_path=run/'protocol.json'
    if protocol_path.exists():
        previous=json.loads(protocol_path.read_text())
        if previous['protocol_sha256']!=signature:raise ValueError('Resume protocol/environment/source mismatch; do not mix runs')
    else:
        atomic_json(protocol_path,dict(**fixed,protocol_sha256=signature))
    atomic_csv(run/'dataset_inventory.csv',pd.DataFrame(records))
    environment={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
    atomic_json(run/'environment.json',dict(python=platform.python_version(),cuda=torch.version.cuda,packages=environment))
    model,sampler=load_diffbir(checkpoint('sd2.1-base-zsnr-laionaes5.ckpt'),checkpoint('DiffBIR_v2.1.pt'),'cuda',version='v2.1')
    pipe=paper2_pipeline(model,sampler)
    cleaner=instantiate_from_config(OmegaConf.load(source_path('DiffBIR')/'configs/inference/swinir.yaml'))
    load_cleaner_weights(cleaner,torch.load(cleaner_path,map_location='cpu',weights_only=True));cleaner.eval().cuda();pipe.cleaner=cleaner
    torch.hub.set_dir(str(cache/'torch_hub'))
    perceptual=lpips.LPIPS(net='alex',version='0.1').eval().cuda()
    protocol=json.loads(protocol_path.read_text());protocol.update(latent_scale_factor=float(model.scale_factor),vae_parameter_dtype=str(next(model.vae.parameters()).dtype))
    atomic_json(protocol_path,protocol)
    expected_per_image=len(conditions)*14
    for index,record in enumerate(records):
        folder=run/record['image_id'];folder.mkdir(exist_ok=True)
        marker=folder/'image_complete.json'
        if marker.exists():
            done=json.loads(marker.read_text())
            if done['protocol_sha256']!=signature or sha(folder/'metrics.csv')!=done['metrics_sha256'] or sha(folder/'checks.csv')!=done['checks_sha256']:
                raise ValueError('Completed image checkpoint mismatch')
            images_intact=all((folder/name).is_file() and sha(folder/name)==checksum for name,checksum in done['image_sha256'].items())
            if images_intact:
                print('Resume: verified completed',record['image_id'],flush=True)
                continue
            print('Missing/damaged images; regenerating this face with the frozen settings',record['image_id'],flush=True)
        rows=[];checks=[]
        def add(name,configuration,post,seed,image,target,**extra):
            if not torch.isfinite(image).all():raise ValueError('Nonfinite output')
            rows.append(dict(image_id=record['image_id'],condition=name,configuration=configuration,postprocessing=post,seed=seed,
                **fidelity_measures(image,target,perceptual),**extra))
        with torch.inference_mode():
            with Image.open(record['path']) as im:array=np.array(im.convert('RGB')).copy()
            target=torch.from_numpy(array).permute(2,0,1)[None].float().cuda()/255
            observed=extra_blur(target,.6)
            clean=pipe.apply_cleaner(observed,False,512,256)
            save_image(target,folder/'reference.png');save_image(observed,folder/'observation.png');save_image(clean,folder/'cleaner.png')
            for name in conditions:
                condition={'oracle':target,'degraded':observed,'restored':clean}[name]
                add(name,'input','none',None,condition.clamp(0,1),target,raw_min=float(condition.min()),raw_max=float(condition.max()))
                latent=model.vae_encode(condition*2-1,sample=False)
                alternate=model.prepare_condition(condition,[''])['c_img']
                error=float((latent-alternate).abs().max())
                checks.append(dict(image_id=record['image_id'],condition=name,max_abs=error,passed=error<=1e-5))
                if error>1e-5:raise ValueError('VAE entry parity failed')
                decoded=(model.vae_decode(latent)+1)/2
                add(name,'vae_roundtrip','raw',None,decoded.clamp(0,1),target);save_image(decoded,folder/f'{name}_vae.png')
                for config_name,config in configs.items():
                    model.cast_dtype(torch.float16)
                    for seed in [11,22,33]:
                        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                        raw=sample(pipe,condition,seed,config);torch.cuda.synchronize();seconds=time.perf_counter()-start
                        if not torch.isfinite(raw).all():raise ValueError('Nonfinite raw decode')
                        for post,image in [('raw',raw.clamp(0,1)),('wavelet',wavelet_reconstruction(raw,condition).clamp(0,1))]:
                            add(name,config_name,post,seed,image,target,seconds=seconds,peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3)
                            save_image(image,folder/f'{name}_{config_name}_seed{seed}_{post}.png')
            del target,observed,clean,condition,latent,alternate,decoded,raw,image;torch.cuda.empty_cache()
        frame=pd.DataFrame(rows)
        if len(frame)!=expected_per_image:raise ValueError('Incomplete image metric coverage')
        atomic_csv(folder/'metrics.csv',frame);atomic_csv(folder/'checks.csv',pd.DataFrame(checks))
        atomic_json(marker,dict(protocol_sha256=signature,metrics_sha256=sha(folder/'metrics.csv'),checks_sha256=sha(folder/'checks.csv'),
            image_sha256={p.name:sha(p) for p in folder.glob('*.png')}))
        atomic_json(run/'status.json',dict(completed=False,n_images=len(records),completed_images=index+1,sampling_calls=(index+1)*len(conditions)*6,stage='cross_domain',dataset=dataset))
        print(f'Completed {index+1}/{len(records)} {record["image_id"]}',flush=True)
        if (index+1)%25==0:
            collect(run);export_tables(run,job)
    frame=collect(run)
    if frame is None or len(frame)!=len(records)*expected_per_image:raise ValueError('Final result coverage mismatch')
    summarize(run,frame)
    atomic_json(run/'status.json',dict(completed=True,n_images=len(records),sampling_calls=len(records)*len(conditions)*6,stage='cross_domain',dataset=dataset))
    # Fixed small visual export selected independently of metric outcomes.
    random=np.random.default_rng(20261031).permutation(len(records))[:min(20,len(records))]
    atomic_json(run/'visual_selection.json',dict(seed=20261031,image_ids=[records[i]['image_id'] for i in random]))
    destination=Path(job).parent.parent/'cross_domain_downloads';destination.mkdir(exist_ok=True)
    visual=destination/(Path(job).name+'_visual20.zip');temp=visual.with_suffix('.partial')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_STORED,allowZip64=True) as archive:
        archive.write(run/'visual_selection.json','visual_selection.json')
        for i in random:
            folder=run/records[i]['image_id']
            for file in folder.glob('*.png'):archive.write(file,folder.name+'/'+file.name)
    temp.replace(visual)
    print('Completed. Download tables AND visual ZIP before session shutdown:',export_tables(run,job),visual,flush=True)
