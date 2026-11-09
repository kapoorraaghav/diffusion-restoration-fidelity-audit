"""Paper 2 fresh reference-path diagnostics and gated held-out confirmation."""
import ast
import json
from pathlib import Path
import shutil
import sys
import time
import types
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import torch

from .opensource import load_diffbir,source_path,save_image,sha256,SOURCES
from .ffhq_worker import download_ffhq
from .conditioning_audit import upstream_pipeline,load_cleaner_weights
from .paper2_design import BASELINE,LEGACY,candidate,frozen_design,validate_frozen,confirmation_ids
from .metrics import bootstrap_ci
from .stage_attribution import fidelity_measures
from .three_pipeline import extra_blur


def load_edm():
    """Delay annotations to accommodate upstream torch.Tuple typo; math unchanged."""
    path=source_path('DiffBIR')/'diffbir/sampler/edm_sampler.py'
    name='diffbir.sampler.edm_sampler'
    if name not in sys.modules:
        tree=ast.parse(path.read_text())
        tree.body.insert(0,ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0))
        ast.fix_missing_locations(tree)
        module=types.ModuleType(name);module.__package__='diffbir.sampler';sys.modules[name]=module
        exec(compile(tree,str(path),'exec'),module.__dict__)
    return sys.modules[name].EDMSampler


def paper2_pipeline(model,sampler):
    pipe=upstream_pipeline(model,sampler)
    # The original Pipeline methods resolve sampler classes from their definition namespace.
    pipe.apply_cldm.__func__.__globals__['EDMSampler']=load_edm()
    return pipe


def upstream_prompts():
    tree=ast.parse((source_path('DiffBIR')/'inference.py').read_text())
    result={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ['DEFAULT_POS_PROMPT','DEFAULT_NEG_PROMPT']:
                    result[target.id]=ast.literal_eval(node.value)
    return result['DEFAULT_POS_PROMPT'],result['DEFAULT_NEG_PROMPT']


@torch.inference_mode()
def sample(pipe,condition,seed,cfg):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    prompts=upstream_prompts() if cfg['prompts']=='upstream_fixed' else ('','')
    return (pipe.apply_cldm(condition,steps=cfg['steps'],strength=cfg['strength'],
        vae_encoder_tiled=False,vae_encoder_tile_size=512,vae_decoder_tiled=False,vae_decoder_tile_size=512,
        cldm_tiled=False,cldm_tile_size=512,cldm_tile_stride=256,
        pos_prompt=prompts[0],neg_prompt=prompts[1],cfg_scale=cfg['cfg'],start_point_type=cfg['start'],
        sampler_type=cfg['sampler'],noise_aug=cfg['noise_aug'],rescale_cfg=False,
        s_churn=0.,s_tmin=0.,s_tmax=300.,s_noise=1.,eta=1.,order=1)+1)/2


def main(settings,job):
    import lpips
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    sys.path.insert(0,str(source_path('DiffBIR')))
    from diffbir.utils.common import instantiate_from_config,wavelet_reconstruction
    # Above diffbir import is resolved after the bootstrap writes and checks pinned sources.
    stage=settings['stage'];run=Path(job)/'experiment';run.mkdir(parents=True,exist_ok=False)
    if stage=='targeted':
        diagnosis=Path(settings.get('diagnosis_status',''))
        if not diagnosis.is_file():raise ValueError('Provide the completed 100-face Notebook 1 status.json before a targeted test')
        earlier=json.loads(diagnosis.read_text())
        if not earlier.get('completed') or earlier.get('stage')!='diagnosis' or earlier.get('n_images')!=100:
            raise ValueError('Targeted testing requires completed 100-face diagnosis')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    cache=Path(settings['cache_root']);models=Path(settings['model_root'])
    def checkpoint(name):return next(p for p in list(models.rglob(name))+[cache/'weights'/name] if p.is_file())
    frozen=None
    if stage=='confirmation':
        frozen=validate_frozen(json.loads(Path(settings['frozen_design']).read_text()))
        if frozen.get('development_test_complete') is not True:raise ValueError('Development test has not been frozen as complete')
        expected=Path(settings['development_status'])
        if not expected.is_file() or not json.loads(expected.read_text()).get('completed'):raise ValueError('Upload/preserve the completed development status.json before confirmation')
        state=json.loads(expected.read_text())
        if state.get('stage')!='targeted' or state.get('n_images')!=100:raise ValueError('Confirmation requires the full targeted development run')
        if sha256(expected)!=frozen.get('development_status_sha256'):raise ValueError('Development status does not match the frozen design')
        count=2000
        # Reuse the same validated downloader with an explicit fixed held-out ID provider.
        from . import ffhq_worker
        original_selection=ffhq_worker.selected_ids
        ffhq_worker.selected_ids=lambda n: confirmation_ids() if n==2000 else original_selection(n)
        selected=download_ffhq(Path(settings['gopro_root']),count)
        configs={'baseline':frozen['baseline'],'intervention':frozen['candidate']}
    else:
        count=int(settings.get('n_images',100))
        if count not in (1,100):raise ValueError('Use 1 for a host check or the fixed 100 for development')
        selected=download_ffhq(Path(settings['gopro_root']),count)
        configs={'baseline':BASELINE,'legacy_context':LEGACY} if stage=='diagnosis' else {'baseline':BASELINE,'intervention':candidate(settings['intervention'])}
        if stage=='targeted' and len(settings.get('rationale','').strip())<30:raise ValueError('Record the hypothesis before testing')
    pd.DataFrame(selected).to_csv(run/'dataset_inventory.csv',index=False)
    model,sampler=load_diffbir(checkpoint('sd2.1-base-zsnr-laionaes5.ckpt'),checkpoint('DiffBIR_v2.1.pt'),'cuda',version='v2.1')
    pipe=paper2_pipeline(model,sampler)
    face_path=hf_hub_download('lxq007/DiffBIR','face_swinir_v1.ckpt',cache_dir=str(cache/'huggingface'))
    cleaner=instantiate_from_config(OmegaConf.load(source_path('DiffBIR')/'configs/inference/swinir.yaml'))
    load_cleaner_weights(cleaner,torch.load(face_path,map_location='cpu',weights_only=True));cleaner.eval().cuda();pipe.cleaner=cleaner
    torch.hub.set_dir(str(cache/'torch_hub'))
    perceptual=lpips.LPIPS(net='alex',version='0.1').eval().cuda()
    protocol=dict(stage=stage,n_images=count,configs=configs,seeds=[11,22,33],sigma=.6,size=512,
        source_revisions=SOURCES,captioner='disabled; fixed upstream positive/negative prompts',
        baseline_role='Author pipeline methods with face cleaner and documented sampler settings; no claim of full CLI/LLAVA parity',
        preprocessing='FFHQ full-frame LANCZOS1024 to512',
        notes=['Oracle is diagnostic, not an identity guarantee','Raw/wavelet are distinct outcomes','VAEs and diffusion errors are not an additive causal decomposition',
               'No model training; no inference without Start','One primary model; second-method replication needed for broad claims'],
        checkpoints={n:sha256(checkpoint(n)) for n in ['sd2.1-base-zsnr-laionaes5.ckpt','DiffBIR_v2.1.pt']},face_cleaner_sha256=sha256(face_path),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),rationale=settings.get('rationale',''))
    (run/'protocol.json').write_text(json.dumps(protocol,indent=2))
    protocol['latent_scale_factor']=float(model.scale_factor)
    protocol['vae_parameter_dtype']=str(next(model.vae.parameters()).dtype)
    if frozen is not None:
        previous=frozen.get('development_protocol',{})
        for key in ['source_revisions','checkpoints','face_cleaner_sha256','latent_scale_factor','captioner','preprocessing']:
            if previous.get(key)!=protocol.get(key):raise ValueError(f'Confirmation differs from frozen development provenance: {key}')
    (run/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];calls=0;checks=[]
    def add(image_id,condition,configuration,post,seed,image,target,**extra):
        if not torch.isfinite(image).all():raise ValueError('Nonfinite output before measurement')
        rows.append(dict(image_id=image_id,condition=condition,configuration=configuration,postprocessing=post,seed=seed,
            **fidelity_measures(image,target,perceptual),**extra))
    with torch.inference_mode():
        for record in selected:
            image_id=record['image_id'];folder=run/image_id;folder.mkdir()
            with Image.open(record['path']) as im:a=np.array(im.convert('RGB')).copy()
            target=torch.from_numpy(a).permute(2,0,1)[None].float().cuda()/255
            observation=extra_blur(target,.6);clean=pipe.apply_cleaner(observation,False,512,256)
            if not torch.isfinite(clean).all():raise ValueError('Nonfinite cleaner output')
            save_image(target,folder/'reference.png');save_image(observation,folder/'observation.png');save_image(clean,folder/'face_cleaner.png')
            for name,condition in [('oracle',target),('face_cleaner',clean)]:
                add(image_id,name,'input','none',None,condition.clamp(0,1),target,raw_min=float(condition.min()),raw_max=float(condition.max()))
                z=model.vae_encode(condition*2-1,sample=False)
                z_again=model.prepare_condition(condition,[''])['c_img']
                error=float((z-z_again).abs().max())
                checks.append(dict(image_id=image_id,condition=name,check='vae_scale_and_normalization_entry_parity',max_abs=error,passed=error<=1e-5))
                if error>1e-5:raise ValueError('VAE normalization/scale entry paths disagree')
                decoded=(model.vae_decode(z)+1)/2
                add(image_id,name,'vae_roundtrip','raw',None,decoded.clamp(0,1),target)
                save_image(decoded,folder/f'{name}_vae.png')
                for cfg_name,cfg in configs.items():
                    model.cast_dtype(torch.float32 if cfg['precision']=='fp32' else torch.float16)
                    for seed in [11,22,33]:
                        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                        counter={'calls':0,'batch_evaluations':0}
                        def count_forward(module,args):
                            counter['calls']+=1;counter['batch_evaluations']+=int(args[0].shape[0])
                        handle=model.register_forward_pre_hook(count_forward)
                        try:raw=sample(pipe,condition,seed,cfg)
                        finally:handle.remove()
                        torch.cuda.synchronize();seconds=time.perf_counter()-start;calls+=1
                        if not torch.isfinite(raw).all():raise ValueError('Nonfinite raw decode before clipping')
                        for post,output in [('raw',raw.clamp(0,1)),('wavelet',wavelet_reconstruction(raw,condition).clamp(0,1))]:
                            add(image_id,name,cfg_name,post,seed,output,target,seconds=seconds,
                                denoiser_forward_calls=counter['calls'],denoiser_batch_evaluations=counter['batch_evaluations'],
                                peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3)
                            # Every output from confirmation is saved for later structural review.
                            save_image(output,folder/f'{name}_{cfg_name}_seed{seed}_{post}.png')
                        pd.DataFrame(rows).to_csv(run/'results.csv',index=False)
                        pd.DataFrame(checks).to_csv(run/'checks.csv',index=False)
                        (run/'status.json').write_text(json.dumps(dict(completed=False,last_image=image_id,sampling_calls=calls)))
            del condition,target,observation,clean,z,z_again,raw,decoded,output;torch.cuda.empty_cache()
    frame=pd.DataFrame(rows)
    fields=['psnr','ssim','lpips','mse']
    image_means=frame.groupby(['image_id','condition','configuration','postprocessing'])[fields].mean().reset_index()
    image_means.to_csv(run/'image_means.csv',index=False)
    image_means.groupby(['condition','configuration','postprocessing'])[fields].mean().to_csv(run/'summary.csv')
    metrics=[]
    for keys,g in image_means.groupby(['condition','configuration','postprocessing']):
        for metric in fields:metrics.append(dict(condition=keys[0],configuration=keys[1],postprocessing=keys[2],metric=metric,**bootstrap_ci(g[metric])))
    pd.DataFrame(metrics).to_csv(run/'image_bootstrap.csv',index=False)
    if stage in ['targeted','confirmation']:
        b=image_means[image_means.configuration=='baseline'];v=image_means[image_means.configuration=='intervention']
        paired=b.merge(v,on=['image_id','condition','postprocessing'],suffixes=('_baseline','_intervention'),validate='one_to_one')
        for metric in fields:paired['delta_'+metric]=paired[metric+'_intervention']-paired[metric+'_baseline']
        paired.to_csv(run/'paired_effects.csv',index=False)
        ci=[]
        for keys,g in paired.groupby(['condition','postprocessing']):
            for metric in fields:ci.append(dict(condition=keys[0],postprocessing=keys[1],metric=metric,**bootstrap_ci(g['delta_'+metric])))
        pd.DataFrame(ci).to_csv(run/'paired_bootstrap.csv',index=False)
    (run/'status.json').write_text(json.dumps(dict(completed=True,n_images=count,sampling_calls=calls,stage=stage)))
    root=Path(settings['output_root']);label=settings['run_label']
    for path in run.glob('*.csv'):shutil.copy2(path,root/f'{label}_PAPER2_{path.name}')
    # Images stay in the job folder; export ZIPs per image to avoid a giant browser download.
    archive_dir=root/f'{label}_image_archives';archive_dir.mkdir(exist_ok=True)
    for record in selected:
        folder=run/record['image_id']
        with zipfile.ZipFile(archive_dir/(folder.name+'.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
            for file in folder.glob('*.png'):archive.write(file,file.name)
    print('Paper2 run completed',run,'sampling calls',calls,flush=True)
