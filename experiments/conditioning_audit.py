"""Expanded paired conditioning audit; no automatic hallucination ground truth."""
import ast
import importlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from .opensource import nafnet, load_diffbir, refine, save_image, sha256, source_path, SOURCES
from .three_pipeline import extra_blur
from .stage_attribution import fidelity_measures
from .metrics import edges

SEEDS=(11,22,33)
SIGMAS=(0.,.6,1.2)


def controlled_error(prior, amplitude=.03):
    """Fixed local edge strokes, independent of reference/outcomes; amplitude before clipping."""
    mask=torch.zeros_like(prior)
    h,w=prior.shape[-2:]
    # Fixed central-quarter ROI, dashed horizontal/vertical strokes, all RGB channels.
    y0,x0=h//4,w//4
    for offset in range(0,h//4,16):
        mask[:,:,y0+offset:y0+offset+2,x0:x0+w//4]=1
    for offset in range(0,w//4,16):
        mask[:,:,y0:y0+h//4,x0+offset:x0+offset+2]=1
    return (prior+amplitude*mask).clamp(0,1),mask


def upstream_pipeline(model,sampler):
    """Run original Pipeline methods, bypassing unrelated broken eager sampler imports.

    Only the sampler import statement is replaced. The selected DDIM/spaced
    algorithm and original stage-2 pipeline methods are unmodified.
    """
    root=source_path('DiffBIR')
    from diffbir.sampler.spaced_sampler import SpacedSampler
    from diffbir.model.gaussian_diffusion import Diffusion
    from omegaconf import OmegaConf
    source=root/'diffbir/pipeline.py'
    tree=ast.parse(source.read_text())
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.level==1 and n.module=='sampler')]
    ns={'__name__':'diffbir._conditioning_audit_pipeline','__package__':'diffbir',
        'DDIMSampler':type(sampler),'SpacedSampler':SpacedSampler}
    exec(compile(tree,str(source),'exec'),ns)
    cfg=OmegaConf.load(root/'configs/inference/diffusion_v2.1.yaml')
    diffusion=Diffusion(**cfg.params)
    return ns['SwinIRPipeline'](None,model,diffusion,None,'cuda')


@torch.inference_mode()
def pipeline_sample(pipe,condition,seed,sampler_type='spaced'):
    if sampler_type not in {'spaced','ddim'}:raise ValueError('Only verified import paths supported')
    # Seed immediately before upstream entry; prepare_condition uses deterministic VAE mode.
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    return (pipe.apply_cldm(condition,steps=20,strength=1.,
        vae_encoder_tiled=False,vae_encoder_tile_size=512,
        vae_decoder_tiled=False,vae_decoder_tile_size=512,
        cldm_tiled=False,cldm_tile_size=512,cldm_tile_stride=256,
        pos_prompt='',neg_prompt='',cfg_scale=1.,start_point_type='noise',
        sampler_type=sampler_type,noise_aug=0,rescale_cfg=False,
        s_churn=0.,s_tmin=0.,s_tmax=float('inf'),s_noise=1.,eta=0.,order=1)+1)/2


def download_kodak(destination,count):
    if count not in (8,16,24):raise ValueError('Choose 8, 16, or 24 images')
    destination.mkdir(parents=True,exist_ok=True)
    records=[]
    for index in range(1,25):
        path=destination/f'kodim{index:02d}.png'
        url=f'https://r0k.us/graphics/kodak/kodak/kodim{index:02d}.png'
        valid=False
        if path.exists():
            try:
                with Image.open(path) as im:im.verify()
                valid=True
            except (OSError,ValueError):pass
        if not valid:
            temp=path.with_suffix('.part')
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 research'})
            with urllib.request.urlopen(req,timeout=90) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst)
            with Image.open(temp) as im:im.verify()
            temp.replace(path)
        with Image.open(path) as im:
            w,h=im.size
            if min(w,h)<512:raise ValueError('Unexpected reference dimensions')
            box=[(w-512)//2,(h-512)//2,(w-512)//2+512,(h-512)//2+512]
        records.append(dict(image_id=path.stem,path=str(path),box=box,sha256=sha256(path),source=url))
    # Fixed ID selection for smaller debugging runs, not outcome-selected.
    ids={8:[1,3,5,8,9,12,13,23],16:list(range(1,17)),24:list(range(1,25))}[count]
    return [r for r in records if int(r['image_id'][-2:]) in ids]


def load_cleaner_weights(cleaner, checkpoint):
    """Unwrap the author checkpoint, as upstream load_model_from_url does."""
    state=checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    if state and all(k.startswith('module.') for k in state):
        state={k.removeprefix('module.'):v for k,v in state.items()}
    cleaner.load_state_dict(state,strict=True)


def load_cleaner(cache):
    """Author-released DiffBIR v2.1 general/SR cleaner; controlled comparison, not deblur benchmark."""
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    from diffbir.utils.common import instantiate_from_config
    path=hf_hub_download('lxq007/DiffBIR-v2','realesrgan_s4_swinir_100k.pth',cache_dir=str(cache/'huggingface'))
    cleaner=instantiate_from_config(OmegaConf.load(source_path('DiffBIR')/'configs/inference/swinir.yaml'))
    load_cleaner_weights(cleaner,torch.load(path,map_location='cpu',weights_only=True))
    return cleaner.eval().cuda(),dict(path=path,sha256=sha256(path),repo='lxq007/DiffBIR-v2',file='realesrgan_s4_swinir_100k.pth')


def main(settings,job,dataset_loader=None,dataset_name='Kodak24',export_tag='AUDIT'):
    import lpips
    from .reference_review import panel,heat,edge_masks
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    run=Path(job)/'experiment';run.mkdir(parents=True,exist_ok=False)
    count=int(settings['n_images']);cache=Path(settings['cache_root']);models=Path(settings['model_root'])
    selected=(dataset_loader or download_kodak)(Path(settings['gopro_root']),count)
    pd.DataFrame(selected).to_csv(run/'dataset_inventory.csv',index=False)
    def checkpoint(name):
        return next(p for p in list(models.rglob(name))+[cache/'weights'/name] if p.is_file())
    def status(**kw):
        (run/'status.json').write_text(json.dumps(dict(completed=False,**kw)))
    protocol=dict(dataset=dataset_name,n_images=count,seeds=SEEDS,sigmas=SIGMAS,size=512,steps=20,
        sources=SOURCES,conditioning=['sharp_oracle','cnn','cnn_plus_fixed_error'],
        added_error='positive 0.03 RGB strokes in fixed [128:256,128:256] ROI, spacing16 width2; preclip amplitude',
        sampling=['legacy_ddim20','upstream_spaced20'],cfg_scale=1,prompts='',strength=1,
        outputs=['raw_clipped','wavelet_then_clipped'],
        cleaner_reference='author SwinIR + upstream spaced20 + wavelet, controlled settings, NOT CLI defaults or deblurring-trained baseline',
        limitations=['Sharp oracle is diagnostic only','Synthetic Gaussian blur differs from NAFNet motion-blur training',
          'Spaced vs DDIM changes the sampler/schedule, not just integration',
          'No automatic hallucination labels; selected patches cannot estimate prevalence',
          f'{count} image units; seeds/blur variants are repeated measurements, not independent images',
          'Dataset changes do not isolate semantic context as a causal factor'],
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        models={n:sha256(checkpoint(n)) for n in ['NAFNet-GoPro-width32.pth','sd2.1-base-zsnr-laionaes5.ckpt','DiffBIR_v2.1.pt']})
    (run/'protocol.json').write_text(json.dumps(protocol,indent=2))
    diagnostics=[]
    cnn=nafnet(checkpoint('NAFNet-GoPro-width32.pth'),'cuda')
    with torch.inference_mode():
        for record in selected:
            with Image.open(record['path']) as im:a=np.array(im.convert('RGB').crop(record['box'])).copy()
            target=torch.from_numpy(a).permute(2,0,1)[None].float().cuda()/255
            cases=[]
            for sigma in SIGMAS:
                observation=extra_blur(target,sigma);raw=cnn(observation)
                if not torch.isfinite(raw).all():raise ValueError('Nonfinite CNN tensor; run stopped')
                prior=raw.clamp(0,1)
                diagnostics.append(dict(image_id=record['image_id'],sigma=sigma,
                    raw_min=float(raw.min()),raw_max=float(raw.max()),
                    out_of_range_fraction=float(((raw<0)|(raw>1)).float().mean()),
                    cnn_mse=float((prior-target).square().mean())))
                cases.append(dict(sigma=sigma,observation=observation.cpu(),prior=prior.cpu()))
            torch.save(dict(target=target.cpu(),cases=cases),run/f"prepared_{record['image_id']}.pt")
            status(stage='CNN preparation',last_image=record['image_id'])
    pd.DataFrame(diagnostics).to_csv(run/'cnn_diagnostics.csv',index=False)
    del cnn,raw,prior,observation,target;torch.cuda.empty_cache()
    torch.hub.set_dir(str(cache/'torch_hub'))
    perceptual=lpips.LPIPS(net='alex',version='0.1').eval().cuda()
    model,sampler=load_diffbir(checkpoint('sd2.1-base-zsnr-laionaes5.ckpt'),checkpoint('DiffBIR_v2.1.pt'),'cuda',version='v2.1')
    # Verify the extra checkpoint before committing to the large sampling run.
    cleaner,info=load_cleaner(cache);cleaner=cleaner.cpu();torch.cuda.empty_cache()
    protocol['author_cleaner']=info
    (run/'protocol.json').write_text(json.dumps(protocol,indent=2))
    pipe=upstream_pipeline(model,sampler)
    from diffbir.utils.common import wavelet_reconstruction
    rows=[];checks=[];patch_rows=[];comparisons=[];calls=0
    def flush():
        pd.DataFrame(rows).to_csv(run/'results.csv',index=False)
        pd.DataFrame(patch_rows).to_csv(run/'candidate_patches.csv',index=False)
        pd.DataFrame(checks).to_csv(run/'checks.csv',index=False)
    with torch.inference_mode():
        # Check wrapper parity under the SAME sampler/settings before comparing different samplers.
        first=torch.load(run/f"prepared_{selected[0]['image_id']}.pt",weights_only=True)
        condition=first['cases'][1]['prior'].cuda()
        wrapped,_=refine(model,sampler,condition,20,11,'cuda')
        direct=pipeline_sample(pipe,condition,11,'ddim').clamp(0,1);calls+=2
        difference=float((wrapped-direct).abs().max())
        checks.append(dict(check='wrapper_vs_upstream_same_ddim',max_abs=difference,passed=difference<=1e-5))
        flush()
        if difference>1e-5:raise ValueError(f'Wrapper parity failed ({difference}); results saved, main audit stopped')
        del wrapped,direct,condition,first
        for record in selected:
            image_id=record['image_id'];folder=run/image_id;folder.mkdir()
            item=torch.load(run/f'prepared_{image_id}.pt',weights_only=True)
            target=item['target'].cuda();save_image(target,folder/'sharp.png')
            def add(method,condition_name,sigma,seed,post,image,**extra):
                if not torch.isfinite(image).all():raise ValueError('Nonfinite image')
                row=dict(image_id=image_id,condition=condition_name,sigma=sigma,seed=seed,
                    method=method,postprocessing=post,**fidelity_measures(image,target,perceptual),**extra)
                rows.append(row)
                return row
            # Oracle appears once/image/seed; do not duplicate by observation blur.
            conditions=[('sharp_oracle',-1.,target,None)]
            for case in item['cases']:
                sigma=case['sigma'];prior=case['prior'].cuda()
                add('observation','observation',sigma,None,'none',case['observation'].cuda())
                perturbed,mask=controlled_error(prior)
                assert not (controlled_error(prior,0)[0]-prior).abs().max()
                save_image(case['observation'],folder/f'observation_s{sigma}.png')
                save_image(prior,folder/f'cnn_s{sigma}.png')
                save_image(perturbed,folder/f'cnn_error_s{sigma}.png')
                save_image(mask,folder/f'injected_mask_s{sigma}.png')
                conditions.extend([('cnn',sigma,prior,None),('cnn_plus_fixed_error',sigma,perturbed,mask)])
            for name,sigma,condition,mask in conditions:
                add('conditioning_input',name,sigma,None,'none',condition,
                    injected_rms=0. if mask is None else float((condition-next(c['prior'].cuda() for c in item['cases'] if c['sigma']==sigma)).square().mean().sqrt()))
                vae=(model.vae_decode(model.vae_encode(condition*2-1,sample=False))+1)/2
                add('vae_roundtrip',name,sigma,None,'raw_clipped',vae.clamp(0,1))
                for seed in SEEDS:
                    for mode in ['legacy_ddim20','upstream_spaced20']:
                        start=time.perf_counter()
                        if mode=='legacy_ddim20':
                            # Original wrapper clips only at final decode; use upstream same-DDIM entry
                            # after verified parity to retain unbounded decode for fair colour correction.
                            raw=pipeline_sample(pipe,condition,seed,'ddim')
                        else:raw=pipeline_sample(pipe,condition,seed,'spaced')
                        if not torch.isfinite(raw).all():raise ValueError('Nonfinite diffusion decode before clipping')
                        torch.cuda.synchronize();seconds=time.perf_counter()-start;calls+=1
                        for post,output in [('raw_clipped',raw.clamp(0,1)),('wavelet_then_clipped',wavelet_reconstruction(raw,condition).clamp(0,1))]:
                            squared=(output-target).square().mean(1,keepdim=True)
                            region=torch.zeros_like(squared);region[:,:,128:256,128:256]=1
                            add(mode,name,sigma,seed,post,output,seconds=seconds,
                                mse_fixed_roi=float((squared*region).sum()/region.sum()),
                                mse_outside_fixed_roi=float((squared*(1-region)).sum()/(1-region).sum()))
                            # Save every raw seed, and seed11 wavelet outputs. Metrics cover all outputs.
                            tag=f'{name}_s{sigma}_{mode}_seed{seed}_{post}'
                            if post=='raw_clipped' or seed==11:save_image(output,folder/f'{tag}.png')
                            if seed==11 and post=='raw_clipped':
                                images=[Image.open(folder/'sharp.png'),Image.fromarray(np.uint8(condition[0].cpu().permute(1,2,0).numpy()*255)),Image.open(folder/f'{tag}.png')]
                                error=(output-target).abs().mean(1)[0].cpu().numpy()
                                panel(images+[heat(error)],['Sharp reference','Conditioning input',mode,'Absolute error (scale .1)']).save(folder/f'{tag}_comparison.png')
                                comparisons.append(f'{image_id}/{tag}_comparison.png')
                                mse=(output-target).square().mean(1)[0].cpu().numpy()
                                base_mse=(condition-target).square().mean(1)[0].cpu().numpy()
                                # Error-based candidate selection only. Human structural labels remain blank.
                                ranked=sorted([(float((mse-base_mse)[y:y+128,x:x+128].mean()),x,y) for y in range(0,512,128) for x in range(0,512,128)],reverse=True)[:2]
                                # One deterministic random grid patch as an outcome-independent control.
                                # Same location across conditioning variants/samplers for an image.
                                rng=np.random.default_rng(int(image_id[-2:]))
                                x,y=(rng.integers(0,4,size=2)*128).tolist()
                                random_coord=(x,y)
                                if (x,y) not in [(p[1],p[2]) for p in ranked]:
                                    ranked.append((float((mse-base_mse)[y:y+128,x:x+128].mean()),x,y))
                                for rank,(delta,x,y) in enumerate(ranked,1):
                                    box=(x,y,x+128,y+128);path=f'{tag}_patch{rank}.png'
                                    panel([im.crop(box) for im in images],['Sharp reference','Conditioning input','Diffusion']).save(folder/path)
                                    patch_rows.append(dict(image_id=image_id,condition=name,sigma=sigma,seed=seed,method=mode,x=x,y=y,width=128,height=128,delta_mse=delta,
                                        selection='largest_mse_increase_and_random_control' if (x,y)==random_coord and rank<=2 else ('random_control' if (x,y)==random_coord else 'largest_mse_increase'),
                                        intersects_injection=bool(mask is not None and mask[:,:,y:y+128,x:x+128].any()),patch_file=f'{image_id}/{path}',reviewer='',label='',notes=''))
                        flush();status(stage='conditioning audit',last_image=image_id,condition=name,sigma=sigma,seed=seed,method=mode,sampling_calls=calls,rows=len(rows))
            del target,condition,vae,raw,output,item,conditions;torch.cuda.empty_cache()
        # Author cleaner comparator changes stage1 only relative to upstream-spaced branch.
        cleaner=cleaner.cuda();pipe.cleaner=cleaner
        for record in selected:
            image_id=record['image_id'];folder=run/image_id
            item=torch.load(run/f'prepared_{image_id}.pt',weights_only=True);target=item['target'].cuda()
            for case in item['cases']:
                sigma=case['sigma'];condition=pipe.apply_cleaner(case['observation'].cuda(),False,512,256)
                # Preserve upstream unclipped cleaner output when conditioning stage2.
                save_image(condition,folder/f'author_cleaner_s{sigma}.png')
                rows.append(dict(image_id=image_id,condition='author_swinir',sigma=sigma,seed=None,method='conditioning_input',postprocessing='display_clipped',
                    **fidelity_measures(condition.clamp(0,1),target,perceptual)))
                for seed in SEEDS:
                    raw=pipeline_sample(pipe,condition,seed,'spaced');calls+=1
                    if not torch.isfinite(raw).all():raise ValueError('Nonfinite author-comparator decode before clipping')
                    for post,output in [('raw_clipped',raw.clamp(0,1)),('wavelet_then_clipped',wavelet_reconstruction(raw,condition).clamp(0,1))]:
                        rows.append(dict(image_id=image_id,condition='author_swinir',sigma=sigma,seed=seed,method='upstream_spaced20',postprocessing=post,**fidelity_measures(output,target,perceptual)))
                        if post=='raw_clipped' or seed==11:save_image(output,folder/f'author_swinir_s{sigma}_seed{seed}_{post}.png')
                        if seed==11:
                            path=f'author_swinir_s{sigma}_{post}_comparison.png'
                            condition_image=Image.open(folder/f'author_cleaner_s{sigma}.png')
                            output_image=Image.open(folder/f'author_swinir_s{sigma}_seed{seed}_{post}.png')
                            error=(output-target).abs().mean(1)[0].cpu().numpy()
                            panel([Image.open(folder/'sharp.png'),condition_image,output_image,heat(error)],
                                ['Sharp reference','Author SwinIR',post,'Absolute error (scale .1)']).save(folder/path)
                            comparisons.append(f'{image_id}/{path}')
                    flush();status(stage='author cleaner comparator',last_image=image_id,sigma=sigma,seed=seed,sampling_calls=calls,rows=len(rows))
    data=pd.DataFrame(rows)
    fields=['psnr','ssim','lpips','mse','false_edge_proxy','valid_edge_recall']
    per_image=data.groupby(['image_id','condition','sigma','method','postprocessing'],dropna=False)[fields].mean().reset_index()
    per_image.groupby(['condition','sigma','method','postprocessing'])[fields].mean().to_csv(run/'summary.csv')
    cnn=data[data.condition=='cnn'];err=data[data.condition=='cnn_plus_fixed_error']
    keys=['image_id','sigma','seed','method','postprocessing']
    paired=cnn.merge(err,on=keys,suffixes=('_cnn','_error'),validate='one_to_one')
    for field in fields:paired[f'delta_{field}_error_minus_cnn']=paired[f'{field}_error']-paired[f'{field}_cnn']
    paired.to_csv(run/'paired_changes.csv',index=False)
    # Input delta MSE is used only for paired intervention response, not named hallucination gain.
    inputs=paired[paired.method=='conditioning_input'][['image_id','sigma','delta_mse_error_minus_cnn']].rename(columns={'delta_mse_error_minus_cnn':'input_delta_mse'})
    response=paired[paired.method.isin(['legacy_ddim20','upstream_spaced20'])].merge(inputs,on=['image_id','sigma'],validate='many_to_one')
    response['output_delta_mse']=response.delta_mse_error_minus_cnn
    response.to_csv(run/'intervention_response.csv',index=False)
    # Summarize image units after seed averaging, rather than treating seeds as independent.
    from .metrics import bootstrap_ci
    ci=[]
    for key,g in response.groupby(['sigma','method','postprocessing']):
        vals=g.groupby('image_id').output_delta_mse.mean()
        ci.append(dict(sigma=key[0],method=key[1],postprocessing=key[2],**bootstrap_ci(vals)))
    pd.DataFrame(ci).to_csv(run/'intervention_image_bootstrap.csv',index=False)
    (run/'visual_review.html').write_text('<!doctype html><meta charset="utf-8"><h1>Expanded conditioning audit</h1><p>Seed11 visual panels. Open after extracting the full ZIP. Oracle conditioning is diagnostic only. Maps and selected patches flag differences, not confirmed hallucinations. Review full images and randomly sampled controls before estimating rates.</p>'+''.join(f'<h3>{p}</h3><img style="max-width:100%" src="{p}">' for p in comparisons),encoding='utf-8')
    # Prepared tensors consume substantial storage and are retained locally, but excluded from portable bundle.
    (run/'status.json').write_text(json.dumps(dict(completed=True,rows=len(rows),sampling_calls=calls,n_images=count)))
    root=Path(settings['output_root']);label=settings['run_label']
    for name in ['results','summary','paired_changes','dataset_inventory','checks','cnn_diagnostics','candidate_patches','intervention_response','intervention_image_bootstrap']:
        shutil.copy2(run/f'{name}.csv',root/f'{label}_{export_tag}_{name}.csv')
    import zipfile
    with zipfile.ZipFile(root/f'{label}_{export_tag}_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in run.rglob('*'):
            if path.is_file() and path.suffix!='.pt':z.write(path,path.relative_to(run))
    print('Audit completed',run,'sampling calls',calls,flush=True)
