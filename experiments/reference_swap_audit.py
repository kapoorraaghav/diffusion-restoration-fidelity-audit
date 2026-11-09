"""Exact post-hoc reference swaps on stored unclipped float diffusion outputs."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch


def load_members(source):
    source=Path(source)
    if source.is_dir():
        if (source/'experiment').is_dir():source=source/'experiment'
        return {str(p.relative_to(source)).replace('\\','/'):p.read_bytes() for p in source.rglob('*') if p.is_file() and p.suffix in ('.json','.npy')}
    with zipfile.ZipFile(source) as z:
        members={}
        for info in z.infolist():
            name=info.filename
            if name.startswith('experiment/') and name.endswith(('.npy','.json')):
                key=name[len('experiment/'):]
                if key in members:raise ValueError('Duplicate archive member')
                members[key]=z.read(info)
        return members


def analyze_frame(frame):
    """CFG1-minus6, then corrected-minus-none, keeping each route separate."""
    metrics=['psnr','ssim','lpips','mse']
    keys=['image_id','seed','route','reference']
    pairs=frame[frame.cfg==1].merge(frame[frame.cfg==6],on=keys,suffixes=('_cfg1','_cfg6'),validate='one_to_one')
    for m in metrics:pairs['cfg_effect_'+m]=pairs[m+'_cfg1']-pairs[m+'_cfg6']
    fields=['image_id','seed','route']
    none=pairs[pairs.reference=='none'][fields+['cfg_effect_'+m for m in metrics]]
    effects=pairs.merge(none,on=fields,suffixes=('','_none'),validate='many_to_one')
    for m in metrics:effects['interaction_'+m]=effects['cfg_effect_'+m]-effects['cfg_effect_'+m+'_none']
    return pairs,effects


def run(source,output_root,upstream_root,progress=None):
    def report(stage,completed=0):
        if progress:progress(dict(stage=stage,completed=completed,total=48))
    report('Reading exact float inputs')
    members=load_members(source)
    state=json.loads(members['status.json'])
    if not state.get('completed') or not state.get('parity_passed'):
        raise ValueError('A completed, passing parity archive is required. Review failures first.')
    root=Path(upstream_root)
    # Standalone setup for a fresh Molab host. No diffusion/model-weight setup.
    from .source_cache import ensure_repository
    expected_revision='5c2d6c177c3f2965e6d33a32931248d242ef7b3d'
    report('Preparing and verifying pinned DiffBIR source')
    ensure_repository(root,'https://github.com/XPixelGroup/DiffBIR.git',expected_revision)
    import subprocess,sys
    revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    protocol=json.loads(members['parity_protocol.json'])
    if revision!=expected_revision or revision!=protocol['source_revisions']['DiffBIR']:raise ValueError('DiffBIR revision differs from parity run')
    sys.path.insert(0,str(root))
    from diffbir.utils.common import wavelet_reconstruction,wavelet_decomposition
    # Import the bundled historical metric implementation after source root is available.
    from experiments.stage_attribution import fidelity_measures
    import lpips
    device='cuda' if torch.cuda.is_available() else 'cpu'
    report('Loading LPIPS metric weights on '+device)
    metric=lpips.LPIPS(net='alex',version='0.1').eval().to(device)
    def tensor(key):
        if key not in members:raise ValueError('Missing exact float artifact '+key+'; use the updated parity notebook, not PNG reconstruction')
        a=np.load(io.BytesIO(members[key]),allow_pickle=False)
        if a.shape!=(1,3,512,512) or not np.isfinite(a).all():raise ValueError('Invalid float image '+key)
        return torch.from_numpy(a).to(device)
    references={name:tensor('canonical/'+name+'.npy') for name in ['oracle','restored','degraded']}
    target=references['oracle'];rows=[];checks=[]
    # Analyze historical and autocast contexts separately; never pool or pick the prettier result.
    with torch.inference_mode():
        for route in ['upstream_noamp','upstream_amp']:
            for generation in ['oracle','restored','degraded']:
                for cfg in [6,1]:
                    key=f'{route}/{generation}_cfg{cfg}_repeat0_raw.npy'
                    raw=tensor(key)
                    original_hash=hashlib.sha256(members[key]).hexdigest()
                    for ref in ['none','degraded','restored','oracle']:
                        # Correction outside autocast is fixed for ALL routes to isolate reference.
                        corrected=raw if ref=='none' else wavelet_reconstruction(raw,references[ref])
                        if ref!='none':
                            _,low_ref=wavelet_decomposition(references[ref]);_,low_raw=wavelet_decomposition(raw)
                            algebra=raw+low_ref-low_raw
                            difference=float((algebra-corrected).abs().max())
                            checks.append(dict(route=route,generation=generation,cfg=cfg,reference=ref,
                                check='linear_operator_identity_before_clipping',max_abs=difference,passed=difference<=1e-5))
                        image=corrected.clamp(0,1)
                        rows.append(dict(image_id='div2k_0801',seed=11,route=route,generation_condition=generation,
                            cfg=cfg,reference=ref,raw_artifact_sha256=original_hash,
                            fraction_clipped=float(((corrected<0)|(corrected>1)).float().mean()),
                            **fidelity_measures(image,target,metric)))
                        report(f'Metrics {route} / {generation} / CFG{cfg} / reference {ref}',len(rows))
    frame=pd.DataFrame(rows);check_frame=pd.DataFrame(checks)
    if not check_frame.passed.all():raise ValueError('Correction operator identity failed; do not interpret the decomposition')
    pairs=[];effects=[]
    for generation,g in frame.groupby('generation_condition'):
        pair,effect=analyze_frame(g);pair['generation_condition']=generation;effect['generation_condition']=generation
        pairs.append(pair);effects.append(effect)
    signature=hashlib.sha256(json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in members.items()},sort_keys=True).encode()).hexdigest()
    report('Writing checked tables and audit ZIP',48)
    out=Path(output_root)/('reference_swap_'+signature[:12]);out.mkdir(parents=True,exist_ok=False)
    frame.to_csv(out/'metrics.csv',index=False)
    pd.concat(pairs,ignore_index=True).to_csv(out/'cfg_effects.csv',index=False)
    pd.concat(effects,ignore_index=True).to_csv(out/'interactions.csv',index=False)
    check_frame.to_csv(out/'checks.csv',index=False)
    metadata=dict(n_images=1,seeds=[11],source_fingerprint=signature,device=device,
        operator='Pinned DiffBIR wavelet_reconstruction, float32 outside autocast, correction before clipping',
        design='Two arithmetic routes separately;3 generation conditions;2 CFG;4 postprocessing references',
        sampling_calls=0,interpretation='Feasibility only. No population confidence intervals or significance claims from one image.',
        oracle='Clean reference is diagnostic only, never a deployable correction',
        scope='No reliability gate, training, threshold tuning or dataset-scale inference')
    (out/'audit_protocol.json').write_text(json.dumps(metadata,indent=2))
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in out.iterdir():z.write(file,file.name)
    report('Completed',48)
    return str(archive),pd.concat(effects,ignore_index=True)
