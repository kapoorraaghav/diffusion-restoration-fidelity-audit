"""Read completed extension table ZIPs; never execute their code or run models."""
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from .revision_analysis import load_preferences, fronts_and_hv, severity_changes
from .revision_design import frozen_protocol


def analyze_archives(source,output):
    source,output=Path(source),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    frames=[];preferences=[];audit=[];tails=[];inventories=[]
    for path in sorted(source.rglob('*_tables.zip')):
        with zipfile.ZipFile(path) as z:
            if z.testzip():raise ValueError('Archive CRC failed: '+path.name)
            protocol=json.loads(z.read('experiment/protocol.json'))
            if protocol.get('version')!='paper2_review_extension_v1':continue
            status=json.loads(z.read('experiment/status.json'))
            if protocol.get('smoke') or not status.get('completed'):continue
            if protocol['design_sha256']!=frozen_protocol()['sha256']:raise ValueError('Design mismatch')
            pref=json.loads(z.read('experiment/frozen_preferences.json'))
            preferences.append(pref)
            g=pd.read_csv(z.open('experiment/image_means.csv'))
            if g.duplicated(['dataset','image_id','profile','strategy']).any() or not np.isfinite(g[['psnr','ssim','lpips','mse','musiq']].to_numpy()).all():
                raise ValueError('Invalid or duplicate source-level outcomes')
            inv=pd.read_csv(z.open('experiment/dataset_inventory.csv'))
            stage=protocol['stage'];design=frozen_protocol()
            expected=design['development'] if stage=='development' else {protocol['dataset']:design['severity' if stage=='severity' else 'final'][protocol['dataset']]}
            if {(r.dataset,int(r.original_id)) for r in inv.itertuples()}!={(d,i) for d,ids in expected.items() for i in ids}:
                raise ValueError('Archived source inventory differs from frozen IDs')
            if set(g.image_id)!=set(inv.image_id) or status['completed_images']!=len(inv):raise ValueError('Incomplete archived source coverage')
            expected_names={c['strategy'] for c in protocol['execution_candidates']}|{'input_y','cleaner','vae_x_oracle','vae_y','vae_z'}
            profiles={p['name'] for p in design['profiles']} if stage=='severity' else {'blur06'}
            if set(g.profile)!=profiles or set(g.strategy)!=expected_names:
                raise ValueError('Archived profile/candidate coverage differs')
            if not (g.groupby(['image_id','profile']).strategy.nunique()==len(expected_names)).all():
                raise ValueError('Some image/profiles have incomplete candidate coverage')
            if len(g)!=len(inv)*len(profiles)*len(expected_names):raise ValueError('Missing source-profile records')
            g['stage']=stage;frames.append(g)
            checks=pd.read_csv(z.open('experiment/checks.csv'))
            if not checks.passed.all():raise ValueError('A VAE parity check failed')
            t=pd.read_csv(z.open('experiment/paired_tail_statistics.csv'));t['stage']=protocol['stage'];tails.append(t)
            inv['stage']=stage;inventories.append(inv)
            audit.append(dict(file=path.name,stage=protocol['stage'],dataset=protocol['dataset'],
                              n_images=status['completed_images'],sampling_calls=status['sampling_calls'],
                              preference_sha256=pref['sha256'],protocol_sha256=protocol['protocol_sha256']))
    if not frames:raise ValueError('Upload completed FULL *_tables.zip files, not smoke ZIPs')
    if len({p['sha256'] for p in preferences})!=1:raise ValueError('Runs use different frozen preferences')
    pref=preferences[0]
    temp=output/'frozen_preferences.json';temp.write_text(json.dumps(pref,indent=2));load_preferences(temp)
    f=pd.concat(frames,ignore_index=True)
    keys=['stage','dataset','image_id','profile','strategy']
    if f.duplicated(keys).any():raise ValueError('Duplicate source records: upload each full run once')
    f.to_csv(output/'all_image_means.csv',index=False)
    pd.concat(tails,ignore_index=True).to_csv(output/'all_paired_tail_statistics.csv',index=False)
    inventory=pd.concat(inventories,ignore_index=True)
    inventory.loc[inventory.dataset=='ffhq','upsampled']=False
    inventory['upsampled']=inventory.upsampled.map(lambda x:str(x).lower()=='true')
    linked=f.merge(inventory[['stage','dataset','image_id','upsampled']],on=['stage','dataset','image_id'],validate='many_to_one')
    linked.groupby(['stage','dataset','upsampled','profile','strategy'])[['psnr','ssim','lpips','mse','musiq']].mean().reset_index().to_csv(output/'resolution_stratified_means.csv',index=False)
    pd.DataFrame(audit).to_csv(output/'run_manifest.csv',index=False)
    summary=f.groupby(['stage','dataset','profile','strategy'])[['psnr','ssim','lpips','mse','musiq']].mean().reset_index()
    summary.to_csv(output/'full_grid_summary.csv',index=False)
    severity=f[f.stage=='severity']
    if len(severity):severity_changes(severity).to_csv(output/'paired_severity_changes.csv',index=False)
    fronts=[];volumes=[]
    for (stage,profile),g in f.groupby(['stage','profile']):
        front,hv=fronts_and_hv(g,pref['normalization']);front['stage']=stage;front['profile']=profile
        hv['stage']=stage;hv['profile']=profile;fronts.append(front);volumes.append(hv)
    pd.concat(fronts,ignore_index=True).to_csv(output/'front_membership.csv',index=False)
    pd.concat(volumes,ignore_index=True).to_csv(output/'hypervolume_reference_sensitivity.csv',index=False)
    choices=pd.DataFrame([dict(preference=k,strategy=v) for k,v in pref['choices'].items()])
    summary.merge(choices,on='strategy').to_csv(output/'frozen_preference_outcomes.csv',index=False)
    completeness={stage:sorted(set(f.loc[f.stage==stage,'dataset'])) for stage in ['final','severity']}
    missing={s:sorted(set(['ffhq','imagenet','div2k','urban100'])-set(ds)) for s,ds in completeness.items()}
    (output/'analysis_status.json').write_text(json.dumps(dict(new_inference=False,missing_domains=missing,
        all_domains_complete=not any(missing.values()),evidence_status='review extension, not untouched held-out confirmation'),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    final=pd.concat(fronts,ignore_index=True);final=final[(final.stage=='final')&(final.profile=='blur06')]
    if len(final):
        fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
        for ax,d in zip(axes.flat,['ffhq','div2k','urban100','imagenet']):
            g=final[final.dataset==d]
            if g.empty:ax.set(title=d+' (not uploaded)');continue
            sc=ax.scatter(g.lpips,g.psnr,c=g.ssim,cmap='viridis',vmin=.35,vmax=.9,s=18)
            h=g[g.nondominated];ax.scatter(h.lpips,h.psnr,facecolors='none',edgecolors='red',s=90)
            ax.set(title=d,xlabel='LPIPS (lower preferred)',ylabel='PSNR dB (higher preferred)');ax.grid(alpha=.15)
        fig.colorbar(sc,ax=axes.ravel().tolist(),label='SSIM (higher preferred)',shrink=.8)
        fig.savefig(output/'sampled_grid_front_projection.pdf');fig.savefig(output/'sampled_grid_front_projection.png',dpi=180);plt.close(fig)
    finalzip=output.parent/'Paper2_Review_Analysis_FINAL.zip'
    with zipfile.ZipFile(finalzip,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for p in output.iterdir():
            if p.is_file():z.write(p,p.name)
    return str(finalzip),summary,missing
