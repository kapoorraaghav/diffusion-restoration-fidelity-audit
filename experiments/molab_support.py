"""Filesystem/job helpers for the reactive Molab notebooks; no GPU imports."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def validate_settings(settings):
    settings=dict(settings)
    for key in ['gopro_root','model_root','cache_root','output_root']:
        path=Path(settings[key]).expanduser().resolve()
        if not settings.get('auto_download',False) and key in {'gopro_root','model_root'} and not path.is_dir():
            raise ValueError(f'{key} is not an existing local directory: {path}')
        settings[key]=str(path)
    root=Path(settings['gopro_root'])
    if not settings.get('auto_download',False) and not next(root.glob('train/*/blur/*.png'),None):
        raise ValueError('GoPro root must contain train/<sequence>/blur/*.png; extract ZIPs first.')
    if not settings.get('auto_download',False) and not list(Path(settings['model_root']).rglob('NAFNet-GoPro-width32.pth')):
        raise ValueError('Model directory must contain NAFNet-GoPro-width32.pth.')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',settings['run_label']):
        raise ValueError('Use letters, numbers, underscores or hyphens for the run label.')
    count=settings['n_images']
    if isinstance(count,bool) or int(count)!=count or not 1<=int(count)<=100:
        raise ValueError('Choose an integer image count between 1 and 100.')
    settings['n_images']=int(count)
    return settings


def read_status(job_dir):
    job_dir=Path(job_dir)
    result={'job_directory':str(job_dir)}
    for name in ['job','exit','launch_error','status']:
        path=(job_dir/'experiment'/'status.json') if name=='status' else job_dir/f'{name}.json'
        if path.exists():
            try:result[name]=json.loads(path.read_text())
            except (ValueError,OSError):result[name]={'state':'being written'}
    logfile=job_dir/'worker.log'
    if logfile.exists():
        with logfile.open('rb') as stream:
            stream.seek(max(0,logfile.stat().st_size-12000))
            result['log_tail']=stream.read().decode('utf-8',errors='replace')
    if not any(key in result for key in ['job','exit','launch_error','status']):
        result['state']='No recorded worker launch. Script-only folders can be retried with Start.'
    return result


def compare_runs(output_root,label):
    """Validate three completed runs before exporting paired B versus C changes."""
    import numpy as np
    import pandas as pd
    output_root=Path(output_root)
    data={};protocols=[];count=None;seed_count=None;sigma_count=None
    for branch in 'ABC':
        run=output_root/f'{label}_{branch}'/'experiment'
        status=json.loads((run/'status.json').read_text())
        if not status.get('completed'):
            raise ValueError(f'Pipeline {branch} is incomplete')
        info=json.loads((run/'protocol.json').read_text())
        protocols.append(info['protocol_hash'])
        count=info['config']['n_images'];seed_count=len(info['config']['seeds']);sigma_count=len(info['config']['sigmas'])
        checks=pd.read_csv(run/'checks.csv')
        if not checks['passed'].all():raise ValueError(f'Pipeline {branch} failed controls')
        data[branch]=pd.read_csv(run/'results.csv',dtype={'image_id':str})
    if len(set(protocols))!=1:raise ValueError('Protocol mismatch across pipelines')
    keys=['image_id','seed']
    a=data['A'][data['A'].condition=='baseline']
    for branch in 'BC':
        baseline=data[branch][data[branch].condition=='baseline']
        pair=a.merge(baseline,on=keys,suffixes=('_a','_other'),validate='one_to_one')
        if len(pair)!=count*seed_count or not np.allclose(pair.mse_a,pair.mse_other,rtol=1e-4,atol=1e-7):
            raise ValueError(f'Pipeline {branch} baseline differs from A')
    before=data['B'][data['B'].condition=='equal_sigma'];result=[]
    for mode in ['equal_sigma','matched_prior_rms']:
        after=data['C'][data['C'].condition==mode]
        paired=before.merge(after,on=keys+['reference_sigma'],suffixes=('_before','_after'),validate='one_to_one')
        if len(paired)!=count*seed_count*sigma_count:raise ValueError('Missing paired rows')
        if mode=='matched_prior_rms':paired=paired[paired.matched_after==True].copy()
        paired['comparison']=mode
        paired['before_minus_after_delta_mse']=paired.delta_output_mse_before-paired.delta_output_mse_after
        result.append(paired)
    destination=output_root/f'{label}_paired_B_vs_C.csv'
    combined=pd.concat(result,ignore_index=True);combined.to_csv(destination,index=False)
    return str(destination),combined


def launch(settings,branch,worker_source):
    settings=validate_settings(settings)
    job_dir=Path(settings['output_root'])/f"{settings['run_label']}_{branch}"
    if job_dir.exists():
        # Molab can restore .py files without restoring logs/job JSON. A folder
        # containing only preparation files is not evidence of a running job.
        if any((job_dir/name).exists() for name in ['job.json','exit.json','worker.log','experiment','launch.lock','launch_error.json']):
            return read_status(job_dir)
        unexpected={p.name for p in job_dir.iterdir()}-{'worker.py','launcher.py','settings.json'}
        if unexpected:raise ValueError(f'Unrecognized files in run folder: {sorted(unexpected)}. Choose a new run label.')
    else:job_dir.mkdir(parents=True,exist_ok=False)
    # Exclusive creation prevents simultaneous reactive executions from launching twice.
    lock=job_dir/'launch.lock'
    try:
        with lock.open('x') as f:f.write(str(os.getpid()))
    except FileExistsError:return read_status(job_dir)
    try:
        return _launch_prepared_job(settings,branch,worker_source,job_dir)
    except Exception as error:
        import traceback
        (job_dir/'launch_error.json').write_text(json.dumps(dict(error=str(error),traceback=traceback.format_exc())))
        raise RuntimeError(f'Worker startup failed: {error}. Details: {job_dir / "launch_error.json"}') from error
    finally:lock.unlink(missing_ok=True)


def _launch_prepared_job(settings,branch,worker_source,job_dir):
    (job_dir/'settings.json').write_text(json.dumps(settings,indent=2))
    worker=job_dir/'worker.py';worker.write_text(worker_source,encoding='utf-8')
    wrapper=job_dir/'launcher.py'
    wrapper.write_text('''import json,runpy,traceback
from pathlib import Path
root=Path(__file__).resolve().parent
code=0
print('Worker started; loading configuration and checking GPU.',flush=True)
try:runpy.run_path(str(root/'worker.py'),run_name='__main__')
except BaseException:
    code=1
    traceback.print_exc()
finally:(root/'exit.json').write_text(json.dumps({'returncode':code}))
raise SystemExit(code)
''')
    env=os.environ.copy()
    env.update(MOLAB_SETTINGS=str(job_dir/'settings.json'),MOLAB_JOB_DIR=str(job_dir),PYTHONUNBUFFERED='1')
    with (job_dir/'worker.log').open('wb') as logfile:
        process=subprocess.Popen([sys.executable,'-u',str(wrapper)],stdout=logfile,stderr=subprocess.STDOUT,
                                 cwd=job_dir,env=env,start_new_session=(os.name!='nt'),
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0) if os.name=='nt' else 0)
    (job_dir/'job.json').write_text(json.dumps(dict(pid=process.pid,branch=branch,settings=settings)))
    return read_status(job_dir)
