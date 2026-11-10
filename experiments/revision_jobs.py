"""Isolated Molab launch and strict settings validation for the review extension."""
import json
from pathlib import Path
import re
import sys
import os
import subprocess
import shutil


def validate(settings):
    s=dict(settings)
    if s.get('stage') not in ['development','final','severity']:
        raise ValueError('Choose development, final, or severity')
    if s['stage']=='development':s['dataset']='development'
    elif s.get('dataset') not in ['ffhq','imagenet','div2k','urban100']:
        raise ValueError('Choose one of the four registered domains')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',s['run_label']):raise ValueError('Use a simple unique run label')
    for key in ['data_root','model_root','cache_root','output_root']:
        s[key]=str(Path(s[key]).expanduser().resolve())
    for key in ['selection_file','imagenet_source','div2k_train_source']:
        if s.get(key):s[key]=str(Path(s[key]).expanduser().resolve())
    if s['stage']!='development':
        p=Path(s.get('selection_file',''))
        if not p.is_file():raise ValueError('Upload frozen_preferences.json from completed FULL Notebook01')
        value=json.loads(p.read_text())
        import hashlib
        content={k:v for k,v in value.items() if k!='sha256'}
        checksum=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if value.get('version')!='paper2_review_extension_v1' or value.get('completed_images')!=100 or value.get('smoke') is not False or checksum!=value.get('sha256'):
            raise ValueError('Not the completed, checksummed review-extension development selection')
        s['selection_sha256']=checksum
    if s['dataset']=='imagenet' and (not s.get('imagenet_source') or not Path(s['imagenet_source']).exists()):
        raise ValueError('Supply your authorized ImageNet validation TAR or canonical extracted directory')
    return s


def status(job):
    job=Path(job);out=dict(job_directory=str(job))
    for key,path in [('job',job/'job.json'),('exit',job/'exit.json'),('progress',job/'experiment/status.json')]:
        if path.is_file():
            try:out[key]=json.loads(path.read_text())
            except (ValueError,OSError):out[key]='being written'
    p=job/'worker.log'
    if p.is_file():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-12000));out['log_tail']=f.read().decode(errors='replace')
    return out


def launch(settings,worker_source):
    s=validate(settings)
    # Pin a separate code tree; never overwrite the historical Molab code cache.
    code_digest=__import__('hashlib').sha256(worker_source.encode()).hexdigest()
    content=dict(s,worker_source_sha256=code_digest)
    job=Path(s['output_root'])/(s['run_label']+'_REVIEW');job.mkdir(parents=True,exist_ok=True)
    lock=job/'launch.lock'
    try:lock.mkdir()
    except FileExistsError:raise ValueError('A launch is in progress; inspect its log before retrying')
    try:
        p=job/'settings.json'
        if p.exists() and json.loads(p.read_text())!=content:
            raise ValueError('Settings/code differ: preserve this run and choose a new label')
        if (job/'experiment/status.json').exists() and json.loads((job/'experiment/status.json').read_text()).get('completed'):
            return dict(job_directory=str(job))
        if (job/'job.json').exists():
            state=json.loads((job/'job.json').read_text());pid=state.get('pid')
            try:
                os.kill(pid,0)
                proc=Path(f'/proc/{pid}/cmdline')
                if not proc.exists() or str(job/'launcher.py').encode() in proc.read_bytes():return dict(job_directory=str(job))
            except ProcessLookupError:pass
        p.write_text(json.dumps(content,indent=2))
        (job/'worker.py').write_text(worker_source,encoding='utf-8')
        launcher="""import json,runpy,traceback
from pathlib import Path
root=Path(__file__).resolve().parent
print('Worker started: pinned-source setup, model/data checks, then resumable inference.',flush=True)
code=0
try:runpy.run_path(str(root/'worker.py'),run_name='__main__')
except BaseException:traceback.print_exc();code=1
(root/'exit.json').write_text(json.dumps(dict(returncode=code)))
raise SystemExit(code)
"""
        (job/'launcher.py').write_text(launcher)
        attempt=len(list(job.glob('worker_attempt_*.log')))+1
        if (job/'worker.log').exists():shutil.copy2(job/'worker.log',job/f'worker_attempt_{attempt:03d}.log')
        if (job/'exit.json').exists():(job/'exit.json').rename(job/f'exit_attempt_{attempt:03d}.json')
        env=dict(os.environ,MOLAB_SETTINGS=str(p),MOLAB_JOB_DIR=str(job),PYTHONUNBUFFERED='1')
        with (job/'worker.log').open('wb') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(job/'launcher.py')],cwd=str(job),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (job/'job.json').write_text(json.dumps(dict(pid=proc.pid,attempt=attempt)))
        return dict(job_directory=str(job))
    finally:lock.rmdir()
