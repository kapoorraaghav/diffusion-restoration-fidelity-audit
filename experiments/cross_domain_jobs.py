"""Molab job control; resume only completed per-image checkpoints."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def validate(settings):
    s=dict(settings)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',s['run_label']):raise ValueError('Run label must contain letters, digits, underscores or hyphens')
    if s['dataset'] not in ['ffhq','imagenet','div2k','urban100']:raise ValueError('Unsupported dataset')
    for key in ['data_root','model_root','cache_root','output_root']:
        s[key]=str(Path(s[key]).expanduser().resolve())
    for key in ['imagenet_source','ffhq_tables']:
        if s.get(key):s[key]=str(Path(s[key]).expanduser().resolve())
    if s['dataset']=='imagenet' and not Path(s['imagenet_source']).exists():
        raise ValueError('Provide the authorized ImageNet validation TAR or extracted folder before Start')
    if s['dataset']=='ffhq' and not Path(s['ffhq_tables']).is_file():
        raise ValueError('Upload your FFHQ2000 complete_tables.zip beside notebook.py before Start')
    return s


def status(job):
    job=Path(job);result={'job_directory':str(job)}
    for key,path in [('job',job/'job.json'),('exit',job/'exit.json'),('progress',job/'experiment/status.json')]:
        if path.exists():
            try:result[key]=json.loads(path.read_text())
            except (OSError,ValueError):result[key]='being written'
    log=job/'worker.log'
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-10000));result['log_tail']=f.read().decode(errors='replace')
    return result


def launch(settings,worker_source):
    s=validate(settings)
    job=Path(s['output_root'])/(s['run_label']+'_CROSS');job.mkdir(parents=True,exist_ok=True)
    lock=job/'launch.lock'
    # Atomic directory creation is a short-lived launch mutex. Worker liveness is separate.
    try:lock.mkdir()
    except FileExistsError:raise ValueError('Launch already in progress. If a prior session died during launch, inspect launch.lock before retrying.')
    try:
        progress=job/'experiment/status.json'
        if progress.exists() and json.loads(progress.read_text()).get('completed'):
            return status(job)
        info=job/'job.json'
        if info.exists():
            old=json.loads(info.read_text());pid=old.get('pid')
            if pid:
                try:
                    os.kill(pid,0)
                    proc=Path(f'/proc/{pid}/cmdline')
                    if not proc.exists() or str(job/'launcher.py').encode() in proc.read_bytes():
                        return status(job)
                except ProcessLookupError:pass
        previous=job/'settings.json'
        if previous.exists() and json.loads(previous.read_text())!=s:
            raise ValueError('Existing run settings differ. Keep the old run; use a new label for a different setup.')
        previous.write_text(json.dumps(s,indent=2))
        (job/'worker.py').write_text(worker_source,encoding='utf-8')
        launcher='''import json,os,runpy,traceback
from pathlib import Path
root=Path(__file__).resolve().parent
print('Worker started; downloads/GPU setup followed by resumable inference.',flush=True)
code=0
try:runpy.run_path(str(root/'worker.py'),run_name='__main__')
except BaseException:
    traceback.print_exc();code=1
(root/'exit.json').write_text(json.dumps({'returncode':code}))
raise SystemExit(code)
'''
        (job/'launcher.py').write_text(launcher)
        # Archive previous attempt logs, never silently discard them.
        attempt=len(list(job.glob('worker_attempt_*.log')))+1
        log=job/'worker.log'
        if log.exists():shutil_copy(log,job/f'worker_attempt_{attempt:03d}.log')
        old_exit=job/'exit.json'
        if old_exit.exists():old_exit.rename(job/f'exit_attempt_{attempt:03d}.json')
        env=os.environ.copy();env.update(MOLAB_SETTINGS=str(previous),MOLAB_JOB_DIR=str(job),PYTHONUNBUFFERED='1')
        with log.open('wb') as f:
            process=subprocess.Popen([sys.executable,'-u',str(job/'launcher.py')],stdout=f,stderr=subprocess.STDOUT,
                cwd=str(job),env=env,start_new_session=True)
        info.write_text(json.dumps(dict(pid=process.pid,attempt=attempt)))
        return status(job)
    finally:
        lock.rmdir()


def shutil_copy(source,destination):
    import shutil
    shutil.copy2(source,destination)
