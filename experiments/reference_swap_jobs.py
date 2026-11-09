"""Background control and read-only refresh for the one-image audit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def state(job):
    job=Path(job);value={'job_directory':str(job)}
    for name in ['progress','result','exit','job']:
        path=job/(name+'.json')
        if path.is_file():
            try:value[name]=json.loads(path.read_text())
            except (OSError,ValueError):value[name]='being written'
    log=job/'worker.log'
    if log.is_file():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-8000));value['log_tail']=f.read().decode(errors='replace')
    return value


def start(settings,cache):
    settings=dict(settings)
    source=Path(settings['source']).expanduser().resolve()
    if not source.exists():raise ValueError('Input archive/folder not found: '+str(source))
    settings['source']=str(source)
    settings['output']=str(Path(settings['output']).expanduser().resolve())
    settings['upstream']=str(Path(settings['upstream']).expanduser().resolve())
    fingerprint=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()[:12]
    job=Path(settings['output'])/'audit_jobs'/fingerprint;job.mkdir(parents=True,exist_ok=True)
    lock=job/'launch.lock'
    try:lock.mkdir()
    except FileExistsError:raise ValueError('A launch is already in progress; refresh instead')
    try:
        existing=state(job)
        if 'result' in existing:return str(job)
        if 'job' in existing:
            pid=existing['job']['pid']
            try:
                os.kill(pid,0)
                proc=Path(f'/proc/{pid}/cmdline')
                if not proc.exists() or str(job/'worker.py').encode() in proc.read_bytes():return str(job)
            except ProcessLookupError:pass
            raise ValueError('Previous attempt ended. Inspect worker.log before retrying; use a new output folder only after resolving the failure.')
        (job/'settings.json').write_text(json.dumps(settings,indent=2))
        script='''import json,sys,traceback
from pathlib import Path
sys.path.insert(0,CACHE)
root=Path(__file__).resolve().parent
def write(name,value):
    path=root/(name+'.json');tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def progress(value):
    write('progress',value)
    print(value['stage'],str(value['completed'])+'/'+str(value['total']),flush=True)
code=0
try:
    progress(dict(stage='Importing audit dependencies',completed=0,total=48))
    from experiments.reference_swap_audit import run
    s=json.loads((root/'settings.json').read_text())
    archive,effects=run(s['source'],s['output'],s['upstream'],progress=progress)
    write('result',dict(archive=archive,completed=True,sampling_calls=0))
except BaseException:
    traceback.print_exc();code=1
    write('progress',dict(stage='Failed; inspect worker.log',completed=0,total=48))
write('exit',dict(returncode=code))
raise SystemExit(code)
'''.replace('CACHE',repr(str(Path(cache).resolve())))
        worker=job/'worker.py';worker.write_text(script)
        with (job/'worker.log').open('wb') as log:
            child=subprocess.Popen([sys.executable,'-u',str(worker)],stdout=log,stderr=subprocess.STDOUT,
                cwd=str(job),start_new_session=True)
        (job/'job.json').write_text(json.dumps({'pid':child.pid}))
        return str(job)
    finally:lock.rmdir()
