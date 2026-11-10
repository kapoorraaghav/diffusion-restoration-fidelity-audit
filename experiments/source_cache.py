"""Pinned upstream checkout cache, preserving incomplete/non-Git directories."""
import json
from pathlib import Path
import subprocess
import uuid


def ensure_repository(target,url,revision):
    target=Path(target)
    def git(path,*args):
        return subprocess.run(['git','-C',str(path),*args],capture_output=True,text=True)
    if (target/'.git').exists():
        head=git(target,'rev-parse','HEAD')
        clean=git(target,'diff','--quiet','HEAD','--')
        if head.returncode==0 and head.stdout.strip()==revision and clean.returncode==0:
            print('Reusing pinned source:',target,flush=True)
            return target
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.clone-'+uuid.uuid4().hex[:12])
    # Clone into a fresh path, never into an existing restored source folder.
    subprocess.run(['git','clone','--no-checkout',url,str(temporary)],check=True)
    subprocess.run(['git','-C',str(temporary),'checkout','--detach',revision],check=True)
    head=git(temporary,'rev-parse','HEAD')
    if head.returncode or head.stdout.strip()!=revision:
        raise RuntimeError('Source revision verification failed; original folder preserved')
    if target.exists():
        preserved=target.with_name(target.name+'.preserved-'+uuid.uuid4().hex[:12])
        target.rename(preserved)
        print('Preserved existing source folder:',preserved,flush=True)
    temporary.rename(target)
    print('Pinned source ready:',target,flush=True)
    return target
