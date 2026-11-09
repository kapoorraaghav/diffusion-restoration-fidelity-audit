"""Dataset-only FFHQ follow-up with the prior conditioning-audit protocol."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .opensource import sha256
from .conditioning_audit import main as run_audit

REPO='marcosv/ffhq-dataset'
REVISION='505f94e2ecc6db64e967e8e6c8e2c2079ea0876b'
SELECTION_SEED=20261026


def selected_ids(count):
    if isinstance(count,bool) or not isinstance(count,int) or not 1<=count<=100:
        raise ValueError('FFHQ audit supports 1–100 fixed validation images')
    return np.random.default_rng(SELECTION_SEED).permutation(np.arange(60000,70000))[:count].tolist()


def download_ffhq(destination,count):
    from huggingface_hub import hf_hub_download
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    records=[]
    for index in selected_ids(count):
        name=f'{index:05d}.png';relative=f'Part7/{name}'
        print('Preparing FFHQ reference',name,flush=True)
        original=Path(hf_hub_download(REPO,relative,repo_type='dataset',revision=REVISION,
            cache_dir=str(destination/'source_cache')))
        original_hash=sha256(original)
        output=destination/f'ffhq_{index:05d}_512.png'
        marker=output.with_suffix('.json')
        signature=dict(repo=REPO,revision=REVISION,file=relative,original_sha256=original_hash,
            transform='Pillow RGB then full-frame LANCZOS downsample 1024x1024 to 512x512')
        valid=False
        if output.exists() and marker.exists():
            try:
                saved=json.loads(marker.read_text())
                valid=all(saved.get(k)==v for k,v in signature.items()) and saved.get('derived_sha256')==sha256(output)
                with Image.open(output) as im:valid=valid and im.size==(512,512);im.verify()
            except (OSError,ValueError):valid=False
        if not valid:
            with Image.open(original) as im:
                if im.size!=(1024,1024):raise ValueError(f'Unexpected FFHQ source size: {im.size}')
                image=im.convert('RGB').resize((512,512),Image.Resampling.LANCZOS)
            temporary=output.with_suffix('.part')
            image.save(temporary,format='PNG');temporary.replace(output)
            marker.write_text(json.dumps(dict(**signature,derived_sha256=sha256(output)),indent=2))
        records.append(dict(image_id=f'ffhq_{index:05d}',original_id=index,path=str(output),box=[0,0,512,512],
            sha256=sha256(output),original_sha256=original_hash,
            source=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{relative}',
            repo_revision=REVISION,selection_seed=SELECTION_SEED,selection_population='FFHQ designated validation IDs 60000–69999',
            preprocessing=signature['transform'],
            dataset_documentation='https://github.com/NVlabs/ffhq-dataset',
            licence='FFHQ source and individual-image terms apply; see official documentation'))
    return records


def main(settings,job):
    run_audit(settings,job,dataset_loader=download_ffhq,
        dataset_name='FFHQ validation subset, mirrored original1024 downsampled512',export_tag='FFHQ')
