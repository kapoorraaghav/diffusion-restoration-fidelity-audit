"""Fixed source manifests for the cross-domain reference-fidelity experiment."""
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request
import zipfile

import numpy as np
from PIL import Image

SEED = 20261031
URBAN_REVISION = '8f6dd8c1d20cb7e8792a7177b4f6fd677633f598'
DIV2K_URL = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def imagenet_ids():
    return np.random.default_rng(SEED).permutation(np.arange(1, 50001))[:1000].tolist()


def download(url, path):
    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.partial')
    print('Downloading', url, flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, temp.open('wb') as out:
        while True:
            chunk = response.read(8*1024*1024)
            if not chunk:
                break
            out.write(chunk)
    temp.replace(path)
    return path


def prepare_reference(source, destination):
    """One center square per source, aspect-preserving resize; no patch inflation."""
    source, destination = Path(source), Path(destination)
    original_hash = sha(source)
    marker = destination.with_suffix('.json')
    transform = 'RGB; center square crop (floor offsets); LANCZOS resize to512; no EXIF transpose'
    if destination.exists() and marker.exists():
        saved = json.loads(marker.read_text())
        if saved['original_sha256'] == original_hash and saved['preprocessing'] == transform and saved['sha256'] == sha(destination):
            return saved
    with Image.open(source) as im:
        im = im.convert('RGB')
        w, h = im.size
        side = min(w, h)
        x, y = (w-side)//2, (h-side)//2
        im = im.crop((x,y,x+side,y+side)).resize((512,512), Image.Resampling.LANCZOS)
        temp = destination.with_suffix('.partial')
        im.save(temp, format='PNG')
        temp.replace(destination)
    saved = dict(original_sha256=original_hash, sha256=sha(destination), original_size=[w,h],
                 crop_box=[x,y,x+side,y+side], upsampled=side<512, preprocessing=transform)
    marker.write_text(json.dumps(saved, indent=2))
    return saved


def prepare(settings):
    dataset = settings['dataset']
    cache = Path(settings['data_root'])/dataset
    cache.mkdir(parents=True, exist_ok=True)
    if dataset == 'ffhq':
        from . import ffhq_worker
        from .paper2_design import confirmation_ids
        old = ffhq_worker.selected_ids
        ffhq_worker.selected_ids = lambda n: confirmation_ids()
        try:
            return ffhq_worker.download_ffhq(cache, 2000)
        finally:
            ffhq_worker.selected_ids = old
    records = []
    if dataset == 'imagenet':
        supplied = Path(settings['imagenet_source']).expanduser()
        ids = imagenet_ids()
        wanted = {f'ILSVRC2012_val_{i:08d}.JPEG': i for i in ids}
        if supplied.is_file():
            archive_hash = sha(supplied)
            original = cache/'originals'
            original.mkdir(exist_ok=True)
            found = set()
            with tarfile.open(supplied, 'r:*') as archive:
                for member in archive:
                    name = Path(member.name).name
                    if name not in wanted or not member.isfile():
                        continue
                    if name in found:
                        raise ValueError('Duplicate selected ImageNet filename in archive')
                    found.add(name)
                    target = original/name
                    with archive.extractfile(member) as stream, target.open('wb') as out:
                        while True:
                            chunk = stream.read(1024*1024)
                            if not chunk: break
                            out.write(chunk)
            if found != set(wanted):
                raise ValueError(f'Archive missing {len(set(wanted)-found)} preselected validation images; no substitutions allowed')
            files = {name: original/name for name in wanted}
        elif supplied.is_dir():
            archive_hash = None
            files = {}
            for p in supplied.rglob('ILSVRC2012_val_*.JPEG'):
                if p.name in wanted:
                    if p.name in files: raise ValueError('Duplicate ImageNet filename')
                    files[p.name] = p
            if set(files) != set(wanted): raise ValueError('ImageNet directory is missing fixed validation filenames')
        else:
            raise ValueError('Upload the authorized ILSVRC2012_img_val.tar, or provide its extracted directory')
        sources = [(f'imagenet_{i:08d}', files[f'ILSVRC2012_val_{i:08d}.JPEG'],
                    dict(source='https://image-net.org/challenges/LSVRC/2012/', original_id=i,
                         archive_sha256=archive_hash, selection_seed=SEED)) for i in ids]
    elif dataset == 'div2k':
        archive = download(DIV2K_URL, cache/'DIV2K_valid_HR.zip')
        archive_hash = sha(archive)
        original = cache/'originals'; original.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for i in range(801,901):
                name = f'DIV2K_valid_HR/{i:04d}.png'
                target = original/f'{i:04d}.png'
                with z.open(name) as stream, target.open('wb') as out:
                    while True:
                        chunk = stream.read(1024*1024)
                        if not chunk: break
                        out.write(chunk)
        sources = [(f'div2k_{i:04d}', original/f'{i:04d}.png',
                    dict(source=DIV2K_URL, original_id=i, archive_sha256=archive_hash)) for i in range(801,901)]
    elif dataset == 'urban100':
        base = f'https://raw.githubusercontent.com/jbhuang0604/SelfExSR/{URBAN_REVISION}/data/Urban100/image_SRF_2/'
        sources = []
        for i in range(1,101):
            name = f'img_{i:03d}_SRF_2_HR.png'
            path = download(base+name, cache/'originals'/name)
            sources.append((f'urban100_{i:03d}', path, dict(source=base+name, original_id=i,
                source_revision=URBAN_REVISION, source_variant='author SRF2 HR; mod-cropped source, not LR or model output')))
    else:
        raise ValueError('Unsupported dataset')
    derived = cache/'references512'; derived.mkdir(exist_ok=True)
    for ident, source, metadata in sources:
        path = derived/f'{ident}.png'
        info = prepare_reference(source,path)
        records.append(dict(image_id=ident,path=str(path),**metadata,**info))
    return records
