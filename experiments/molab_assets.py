"""Direct remote asset preparation for Molab, with verified NAFNet and bounded extraction."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

NAF_SHA='19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c'
MODELS={
    'NAFNet-GoPro-width32.pth':dict(repo_id='mikestealth/nafnet-models',
        revision='9526c38b626f6e8ca0c02e4a282859ac84d240a2',source='community mirror; exact official-file SHA256 verified'),
    'sd2.1-base-zsnr-laionaes5.ckpt':dict(repo_id='lxq007/DiffBIR-v2',source='DiffBIR authors'),
    'DiffBIR_v2.1.pt':dict(repo_id='lxq007/DiffBIR-v2',source='DiffBIR authors'),
}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def extract_training_pairs(archive,destination):
    """Extract only train/<sequence>/{blur,sharp} PNGs, rejecting unsafe ZIP paths.

    Reading ZipExtFile to EOF verifies member CRC; marker is written only after
    the complete paired inventory has been checked.
    """
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    manifest=[]
    with zipfile.ZipFile(archive) as z:
        members=[]
        for entry in z.infolist():
            path=PurePosixPath(entry.filename.replace('\\','/'))
            if path.is_absolute() or '..' in path.parts or any(':' in p for p in path.parts):
                raise ValueError(f'Unsafe archive member: {entry.filename}')
            parts=path.parts
            if 'train' not in parts:continue
            start=parts.index('train');relative=PurePosixPath(*parts[start:])
            if len(relative.parts)!=4 or relative.parts[2] not in {'blur','sharp'} or relative.suffix.lower()!='.png':continue
            if (entry.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Symlink archive members are not accepted')
            members.append((entry,relative))
        if not members:raise ValueError('No GoPro training blur/sharp pairs in archive')
        relative_names=[str(relative) for _,relative in members]
        if len(relative_names)!=len(set(relative_names)):raise ValueError('Duplicate training archive paths')
        required=sum(entry.file_size for entry,_ in members)
        if shutil.disk_usage(destination).free < required+256*1024**2:
            raise OSError(f'Insufficient free disk for extraction: need about {required/1024**3:.1f} GiB plus margin')
        for i,(entry,relative) in enumerate(members):
            target=destination/Path(*relative.parts);target.parent.mkdir(parents=True,exist_ok=True)
            # Always stream from the validated member, so a partial extraction is repaired.
            with z.open(entry) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,length=1024*1024)
            manifest.append(dict(path=str(relative),bytes=entry.file_size,crc=entry.CRC))
            if i%200==0:print(f'Extracted {i+1}/{len(members)} training images',flush=True)
    blurred=[r for r in manifest if '/blur/' in r['path']]
    sharp={r['path'] for r in manifest if '/sharp/' in r['path']}
    if not blurred or any(r['path'].replace('/blur/','/sharp/') not in sharp for r in blurred):
        raise ValueError('Training image pairs are incomplete')
    result=dict(completed=True,pairs=len(blurred),files=manifest)
    (destination/'extraction_manifest.json').write_text(json.dumps(result,indent=2))
    return result


def dataset_ready(destination,expected_pairs=2103):
    destination=Path(destination);marker=destination/'extraction_manifest.json'
    if not marker.is_file():return False
    try:
        info=json.loads(marker.read_text())
        return bool(info['completed'] and info['pairs']==expected_pairs and all(
            (destination/f['path']).is_file() and (destination/f['path']).stat().st_size==f['bytes']
            for f in info['files']))
    except (ValueError,KeyError,OSError):return False


def ensure_assets(settings,include_dataset=True):
    from huggingface_hub import hf_hub_download
    cache=Path(settings['cache_root']);cache.mkdir(parents=True,exist_ok=True)
    models=Path(settings['model_root']);models.mkdir(parents=True,exist_ok=True)
    dataset=Path(settings['gopro_root'])
    provenance={}
    for name,source in MODELS.items():
        existing=sorted(models.rglob(name))+[cache/'weights'/name]
        valid=None
        for path in existing:
            if not path.is_file() or path.stat().st_size<1_000_000:continue
            if name.startswith('NAFNet') and digest(path)!=NAF_SHA:
                print('Ignoring invalid NAFNet checkpoint:',path,flush=True);continue
            valid=path;break
        if valid is None:
            print('Downloading model:',name,'from',source['repo_id'],flush=True)
            kwargs=dict(repo_id=source['repo_id'],filename=name,local_dir=models)
            if 'revision' in source:kwargs['revision']=source['revision']
            valid=Path(hf_hub_download(**kwargs))
        actual=digest(valid)
        if name.startswith('NAFNet') and actual!=NAF_SHA:
            raise ValueError('NAFNet checksum does not match the official checkpoint')
        provenance[name]=dict(path=str(valid),sha256=actual,bytes=valid.stat().st_size,**source)
        print('Model ready:',valid,flush=True)
    if not include_dataset:
        report=dict(models=provenance,dataset_downloaded=False)
        (cache/'asset_sources_models_only.json').write_text(json.dumps(report,indent=2))
        return report
    # A user-supplied extracted dataset remains supported; auto extraction uses a marker.
    external_pairs=next(dataset.glob('train/*/blur/*.png'),None)
    marker=dataset/'extraction_manifest.json'
    if dataset_ready(dataset):
        print('Reusing complete extracted GoPro training pairs:',dataset,flush=True)
    elif external_pairs and not marker.exists():
        # A prior interrupted automatic extraction must never look externally prepared.
        if (dataset/'.automatic_extraction_started').exists():
            external_pairs=None
        else:print('Using existing extracted GoPro folder:',dataset,flush=True)
    else:external_pairs=None
    if not dataset_ready(dataset) and not external_pairs:
        print('Downloading official GOPRO_Large.zip (~9.54 GB); not GOPRO_Large_all.zip.',flush=True)
        archive=Path(hf_hub_download(repo_id='snah/GOPRO_Large',repo_type='dataset',
            filename='GOPRO_Large.zip',local_dir=cache/'dataset_archive'))
        dataset.mkdir(parents=True,exist_ok=True)
        (dataset/'.automatic_extraction_started').write_text('Incomplete until extraction_manifest.json validates')
        info=extract_training_pairs(archive,dataset)
        if info['pairs']!=2103:raise ValueError(f"Expected 2103 GoPro training pairs, found {info['pairs']}")
    report=dict(models=provenance,gopro_root=str(dataset),
        dataset_source='https://huggingface.co/datasets/snah/GOPRO_Large',
        dataset_file='GOPRO_Large.zip',extracted_split='train only',
        storage_note='Cache is reused only while files remain on the host; no persistence is guaranteed.')
    (cache/'asset_sources.json').write_text(json.dumps(report,indent=2))
    return report
