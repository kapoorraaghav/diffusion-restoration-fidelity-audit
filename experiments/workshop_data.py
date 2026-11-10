"""Disjoint fixed Paper 2 development/final data, reusing established loaders."""
from pathlib import Path
import types

from .cross_domain_data import prepare as old_prepare, download, prepare_reference, sha
from .workshop_design import splits


def faces(ids, destination):
    from . import ffhq_worker
    # Isolated globals preserve the old loader without mutating its module.
    namespace = dict(ffhq_worker.__dict__)
    namespace['selected_ids'] = lambda _: ids
    loader = types.FunctionType(ffhq_worker.download_ffhq.__code__, namespace)
    return loader(destination, len(ids))


def prepare(settings):
    split = splits()
    stage, dataset = settings['stage'], settings['dataset']
    root = Path(settings['data_root'])
    if stage == 'development':
        face_ids = split['development']['ffhq'][:1] if settings['smoke'] else split['development']['ffhq']
        face_records = faces(face_ids, root/'ffhq')
        for r in face_records:
            r['dataset'] = 'ffhq'
        natural_records = []
        url = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'
        supplied = settings.get('div2k_train_source', '').strip()
        archive = Path(supplied).expanduser() if supplied else download(url, root/'div2k'/'DIV2K_train_HR.zip')
        if not archive.is_file():
            raise ValueError('DIV2K development source must be the official training HR ZIP')
        checksum = sha(archive)
        import zipfile
        with zipfile.ZipFile(archive) as z:
            train_ids = split['development']['div2k'][:1] if settings['smoke'] else split['development']['div2k']
            for i in train_ids:
                source = root/'div2k'/'train_originals'/f'{i:04d}.png'
                source.parent.mkdir(parents=True, exist_ok=True)
                with z.open(f'DIV2K_train_HR/{i:04d}.png') as inp, source.open('wb') as out:
                    import shutil
                    shutil.copyfileobj(inp, out)
                target = root/'div2k'/'train512'/f'{i:04d}.png'
                target.parent.mkdir(parents=True, exist_ok=True)
                natural_records.append(dict(dataset='div2k', image_id=f'div2k_{i:04d}', original_id=i,
                                   path=str(target), source=url, archive_sha256=checksum,
                                   **prepare_reference(source, target)))
        result = [v for pair in zip(face_records, natural_records) for v in pair]
    elif dataset == 'ffhq':
        face_ids = split['final']['ffhq'][:1] if settings['smoke'] else split['final']['ffhq']
        result = faces(face_ids, root/'ffhq')
        for r in result:
            r['dataset'] = 'ffhq'
    else:
        result = old_prepare(dict(settings, dataset=dataset))
        allowed = set(split['final'][dataset])
        result = [dict(r, dataset=dataset) for r in result if int(r['original_id']) in allowed]
        if {int(r['original_id']) for r in result} != allowed:
            raise ValueError('Final source IDs do not match frozen split')
    for r in result:
        r['role'] = stage
    return result[:2] if settings['smoke'] and stage == 'development' else result[:1] if settings['smoke'] else result
