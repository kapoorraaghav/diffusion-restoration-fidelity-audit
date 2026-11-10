"""Paper 2 finite configuration study on pinned pretrained components."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import torch

from .cross_domain_data import sha
from .cross_domain_worker import EXPECTED_MODELS, FACE_HASH, GENERAL_HASH, atomic_csv, atomic_json
from .workshop_design import VERSION, SEEDS, ALPHAS, splits, digest, key, corrected, load_selection
from .workshop_data import prepare
from .workshop_analysis import analyze
from .paper2_design import BASELINE, candidate
from .paper2_worker import paper2_pipeline, sample
from .opensource import load_diffbir, source_path, save_image, SOURCES
from .conditioning_audit import load_cleaner_weights
from .stage_attribution import fidelity_measures
from .three_pipeline import extra_blur


def export(run, job, complete=False):
    destination = Path(job).parent.parent/'paper2_workshop_downloads'
    destination.mkdir(parents=True, exist_ok=True)
    path = destination/(Path(job).name+'_tables.zip')
    temp = path.with_suffix('.partial')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(run.iterdir()):
            if p.is_file() and p.suffix in ['.csv', '.json', '.txt']:
                z.write(p, 'experiment/'+p.name)
        for name in ['settings.json', 'job.json', 'worker.py', 'launcher.py', 'worker.log', 'exit.json']:
            p = Path(job)/name
            if p.is_file():
                z.write(p, 'job_logs/'+name)
    temp.replace(path)
    selection = run/'frozen_selection.json'
    if selection.is_file():
        (destination/'frozen_selection.json').write_bytes(selection.read_bytes())
    if not complete:
        return str(path)
    files = sorted(p for p in run.glob('images/**/*') if p.is_file())
    manifests = []
    # Files are assigned to bounded parts, not one ZIP per image. No float quantization.
    buckets, current, total = [], [], 0
    for p in files:
        if current and total+p.stat().st_size > 75*1024**2:
            buckets.append(current)
            current, total = [], 0
        current.append(p)
        total += p.stat().st_size
    if current:
        buckets.append(current)
    for i, bucket in enumerate(buckets, 1):
        part = destination/f'{Path(job).name}_review_part{i:02d}.zip'
        temp = part.with_suffix('.partial')
        with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            for p in bucket:
                z.write(p, p.relative_to(run).as_posix())
        temp.replace(part)
        manifests.append(dict(file=part.name, bytes=part.stat().st_size, sha256=sha(part)))
    atomic_json(destination/f'{Path(job).name}_backup_manifest.json',
                dict(tables=dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path)), review_parts=manifests))
    return str(path)


def collect(run):
    folders = sorted(p for p in (run/'records').iterdir() if (p/'complete.json').is_file())
    for source, target in [('metrics.csv', 'results.csv'), ('checks.csv', 'checks.csv'),
                           ('raw_hashes.csv', 'raw_hashes.csv')]:
        atomic_csv(run/target, pd.concat([pd.read_csv(p/source) for p in folders], ignore_index=True))
    swaps = [pd.read_csv(p/'reference_swaps.csv') for p in folders if (p/'reference_swaps.csv').is_file()]
    if swaps:
        atomic_csv(run/'reference_swaps.csv', pd.concat(swaps, ignore_index=True))


def main(settings, job):
    import lpips
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    sys.path.insert(0, str(source_path('DiffBIR')))
    from diffbir.utils.common import instantiate_from_config, wavelet_reconstruction
    selection = load_selection(settings['selection_file']) if settings['stage'] == 'final' else None
    if selection:
        for name in ['workshop_design.py', 'workshop_worker.py', 'workshop_analysis.py', 'workshop_data.py',
                     'paper2_worker.py', 'metrics.py', 'stage_attribution.py', 'three_pipeline.py', 'opensource.py']:
            if selection.get('source_hashes', {}).get(name) != sha(Path(__file__).parent/name):
                raise ValueError('Development/final source mismatch: '+name)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run = Path(job)/'experiment'
    (run/'records').mkdir(parents=True, exist_ok=True)
    records = prepare(settings)
    if not records:
        raise ValueError('Empty source manifest')
    if not settings['smoke']:
        expected_domains = splits()['development'] if settings['stage'] == 'development' else {settings['dataset']: splits()['final'][settings['dataset']]}
        expected_ids = {(domain, int(i)) for domain, ids in expected_domains.items() for i in ids}
        actual_ids = [(r['dataset'], int(r['original_id'])) for r in records]
        if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
            raise ValueError('Source inventory differs from registered split')
    models = Path(settings['model_root'])
    cache = Path(settings['cache_root'])

    def checkpoint(name):
        options = list(models.rglob(name)) + [cache/'weights'/name]
        return next(p for p in options if p.is_file())

    hashes = {name: sha(checkpoint(name)) for name in EXPECTED_MODELS}
    if hashes != EXPECTED_MODELS:
        raise ValueError('Diffusion checkpoint hash mismatch')
    domain_names = sorted({r['dataset'] for r in records})
    cleaner_assets = {}
    for domain in domain_names:
        face = domain == 'ffhq'
        filename = 'face_swinir_v1.ckpt' if face else 'realesrgan_s4_swinir_100k.pth'
        repo = 'lxq007/DiffBIR' if face else 'lxq007/DiffBIR-v2'
        path = Path(hf_hub_download(repo, filename, cache_dir=str(cache/'huggingface')))
        expected = FACE_HASH if face else GENERAL_HASH
        if sha(path) != expected:
            raise ValueError('Cleaner checkpoint mismatch')
        cleaner_assets[domain] = dict(path=str(path), filename=filename, sha256=expected)
    review, floats, oracle = set(), set(), set()
    for domain in domain_names:
        ids = [r['image_id'] for r in records if r['dataset'] == domain]
        order = np.random.default_rng(20261031).permutation(ids).tolist()
        review.update(order[:20])
        floats.update(order[:5])
        if settings['stage'] == 'final' or settings['smoke']:
            oracle.update(order[:20])
    fixed = dict(version=VERSION, stage=settings['stage'], dataset=settings['dataset'], smoke=settings['smoke'],
                 n_images=len(records), configs=dict(baseline=BASELINE, intervention=candidate('cfg1')),
                 seeds=SEEDS, alphas=ALPHAS, sigma=.6, size=512, source_revisions=SOURCES,
                 autocast=False, correction='float32 outside autocast; before clipping; reference=z',
                 checkpoints=hashes, cleaners={d: {k: v for k, v in a.items() if k != 'path'} for d, a in cleaner_assets.items()},
                 split_sha256=splits()['sha256'], selection_sha256=selection['sha256'] if selection else None,
                 manifest=[{k: r[k] for k in ['dataset', 'image_id', 'sha256']} for r in records],
                 review_ids=sorted(review), exact_float_ids=sorted(floats), oracle_ids=sorted(oracle),
                 code_hashes={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
                 torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
                 gpu=torch.cuda.get_device_name(),
                 packages={n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'pillow', 'torchvision', 'scipy', 'lpips', 'torchsde']},
                 limitations=['One diffusion backbone; fixed synthetic blur, not standard published SR protocol',
                              'Cleaner and crop differ between face and natural domains',
                              'Only preselected review floats preserved; hashes cannot reconstruct unsaved arrays'])
    signature = digest(fixed)
    protocol_path = run/'protocol.json'
    if protocol_path.is_file() and json.loads(protocol_path.read_text())['protocol_sha256'] != signature:
        raise ValueError('Resume source/data/environment/settings mismatch; preserve old run and use new label')
    atomic_json(protocol_path, dict(**fixed, protocol_sha256=signature))
    atomic_json(run/'split_manifest.json', splits())
    if selection:
        atomic_json(run/'frozen_selection.json', selection)
    atomic_csv(run/'dataset_inventory.csv', pd.DataFrame(records))
    atomic_json(run/'environment.json', dict(packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}))
    model, sampler = load_diffbir(checkpoint('sd2.1-base-zsnr-laionaes5.ckpt'), checkpoint('DiffBIR_v2.1.pt'), 'cuda', version='v2.1')
    pipe = paper2_pipeline(model, sampler)
    atomic_json(run/'runtime_model.json', dict(latent_scale_factor=float(model.scale_factor),
                                             vae_parameter_dtype=str(next(model.vae.parameters()).dtype)))
    cleaner_models = {}
    for domain, asset in cleaner_assets.items():
        cleaner = instantiate_from_config(OmegaConf.load(source_path('DiffBIR')/'configs/inference/swinir.yaml'))
        load_cleaner_weights(cleaner, torch.load(asset['path'], map_location='cpu', weights_only=True))
        cleaner_models[domain] = cleaner.eval().cuda()
    torch.hub.set_dir(str(cache/'torch_hub'))
    perceptual = lpips.LPIPS(net='alex', version='0.1').eval().cuda()
    calls = 0
    for index, record in enumerate(records):
        ident, domain = record['image_id'], record['dataset']
        folder = run/'records'/ident
        folder.mkdir(exist_ok=True)
        marker = folder/'complete.json'
        if marker.is_file():
            done = json.loads(marker.read_text())
            if done['protocol_sha256'] != signature:
                raise ValueError('Completed image protocol mismatch')
            if all((run/p).is_file() and sha(run/p) == h for p, h in done['files'].items()):
                calls += done['sampling_calls']
                print('Resume verified:', ident, flush=True)
                continue
            raise ValueError('Completed image files damaged/missing; preserve run and use a fresh label')
        rows, checks, swaps, raw_hashes = [], [], [], []
        image_dir = run/'images'/ident
        if ident in review:
            image_dir.mkdir(parents=True, exist_ok=True)

        def metric(condition, strategy, seed, image, extra=None):
            if not torch.isfinite(image).all():
                raise ValueError('Nonfinite metric input')
            return dict(dataset=domain, image_id=ident, condition=condition, strategy=strategy, seed=seed,
                        **fidelity_measures(image, target, perceptual), **(extra or {}))

        review_strategies = ({key(c, a) for c in [6, 1] for a in ALPHAS} | {'blend_cfg6_beta0.5', 'blend_cfg1_beta0.5'}
                             if selection is None else {key(6, 1), selection['selected'], selection['tuned_blend'], selection['tuned_full_cfg']})

        def preserve(name, tensor, exact=False, preview=True):
            if ident in review and preview:
                save_image(tensor, image_dir/(name+'.png'))
            if exact and ident in floats:
                np.save(image_dir/(name+'.npy'), tensor.detach().float().cpu().numpy(), allow_pickle=False)

        with torch.inference_mode():
            with Image.open(record['path']) as im:
                array = np.array(im.convert('RGB')).copy()
            target = torch.from_numpy(array).permute(2, 0, 1)[None].float().cuda()/255
            if target.shape != (1, 3, 512, 512):
                raise ValueError('Unexpected reference shape')
            observed = extra_blur(target, .6)
            pipe.cleaner = cleaner_models[domain]
            clean = pipe.apply_cleaner(observed, False, 512, 256)
            for label, tensor in [('reference', target), ('observation', observed), ('cleaner', clean)]:
                preserve(label, tensor, exact=True)
            branches = [('restored', clean), ('degraded', observed)]
            if ident in oracle:
                branches.append(('oracle', target))
            image_calls = 0
            for condition, cond in branches:
                rows.append(metric(condition, 'cleaner', None, clean.clamp(0, 1)))
                latent = model.vae_encode(cond*2-1, sample=False)
                alternate = model.prepare_condition(cond, [''])['c_img']
                error = float((latent-alternate).abs().max())
                checks.append(dict(image_id=ident, condition=condition, check='vae_entry_parity', max_abs=error, passed=error <= 1e-5))
                if error > 1e-5:
                    raise ValueError('VAE entry parity failed')
                for cfg in [6, 1]:
                    config = BASELINE if cfg == 6 else candidate('cfg1')
                    model.cast_dtype(torch.float16)
                    for seed in SEEDS:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        raw = sample(pipe, cond, seed, config).float()
                        torch.cuda.synchronize()
                        seconds = time.perf_counter()-start
                        image_calls += 1
                        if not torch.isfinite(raw).all():
                            raise ValueError('Nonfinite raw output')
                        content = raw.detach().cpu().contiguous().numpy().tobytes()
                        checksum = hashlib.sha256(content).hexdigest()
                        raw_hashes.append(dict(image_id=ident, condition=condition, cfg=cfg, seed=seed,
                                               tensor_bytes_sha256=checksum, shape='1,3,512,512', dtype='float32'))
                        full = wavelet_reconstruction(raw, clean).float()
                        for alpha in ALPHAS:
                            output = corrected(raw, full, alpha)
                            name = key(cfg, alpha)
                            rows.append(metric(condition, name, seed, output,
                                               dict(seconds=seconds, raw_sha256=checksum)))
                            preserve(f'{condition}_{name}_seed{seed}', output,
                                     preview=seed == 11 and condition == 'restored' and name in review_strategies)
                        if condition != 'oracle':
                            output = (.5*clean+.5*corrected(raw, full, 1.)).clamp(0, 1)
                            name = f'blend_cfg{cfg}_beta0.5'
                            rows.append(metric(condition, name, seed, output, dict(seconds=seconds, raw_sha256=checksum)))
                            preserve(f'{condition}_{name}_seed{seed}', output,
                                     preview=seed == 11 and condition == 'restored' and name in review_strategies)
                        preserve(f'{condition}_cfg{cfg}_seed{seed}_raw', raw, exact=True,
                                 preview=seed == 11 and condition == 'restored')
                        if condition == 'restored' and ident in oracle:
                            for reference, ref in [('none', None), ('y', observed), ('z', clean), ('x', target)]:
                                shifted = raw if ref is None else wavelet_reconstruction(raw, ref).float()
                                swaps.append(dict(reference=reference, cfg=cfg, raw_sha256=checksum,
                                                  fraction_clipped=float(((shifted < 0) | (shifted > 1)).float().mean()),
                                                  **metric(condition, 'reference_swap', seed, shifted.clamp(0, 1))))
                del latent, alternate
            expected = 69 if ident in oracle else 50
            if len(rows) != expected or image_calls != len(branches)*6:
                raise ValueError('Image candidate coverage mismatch')
            atomic_csv(folder/'metrics.csv', pd.DataFrame(rows))
            atomic_csv(folder/'checks.csv', pd.DataFrame(checks))
            atomic_csv(folder/'raw_hashes.csv', pd.DataFrame(raw_hashes))
            if swaps:
                atomic_csv(folder/'reference_swaps.csv', pd.DataFrame(swaps))
            files = list(folder.glob('*.csv')) + (list(image_dir.glob('*')) if image_dir.exists() else [])
            atomic_json(marker, dict(protocol_sha256=signature, sampling_calls=image_calls,
                                     files={p.relative_to(run).as_posix(): sha(p) for p in files}))
            calls += image_calls
            del target, observed, clean, cond, raw, full, output, branches
        atomic_json(run/'status.json', dict(completed=False, completed_images=index+1, n_images=len(records), sampling_calls=calls))
        print(f'Completed {index+1}/{len(records)}: {ident}; calls={calls}', flush=True)
        # Small tables backup after every image, not only after an entire domain.
        collect(run)
        export(run, job)
        torch.cuda.empty_cache()
    collect(run)
    analyze(run, settings, selection)
    atomic_json(run/'status.json', dict(completed=True, completed_images=len(records), n_images=len(records), sampling_calls=calls))
    print('Complete. Verify and download:', export(run, job, complete=True), flush=True)
