"""Finite Paper 2 operating-point study; no learned sampling method."""
import hashlib
import json
from pathlib import Path

import numpy as np

VERSION = 'paper2_workshop_v1'
SEEDS = [11, 22, 33]
ALPHAS = [0., .5, 1.]
METRICS = ['psnr', 'ssim', 'lpips', 'mse']
PSNR_MARGIN = .2
TIE_TOL = 1e-6
METHOD = dict(cfg=[6, 1], seeds=SEEDS, alphas=ALPHAS, sigma=.6, size=512,
              sampler='edm_dpm++_3m_sde', steps=10, precision='fp16_no_autocast',
              correction='float32 before clipping; reference=z', psnr_margin_db=PSNR_MARGIN,
              lpips_tie_tolerance=TIE_TOL)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def splits():
    from .paper2_design import confirmation_ids
    from .ffhq_worker import selected_ids
    excluded = sorted(set(confirmation_ids()) | set(selected_ids(100)))
    rng = np.random.default_rng(20261031)
    faces = rng.permutation(np.setdiff1d(np.arange(60000, 70000), excluded)).tolist()
    train = rng.permutation(np.arange(1, 801))[:50].tolist()
    imagenet = np.random.default_rng(20261031).permutation(np.arange(1, 50001))[:1000].tolist()
    value = dict(version=VERSION, seed=20261031, excluded_historical_ffhq=excluded,
                 development=dict(ffhq=faces[:50], div2k=train),
                 final=dict(ffhq=faces[50:250], div2k=list(range(802, 901)),
                            urban100=list(range(1, 101)), imagenet=imagenet),
                 excluded_inspected_div2k=[801])
    value['sha256'] = digest(value)
    return value


def key(cfg, alpha):
    return f'cfg{int(cfg)}_alpha{alpha:g}'


DEFAULT = key(6, 1)
RAW1, RAW6 = key(1, 0), key(6, 0)
FULL1, FULL6 = key(1, 1), key(6, 1)
CORRECTIONS = [key(c, a) for c in [6, 1] for a in ALPHAS]
BLENDS = ['cleaner', key(6, 1), key(1, 1), 'blend_cfg6_beta0.5', 'blend_cfg1_beta0.5']


def corrected(raw, full, alpha):
    """Interpolate UNCLIPPED raw/corrected tensors; clip once at the end."""
    if alpha not in ALPHAS:
        raise ValueError('Unregistered correction strength')
    return (raw + alpha * (full - raw)).clamp(0, 1)


def select_configuration(image_means):
    """One global choice; equally weighted development domains, image means first."""
    import pandas as pd
    required = {'dataset', 'image_id', 'condition', 'strategy', *METRICS}
    if not required <= set(image_means):
        raise ValueError('Missing development fields')
    frame = image_means[image_means.condition == 'restored'].copy()
    if set(frame.dataset) != {'ffhq', 'div2k'}:
        raise ValueError('Development must contain both registered domains')
    if frame.duplicated(['dataset', 'image_id', 'strategy']).any():
        raise ValueError('Development records must be seed-averaged, unique per image/strategy')
    domains = sorted(frame.dataset.unique())
    for domain in domains:
        sub = frame[frame.dataset == domain]
        if sub.image_id.nunique() != 50:
            raise ValueError('Full development selection requires 50 images in each domain')
        for strategy in set(CORRECTIONS + BLENDS):
            if set(sub[sub.strategy == strategy].image_id) != set(sub.image_id):
                raise ValueError('Incomplete candidate image coverage')
    values = frame.groupby(['dataset', 'strategy'])[METRICS].mean()
    default = values.xs(DEFAULT, level='strategy')
    table = []
    for strategy in sorted(set(CORRECTIONS + BLENDS)):
        stats = values.xs(strategy, level='strategy')
        change = stats.psnr - default.psnr
        table.append(dict(strategy=strategy, lpips=float(stats.lpips.mean()),
                          psnr=float(stats.psnr.mean()), feasible=bool((change >= -PSNR_MARGIN).all()),
                          **{f'psnr_delta_{d}': float(change[d]) for d in domains}))
    table = pd.DataFrame(table)

    def choose(candidates):
        rows = table[table.strategy.isin(candidates) & table.feasible]
        best = rows.lpips.min()
        ties = rows[rows.lpips <= best + TIE_TOL].strategy.tolist()
        return DEFAULT if DEFAULT in ties else sorted(ties)[0]

    return dict(selected=choose(CORRECTIONS), tuned_blend=choose(BLENDS),
                tuned_full_cfg=choose([FULL1, FULL6]), psnr_margin_db=PSNR_MARGIN,
                lpips_tie_tolerance=TIE_TOL, selection_rule='equal-domain mean LPIPS; per-domain PSNR guard'), table


def load_selection(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    content = {k: v for k, v in value.items() if k != 'sha256'}
    if digest(content) != value.get('sha256'):
        raise ValueError('Selection file checksum mismatch')
    if value.get('version') != VERSION or value.get('split_sha256') != splits()['sha256']:
        raise ValueError('Selection was made with a different protocol/split')
    if value.get('method') != METHOD:
        raise ValueError('Selection method/settings differ from this runner')
    if value.get('smoke') is not False or value.get('completed_images') != 100:
        raise ValueError('Only the completed full development run can select a configuration')
    if value['selected'] not in CORRECTIONS or value['tuned_blend'] not in BLENDS or value['tuned_full_cfg'] not in [FULL1, FULL6]:
        raise ValueError('Unregistered selected method')
    return value
