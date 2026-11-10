"""Seed-averaged paired Paper 2 outcomes and frozen development selection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .workshop_design import (METRICS, DEFAULT, FULL1, FULL6, RAW1, RAW6,
                              VERSION, METHOD, splits, digest, select_configuration)


def interval(values):
    a = np.asarray(values, dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('Invalid image effects')
    if len(a) == 1:
        return dict(n_images=1, mean=float(a[0]), ci95_low=None, ci95_high=None)
    rng = np.random.default_rng(20261031)
    means = np.concatenate([a[rng.integers(0, len(a), size=(100, len(a)))].mean(1) for _ in range(20)])
    low, high = np.quantile(means, [.025, .975])
    return dict(n_images=len(a), mean=float(a.mean()), ci95_low=float(low), ci95_high=float(high))


def analyze(run, settings, selection=None):
    run = Path(run)
    frame = pd.read_csv(run/'results.csv')
    group = ['dataset', 'image_id', 'condition', 'strategy']
    means = frame.groupby(group, dropna=False)[METRICS].mean().reset_index()
    means.to_csv(run/'image_means.csv', index=False)
    means.groupby(['dataset', 'condition', 'strategy'])[METRICS].mean().reset_index().to_csv(run/'summary.csv', index=False)
    rows, summaries, tests = [], [], []
    from scipy.stats import ttest_1samp
    for (dataset, condition), g in means.groupby(['dataset', 'condition']):
        wide = {m: g.pivot(index='image_id', columns='strategy', values=m) for m in METRICS}
        for metric, table in wide.items():
            comparisons = {'cfg_raw': table[RAW1] - table[RAW6],
                           'cfg_full': table[FULL1] - table[FULL6],
                           'interaction': (table[FULL1]-table[FULL6])-(table[RAW1]-table[RAW6])}
            if selection is not None and condition != 'oracle':
                comparisons.update(selected_minus_default=table[selection['selected']]-table[DEFAULT],
                                   selected_minus_tuned_blend=table[selection['selected']]-table[selection['tuned_blend']],
                                   selected_minus_tuned_full=table[selection['selected']]-table[selection['tuned_full_cfg']],
                                   selected_minus_cleaner=table[selection['selected']]-table['cleaner'])
            for name, effect in comparisons.items():
                if effect.isna().any():
                    raise ValueError('Unpaired image outcomes')
                summaries.append(dict(dataset=dataset, condition=condition, metric=metric, contrast=name, **interval(effect)))
                rows.extend(dict(dataset=dataset, condition=condition, metric=metric, contrast=name,
                                 image_id=i, effect=float(v)) for i, v in effect.items())
                if condition == 'restored' and metric == 'lpips' and name in ['interaction', 'selected_minus_default']:
                    a = effect.to_numpy()
                    if len(a) < 2:
                        p = None
                    elif np.ptp(a) == 0:
                        p = 1. if a[0] == 0 else 0.
                    else:
                        p = float(ttest_1samp(a, 0.).pvalue)
                    tests.append(dict(dataset=dataset, contrast=name, metric=metric, n_images=len(a),
                                      mean=float(a.mean()), p_unadjusted=p,
                                      test='two-sided one-sample t test on paired image effects; IID images/CLT assumption'))
    pd.DataFrame(rows).to_csv(run/'paired_image_effects.csv', index=False)
    pd.DataFrame(summaries).to_csv(run/'paired_intervals.csv', index=False)
    pd.DataFrame(tests).to_csv(run/'primary_tests.csv', index=False)
    if settings['stage'] == 'development' and not settings['smoke']:
        chosen, candidates = select_configuration(means)
        candidates.to_csv(run/'development_candidates.csv', index=False)
        artifact = dict(**chosen, version=VERSION, split_sha256=splits()['sha256'], smoke=False,
                        method=METHOD,
                        completed_images=int(means[['dataset', 'image_id']].drop_duplicates().shape[0]),
                        development_protocol_sha256=json.loads((run/'protocol.json').read_text())['protocol_sha256'],
                        source_hashes=json.loads((run/'protocol.json').read_text())['code_hashes'],
                        image_means_sha256=__import__('hashlib').sha256((run/'image_means.csv').read_bytes()).hexdigest())
        artifact['sha256'] = digest(artifact)
        (run/'frozen_selection.json').write_text(json.dumps(artifact, indent=2), encoding='utf-8')


def combined_primary_tests(table_archives, output):
    """Only after all four FINAL runs: adjust the registered eight-test family."""
    import zipfile
    from .workshop_design import load_selection
    frames, fingerprints, choices = [], set(), set()
    for path in table_archives:
        with zipfile.ZipFile(path) as z:
            status = json.loads(z.read('experiment/status.json'))
            protocol = json.loads(z.read('experiment/protocol.json'))
            if not status.get('completed') or protocol['stage'] != 'final' or protocol['smoke']:
                raise ValueError('Combine only completed final tables, not smoke/development/historical results')
            if status['completed_images'] != len(splits()['final'][protocol['dataset']]):
                raise ValueError('Incomplete final domain')
            fingerprints.add(protocol['split_sha256'])
            choices.add(protocol['selection_sha256'])
            frames.append(pd.read_csv(z.open('experiment/primary_tests.csv')))
    all_tests = pd.concat(frames, ignore_index=True)
    if len(fingerprints) != 1 or len(choices) != 1 or next(iter(fingerprints)) != splits()['sha256']:
        raise ValueError('Runs use different splits or operating-point selections')
    expected = {(d, c) for d in ['ffhq', 'div2k', 'urban100', 'imagenet']
                for c in ['interaction', 'selected_minus_default']}
    if len(all_tests) != 8 or set(zip(all_tests.dataset, all_tests.contrast)) != expected:
        raise ValueError('Need exactly the eight registered tests across four final datasets')
    if all_tests.p_unadjusted.isna().any():
        raise ValueError('Undefined primary p-values')
    order = all_tests.p_unadjusted.to_numpy().argsort()
    adjusted = np.empty(8)
    running = 0.
    for rank, idx in enumerate(order):
        running = max(running, min(1., (8-rank)*float(all_tests.iloc[idx].p_unadjusted)))
        adjusted[idx] = running
    all_tests['p_holm_eight'] = adjusted
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    all_tests.to_csv(output, index=False)
    return all_tests
