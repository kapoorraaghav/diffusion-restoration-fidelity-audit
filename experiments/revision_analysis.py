"""Auditable finite-set fronts, preferences, hypervolume and per-image diagnostics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .revision_design import PREFERENCES, VERSION, digest, frozen_protocol

OBJECTIVES = ['psnr', 'ssim', 'lpips']
SIGNS = np.array([-1., -1., 1.])  # minimization


def nondominated(loss):
    a = np.asarray(loss, float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('Expected finite objective matrix')
    delta = a[:, None, :] - a[None, :, :]
    dominates = (delta <= 1e-10).all(-1) & (delta < -1e-10).any(-1)
    return ~dominates.any(0)


def hypervolume(loss, reference):
    """Exact union volume of minimization boxes; points outside reference contribute zero."""
    a, r = np.asarray(loss, float), np.asarray(reference, float)
    if a.ndim != 2 or a.shape[1] != len(r) or not np.isfinite(a).all() or not np.isfinite(r).all():
        raise ValueError('Invalid hypervolume inputs')
    a = a[(a < r).all(1)]
    if not len(a):
        return 0.
    a = a[nondominated(a)]
    if len(r) == 1:
        return float(r[0] - a[:, 0].min())
    edges = sorted(set(a[:, 0].tolist() + [float(r[0])]))
    volume = 0.
    for low, high in zip(edges[:-1], edges[1:]):
        active = a[a[:, 0] <= low + 1e-14, 1:]
        volume += (high-low)*hypervolume(active, r[1:])
    return float(volume)


def normalization(development):
    d = development[development.strategy!='vae_x_oracle']
    # First images within domain, then equally weighted domains. Never fit scaling on test data.
    means = d.groupby(['dataset', 'strategy'])[OBJECTIVES].mean().groupby('strategy').mean()
    loss = means.to_numpy()*SIGNS
    low = loss.min(0)
    span = loss.max(0)-low
    if (span < 1e-12).any():
        raise ValueError('Development range is degenerate; do not silently alter normalization')
    return dict(ideal=low.tolist(), scale=span.tolist(), objectives=OBJECTIVES, signs=SIGNS.tolist())


def normalized(frame, scaling):
    return (frame[OBJECTIVES].to_numpy()*SIGNS-np.array(scaling['ideal']))/np.array(scaling['scale'])


def freeze_preferences(development, protocol_sha256, output):
    from .revision_design import candidates
    deployable=['cleaner','input_y','vae_y','vae_z']+[c['strategy'] for c in candidates()]
    d = development[development.strategy.isin(deployable)].copy()
    if d.duplicated(['dataset', 'image_id', 'strategy']).any():
        raise ValueError('Preferences require seed-averaged image records')
    expected = frozen_protocol()['development']
    if set(d.dataset) != set(expected):
        raise ValueError('Missing development domains')
    needed = set(deployable)
    for domain, g in d.groupby('dataset'):
        if g.image_id.nunique() != 50 or set(g.strategy) != needed:
            raise ValueError('Incomplete 50+50 development grid')
        if {int(str(i).rsplit('_',1)[-1]) for i in g.image_id.unique()} != set(expected[domain]):
            raise ValueError('Development source IDs differ from the frozen split')
        if not (g.groupby('image_id').strategy.nunique() == len(needed)).all():
            raise ValueError('Incomplete candidate coverage')
    scaling = normalization(d)
    scaling['hv_references']=[1.1,2.,4.]
    per_domain = d.groupby(['dataset', 'strategy'])[OBJECTIVES].mean()
    base = per_domain.xs('cfg6_alpha1', level='strategy').psnr
    feasible = per_domain.psnr.unstack('dataset').subtract(base, axis='columns').ge(-.2).all(1)
    means = per_domain.groupby('strategy').mean()
    loss = normalized(means, scaling)
    choices = {}
    for name, weights in PREFERENCES.items():
        scores = pd.Series(loss@np.array(weights), index=means.index)
        scores = scores[feasible]
        ties = sorted(scores[scores <= scores.min()+1e-6].index)
        choices[name] = 'cfg6_alpha1' if 'cfg6_alpha1' in ties else ties[0]
    value = dict(version=VERSION, smoke=False, completed_images=100, normalization=scaling,
                 choices=choices, weights=PREFERENCES, protocol_sha256=protocol_sha256,
                 design_sha256=frozen_protocol()['sha256'], development_data_sha256=digest(d.sort_values(['dataset','image_id','strategy']).to_dict('records')))
    value['sha256'] = digest(value)
    Path(output).write_text(json.dumps(value, indent=2))
    return value


def load_preferences(path):
    value = json.loads(Path(path).read_text())
    if digest({k:v for k,v in value.items() if k!='sha256'}) != value.get('sha256'):
        raise ValueError('Preference file checksum mismatch')
    if value.get('version') != VERSION or value.get('smoke') is not False or value.get('completed_images') != 100:
        raise ValueError('Need completed full development selection, not smoke')
    if value.get('design_sha256') != frozen_protocol()['sha256']:
        raise ValueError('Preference file design differs')
    return value


def effect_summary(effect, signs=1.):
    a = np.asarray(effect, float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('Empty/nonfinite image effects')
    harm = a*signs  # positive = candidate worsens this metric
    rng = np.random.default_rng(20261102)
    samples = np.concatenate([a[rng.integers(len(a), size=(100, len(a)))].mean(1) for _ in range(20)])
    low, high = np.quantile(samples, [.025, .975])
    return dict(n_images=len(a), mean=float(a.mean()), ci95_low=float(low), ci95_high=float(high),
                q05=float(np.quantile(a,.05)), median=float(np.median(a)), q95=float(np.quantile(a,.95)),
                win_fraction=float((harm < -1e-10).mean()), tie_fraction=float((abs(harm)<=1e-10).mean()),
                loss_fraction=float((harm>1e-10).mean()), worst5pct_mean_harm=float(np.sort(harm)[-max(1,int(np.ceil(.05*len(harm)))):].mean()))


def paired_diagnostics(image_means):
    rows = []
    group = ['dataset', 'condition'] + (['profile'] if 'profile' in image_means else [])
    for labels,g in image_means.groupby(group):
        fields = dict(zip(group, labels))
        for metric in ['psnr','ssim','lpips','mse','musiq']:
            if metric not in g:
                continue
            table = g.pivot(index='image_id', columns='strategy', values=metric)
            if table.isna().any().any():
                raise ValueError('Unpaired candidate coverage')
            harm_sign = -1 if metric in ['psnr','ssim','musiq'] else 1
            for baseline in ['cleaner','input_y','vae_z','cfg6_alpha1']:
                if baseline not in table:
                    continue
                for name in table:
                    if name == baseline or name == 'vae_x_oracle':
                        continue
                    rows.append(dict(**fields, metric=metric, contrast=f'{name}_minus_{baseline}',
                                     **effect_summary(table[name]-table[baseline], harm_sign)))
            keys=['cfg1_alpha0','cfg6_alpha0','cfg1_alpha1','cfg6_alpha1']
            if all(k in table for k in keys):
                delta = (table[keys[2]]-table[keys[3]])-(table[keys[0]]-table[keys[1]])
                rows.append(dict(**fields, metric=metric, contrast='cfg_by_correction_interaction',
                                 **effect_summary(delta, harm_sign)))
    return pd.DataFrame(rows)


def fronts_and_hv(image_means, scaling):
    means = image_means.groupby(['dataset','condition','strategy'])[OBJECTIVES].mean().reset_index()
    rows, hv = [], []
    for (domain,condition),g in means.groupby(['dataset','condition']):
        # All deployable controls remain eligible. Only the clean-target VAE oracle is excluded.
        g = g[g.strategy!='vae_x_oracle'].copy()
        loss = normalized(g,scaling)
        g['nondominated'] = nondominated(loss)
        rows.append(g)
        sets={'all_evaluated':np.ones(len(g),bool), 'diffusion_only':g.strategy.str.match(r'cfg\d+_alpha[\d.]+$').to_numpy(),
              'historical_nine':g.strategy.isin(['cleaner','cfg1_alpha0','cfg1_alpha0.5','cfg1_alpha1','cfg6_alpha0','cfg6_alpha0.5','cfg6_alpha1','blend_cfg1_beta0.5','blend_cfg6_beta0.5']).to_numpy()}
        for name,mask in sets.items():
            if not mask.any():continue
            for r in scaling.get('hv_references',[1.,1.1,1.2]):
                hv.append(dict(dataset=domain,condition=condition,candidate_set=name,reference=r,
                               hypervolume=hypervolume(loss[mask],[r]*3),n_candidates=int(mask.sum()),
                               outside_reference=int((~(loss[mask]<r).all(1)).sum())))
    return pd.concat(rows,ignore_index=True),pd.DataFrame(hv)


def analyze_run(run, stage, protocol_sha256):
    run=Path(run)
    frame=pd.read_csv(run/'results.csv')
    groups=['dataset','image_id','condition','profile','strategy']
    means=frame.groupby(groups)[['psnr','ssim','lpips','mse','musiq']].mean().reset_index()
    means.to_csv(run/'image_means.csv',index=False)
    means.groupby(['dataset','condition','profile','strategy'])[['psnr','ssim','lpips','mse','musiq']].mean().reset_index().to_csv(run/'summary.csv',index=False)
    paired_diagnostics(means).to_csv(run/'paired_tail_statistics.csv',index=False)
    if stage=='severity':severity_changes(means).to_csv(run/'paired_severity_changes.csv',index=False)
    return means


def severity_changes(means):
    """Paired change in a candidate-vs-default effect, relative to the base degradation."""
    rows=[]
    for domain,g in means.groupby('dataset'):
        for metric in ['psnr','ssim','lpips','mse','musiq']:
            t=g.pivot(index=['image_id','profile'],columns='strategy',values=metric)
            for strategy in t:
                if strategy in ['cfg6_alpha1','vae_x_oracle']:continue
                effects=(t[strategy]-t.cfg6_alpha1).unstack('profile')
                if 'blur06' not in effects:raise ValueError('Severity analysis requires the matched base profile')
                for profile in effects:
                    if profile=='blur06':continue
                    delta=effects[profile]-effects.blur06
                    if delta.isna().any():raise ValueError('Severity profiles lack matched source coverage')
                    rows.append(dict(dataset=domain,strategy=strategy,metric=metric,profile=profile,
                        contrast='candidate_minus_default_change_from_blur06',
                        **effect_summary(delta,-1 if metric in ['psnr','ssim','musiq'] else 1)))
    return pd.DataFrame(rows)
