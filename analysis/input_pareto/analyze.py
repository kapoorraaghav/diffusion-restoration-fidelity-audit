"""Retrospective, matched-source Pareto analysis using v32 audit archives."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT.parent/'Paper2_v32_Evidence_Audit'))
from experiments.revision_analysis import nondominated
from recompute import BASE, CORE, DOMAINS, FOLLOW, METRICS, load

SIGN = np.array([1,1,-1,-1])
SEED = 20261102
RESAMPLES = 2000
TOL = 1e-10
means, effects, relations, fronts = [], [], [], []


def analyze(frame, ids, actions, metrics, domain, sigma, series):
    assert set(frame.source_id) == set(ids)
    assert set(frame.candidate) == set(actions)
    assert not frame.duplicated(['source_id','candidate','seed']).any()
    img = frame.groupby(['source_id','candidate'])[metrics].mean()
    ids = sorted(ids)
    panel = np.stack([img[m].unstack('candidate').loc[ids,actions].to_numpy() for m in metrics], axis=-1)
    assert np.isfinite(panel).all()
    n = len(ids)
    weights = np.random.default_rng(SEED).multinomial(n,np.full(n,1/n),size=RESAMPLES)/n
    boot = (weights@panel.reshape(n,-1)).reshape(RESAMPLES,len(actions),len(metrics))
    mean = panel.mean(axis=0)
    signs = SIGN[:len(metrics)]
    oriented = mean*signs
    desc = dict(series=series,domain=domain,sigma=sigma,n=n)
    for i,action in enumerate(actions):
        means.append(dict(**desc,action=action,**dict(zip(metrics,mean[i]))))
        delta = (panel[:,i]-panel[:,actions.index('input')])*signs
        boot_delta = weights@delta
        lo,hi = np.quantile(boot_delta,[.025,.975],axis=0)
        for j,metric in enumerate(metrics):
            win,tie,harm = ((delta[:,j]>0).sum(),(delta[:,j]==0).sum(),(delta[:,j]<0).sum())
            effects.append(dict(**desc,action=action,metric=metric,mean_improvement=delta[:,j].mean(),
                ci95_low=lo[j],ci95_high=hi[j],wins=win,ties=tie,harmed=harm,harmed_fraction=harm/n))
    for k in [3,4] if len(metrics)==4 else [3]:
        selected = nondominated(-oriented[:,:k])
        fronts.append(dict(**desc,objective_count=k,actions=';'.join(actions),
            frontier=';'.join(a for a,keep in zip(actions,selected) if keep)))
        for i,a in enumerate(actions):
            for j,b in enumerate(actions):
                if i==j:
                    continue
                delta = oriented[i,:k]-oriented[j,:k]
                draws = (boot[:,i,:k]-boot[:,j,:k])*signs[:k]
                wins = (draws>=-TOL).all(axis=1)&(draws>TOL).any(axis=1)
                relations.append(dict(**desc,objective_count=k,action=a,comparator=b,
                    mean_dominates=bool((delta>=-TOL).all() and (delta>TOL).any()),
                    worse_metrics=';'.join(m for m,d in zip(metrics[:k],delta) if d < -TOL),
                    better_metrics=';'.join(m for m,d in zip(metrics[:k],delta) if d > TOL),
                    dominance_resamples=int(wins.sum()),resamples=RESAMPLES,descriptive_frequency=wins.mean()))


def main():
    downloads=Path('<LOCAL_HOME>/Downloads')
    baseline=load(downloads/'p2input_all_full_v1_FOCUS_FINAL.zip','experiment/metrics.csv')
    sharp_names={
        'ffhq':'p2focus_sharpen_final_ffhq_full_v1_FOCUS_FINAL.zip',
        'div2k':'p2focus_div2k_full_v1_FOCUS_FINAL.zip',
        'urban100':'p2focus_urban100_full_v1_FOCUS_FINAL.zip',
        'imagenet':'p2focus_imagenet_full_v3_FOCUS_FINAL.zip'}
    for domain,n in DOMAINS.items():
        b=load(BASE/f'p2w_{domain}_full_v1_CROSS_tables.zip','experiment/results.csv')
        b=b.query('condition == "restored"').rename(columns={'strategy':'candidate','image_id':'source_id'})
        h=load(downloads/sharp_names[domain],'experiment/metrics.csv').query('candidate != "cleaner"')
        inp=baseline.query('domain == @domain and candidate == "input"')
        frame=pd.concat([b,h,inp],ignore_index=True)
        actions=['input',*sorted(b.candidate.unique()),*sorted(h.candidate.unique())]
        assert len(actions)==25 and len(inp)==n
        analyze(frame,inp.source_id.tolist(),actions,CORE,domain,.6,'domain_primary')
    archives=[
        FOLLOW/'returned_archives/p2review_div2k_severity_dists_full_v1_FOCUS_FINAL.zip',
        FOLLOW/'div2k_sigma3/p2review_div2k_sigma3_full_v1_FOCUS_FINAL.zip']
    severity=pd.concat([load(p,'experiment/metrics.csv') for p in archives],ignore_index=True)
    ids=sorted(severity.source_id.unique())
    actions=['input','cleaner','sharpen','cfg1_raw','cfg1_full','cfg6_raw','cfg6_full','blend1']
    assert len(ids)==99 and set(severity.sigma)=={.6,1.2,3.}
    for sigma,g in severity.groupby('sigma'):
        analyze(g,ids,actions,METRICS,'div2k',sigma,'severity_matched')
    for name,rows in [('frontiers',fronts),('action_means',means),('input_effects',effects),('dominance',relations)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    print(pd.DataFrame(fronts).drop(columns='actions').to_string(index=False))


if __name__=='__main__':
    main()
