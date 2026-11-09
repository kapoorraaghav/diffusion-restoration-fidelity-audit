"""Read score archives and write only the v32 audit outputs."""
from pathlib import Path
import re
import hashlib
import json
import math
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PAPER = ROOT/'outputs/Paper2_MultiRobust_Submission_v32'
BASE = ROOT/'work/consolidation_reproduction_check/source_tables'
FOLLOW = ROOT/'followup_v9'
DOMAINS = {'ffhq':200,'div2k':99,'urban100':100,'imagenet':1000}
CORE = ['psnr','ssim','lpips']
METRICS = CORE + ['dists']
PROVENANCE = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, member):
    PROVENANCE[str(path)] = digest(path)
    with zipfile.ZipFile(path) as z:
        return pd.read_csv(z.open(member))


def read_json(path, member):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read(member))


def stats(delta, metric, seed=20261108):
    values = delta.sort_index().to_numpy()
    assert np.isfinite(values).all()
    benefit = values if metric in ['psnr','ssim'] else -values
    rng = np.random.default_rng(seed)
    lo, hi = np.quantile(values[rng.integers(0,len(values),(2000,len(values)))].mean(axis=1), [.025,.975])
    tail = max(1,math.ceil(.05*len(values)))
    return dict(n=len(values),mean_delta=values.mean(),ci_low=lo,ci_high=hi,
                wins=int((benefit>0).sum()),ties=int((benefit==0).sum()),losses=int((benefit<0).sum()),
                worst_n=tail,worst_mean_delta=values[np.argsort(benefit,kind='stable')[:tail]].mean(),
                bootstrap_seed=seed)


def compare(candidate, baseline, metric, seed=20261108):
    pairs=pd.concat([candidate.rename('candidate'),baseline.rename('input')],axis=1).dropna()
    return dict(candidate_mean=pairs.candidate.mean(),baseline_mean=pairs.input.mean(),
                missing_candidate=len(baseline.index.difference(candidate.index)),
                missing_baseline=len(candidate.index.difference(baseline.index)),
                **stats(pairs.candidate-pairs.input,metric,seed))


def verify_seeds(frame, ids, deterministic):
    assert not frame.duplicated(['source_id','seed']).any()
    assert set(frame.source_id)==set(ids)
    expected = {-1} if deterministic else {11,22,33}
    assert frame.assign(seed=frame.seed.fillna(-1)).groupby('source_id').seed.agg(set).map(lambda x:x==expected).all()


def check_refs(inv, expected):
    joined=inv.set_index('image_id').sort_index()
    original=expected.set_index('image_id').sort_index()
    assert joined.index.equals(original.index)
    assert joined.sha256.equals(original.sha256)
    for field in ['original_sha256','preprocessing','crop_box','box','original_size','upsampled']:
        if field in joined and field in original:
            pd.testing.assert_series_equal(joined[field].astype('string').fillna(''),original[field].astype('string').fillna(''),check_dtype=False)


def parity(path, inv):
    errors=[]
    with zipfile.ZipFile(path) as z:
        for row in inv.itertuples():
            record=json.loads(z.read(f'experiment/records/{row.image_id}/cleaner_parity.json'))
            assert record['passed'] and record['reference_sha256']==row.sha256
            errors.extend(record['errors'].values())
    return max(errors)


def validate_tables(tex, table2, table3, input_table, division):
    def cells(label):
        end=tex.index(r'\label{tab:'+label+'}')
        start=tex.rfind(r'\midrule',0,end)+len(r'\midrule')
        stop=tex.index(r'\bottomrule',start)
        clean=re.sub(r'\\textbf\{([^}]+)\}',r'\1',tex[start:stop])
        return [line.strip().removesuffix(r'\\').strip().split(' & ')
                for line in clean.splitlines() if ' & ' in line]

    domain_keys={'ffhq':'ffhq','div2k':'div2k','urban100':'urban100','imagenet':'imagenet'}
    candidates=['cleaner','cfg6_alpha0','cfg6_alpha0.5','cfg6_alpha1',
                'cfg1_alpha0','cfg1_alpha0.5','cfg1_alpha1','blend_cfg6_beta0.5','blend_cfg1_beta0.5']
    t2=pd.DataFrame(table2).set_index('domain')
    for domain,up,cost,win,tail,dom in cells('cost'):
        r=t2.loc[domain_keys[domain.lower()]]
        assert up==f'{int(r.upsampled):,}/{int(r.n):,}'
        assert [float(cost),float(win),float(tail),float(dom)]==[float(f'{r.psnr_cost:.3f}'),float(f'{r.lpips_win_pct:.1f}'),float(f'{r.worst_tail:.4f}'),float(f'{r.dominance_pct:.1f}')]
    t3=pd.DataFrame(table3).set_index(['domain','candidate'])
    for candidate, row in zip(candidates,cells('results'),strict=True):
        for domain,entry in zip(DOMAINS,row[1:],strict=True):
            expected=t3.loc[(domain,candidate)]
            for metric,number,digits in zip(CORE,entry.split('/'),[2,3,3],strict=True):
                assert float(number)==float(f'{expected[metric]:.{digits}f}'),(domain,candidate,metric)
    t6=input_table.set_index('domain')
    for domain,*numbers in cells('input'):
        for metric,number,digits in zip(METRICS,numbers,[3,4,4,4],strict=True):
            assert float(number)==float(f'{t6.loc[domain.lower(),metric]:.{digits}f}')
    t7=pd.DataFrame(division).set_index(['sigma','candidate','metric'])
    names={'Input only':'input','Cleaner':'cleaner','Frozen sharpening':'sharpen','CFG6/full':'cfg6_full','CFG1/full':'cfg1_full','Blend1':'blend1'}
    for method,*numbers in cells('severity'):
        for metric,entry,digits in zip(METRICS,numbers,[3,4,4,4],strict=True):
            for sigma,number in zip([.6,1.2,3.],entry.split('/'),strict=True):
                assert float(number)==float(f'{t7.loc[sigma,names[method],metric].candidate_mean:.{digits}f}')
    print('Printed tables 2, 3, 6 and 7 reproduce at display precision.')

def main():
    tex=PAPER/'main.tex'
    before=digest(tex)
    audit=Path('<LOCAL_HOME>/Downloads/p2input_all_full_v1_FOCUS_FINAL.zip')
    input_scores=load(audit,'experiment/metrics.csv')
    input_inv=load(audit,'experiment/source_inventory.csv')
    input_protocol=read_json(audit,'experiment/protocol.json')
    assert input_protocol['sigma']==.6 and input_protocol['resolution']==512
    input_parity=parity(audit,input_inv)
    extension=[
        FOLLOW/'returned_archives/p2review_div2k_severity_dists_full_v1_FOCUS_FINAL.zip',
        FOLLOW/'div2k_sigma3/p2review_div2k_sigma3_full_v1_FOCUS_FINAL.zip']
    original={}
    coverage=[]
    mild=[]
    cross_checks=[]
    table2=[]
    table3=[]
    headline=[]
    for domain,n in DOMAINS.items():
        path=BASE/f'p2w_{domain}_full_v1_CROSS_tables.zip'
        frame=load(path,'experiment/results.csv').rename(columns={'image_id':'source_id'})
        inv=load(path,'experiment/dataset_inventory.csv')
        original[domain]=inv
        current=input_inv[input_inv.dataset==domain]
        check_refs(current,inv)
        protocol=read_json(path,'experiment/protocol.json')
        assert protocol['sigma']==.6 and protocol['size']==512 and protocol['seeds']==[11,22,33]
        inp=input_scores.query('domain == @domain and candidate == "input"').set_index('source_id')
        assert len(inp)==n and inp.index.is_unique
        for (branch,candidate),group in frame.groupby(['condition','strategy']):
            ids=inv.image_id if branch!='oracle' else group.source_id.unique()
            assert len(ids)==(20 if branch=='oracle' else n)
            verify_seeds(group,ids,candidate=='cleaner')
            score=group.groupby('source_id')[CORE].mean()
            coverage.append(dict(domain=domain,sigma=.6,branch=branch,candidate=candidate,n=len(score),metrics='/'.join(CORE),extra_output_only='mse/false_edge_proxy/valid_edge_recall',score_archive=path.name))
            for metric in CORE:
                ref=inp[metric].reindex(score.index) if branch=='oracle' else inp[metric]
                result=compare(score[metric],ref,metric)
                mild.append(dict(domain=domain,sigma=.6,branch=branch,candidate=candidate,metric=metric,**result))
            if branch=='restored':
                table3.append(dict(domain=domain,candidate=candidate,n=n,**score.mean().to_dict()))
        restored=frame.query('condition == "restored"').groupby(['source_id','strategy'])[CORE].mean()
        blend=restored.xs('blend_cfg1_beta0.5',level='strategy').sort_index()
        cleaner=restored.xs('cleaner',level='strategy').sort_index()
        replay=input_scores.query('domain == @domain and candidate == "cleaner"').set_index('source_id').sort_index()
        cross_checks.append(dict(check='archived vs input-audit cleaner',domain=domain,n=n,**(cleaner-replay[CORE]).abs().max().to_dict()))
        coverage.append(dict(domain=domain,sigma=.6,branch='deterministic',candidate='input/cleaner',n=n,metrics='/'.join(METRICS),score_archive=audit.name))
        cfg1=restored.xs('cfg1_alpha1',level='strategy').sort_index()
        weights=np.random.default_rng(20261102).multinomial(n,np.full(n,1/n),size=2000)/n
        vec=(weights@(blend-cfg1).to_numpy())*np.array([1,1,-1])
        old=compare(blend.lpips,cleaner.lpips,'lpips',20261102)
        table2.append(dict(domain=domain,n=n,upsampled=int(inv.upsampled.fillna(False).sum()) if 'upsampled' in inv else 0,
                           psnr_cost=(cleaner.psnr-blend.psnr).mean(),lpips_win_pct=100*old['wins']/n,
                           worst_tail=old['worst_mean_delta'],dominance_pct=100*((vec>=-1e-10).all(1)&(vec>1e-10).any(1)).mean()))
        if domain=='imagenet':
            headline=[dict(comparator='cleaner',**old),dict(comparator='input',**compare(blend.lpips,inp.lpips,'lpips'))]
        names={'ffhq':'p2focus_sharpen_final_ffhq_full_v1_FOCUS_FINAL.zip','div2k':'p2focus_div2k_full_v1_FOCUS_FINAL.zip','urban100':'p2focus_urban100_full_v1_FOCUS_FINAL.zip','imagenet':'p2focus_imagenet_full_v3_FOCUS_FINAL.zip'}
        sharp=Path('<LOCAL_HOME>/Downloads')/names[domain]
        sharp_inv=load(sharp,'experiment/source_inventory.csv')
        check_refs(sharp_inv,inv)
        sharp_scores=load(sharp,'experiment/metrics.csv')
        cross_checks.append(dict(check='sharpen reference hashes/cleaner parity',domain=domain,n=n,max_parity_error=parity(sharp,sharp_inv)))
        for candidate,group in sharp_scores.groupby('candidate'):
            if candidate=='cleaner':
                continue
            verify_seeds(group,inv.image_id,True)
            coverage.append(dict(domain=domain,sigma=.6,branch='deterministic cleaner sharpening',candidate=candidate,n=n,metrics='/'.join(CORE),score_archive=sharp.name))
            for metric in CORE:
                mild.append(dict(domain=domain,sigma=.6,branch='deterministic cleaner sharpening',candidate=candidate,metric=metric,**compare(group.set_index('source_id')[metric],inp[metric],metric)))

    division=[]
    for path in extension:
        inv=load(path,'experiment/source_inventory.csv')
        check_refs(inv,original['div2k'])
        cross_checks.append(dict(check='severity reference hashes/mild cleaner parity',domain='div2k',n=99,max_parity_error=parity(path,inv)))
        data=load(path,'experiment/metrics.csv')
        p=read_json(path,'experiment/protocol.json')
        assert p['seeds']==[11,22,33]
        assert set(data.sigma)==set(p['sigma'])
        for sigma,group in data.groupby('sigma'):
            inp=group.query('candidate=="input"').set_index('source_id')
            for candidate,rows in group.groupby('candidate'):
                deterministic=candidate in ['input','cleaner','sharpen']
                verify_seeds(rows,inv.image_id,deterministic)
                score=rows.groupby('source_id')[METRICS].mean()
                coverage.append(dict(domain='div2k',sigma=sigma,branch='deterministic' if deterministic else 'restored',candidate=candidate,n=99,metrics='/'.join(METRICS),score_archive=path.name))
                for metric in METRICS:
                    division.append(dict(sigma=sigma,candidate=candidate,metric=metric,**compare(score[metric],inp[metric],metric)))
            if sigma==.6:
                b=load(BASE/'p2w_div2k_full_v1_CROSS_tables.zip','experiment/results.csv')
                for candidate,strategy in {'cleaner':'cleaner','cfg1_raw':'cfg1_alpha0','cfg6_raw':'cfg6_alpha0','cfg1_full':'cfg1_alpha1','cfg6_full':'cfg6_alpha1','blend1':'blend_cfg1_beta0.5'}.items():
                    old=b.query('condition=="restored" and strategy==@strategy').groupby('image_id')[CORE].mean()
                    new=group.query('candidate==@candidate').groupby('source_id')[CORE].mean()
                    cross_checks.append(dict(check=f'DIV2K mild replay vs original {candidate}',domain='div2k',n=99,**(new-old).abs().max().to_dict()))
    baseline_table=input_scores.query('candidate == "input"').groupby('domain')[METRICS].mean().reset_index()
    validate_tables(tex.read_text(encoding='utf-8'),table2,table3,baseline_table,division)
    assert digest(tex)==before
    baseline_table.to_csv(OUT/'table6_recomputed.csv',index=False)
    for name,rows in [('imagenet_lpips',headline),('div2k_all_methods',division),('mild_input_comparisons',mild),('score_coverage',coverage),('cross_archive_checks',cross_checks),('table2_recomputed',table2),('table3_recomputed',table3)]:
        pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(manuscript_sha256=before,input_parity_max_error=input_parity,archives=PROVENANCE),indent=2))
    print(pd.DataFrame(headline).to_string(index=False))
    print(pd.DataFrame(cross_checks).to_string(index=False))
    print(pd.DataFrame(table2).to_string(index=False))


if __name__=='__main__':
    main()
