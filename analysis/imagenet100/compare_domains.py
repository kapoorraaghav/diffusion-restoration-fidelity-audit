"""Validate the frozen ImageNet run and compare with the matched DIV2K scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'outputs/Paper2_Input_Pareto_Analysis'))
import analyze as pareto

ACTIONS=['input','cleaner','cfg1_full','cfg6_full','blend1']
METRICS=['psnr','ssim','lpips','dists']
SIGMAS={.6,1.2,3.}


def archive_frame(path,manifest):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        status=json.loads(z.read('experiment/status.json'))
        assert status['completed'] and not status['smoke'] and status['sampling_calls']==1800
        assert status['completed_sources']==100 and status['metric_rows']==5400
        protocol=json.loads(z.read('experiment/protocol.json'))
        assert protocol['frozen_subset_sha256']==hashlib.sha256((HERE/'frozen_imagenet100.json').read_bytes()).hexdigest()
        assert protocol['code_sha256']==json.loads((HERE/'RELEASE.json').read_text())['code_sha256']
        assert protocol['input_sha256']==manifest['input_archive_sha256']
        prior=json.loads((ROOT/'followup_v9/reviewer_extension/results_review/protocol.json').read_text())
        for k in ['seeds','cfg','steps','sampler','blur_boundary','blur_radius','models','metric_weights','dists_source_sha256','dists_weights_sha256','dists_resolution','sharpening','blend_order']:
            assert protocol[k]==prior[k],k
        ids={r['image_id'] for r in manifest['sources']}
        assert set(protocol['sources'])==ids
        all_scores=pd.read_csv(z.open('experiment/metrics.csv'))
        assert len(all_scores)==5400 and set(all_scores.source_id)==ids and set(all_scores.sigma)==SIGMAS
        assert not all_scores.duplicated(['source_id','sigma','candidate','seed']).any()
        records=[]
        for row in manifest['sources']:
            pre=f"experiment/records/{row['image_id']}/"
            mark=json.loads(z.read(pre+'complete.json'))
            parity=json.loads(z.read(pre+'cleaner_parity.json'))
            assert parity['reference_sha256']==row['sha256'] and parity['passed']
            assert max(parity['errors'].values())<=1e-3
            assert mark['sampling_calls']==18 and mark['protocol_sha256']==protocol['sha256']
            for name,key in [('metrics.csv','metrics_sha256'),('cleaner_parity.json','parity_sha256')]:
                assert hashlib.sha256(z.read(pre+name)).hexdigest()==mark[key]
            records.append(pd.read_csv(z.open(pre+'metrics.csv')))
        cols=['source_id','sigma','candidate','seed']
        def sorted_rows(df):return df.sort_values(cols).reset_index(drop=True)
        pd.testing.assert_frame_equal(sorted_rows(all_scores),sorted_rows(pd.concat(records)))
        return all_scores


def run_panel(frame,domain,n):
    frame=frame[frame.candidate.isin(ACTIONS)]
    ids=sorted(frame.source_id.unique())
    assert len(ids)==n and set(frame.sigma)==SIGMAS
    for sigma,group in frame.groupby('sigma'):
        for action,rows in group.groupby('candidate'):
            expected={-1} if action in ['input','cleaner'] else {11,22,33}
            assert rows.groupby('source_id').seed.agg(set).map(lambda s:s==expected).all()
        pareto.analyze(group,ids,ACTIONS,METRICS,domain,sigma,'matched_domain_comparison')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('imagenet_final_zip',nargs='?',type=Path)
    args=parser.parse_args()
    manifest=json.loads((HERE/'frozen_imagenet100.json').read_text())
    p=ROOT/'outputs/Paper2_MultiRobust_Submission_v37'
    div=pd.concat([pd.read_csv(p/'severity_metrics.csv'),pd.read_csv(p/'supplementary/sigma3/metrics.csv')])
    run_panel(div,'div2k',99)
    suffix='div2k_only'
    if args.imagenet_final_zip:
        imagenet=archive_frame(args.imagenet_final_zip,manifest)
        run_panel(imagenet,'imagenet',100)
        suffix='both_domains'
        known=pd.read_csv(p/'supplementary/input_audit/metrics.csv').query('domain=="imagenet"')
        mild=imagenet.query('sigma==0.6 and candidate in ["input","cleaner"]')
        cols=['source_id','candidate']
        d=mild.set_index(cols)[METRICS]-known.set_index(cols)[METRICS].reindex(mild.set_index(cols).index)
        if not np.isfinite(d).all().all():
            raise ValueError('Missing mild ImageNet baseline parity pairs')
        parity=d.abs().max()
        print('Mild baseline max errors:',parity.to_dict())
        if parity.psnr>1e-3 or parity[['ssim','lpips','dists']].max()>1e-3:
            raise ValueError('Unexpected mild baseline discrepancy. Inspect before reporting.')
    for name,data in [('frontiers',pareto.fronts),('paired_effects',pareto.effects),('means',pareto.means),('dominance',pareto.relations)]:
        pd.DataFrame(data).to_csv(HERE/f'{suffix}_{name}.csv',index=False)
    print(pd.DataFrame(pareto.fronts).drop(columns='actions').to_string(index=False))


if __name__=='__main__':
    main()
