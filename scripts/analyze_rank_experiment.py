"""Paired session bootstrap and source-preserving ranking diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from otto.evaluate import WEIGHTS
from otto.rank_experiment import load_queries, targets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['dev','audit'],default='dev')
    args=p.parse_args();root=Path('artifacts/rank-v1');base=root/args.split
    cfg=json.loads((root/'config.json').read_text())
    queries=load_queries('data/sample/train.jsonl' if args.split=='dev' else 'data/sample10/train.jsonl',
                         cfg['eval_graph_before'] if args.split=='dev' else cfg['audit_start'],
                         cfg['dev_queries_end'],args.split)
    names=['baseline','ranker','rules_adjacent','ranker_without_relation_features']
    hits={name:np.zeros((len(queries),3)) for name in names};den=np.zeros((len(queries),3))
    by_sid={sid:i for i,(sid,_,_) in enumerate(queries)}
    segments={name:{group:{kind:{'hits':0,'targets':0} for kind in WEIGHTS}
                    for group in ['seen_target','unseen_target']} for name in names}
    for i,(_,_,truth) in enumerate(queries):
        for j,kind in enumerate(WEIGHTS):den[i,j]=min(20,len(targets(truth,kind)))
    for name in names:
        with (base/f'{name}.csv').open() as f:
            for row in csv.DictReader(f):
                sid,kind=row['session_type'].rsplit('_',1);i=by_sid[int(sid)]
                _,events,truth=queries[i];target=targets(truth,kind)
                predicted=set(map(int,row['labels'].split()[:20]));seen={e['aid'] for e in events}
                hits[name][i,list(WEIGHTS).index(kind)]=len(target&predicted)
                for group,part in [('seen_target',target&seen),('unseen_target',target-seen)]:
                    segments[name][group][kind]['hits']+=len(part&predicted)
                    segments[name][group][kind]['targets']+=len(part)
    weights=np.asarray(list(WEIGHTS.values()));rng=np.random.default_rng(20260920)
    comparisons={}
    for other in ['baseline','rules_adjacent','ranker_without_relation_features']:
        delta=hits['ranker']-hits[other];draws=[]
        for _ in range(1000):
            ix=rng.integers(0,len(queries),len(queries));d=den[ix].sum(0)
            if np.all(d>0):draws.append(float((delta[ix].sum(0)/d)@weights))
        comparisons['ranker_minus_'+other]={'difference':float((delta.sum(0)/den.sum(0))@weights),
                                          'session_bootstrap_95pct':np.quantile(draws,[.025,.975]).tolist()}
    report={'sessions':len(queries),'bootstrap_samples':1000,'seed':20260920,
            'comparisons':comparisons,'target_segments':segments,
            'segment_note':'Uncapped unique target counts within each session, descriptive only; not official capped metric or a causal estimate.',
            'interval_note':'Session resampling assumes sessions independent; shared item/time correlations and model-selection uncertainty are not captured.'}
    Path(f'reports/rank_v1_{args.split}_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()
