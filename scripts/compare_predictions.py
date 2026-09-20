"""Paired session bootstrap of official weighted Recall@20 differences."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from otto.evaluate import WEIGHTS,evaluate
from otto.rank_experiment import targets


def compare(labels, base, candidate, repeats=1000):
    # Strictly validate both inputs before diagnostics.
    br=evaluate(labels,base);cr=evaluate(labels,candidate)
    truth=[json.loads(s) for s in Path(labels).read_text().splitlines()]
    index={r['session']:i for i,r in enumerate(truth)}
    counts=[]
    den=np.asarray([[min(20,len(targets(r['labels'],k))) for k in WEIGHTS] for r in truth],dtype=float)
    for path in [base,candidate]:
        hit=np.zeros_like(den)
        with open(path) as f:
            for row in csv.DictReader(f):
                sid,k=row['session_type'].rsplit('_',1);i=index[int(sid)]
                hit[i,list(WEIGHTS).index(k)]=len(set(map(int,row['labels'].split()[:20]))&targets(truth[i]['labels'],k))
        counts.append(hit)
    delta=counts[1]-counts[0];weights=np.asarray(list(WEIGHTS.values()));rng=np.random.default_rng(20260920)
    draws=[]
    for _ in range(repeats):
        ix=rng.integers(0,len(truth),len(truth));d=den[ix].sum(0)
        if np.all(d>0):draws.append(float((delta[ix].sum(0)/d)@weights))
    return {'sessions':len(truth),'base_score':br['score'],'candidate_score':cr['score'],
            'difference':cr['score']-br['score'],'session_bootstrap_95pct':np.quantile(draws,[.025,.975]).tolist(),
            'repeats':repeats,'seed':20260920,'caveat':'Assumes independent sessions; does not model shared item/time correlations or adaptive model selection.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['labels','base','candidate','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();r=compare(a.labels,a.base,a.candidate)
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

if __name__=='__main__':main()
