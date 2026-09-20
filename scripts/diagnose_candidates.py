import json,pickle
from collections import Counter
from pathlib import Path
from otto.baseline import split_session,labels_for
from otto.evaluate import WEIGHTS
ref=json.loads(Path('artifacts/baseline/report.json').read_text()); cutoff=ref['cutoff_ms']; seed=ref['seed']
with open('artifacts/scale10/model.pkl','rb') as f: model=pickle.load(f)
model.policy='pooled'
counts={k:Counter() for k in WEIGHTS}; lengths=Counter(); n=0
for line in open('data/sample/train.jsonl'):
    row=json.loads(line)
    if row['events'][0]['ts']<cutoff: continue
    split=split_session(row['events'],seed,row['session'])
    if not split:continue
    prefix,future=split; truth=labels_for(future); n+=1
    lengths['history_unique_ge20']+=len({e['aid'] for e in prefix})>=20
    for kind in WEIGHTS:
        target=truth.get(kind)
        if target is None: continue
        target={target} if isinstance(target,int) else set(target)
        counts[kind]['denominator']+=min(20,len(target))
        pred=model.predict(prefix,kind,budget=200)
        for k in [20,50,100,200]:counts[kind][k]+=min(20,len(target.intersection(pred[:k])))
result={'validation_sessions':n,'history_unique_ge20':lengths['history_unique_ge20'],'candidate_oracle':{}}
for k in [20,50,100,200]:
    rec={kind:c[k]/c['denominator'] for kind,c in counts.items()}
    result['candidate_oracle'][k]={'recall':rec,'weighted':sum(WEIGHTS[t]*rec[t] for t in WEIGHTS)}
Path('reports/scale10_candidate_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
