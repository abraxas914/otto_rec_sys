"""Controlled scaling experiment with fixed M0 validation queries."""
import argparse,csv,json,pickle,time,hashlib
from pathlib import Path
from collections import Counter
from otto.covis import Covis
from otto.baseline import split_session,labels_for
from otto.evaluate import evaluate,WEIGHTS

class HybridCovis(Covis):
    def __init__(self):
        super().__init__()
        self.policy='typed'
    def predict(self,events,kind,weighted=True,budget=20):
        if self.policy=='typed':
            return super().predict(events,kind,weighted,budget)
        recent_scores=Counter()
        for pos,event in enumerate(reversed(events)):
            recent_scores[event['aid']]+={'clicks':1,'carts':6,'orders':3}[event['type']]*(0.95**pos)
        recent=sorted(recent_scores,key=lambda a:(-recent_scores[a],a))
        weights={'clicks':1.0} if kind=='clicks' else {'clicks':0.25,'carts':1.0,'orders':1.0}
        if self.policy=='click_fallback':
            weights={kind:1.0,'clicks':0.3} if kind!='clicks' else {'clicks':1.0}
        scores=Counter()
        for pos,aid in enumerate(list(dict.fromkeys(e['aid'] for e in reversed(events)))[:20]):
            for source,weight in weights.items():
                neighbors=self.graph[source].get(aid,[])
                norm=max((s for _,s in neighbors),default=1)
                for target,strength in neighbors:
                    scores[target]+=weight*strength/norm/(1+pos)
        candidates=sorted(scores,key=lambda a:(-scores[a],a))
        return list(dict.fromkeys(recent+candidates+self.pop[kind]))[:budget]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',default='data/sample10/train.jsonl')
    p.add_argument('--output',default='artifacts/scale10')
    p.add_argument('--fit-final',action='store_true')
    p.add_argument('--policy',default='typed',choices=['typed','click_fallback','pooled'])
    args=p.parse_args()
    from otto.scale import HybridCovis
    ref=json.loads(Path('artifacts/baseline/report.json').read_text())
    cutoff,seed=ref['cutoff_ms'],ref['seed']
    model=HybridCovis(); start=time.perf_counter(); nfit=nevents=0
    with open(args.train) as stream:
        for i,line in enumerate(stream,1):
            row=json.loads(line); events=row['events']
            if args.fit_final or events[0]['ts']<cutoff:
                events=events if args.fit_final else [e for e in events if e['ts']<cutoff]
                model.add(events);nfit+=1;nevents+=len(events)
            if i%200000==0: print(f'Scanned {i:,}; fitted {nfit:,}',flush=True)
    model.finish();model.policy=args.policy
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with (out/'model.pkl').open('wb') as f:pickle.dump(model,f)
    report={'training':args.train,'fit_sessions':nfit,'fit_events':nevents,'cutoff_ms':None if args.fit_final else cutoff,'fit_seconds':time.perf_counter()-start,'policy':args.policy}
    if not args.fit_final:
        queries=[]
        with open('data/sample/train.jsonl') as f:
            for line in f:
                row=json.loads(line)
                if row['events'][0]['ts']>=cutoff:
                    split=split_session(row['events'],seed,row['session'])
                    if split:queries.append((row['session'],split[0],labels_for(split[1])))
        with (out/'labels.jsonl').open('w') as f:
            for sid,_,truth in queries:f.write(json.dumps({'session':sid,'labels':truth})+'\n')
        assert (out/'labels.jsonl').read_bytes()==Path('artifacts/covis/labels.jsonl').read_bytes()
        report['validation_sessions']=len(queries);report['models']={}
        for policy in ['typed','click_fallback','pooled']:
            model.policy=policy
            with (out/f'{policy}.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['session_type','labels'])
                for sid,events,_ in queries:
                    for kind in WEIGHTS:writer.writerow([f'{sid}_{kind}',' '.join(map(str,model.predict(events,kind)))])
            result=evaluate(out/'labels.jsonl',out/f'{policy}.csv');report['models'][policy]=result
            print(policy,result['score'],flush=True)
    report['elapsed_seconds']=time.perf_counter()-start
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
