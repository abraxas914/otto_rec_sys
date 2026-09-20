"""Fixed-candidate test of 2026 PCTM evidence versus frequency-only controls."""
import argparse,csv,gc,json,pickle,time
from pathlib import Path
import numpy as np
from otto.pctm import TypedPCTM,PCTM_FEATURES
from otto.relations import FEATURES
from otto.rank_experiment import load_queries,bounded_query,targets
from otto.prepare import selected
from otto.evaluate import WEIGHTS,evaluate


def fit(cutoff,path):
    if path.exists():
        with path.open('rb') as f:model=pickle.load(f)
        assert model.max_fit_ts<cutoff, 'Cached graph violates time boundary'
        return model
    model=TypedPCTM()
    for line in open('data/sample10/train.jsonl'):
        events=json.loads(line)['events'];model.add([e for e in events if e['ts']<cutoff])
    assert model.max_fit_ts<cutoff
    model.finish()
    with path.open('wb') as f:pickle.dump(model,f)
    print('PCTM fitted',cutoff,model.catalog_size,flush=True)
    return model


def fresh_query(sid):
    return not selected(sid,42,.01) and not selected(sid,991,.15) and selected(sid,991,.30)


def fresh_queries(cfg):
    queries=[]
    for line in open('data/sample10/train.jsonl'):
        row=json.loads(line)
        if fresh_query(row['session']):
            q=bounded_query(row,cfg['audit_start'],cfg['dev_queries_end'])
            if q:queries.append(q)
    return queries


def train_features(builder,probe,queries,kind,cache,old):
    if cache.exists():
        extra=np.load(cache)
        assert extra.shape==(len(old),len(PCTM_FEATURES)), 'Stale feature cache'
        return extra
    chunks=[];offset=0
    for _,events,truth in queries:
        target=targets(truth,kind)
        if not target:continue
        aids,rows,_=builder.make(events,kind)
        if not set(aids)&target:continue
        # Exact same order and 31 original features as v1, isolating new evidence.
        assert np.array_equal(np.asarray(rows,dtype=np.float32),old[offset:offset+len(aids)])
        chunks.append(np.asarray(probe.features(aids,kind,probe.evidence(events)),dtype=np.float32))
        offset+=len(aids)
    assert offset==len(old)
    extra=np.concatenate(chunks);np.save(cache,extra);return extra


def evaluate_split(builder,probe,queries,models,out):
    out.mkdir(exist_ok=True)
    labels=out/'labels.jsonl'
    with labels.open('w') as f:
        for sid,_,truth in queries:f.write(json.dumps({'session':sid,'labels':truth})+'\n')
    names=['rank_v1','frequency_only','pctm_features']
    files={n:(out/f'{n}.csv').open('w',newline='') for n in names};writers={n:csv.writer(f) for n,f in files.items()}
    for w in writers.values():w.writerow(['session_type','labels'])
    candidate_diagnostic={k:{'new_hits_if_add_top50':0,'denominator':0} for k in WEIGHTS}
    try:
        for start in range(0,len(queries),128):
            batch=queries[start:start+128]
            ev=[builder.evidence(e) for _,e,_ in batch];pv=[probe.evidence(e) for _,e,_ in batch]
            for kind in WEIGHTS:
                entries=[builder.make(e,kind,v) for (_,e,_),v in zip(batch,ev)]
                old=np.asarray([r for _,rows,_ in entries for r in rows],dtype=np.float32)
                extra=np.asarray([r for (aids,_,_),v in zip(entries,pv) for r in probe.features(aids,kind,v)],dtype=np.float32)
                new=np.column_stack([old,extra]);freq=np.column_stack([old,extra[:,-2:]])
                pred={'rank_v1':models[kind]['rank_v1'].predict(old,num_threads=4),
                      'frequency_only':models[kind]['frequency_only'].predict(freq,num_threads=4),
                      'pctm_features':models[kind]['pctm_features'].predict(new,num_threads=4)}
                begin=0
                for (sid,_,truth),(aids,_,_),v in zip(batch,entries,pv):
                    for name,values in pred.items():
                        ids=sorted(range(len(aids)),key=lambda i:(-values[begin+i],aids[i]))[:20]
                        writers[name].writerow([f'{sid}_{kind}',' '.join(str(aids[i]) for i in ids)])
                    target=targets(truth,kind);m=candidate_diagnostic[kind]
                    m['denominator']+=min(20,len(target))
                    m['new_hits_if_add_top50']+=len((set(list(v[kind][1])[:50])-set(aids))&target)
                    begin+=len(aids)
    finally:
        for f in files.values():f.close()
    return {'sessions':len(queries),'models':{n:evaluate(labels,out/f'{n}.csv') for n in names},
            'candidate_diagnostic':candidate_diagnostic,
            'note':'Identical v1 candidates for all evaluated models; diagnostic extra candidates are NOT used. No standalone full-catalog PCTM benchmark.'}


def main():
    import lightgbm as lgb
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--confirm',action='store_true');args=p.parse_args()
    out=Path('artifacts/pctm-v1');out.mkdir(parents=True,exist_ok=True)
    oldroot=Path('artifacts/rank-v1');cfg=json.loads((oldroot/'config.json').read_text());started=time.perf_counter()
    config={'parent':cfg,'distance':5,'decay':.7,'history_decay':.7,'tau':15000.,'neighbors':80,
            'candidate_policy':'unchanged rank-v1 candidates','confirm_hash_band':[.15,.30],
            'model_names':['rank_v1','frequency_only','pctm_features'],'trees':120,'leaves':15}
    config_path=out/'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text())==config, 'Config changed: use a new experiment directory'
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    models={k:{} for k in WEIGHTS}
    if not args.confirm:
        probe=fit(cfg['train_graph_before'],out/'train_probe.pkl')
        with (oldroot/'train_graph.pkl').open('rb') as f:builder=pickle.load(f)
        queries=load_queries('data/sample10/train.jsonl',cfg['rank_queries_start'],cfg['rank_queries_end'],'train')
        for kind in WEIGHTS:
            with np.load(oldroot/f'train_{kind}.npz') as a:x=a['x'];y=a['y'];groups=a['groups']
            extra=train_features(builder,probe,queries,kind,out/f'{kind}_features.npy',x)
            for name,cols in [('frequency_only',extra[:,-2:]),('pctm_features',extra)]:
                path=out/f'{kind}_{name}.txt'
                if path.exists():model=lgb.Booster(model_file=str(path))
                else:
                    data=np.column_stack([x,cols]);fn=FEATURES+(PCTM_FEATURES[-2:] if name=='frequency_only' else PCTM_FEATURES)
                    ds=lgb.Dataset(data,label=y,group=groups,feature_name=fn)
                    model=lgb.train({'objective':'lambdarank','metric':'ndcg','ndcg_eval_at':[20],
                         'learning_rate':.05,'num_leaves':15,'min_data_in_leaf':100,'num_threads':6,
                         'verbosity':-1,'seed':42,'deterministic':True,'force_col_wise':True,
                         'lambdarank_truncation_level':25},ds,num_boost_round=120)
                    model.save_model(str(path));del data,ds
                models[kind][name]=model
                print('Fitted',kind,name,flush=True)
            del x,y,groups,extra;gc.collect()
        del builder,probe,queries;gc.collect()
    for kind in WEIGHTS:
        for name in ['frequency_only','pctm_features']:
            if name not in models[kind]:models[kind][name]=lgb.Booster(model_file=str(out/f'{kind}_{name}.txt'))
        models[kind]['rank_v1']=lgb.Booster(model_file=str(oldroot/f'{kind}_ranker.txt'))
    probe=fit(cfg['eval_graph_before'],out/'eval_probe.pkl')
    with (oldroot/'eval_graph.pkl').open('rb') as f:builder=pickle.load(f)
    split='confirm' if args.confirm else 'dev'
    queries=fresh_queries(cfg) if args.confirm else load_queries('data/sample/train.jsonl',cfg['eval_graph_before'],cfg['dev_queries_end'],'dev')
    print('Evaluating',split,len(queries),flush=True)
    report=evaluate_split(builder,probe,queries,models,out/split)
    if not args.confirm:
        expected=json.loads(Path('reports/rank_v1_dev.json').read_text())['models']['ranker']
        assert report['models']['rank_v1']==expected
    report.update(config=config,seconds=time.perf_counter()-started,
                  max_fit_ts=probe.max_fit_ts,catalog_size=probe.catalog_size)
    Path(f'reports/pctm_v1_{split}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({n:r['score'] for n,r in report['models'].items()},indent=2))

if __name__=='__main__':main()
