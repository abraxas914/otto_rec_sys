"""Time-isolated candidate/ranker experiment. Never uses competition test labels."""
import argparse
import csv
import gc
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
from otto.baseline import split_session, labels_for
from otto.prepare import selected
from otto.scale import HybridCovis
from otto.relations import Relations, CandidateFeatures, FEATURES, RELATIONS
from otto.evaluate import WEIGHTS, evaluate

DAY = 86400000


def bounded_query(row, start, end, seed=42):
    """Exclude pre-window sessions and truncate suffix at the window end."""
    if not start <= row['events'][0]['ts'] < end:
        return None
    events = [e for e in row['events'] if e['ts'] < end]
    pair = split_session(events, seed, row['session'])
    if pair is None:
        return None
    prefix, suffix = pair
    return row['session'], prefix, labels_for(suffix)


def targets(truth, kind):
    v = truth.get(kind, [])
    return {v} if isinstance(v, int) else set(v)


def fit_graphs(path, cutoff, out, base_path=None):
    if out.exists():
        with out.open('rb') as f:
            return pickle.load(f)
    base = HybridCovis() if base_path is None else None
    extra = Relations(); max_ts = -1; sessions = 0
    with open(path) as f:
        for line in f:
            events = json.loads(line)['events']
            events = [e for e in events if e['ts'] < cutoff]
            if not events:
                continue
            max_ts = max(max_ts, events[-1]['ts']); sessions += 1
            if base is not None:
                base.add(events)
            extra.add(events)
    if base is None:
        with open(base_path, 'rb') as f:
            base = pickle.load(f)
    else:
        base.finish()
    extra.finish()
    assert max_ts < cutoff
    builder = CandidateFeatures(base, extra)
    with out.open('wb') as f:
        pickle.dump(builder, f)
    (out.with_suffix('.json')).write_text(json.dumps({'cutoff_ms': cutoff, 'max_fit_ts': max_ts,
                                                    'fit_sessions': sessions}, indent=2)+'\n')
    print('Fitted graphs', out.name, sessions, flush=True)
    return builder


def load_queries(path, start, end, mode):
    result = []
    for line in open(path):
        row = json.loads(line); sid = row['session']
        if mode == 'train' and not selected(sid, 1907, .12):
            continue
        if mode == 'audit' and (selected(sid, 42, .01) or not selected(sid, 991, .15)):
            continue
        query = bounded_query(row, start, end)
        if query:
            result.append(query)
    return result


def build_training(builder, queries, kind, cache):
    if cache.exists():
        a = np.load(cache)
        return a['x'], a['y'], a['groups'], int(a['dropped'])
    xs = []; ys = []; groups = []; dropped = 0
    for _, events, truth in queries:
        target = targets(truth, kind)
        if not target:
            continue
        aids, rows, _ = builder.make(events, kind)
        labels = [int(a in target) for a in aids]
        if not any(labels):
            dropped += 1
            continue
        xs.append(np.asarray(rows, dtype=np.float32)); ys.extend(labels); groups.append(len(aids))
    x = np.concatenate(xs); y = np.asarray(ys, dtype=np.int8); groups = np.asarray(groups)
    np.savez(cache, x=x, y=y, groups=groups, dropped=dropped)
    return x, y, groups, dropped


def eval_models(builder, queries, boosters, out):
    out.mkdir(exist_ok=True)
    label_path = out/'labels.jsonl'
    with label_path.open('w') as f:
        for sid, _, truth in queries:
            f.write(json.dumps({'session': sid, 'labels': truth})+'\n')
    names = ['baseline', 'ranker', 'ranker_without_relation_features']
    names += ['rules_'+k for k in RELATIONS]
    streams = {n: (out/f'{n}.csv').open('w', newline='') for n in names}
    writers = {n: csv.writer(f) for n, f in streams.items()}
    for writer in writers.values():
        writer.writerow(['session_type','labels'])
    metrics = {k: {'denominator': 0, 'baseline_oracle_hits': 0, 'union_oracle_hits': 0,
                   'new_target_hits': 0, 'lost_target_hits': 0} for k in WEIGHTS}
    try:
        for offset in range(0, len(queries), 128):
            batch = queries[offset:offset+128]
            evidence = [builder.evidence(events) for _, events, _ in batch]
            for kind in WEIGHTS:
                entries = [builder.make(e, kind, v) for (_, e, _), v in zip(batch, evidence)]
                matrix = np.asarray([r for _, rows, _ in entries for r in rows], dtype=np.float32)
                preds = {name: booster.predict(matrix, num_threads=4) for name, booster in boosters[kind].items()}
                begin = 0
                for (sid, events, truth), (aids, rows, baseline), ev in zip(batch, entries, evidence):
                    target = targets(truth, kind); stop = begin+len(aids)
                    choices = {'baseline': baseline[:20]}
                    for name, values in preds.items():
                        order = sorted(range(len(aids)), key=lambda i: (-values[begin+i], aids[i]))
                        choices[name] = [aids[i] for i in order[:20]]
                    for source in RELATIONS:
                        # Controlled additive rule, preserving history-first baseline behavior.
                        old = builder.base.predict(events, kind, budget=200)
                        seen = set(e['aid'] for e in events)
                        history = [a for a in old if a in seen]
                        score = {a: 1/(20+i) for i,a in enumerate(old) if a not in seen}
                        for a, rank in ev[1][source].items():
                            if a not in seen:
                                score[a] = score.get(a,0) + 1/(20+rank)
                        choices['rules_'+source] = list(dict.fromkeys(history+sorted(score,key=lambda a:(-score[a],a))))[:20]
                    for name, pred in choices.items():
                        writers[name].writerow([f'{sid}_{kind}', ' '.join(map(str,pred))])
                    m = metrics[kind]; before=set(baseline)&target; after=set(aids)&target
                    m['denominator'] += min(20,len(target))
                    m['baseline_oracle_hits'] += min(20,len(before))
                    m['union_oracle_hits'] += min(20,len(after))
                    m['new_target_hits'] += len(after-before); m['lost_target_hits'] += len(before-after)
                    begin=stop
    finally:
        for f in streams.values(): f.close()
    return {'sessions':len(queries), 'models':{n:evaluate(label_path,out/f'{n}.csv') for n in names},
            'candidate_metrics':metrics}


def main():
    import lightgbm as lgb
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='artifacts/rank-v1')
    p.add_argument('--audit',action='store_true',help='Evaluate frozen models on previously unused later-window sessions')
    args=p.parse_args(); out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); ref=json.loads(Path('artifacts/baseline/report.json').read_text())
    dev_start=ref['cutoff_ms']; train_start=dev_start-7*DAY; end=ref['max_ts_ms']+1
    config={'train_graph_before':train_start,'rank_queries_start':train_start,'rank_queries_end':dev_start,
            'eval_graph_before':dev_start,'dev_queries_end':end,'audit_start':end-2*DAY,
            'train_query_sample':.12,'sample_seed':1907,'trees':120,'leaves':15,'candidate_budget':200,
            'note':'Audit sessions disjoint from historical 1% development set; later time within same week, not a new unseen calendar week.'}
    config_path=out/'config.json'
    if config_path.exists(): assert json.loads(config_path.read_text())==config,'Use a new output directory for new config'
    else: config_path.write_text(json.dumps(config,indent=2)+'\n')
    boosters={k:{} for k in WEIGHTS}; train_stats={}
    if not args.audit:
        train=load_queries('data/sample10/train.jsonl',train_start,dev_start,'train')
        print('Training queries',len(train),flush=True)
        builder=fit_graphs('data/sample10/train.jsonl',train_start,out/'train_graph.pkl')
        for kind in WEIGHTS:
            x,y,groups,dropped=build_training(builder,train,kind,out/f'train_{kind}.npz')
            print('Training',kind,len(groups),len(y),'dropped',dropped,flush=True)
            train_stats[kind]={'groups':len(groups),'rows':len(y),'positives':int(y.sum()),'no_candidate_positive_dropped':dropped}
            for name in ('ranker','ranker_without_relation_features'):
                path=out/f'{kind}_{name}.txt'
                if path.exists(): model=lgb.Booster(model_file=str(path))
                else:
                    # Exclude new relation feature columns, retaining same candidate set.
                    data=x.copy() if name.endswith('features') else x
                    if name.endswith('features'): data[:,-9:]=0
                    ds=lgb.Dataset(data,label=y,group=groups,feature_name=FEATURES)
                    model=lgb.train({'objective':'lambdarank','metric':'ndcg','ndcg_eval_at':[20],
                                     'learning_rate':.05,'num_leaves':15,'min_data_in_leaf':100,
                                     'num_threads':6,'verbosity':-1,'seed':42,'deterministic':True,
                                     'force_col_wise':True,'lambdarank_truncation_level':25},ds,num_boost_round=120)
                    model.save_model(str(path))
                    del ds,data
                boosters[kind][name]=model
            del x,y,groups;gc.collect()
        del builder,train;gc.collect()
        (out/'training.json').write_text(json.dumps(train_stats,indent=2)+'\n')
    else:
        for kind in WEIGHTS:
            for name in ('ranker','ranker_without_relation_features'):
                boosters[kind][name]=lgb.Booster(model_file=str(out/f'{kind}_{name}.txt'))
    builder=fit_graphs('data/sample10/train.jsonl',dev_start,out/'eval_graph.pkl','artifacts/scale10/model.pkl')
    if args.audit:
        queries=load_queries('data/sample10/train.jsonl',end-2*DAY,end,'audit'); split='audit'
    else:
        queries=load_queries('data/sample/train.jsonl',dev_start,end,'dev');split='dev'
        expected=Path('artifacts/covis/labels.jsonl').read_text().splitlines()
        assert [json.loads(v) for v in expected]==[{'session':s,'labels':t} for s,_,t in queries]
    print('Evaluate',split,len(queries),flush=True)
    report=eval_models(builder,queries,boosters,out/split)
    report.update(config=config,elapsed_seconds=time.perf_counter()-start,lightgbm=lgb.__version__)
    (out/f'{split}_report.json').write_text(json.dumps(report,indent=2)+'\n')
    Path(f'reports/rank_v1_{split}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({n:r['score'] for n,r in report['models'].items()},indent=2),flush=True)

if __name__=='__main__':main()
