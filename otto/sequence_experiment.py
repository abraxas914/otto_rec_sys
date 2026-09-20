"""Time-isolated, fixed-candidate neural representation experiment."""
import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np
import torch
from otto.baseline import split_session, labels_for
from otto.evaluate import evaluate, WEIGHTS
from otto.prepare import selected
from otto.rank_experiment import load_queries, bounded_query, targets
from otto.relations import FEATURES
from otto.pctm import PCTM_FEATURES
from otto.sequence import (BehaviorEncoder, KINDS, NEURAL_FEATURES, encode_events,
                           negative_mask, task_balanced_loss)

OUT = Path('artifacts/sequence-v1')
OLD = Path('artifacts/rank-v1')
PCTM = Path('artifacts/pctm-v1')
SOURCE = Path('data/sample10/train.jsonl')
CONFIG = {'version': 1, 'neural_fit_before': 1660514390707, 'vocab_limit': 200000,
          'query_fraction': .5, 'query_seed': 2701, 'split_seed': 2718, 'seed': 2718,
          'length': 30, 'dim': 64, 'epochs': 3, 'batch': 512, 'uniform_negatives': 512,
          'temperature': .12, 'lr': .001, 'weight_decay': .0001,
          'confirm_hash_band': [.30, .45], 'candidate_policy': 'unchanged rank-v1',
          'neural_snapshot': 'frozen before ranking training window, no eval refit'}


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    config = dict(CONFIG)
    config['data_manifest'] = hashlib.sha256(Path('data/sample10/manifest.json').read_bytes()).hexdigest()
    path = OUT/'config.json'
    if path.exists():
        assert json.loads(path.read_text()) == config, 'Use a new directory after changing config'
    else: path.write_text(json.dumps(config, indent=2)+'\n')


def prepare():
    if (OUT/'training.npz').exists(): return
    start = time.perf_counter(); counts = Counter(); max_ts = -1
    for line in SOURCE.open():
        events = [e for e in json.loads(line)['events'] if e['ts'] < CONFIG['neural_fit_before']]
        counts.update(e['aid'] for e in events)
        if events: max_ts = max(max_ts, events[-1]['ts'])
    vocab = {aid: i+2 for i, (aid, _) in enumerate(sorted(counts.items(), key=lambda p: (-p[1], p[0]))[:CONFIG['vocab_limit']])}
    with (OUT/'vocab.pkl').open('wb') as f: pickle.dump(vocab, f)
    print('Vocabulary', len(vocab), 'event coverage', sum(counts[a] for a in vocab)/sum(counts.values()), flush=True)
    xs = []; bs = []; ts = []; row_queries = []; kinds = []; values = []; offsets = [0]
    stats = {k: {'all_targets': 0, 'known_targets': 0, 'groups': 0, 'unknown_only_groups': 0} for k in KINDS}
    for line in SOURCE.open():
        row = json.loads(line)
        if not selected(row['session'], CONFIG['query_seed'], CONFIG['query_fraction']): continue
        events = [e for e in row['events'] if e['ts'] < CONFIG['neural_fit_before']]
        pair = split_session(events, CONFIG['split_seed'], row['session'])
        if pair is None: continue
        prefix, suffix = pair; labels = labels_for(suffix)
        x, b, t = encode_events(prefix, vocab, CONFIG['length'])
        qi = len(xs); xs.append(x); bs.append(b); ts.append(t)
        for k, kind in enumerate(KINDS):
            truth = targets(labels, kind); known = sorted(vocab[a] for a in truth if a in vocab)
            stats[kind]['all_targets'] += len(truth); stats[kind]['known_targets'] += len(known)
            if truth and not known: stats[kind]['unknown_only_groups'] += 1
            if not known: continue
            stats[kind]['groups'] += 1
            row_queries.append(qi); kinds.append(k); values.extend(known); offsets.append(len(values))
    np.savez(OUT/'training.npz', items=np.asarray(xs, np.int32), types=np.asarray(bs, np.int8),
             ages=np.asarray(ts, np.int8), row_queries=np.asarray(row_queries, np.int32),
             kinds=np.asarray(kinds, np.int8), values=np.asarray(values, np.int32), offsets=np.asarray(offsets, np.int64))
    report = {'max_fit_ts': max_ts, 'cutoff': CONFIG['neural_fit_before'], 'vocabulary': len(vocab),
              'catalog_before_cutoff': len(counts), 'event_coverage': sum(counts[a] for a in vocab)/sum(counts.values()),
              'prefixes': len(xs), 'rows': len(kinds), 'tasks': stats, 'seconds': time.perf_counter()-start}
    assert max_ts < CONFIG['neural_fit_before']
    (OUT/'data_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


def device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def load_vocab():
    with (OUT/'vocab.pkl').open('rb') as f: return pickle.load(f)


def train():
    prepare(); vocab = load_vocab(); dev = device(); torch.set_num_threads(4)
    data = dict(np.load(OUT/'training.npz'))
    for name in ('pool', 'transformer'):
        if (OUT/f'{name}.pt').exists(): continue
        torch.manual_seed(CONFIG['seed']); rng = np.random.default_rng(CONFIG['seed'])
        model = BehaviorEncoder(len(vocab)+2, name, CONFIG['dim'], CONFIG['length']).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        model.train(); history = []; start = time.perf_counter()
        for epoch in range(CONFIG['epochs']):
            order = rng.permutation(len(data['kinds'])); losses = []; epoch_start = time.perf_counter()
            for start_row in range(0, len(order), CONFIG['batch']):
                ix = order[start_row:start_row+CONFIG['batch']]; qi = data['row_queries'][ix]
                positives = [data['values'][data['offsets'][i]:data['offsets'][i+1]] for i in ix]
                chosen = np.asarray([v[rng.integers(len(v))] for v in positives], np.int64)
                pool = np.concatenate([chosen, rng.integers(2, len(vocab)+2, CONFIG['uniform_negatives'])])
                mask = torch.from_numpy(negative_mask(positives, pool)).to(dev)
                inp = [torch.from_numpy(data[key][qi].astype(np.int64)).to(dev) for key in ('items', 'types', 'ages')]
                kk = torch.from_numpy(data['kinds'][ix].astype(np.int64)).to(dev)
                h = model.encode(*inp)[torch.arange(len(ix), device=dev), kk]
                pos = (h*model.vectors(torch.from_numpy(chosen).to(dev))).sum(-1)
                neg = h@model.vectors(torch.from_numpy(pool).to(dev)).T
                loss = task_balanced_loss(pos, neg, mask, kk, CONFIG['temperature'])
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); value = float(loss.detach().cpu()); losses.append(value)
                if not np.isfinite(value): raise RuntimeError('Non-finite training loss')
                if start_row//CONFIG['batch'] % 100 == 0:
                    print(name, 'epoch', epoch+1, 'step', start_row//CONFIG['batch'], 'loss', round(value, 4),
                          'seconds', round(time.perf_counter()-epoch_start, 1), flush=True)
            history.append({'epoch': epoch+1, 'loss': float(np.mean(losses)), 'seconds': time.perf_counter()-epoch_start})
        model.eval(); state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state, OUT/f'{name}.pt')
        report = {'encoder': name, 'device': str(dev), 'torch': torch.__version__, 'parameters': sum(p.numel() for p in model.parameters()),
                  'history': history, 'seconds': time.perf_counter()-start, 'config': CONFIG}
        (OUT/f'{name}_training.json').write_text(json.dumps(report, indent=2)+'\n')
        print('FINISHED', name, json.dumps(report), flush=True)
        del model, optimizer, state; gc.collect()
        if dev.type == 'mps': torch.mps.empty_cache()


def load_models():
    vocab = load_vocab(); dev = device(); models = {}
    for name in ('pool', 'transformer'):
        model = BehaviorEncoder(len(vocab)+2, name, CONFIG['dim'], CONFIG['length'])
        model.load_state_dict(torch.load(OUT/f'{name}.pt', map_location='cpu', weights_only=True))
        models[name] = model.to(dev).eval()
    return vocab, models, dev


def encode_batch(batch, vocab, models, dev):
    inp = list(zip(*(encode_events(e, vocab, CONFIG['length']) for _, e, _ in batch)))
    inp = [torch.from_numpy(np.asarray(v)).to(dev) for v in inp]
    with torch.no_grad(): return {name: model.encode(*inp) for name, model in models.items()}


def feature_batch(entries, kind, vocab, models, encoded, dev):
    lengths = [len(aids) for aids, _, _ in entries]
    ids = np.zeros((len(entries), max(lengths)), np.int64)
    for i, (aids, _, _) in enumerate(entries): ids[i, :len(aids)] = [vocab.get(a, 1) for a in aids]
    kk = torch.full((len(entries),), KINDS.index(kind), device=dev)
    candidates = torch.from_numpy(ids).to(dev); result = {}
    with torch.no_grad():
        for name, model in models.items():
            scores = model.candidate_scores(encoded[name], kk, candidates).cpu().numpy()
            rows = []
            for i, (aids, _, _) in enumerate(entries):
                s = scores[i, :len(aids)]
                order = sorted(range(len(aids)), key=lambda j: (-s[j], aids[j]))
                ranks = np.empty(len(aids)); ranks[order] = np.arange(1, len(aids)+1)
                rows.append(np.column_stack([s, ranks, ids[i, :len(aids)] >= 2]).astype(np.float32))
            result[name] = rows
    return result


def fit_rankers():
    import os
    import sys
    if all((OUT/f'{k}_features.npz').exists() for k in KINDS):
        os.execv(sys.executable, [sys.executable, '-u', '-m', 'otto.sequence_rank'])
    cfg = json.loads((OLD/'config.json').read_text())
    vocab, models, dev = load_models()
    with (OLD/'train_graph.pkl').open('rb') as f: builder = pickle.load(f)
    queries = load_queries(SOURCE, cfg['rank_queries_start'], cfg['rank_queries_end'], 'train')
    if not all((OUT/f'{k}_features.npz').exists() for k in KINDS):
        original = {k: np.load(OLD/f'train_{k}.npz')['x'] for k in KINDS}
        offset = dict.fromkeys(KINDS, 0); chunks = {k: {n: [] for n in models} for k in KINDS}
        for start in range(0, len(queries), 128):
            batch = queries[start:start+128]; encoded = encode_batch(batch, vocab, models, dev)
            ev = [builder.evidence(e) for _, e, _ in batch]
            for kind in KINDS:
                entries = [builder.make(e, kind, v) for (_, e, _), v in zip(batch, ev)]
                features = feature_batch(entries, kind, vocab, models, encoded, dev)
                for i, ((_, _, truth), (aids, rows, _)) in enumerate(zip(batch, entries)):
                    if not set(aids)&targets(truth, kind): continue
                    j = offset[kind]; n = len(aids)
                    assert np.array_equal(np.asarray(rows, np.float32), original[kind][j:j+n])
                    for name in models: chunks[kind][name].append(features[name][i])
                    offset[kind] += n
            if start % 2560 == 0: print('Training features', start, '/', len(queries), flush=True)
        for kind in KINDS:
            assert offset[kind] == len(original[kind])
            np.savez(OUT/f'{kind}_features.npz', **{n: np.concatenate(c) for n, c in chunks[kind].items()})
        del original, chunks
    print('Features saved; starting isolated CPU ranking process', flush=True)
    os.execv(sys.executable, [sys.executable, '-u', '-m', 'otto.sequence_rank'])


def fresh_query(sid):
    return not selected(sid, 42, .01) and not selected(sid, 991, .30) and selected(sid, 991, .45)


def evaluate_split(confirm=False):
    import subprocess
    import sys
    start_time = time.perf_counter(); cfg = json.loads((OLD/'config.json').read_text())
    if confirm:
        queries = []
        for line in SOURCE.open():
            row = json.loads(line)
            if fresh_query(row['session']):
                q = bounded_query(row, cfg['audit_start'], cfg['dev_queries_end'])
                if q: queries.append(q)
    else: queries = load_queries('data/sample/train.jsonl', cfg['eval_graph_before'], cfg['dev_queries_end'], 'dev')
    vocab, models, dev = load_models()
    with (OLD/'eval_graph.pkl').open('rb') as f: builder = pickle.load(f)
    with (PCTM/'eval_probe.pkl').open('rb') as f: probe = pickle.load(f)
    names = ['rank_v1', 'pctm', 'coverage', 'pool', 'transformer', 'transformer_pctm', 'pool_only', 'transformer_only']
    split = 'confirm' if confirm else 'dev'; out = OUT/split; out.mkdir(exist_ok=True)
    label_path = out/'labels.jsonl'
    with label_path.open('w') as f:
        for sid, _, truth in queries: f.write(json.dumps({'session': sid, 'labels': truth})+'\n')
    handles = {n: (out/f'{n}.csv').open('w', newline='') for n in names}
    writers = {n: csv.writer(f) for n, f in handles.items()}
    for w in writers.values(): w.writerow(['session_type', 'labels'])
    coverage = {k: {'targets': 0, 'known_targets': 0, 'candidates': 0, 'known_candidates': 0} for k in KINDS}
    worker = subprocess.Popen([sys.executable, '-u', '-m', 'otto.sequence_rank', '--predict'],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert pickle.load(worker.stdout) == 'ready'
    try:
        for start in range(0, len(queries), 128):
            batch = queries[start:start+128]; encoded = encode_batch(batch, vocab, models, dev)
            ev = [builder.evidence(e) for _, e, _ in batch]; pv = [probe.evidence(e) for _, e, _ in batch]
            for kind in KINDS:
                entries = [builder.make(e, kind, v) for (_, e, _), v in zip(batch, ev)]
                extras = feature_batch(entries, kind, vocab, models, encoded, dev)
                old = np.asarray([r for _, rows, _ in entries for r in rows], np.float32)
                seq = np.concatenate(extras['transformer']); pool = np.concatenate(extras['pool'])
                pctm = np.asarray([r for (aids, _, _), v in zip(entries, pv) for r in probe.features(aids, kind, v)], np.float32)
                pickle.dump((kind, old, pctm, pool, seq), worker.stdin, protocol=5); worker.stdin.flush()
                predictions = pickle.load(worker.stdout)
                predictions.update(pool_only=pool[:, 0], transformer_only=seq[:, 0])
                offset = 0
                for (sid, _, truth), (aids, _, _) in zip(batch, entries):
                    for name, scores in predictions.items():
                        order = sorted(range(len(aids)), key=lambda j: (-scores[offset+j], aids[j]))[:20]
                        writers[name].writerow([f'{sid}_{kind}', ' '.join(str(aids[j]) for j in order)])
                    target = targets(truth, kind); c = coverage[kind]
                    c['targets'] += len(target); c['known_targets'] += sum(a in vocab for a in target)
                    c['candidates'] += len(aids); c['known_candidates'] += sum(a in vocab for a in aids)
                    offset += len(aids)
            if start % 2560 == 0: print('Evaluate', split, start, '/', len(queries), flush=True)
    finally:
        for f in handles.values(): f.close()
        try:
            pickle.dump(None, worker.stdin, protocol=5); worker.stdin.flush()
            worker.wait(timeout=15)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            worker.terminate(); worker.wait(timeout=5)
        finally:
            worker.stdin.close(); worker.stdout.close()
    assert worker.returncode == 0, 'CPU prediction worker failed'
    report = {'sessions': len(queries), 'models': {n: evaluate(label_path, out/f'{n}.csv') for n in names},
              'coverage': coverage, 'seconds': time.perf_counter()-start_time, 'config': CONFIG,
              'note': 'Fixed candidates; one frozen early neural snapshot. New sessions in the same calendar period, not a new week.'}
    if not confirm:
        assert report['models']['rank_v1'] == json.loads(Path('reports/rank_v1_dev.json').read_text())['models']['ranker']
        assert report['models']['pctm'] == json.loads(Path('reports/pctm_v1_dev.json').read_text())['models']['pctm_features']
    Path(f'reports/sequence_v1_{split}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({n: v['score'] for n, v in report['models'].items()}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['prepare', 'train', 'rank', 'dev', 'confirm'])
    args = p.parse_args(); initialize()
    if args.phase == 'prepare': prepare()
    elif args.phase == 'train': train()
    elif args.phase == 'rank': fit_rankers()
    else: evaluate_split(args.phase == 'confirm')

if __name__ == '__main__': main()
