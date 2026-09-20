"""Bounded session co-visitation baseline with chronological evaluation."""
import argparse
import csv
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from otto.baseline import split_session, labels_for
from otto.evaluate import WEIGHTS, evaluate


class Covis:
    def __init__(self, neighbors=40):
        self.neighbors = neighbors
        self.graph = {k: defaultdict(Counter) for k in WEIGHTS}
        self.pop = {k: Counter() for k in WEIGHTS}

    def add(self, events):
        for e in events:
            self.pop[e['type']][e['aid']] += 1
        events = events[-30:]
        pairs = set()
        for a in events:
            for b in events:
                if a['aid'] != b['aid'] and abs(a['ts']-b['ts']) <= 86400000:
                    pairs.add((a['aid'], b['aid'], b['type']))
        for a, b, kind in pairs:
            self.graph[kind][a][b] += 1

    def finish(self):
        self.graph = {kind: {a: sorted(count.items(), key=lambda x: (-x[1], x[0]))[:self.neighbors]
                            for a, count in graph.items()} for kind, graph in self.graph.items()}
        self.pop = {k: [a for a, _ in sorted(count.items(), key=lambda x: (-x[1], x[0]))[:200]]
                    for k, count in self.pop.items()}

    def predict(self, events, kind, weighted=False, budget=20):
        recent = list(dict.fromkeys(e['aid'] for e in reversed(events)))
        if weighted:
            scores = Counter()
            for pos, event in enumerate(reversed(events)):
                behavior = {'clicks': 1, 'carts': 6, 'orders': 3}[event['type']]
                scores[event['aid']] += behavior * (0.95 ** pos)
            recent = sorted(scores, key=lambda a: (-scores[a], a))
        scores = Counter()
        for pos, aid in enumerate(list(dict.fromkeys(e['aid'] for e in reversed(events)))[:20]):
            for target, strength in self.graph[kind].get(aid, []):
                scores[target] += strength / (1 + pos)
        candidates = sorted(scores, key=lambda a: (-scores[a], a))
        return list(dict.fromkeys(recent + candidates + self.pop[kind]))[:budget]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='data/sample/train.jsonl')
    p.add_argument('--reference', default='artifacts/baseline/report.json')
    p.add_argument('--output', default='artifacts/covis')
    args = p.parse_args()
    start = time.perf_counter()
    ref = json.loads(Path(args.reference).read_text())
    cutoff, seed = ref['cutoff_ms'], ref['seed']
    model, queries = Covis(), []
    with open(args.input) as stream:
        for n, line in enumerate(stream, 1):
            row = json.loads(line)
            events = row['events']
            if events[0]['ts'] < cutoff:
                model.add([e for e in events if e['ts'] < cutoff])
            else:
                split = split_session(events, seed, row['session'])
                if split:
                    prefix, suffix = split
                    queries.append((row['session'], prefix, labels_for(suffix)))
            if n % 25000 == 0:
                print(f'Processed {n:,} sessions', flush=True)
    model.finish()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'model.pkl').open('wb') as stream:
        pickle.dump(model, stream)
    with (out/'labels.jsonl').open('w') as stream:
        for sid, _, truth in queries:
            stream.write(json.dumps({'session': sid, 'labels': truth})+'\n')
    report = {'scope': 'Same 1% development split as M0', 'cutoff_ms': cutoff, 'seed': seed,
              'config': {'last_events': 30, 'pair_window_ms': 86400000, 'neighbors': 40},
              'models': {}, 'candidate_upper_bound': {}}
    for weighted in (False, True):
        name = 'weighted_covis' if weighted else 'recent_covis'
        with (out/f'{name}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['session_type', 'labels'])
            for sid, prefix, _ in queries:
                for kind in WEIGHTS:
                    writer.writerow([f'{sid}_{kind}', ' '.join(map(str, model.predict(prefix, kind, weighted)))])
        report['models'][name] = evaluate(out/'labels.jsonl', out/f'{name}.csv')
    for budget in (50, 100, 200):
        hits, denom, incremental = Counter(), Counter(), Counter()
        for _, prefix, truth in queries:
            seen = {e['aid'] for e in prefix}
            for kind in WEIGHTS:
                values = truth.get(kind, [])
                target = {values} if isinstance(values, int) else set(values)
                found = set(model.predict(prefix, kind, False, budget)) & target
                hits[kind] += min(20, len(found))
                denom[kind] += min(20, len(target))
                incremental[kind] += len(found-seen)
        report['candidate_upper_bound'][budget] = {
            'recall': {k: hits[k]/denom[k] for k in WEIGHTS},
            'unseen_in_prefix_target_hits': dict(incremental)}
    report['elapsed_seconds'] = time.perf_counter()-start
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
