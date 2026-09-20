"""Time-isolated popularity and revisit baselines on the development sample."""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from otto.evaluate import WEIGHTS, evaluate


def labels_for(events):
    result = {}
    for kind in WEIGHTS:
        items = list(dict.fromkeys(x['aid'] for x in events if x['type'] == kind))
        if items:
            result[kind] = items[0] if kind == 'clicks' else items
    return result


def split_session(events, seed, sid):
    # Keep equal-timestamp events together: no artificial order within a batch.
    boundaries = [i for i in range(1, len(events)) if events[i-1]['ts'] < events[i]['ts']]
    if not boundaries:
        return None
    value = int.from_bytes(hashlib.blake2b(f'{seed}:{sid}'.encode(), digest_size=8).digest(), 'big')
    cut = boundaries[value % len(boundaries)]
    return events[:cut], events[cut:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/sample/train.jsonl')
    parser.add_argument('--output', default='artifacts/baseline')
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.days < 1:
        parser.error('days must be positive')
    source = Path(args.input)
    started = time.perf_counter()
    max_ts = 0
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for line in stream:
            digest.update(line)
            max_ts = max(max_ts, max(x['ts'] for x in json.loads(line)['events']))
    cutoff = max_ts + 1 - args.days * 86400000
    counts = {kind: Counter() for kind in WEIGHTS}
    queries = []
    seen = set()
    training_events = skipped = 0
    with source.open() as stream:
        for line in stream:
            row = json.loads(line)
            sid, events = row['session'], row['events']
            if sid in seen:
                raise ValueError('Duplicate session')
            seen.add(sid)
            if any(a['ts'] > b['ts'] for a, b in zip(events, events[1:])):
                raise ValueError('Unordered events')
            if events[0]['ts'] < cutoff:
                for event in events:
                    if event['ts'] < cutoff:
                        counts[event['type']][event['aid']] += 1
                        training_events += 1
            else:
                split = split_session(events, args.seed, sid)
                if split is None:
                    skipped += 1
                else:
                    prefix, suffix = split
                    queries.append((sid, prefix, labels_for(suffix)))
    popular = {kind: [aid for aid, _ in sorted(count.items(), key=lambda x: (-x[1], x[0]))[:20]]
               for kind, count in counts.items()}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    labels_path = out / 'labels.jsonl'
    with labels_path.open('w') as stream:
        for sid, _, labels in queries:
            stream.write(json.dumps({'session': sid, 'labels': labels}) + '\n')
    results = {'scope': '1% development sample by default; not leaderboard-comparable',
               'input_sha256': digest.hexdigest(), 'cutoff_ms': cutoff, 'max_ts_ms': max_ts,
               'seed': args.seed, 'holdout_days': args.days, 'train_events': training_events,
               'validation_sessions': len(queries), 'unsplittable_sessions': skipped, 'models': {}}
    for model in ['popularity', 'recent_then_popular']:
        path = out / f'{model}.csv'
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['session_type', 'labels'])
            for sid, prefix, _ in queries:
                recent = [event['aid'] for event in reversed(prefix)] if model.startswith('recent') else []
                for kind in WEIGHTS:
                    items = list(dict.fromkeys(recent + popular[kind]))[:20]
                    writer.writerow([f'{sid}_{kind}', ' '.join(map(str, items))])
        results['models'][model] = evaluate(labels_path, path)
    results['elapsed_seconds'] = time.perf_counter() - started
    (out / 'report.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
