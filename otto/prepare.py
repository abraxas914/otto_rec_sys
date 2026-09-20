"""Inspect the official archive and take a deterministic session sample."""
import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path


def selected(session, seed, fraction):
    value = int.from_bytes(hashlib.blake2b(f'{seed}:{session}'.encode(), digest_size=8).digest(), 'big')
    return value / 2**64 < fraction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', default='data/raw/recsys-dataset.zip')
    parser.add_argument('--fraction', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='data/sample')
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error('fraction must be in (0, 1]')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive)
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    manifest = {'source': 'https://www.kaggle.com/datasets/otto/recsys-dataset',
                'archive_sha256': digest.hexdigest(), 'archive_bytes': archive.stat().st_size,
                'fraction': args.fraction, 'seed': args.seed, 'files': [],
                'scope': 'Development sample; not a full-dataset benchmark. Test contents not inspected.'}
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            entry = {'name': info.filename, 'bytes': info.file_size, 'crc32': info.CRC}
            manifest['files'].append(entry)
            if Path(info.filename).name != 'otto-recsys-train.jsonl':
                continue
            counts = Counter()
            low, high = None, None
            sampled = total = unordered = 0
            with zipped.open(info) as stream, (out / 'train.jsonl').open('wb') as dest:
                for line in stream:
                    total += 1
                    match = re.search(rb'"session"\s*:\s*(\d+)', line[:200])
                    if not match:
                        raise ValueError('Missing session ID')
                    if not selected(int(match[1]), args.seed, args.fraction):
                        continue
                    record = json.loads(line)
                    events = record['events']
                    stamps = [x['ts'] for x in events]
                    if not stamps:
                        raise ValueError('Empty session')
                    unordered += int(stamps != sorted(stamps))
                    counts.update(x['type'] for x in events)
                    low = min(stamps) if low is None else min(low, min(stamps))
                    high = max(stamps) if high is None else max(high, max(stamps))
                    sampled += 1
                    dest.write(line)
                    if sampled % 20000 == 0:
                        print(f'Scanned {total:,}; sampled {sampled:,}', flush=True)
            entry.update(total_sessions=total, sampled_sessions=sampled,
                         sampled_event_counts=dict(counts), sampled_min_ts=low,
                         sampled_max_ts=high, sampled_unordered_sessions=unordered)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
