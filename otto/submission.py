"""Fit on training sessions and stream predictions for ORIGINAL competition test."""
import argparse
import csv
import gzip
import hashlib
import json
import pickle
import time
import zipfile
from pathlib import Path
from otto.covis import Covis
from otto.evaluate import WEIGHTS


def records(path):
    path = Path(path)
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z:
            members = [n for n in z.namelist() if Path(n).name == 'test.jsonl']
            if len(members) != 1:
                raise ValueError('Require original competition test.jsonl, not public research future sequences')
            with z.open(members[0]) as f:
                for line in f:
                    yield json.loads(line)
    else:
        with path.open() as f:
            for line in f:
                yield json.loads(line)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', default='data/sample/train.jsonl')
    p.add_argument('--test')
    p.add_argument('--output', default='artifacts/submission-v1')
    p.add_argument('--fit-only', action='store_true')
    p.add_argument('--load-model', action='store_true')
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if args.load_model:
        with (out/'model.pkl').open('rb') as f:
            model = pickle.load(f)  # Only load our own trusted local model.
    else:
        model = Covis()
        for n, row in enumerate(records(args.train), 1):
            model.add(row['events'])
            if n % 100000 == 0:
                print(f'Fit {n:,} sessions', flush=True)
        model.finish()
        with (out/'model.pkl').open('wb') as f:
            pickle.dump(model, f)
    if args.fit_only:
        print('Fitted model saved', flush=True)
        return
    if not args.test:
        p.error('--test required unless --fit-only')
    dest = out/'submission.csv.gz'
    sessions, last = 0, None
    with gzip.open(dest, 'wt', compresslevel=1, newline='') as f:
        writer = csv.writer(f); writer.writerow(['session_type','labels'])
        for row in records(args.test):
            sid = row['session']
            if last is not None and sid <= last:
                raise ValueError('Expected strictly increasing original competition session IDs')
            last = sid; sessions += 1
            for kind in WEIGHTS:
                pred = model.predict(row['events'], kind, weighted=True)
                if len(pred) != 20 or len(set(pred)) != 20:
                    raise ValueError('Need 20 distinct predictions')
                writer.writerow([f'{sid}_{kind}',' '.join(map(str,pred))])
            if sessions % 100000 == 0:
                print(f'Predicted {sessions:,} sessions', flush=True)
    if sessions != 1671803:
        raise ValueError(f'Unexpected test session count: {sessions}')
    digest=hashlib.sha256()
    with dest.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            digest.update(block)
    report={'sessions':sessions,'rows':sessions*3,'sha256':digest.hexdigest(),
            'bytes':dest.stat().st_size,'elapsed_seconds':time.perf_counter()-start,
            'model':type(model).__name__, 'policy':getattr(model,'policy','typed'),
            'training_input':args.train,'test_input':args.test}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
