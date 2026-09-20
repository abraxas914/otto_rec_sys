"""Fit CPU rankers in a clean process, isolated from the Apple GPU runtime."""
import gc
import pickle
import sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
from otto.relations import FEATURES
from otto.pctm import PCTM_FEATURES
from otto.sequence_schema import KINDS, NEURAL_FEATURES

OUT = Path('artifacts/sequence-v1')
OLD = Path('artifacts/rank-v1')
PCTM = Path('artifacts/pctm-v1')


def main():
    for kind in KINDS:
        with np.load(OLD/f'train_{kind}.npz') as a: x, y, groups = a['x'], a['y'], a['groups']
        extra = dict(np.load(OUT/f'{kind}_features.npz')); pctm = np.load(PCTM/f'{kind}_features.npy')
        additions = {'coverage': extra['transformer'][:, -1:], 'pool': extra['pool'],
                     'transformer': extra['transformer'], 'transformer_pctm': np.column_stack([pctm, extra['transformer']])}
        for name, cols in additions.items():
            path = OUT/f'{kind}_{name}.txt'
            if path.exists(): continue
            names = ([NEURAL_FEATURES[-1]] if name == 'coverage' else PCTM_FEATURES+NEURAL_FEATURES if name.endswith('pctm') else NEURAL_FEATURES)
            ds = lgb.Dataset(np.column_stack([x, cols]), label=y, group=groups, feature_name=FEATURES+names)
            model = lgb.train({'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [20],
                              'learning_rate': .05, 'num_leaves': 15, 'min_data_in_leaf': 100, 'num_threads': 6,
                              'verbosity': -1, 'seed': 42, 'deterministic': True, 'force_col_wise': True,
                              'lambdarank_truncation_level': 25}, ds, num_boost_round=120)
            model.save_model(str(path)); print('Ranker fitted', kind, name, flush=True)
            del ds, model
        del x, y, groups, extra, pctm, additions; gc.collect()


def predict_main():
    names = ['rank_v1', 'pctm', 'coverage', 'pool', 'transformer', 'transformer_pctm']
    boosts = {k: {n: lgb.Booster(model_file=str(OLD/f'{k}_ranker.txt' if n == 'rank_v1' else
                PCTM/f'{k}_pctm_features.txt' if n == 'pctm' else OUT/f'{k}_{n}.txt')) for n in names} for k in KINDS}
    source, sink = sys.stdin.buffer, sys.stdout.buffer
    pickle.dump('ready', sink, protocol=5); sink.flush()
    while True:
        message = pickle.load(source)
        if message is None: break
        kind, old, pctm, pool, seq = message
        matrices = {'rank_v1': old, 'pctm': np.column_stack([old, pctm]), 'coverage': np.column_stack([old, seq[:, -1:]]),
                    'pool': np.column_stack([old, pool]), 'transformer': np.column_stack([old, seq]),
                    'transformer_pctm': np.column_stack([old, pctm, seq])}
        predictions = {n: boosts[kind][n].predict(matrices[n], num_threads=4) for n in names}
        pickle.dump(predictions, sink, protocol=5); sink.flush()


if __name__ == '__main__':
    if '--predict' in sys.argv: predict_main()
    else: main()
