"""Paired uncertainty and descriptive diagnostics for the frozen neural experiment."""
import argparse
import csv
import json
import numpy as np
from pathlib import Path
from scripts.compare_predictions import compare
from otto.rank_experiment import load_queries, bounded_query, targets
from otto.sequence_experiment import fresh_query, load_vocab
from otto.sequence import KINDS
from otto.evaluate import WEIGHTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['dev', 'confirm'], default='dev')
    args = p.parse_args(); root = Path('artifacts/sequence-v1')/args.split
    cfg = json.loads(Path('artifacts/rank-v1/config.json').read_text())
    if args.split == 'dev':
        queries = load_queries('data/sample/train.jsonl', cfg['eval_graph_before'], cfg['dev_queries_end'], 'dev')
    else:
        queries = []
        for line in open('data/sample10/train.jsonl'):
            row = json.loads(line)
            if fresh_query(row['session']):
                q = bounded_query(row, cfg['audit_start'], cfg['dev_queries_end'])
                if q: queries.append(q)
    expected = [{'session': sid, 'labels': truth} for sid, _, truth in queries]
    assert expected == [json.loads(line) for line in (root/'labels.jsonl').read_text().splitlines()]
    comparisons = {}
    for base, candidate in [('rank_v1', 'transformer'), ('coverage', 'transformer'),
                            ('pool', 'transformer'), ('pctm', 'transformer_pctm')]:
        key = candidate+'_minus_'+base
        comparisons[key] = compare(root/'labels.jsonl', root/f'{base}.csv', root/f'{candidate}.csv')
    lookup = {sid: (events, truth) for sid, events, truth in queries}; vocab = load_vocab()
    names = ['rank_v1', 'pctm', 'coverage', 'pool', 'transformer', 'transformer_pctm']
    segments = {}; lengths = {}
    for name in names:
        segments[name] = {g: {k: {'hits': 0, 'targets': 0} for k in KINDS}
                          for g in ['seen', 'unseen', 'in_vocabulary', 'outside_vocabulary']}
        lengths[name] = {g: {k: {'hits': 0, 'denominator': 0} for k in KINDS}
                         for g in ['1-5', '6-20', '21+']}
        with (root/f'{name}.csv').open() as f:
            for row in csv.DictReader(f):
                sid, kind = row['session_type'].rsplit('_', 1); events, truth = lookup[int(sid)]
                target = targets(truth, kind); seen = {e['aid'] for e in events}
                pred = set(map(int, row['labels'].split()[:20])); known = {a for a in target if a in vocab}
                for group, part in [('seen', target&seen), ('unseen', target-seen),
                                    ('in_vocabulary', known), ('outside_vocabulary', target-known)]:
                    item = segments[name][group][kind]; item['hits'] += len(part&pred); item['targets'] += len(part)
                group = '1-5' if len(events) <= 5 else '6-20' if len(events) <= 20 else '21+'
                item = lengths[name][group][kind]; item['hits'] += len(target&pred); item['denominator'] += min(20, len(target))
        for group, tasks in lengths[name].items():
            values = [WEIGHTS[k]*v['hits']/v['denominator'] for k, v in tasks.items() if v['denominator']]
            tasks['weighted_recall'] = sum(values) if len(values) == 3 else None
    train = np.load('artifacts/sequence-v1/training.npz')
    value_kinds = np.repeat(train['kinds'], np.diff(train['offsets']))
    frequency = {k: np.bincount(train['values'][value_kinds == i], minlength=len(vocab)+2)
                 for i, k in enumerate(KINDS)}
    exposure = {k: {'outside_vocabulary': 0, 'zero_same_task_positive': 0,
                    'one_to_four_same_task_positives': 0, 'five_plus_same_task_positives': 0} for k in KINDS}
    for _, _, truth in queries:
        for kind in KINDS:
            for aid in targets(truth, kind):
                if aid not in vocab: group = 'outside_vocabulary'
                else:
                    count = frequency[kind][vocab[aid]]
                    group = 'zero_same_task_positive' if count == 0 else 'one_to_four_same_task_positives' if count < 5 else 'five_plus_same_task_positives'
                exposure[kind][group] += 1
    report = {'sessions': len(queries), 'comparisons': comparisons, 'target_segments': segments,
              'prefix_length': lengths, 'training_positive_exposure': exposure, 'primary_comparison': 'transformer_minus_rank_v1',
              'note': 'Multiple descriptive comparisons, no multiplicity correction. Session bootstrap ignores item/time correlations and model selection. Exposure counts refer to available same-task positive sets, not necessarily the positive sampled in each epoch. Target segments use uncapped unique counts; length cohorts use official denominators.'}
    Path(f'reports/sequence_v1_{args.split}_diagnostics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: {'difference': v['difference'], 'interval': v['session_bootstrap_95pct']} for k, v in comparisons.items()}, indent=2))

if __name__ == '__main__': main()
