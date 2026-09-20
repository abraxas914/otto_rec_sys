"""Compare scores while recording the upstream click-ID-zero edge case."""
import argparse
import importlib.util
import json
from pathlib import Path

from otto.evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--directory', default='artifacts/baseline')
args = parser.parse_args()
root = Path(args.directory)
spec = importlib.util.spec_from_file_location('official', 'references/official_evaluate.py')
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)
label_lines = (root/'labels.jsonl').read_text().splitlines()
labels = official.prepare_labels(label_lines)
zero_clicks = sum(json.loads(line)['labels'].get('clicks') == 0 for line in label_lines)
report = {'upstream_source': json.loads(Path('references/source.json').read_text()),
          'click_zero_label_count': zero_clicks, 'models': {},
          'known_difference': 'Upstream truthiness checks exclude click aid=0; local code follows metric formula.'}
for path in sorted(root.glob('*.csv')):
    predictions = official.prepare_predictions(path.read_text().splitlines()[1:])
    expected = official.get_scores(labels, predictions)
    actual = evaluate(root/'labels.jsonl', path)
    delta = {k: actual['tasks'][k]['recall@20']-expected[k] for k in ('clicks','carts','orders')}
    delta['total'] = actual['score'] - expected['total']
    report['models'][path.stem] = {'official': expected, 'local_minus_official': delta}
    for kind in ('carts', 'orders'):
        assert abs(delta[kind]) < 1e-12, (path, kind, delta)
    if zero_clicks == 0:
        assert all(abs(value)<1e-12 for value in delta.values()), delta
(root/'official_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
