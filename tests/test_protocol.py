import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from otto.baseline import main
from otto.baseline import labels_for, split_session
from otto.prepare import selected


class ProtocolTests(unittest.TestCase):
    def test_labels_next_click_not_all_clicks(self):
        events = [{'aid': aid, 'type': kind} for aid, kind in
                  [(0, 'clicks'), (1, 'clicks'), (2, 'carts'), (2, 'carts'), (3, 'orders')]]
        self.assertEqual(labels_for(events), {'clicks': 0, 'carts': [2], 'orders': [3]})

    def test_equal_timestamps_never_split(self):
        events = [{'ts': ts} for ts in [1, 1, 2, 2, 3, 3]]
        for seed in range(20):
            before, after = split_session(events, seed, 1)
            self.assertLess(before[-1]['ts'], after[0]['ts'])
        self.assertIsNone(split_session([{'ts': 1}, {'ts': 1}], 42, 1))

    def test_sample_stable_and_nested(self):
        small = {sid for sid in range(1000) if selected(sid, 42, 0.1)}
        large = {sid for sid in range(1000) if selected(sid, 42, 0.2)}
        self.assertTrue(small < large)
        self.assertEqual(small, {sid for sid in reversed(range(1000)) if selected(sid, 42, 0.1)})

    def test_future_event_excluded_from_training(self):
        day = 86400000
        def event(aid, ts, kind):
            return {'aid': aid, 'ts': ts, 'type': kind}
        rows = [
            {'session': 1, 'events': [event(1, 0, 'clicks'), event(2, 1, 'carts'),
                                     event(3, 2, 'orders'), event(999, 3*day, 'orders')]},
            {'session': 2, 'events': [event(4, 3*day, 'clicks'), event(5, 3*day+1, 'clicks'),
                                     event(6, 3*day+1, 'carts'), event(7, 3*day+1, 'orders')]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'train.jsonl'
            source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            argv = ['baseline', '--input', str(source), '--output', str(root/'out'), '--days', '1']
            with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            report = json.loads((root/'out/report.json').read_text())
            self.assertEqual(report['train_events'], 3)
            self.assertEqual(report['validation_sessions'], 1)
            self.assertNotIn('999', (root/'out/popularity.csv').read_text())
