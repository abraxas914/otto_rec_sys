import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import numpy as np
    import torch
    from otto.sequence import BehaviorEncoder, encode_events, negative_mask, task_balanced_loss
    from otto.sequence_experiment import fresh_query
    from otto.prepare import selected


@unittest.skipUnless(HAS_TORCH, 'Install requirements-neural.txt for neural tests')
class SequenceTests(unittest.TestCase):
    def test_visible_prefix_and_zero_item(self):
        events = [{'aid': 0, 'ts': 0, 'type': 'clicks'}, {'aid': 99, 'ts': 120000, 'type': 'orders'}]
        items, kinds, age = encode_events(events, {0: 2}, 4)
        self.assertEqual(items.tolist(), [2, 1, 0, 0])
        self.assertEqual(kinds.tolist(), [1, 3, 0, 0])
        self.assertEqual(age.tolist(), [2, 1, 0, 0])

    def test_all_positives_masked_and_loss_ignores_them(self):
        mask = negative_mask([[2, 3], [4]], [2, 3, 4, 5])
        self.assertEqual(mask.tolist(), [[True, True, False, False], [False, False, True, False]])
        pos = torch.tensor([.3, .5]); neg = torch.zeros(2, 4); kinds = torch.tensor([0, 2])
        a = task_balanced_loss(pos, neg, torch.tensor(mask), kinds)
        neg[torch.tensor(mask)] = 999
        self.assertAlmostEqual(a.item(), task_balanced_loss(pos, neg, torch.tensor(mask), kinds).item())

    def test_padding_and_unknown_candidate(self):
        torch.manual_seed(1)
        for name in ['pool', 'transformer']:
            model = BehaviorEncoder(10, name, dim=8, length=4).eval()
            with torch.no_grad():
                h1 = model.encode(torch.tensor([[2, 3]]), torch.tensor([[1, 2]]), torch.tensor([[2, 1]]))
                h2 = model.encode(torch.tensor([[2, 3, 0, 0]]), torch.tensor([[1, 2, 0, 0]]), torch.tensor([[2, 1, 0, 0]]))
                self.assertTrue(torch.allclose(h1, h2, atol=1e-6))
                scores = model.candidate_scores(h2, torch.tensor([0]), torch.tensor([[1, 2]]))
                self.assertAlmostEqual(scores[0, 0].item(), -1.01, places=5)
                self.assertTrue(torch.isfinite(h2).all())

    def test_new_confirmation_excludes_previous_sets(self):
        accepted = [s for s in range(10000) if fresh_query(s)]
        self.assertTrue(accepted)
        self.assertTrue(all(not selected(s, 42, .01) and not selected(s, 991, .30) for s in accepted))
