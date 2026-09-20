import unittest
from otto.scale import HybridCovis

class HybridTests(unittest.TestCase):
    def model(self, policy):
        m = HybridCovis(); m.policy = policy
        m.graph = {'clicks': {1: [(2, 10), (3, 5)]}, 'carts': {}, 'orders': {1: [(4, 2)]}}
        m.pop = {kind: [2, 3, 4, 5] for kind in m.graph}
        return m

    def test_cross_behavior_backfills_sparse_cart_graph(self):
        m = self.model('pooled')
        events = [{'aid': 1, 'ts': 0, 'type': 'clicks'}]
        self.assertEqual(m.predict(events, 'carts', budget=4), [1, 4, 2, 3])

    def test_click_ranking_ignores_order_strength(self):
        m = self.model('pooled')
        events = [{'aid': 1, 'ts': 0, 'type': 'clicks'}]
        self.assertEqual(m.predict(events, 'clicks', budget=3), [1, 2, 3])

    def test_recent_and_fallback_are_unique(self):
        m = self.model('click_fallback')
        events = [{'aid': 2, 'ts': 0, 'type': 'carts'}, {'aid': 1, 'ts': 1, 'type': 'clicks'}]
        result = m.predict(events, 'orders')
        self.assertEqual(result[:2], [2, 1])
        self.assertEqual(len(result), len(set(result)))
