import unittest
from otto.relations import Relations, CandidateFeatures, FEATURES
from otto.rank_experiment import bounded_query
from otto.scale import HybridCovis


def event(aid, ts, kind='clicks'):
    return {'aid':aid,'ts':ts,'type':kind}


class RelationsTests(unittest.TestCase):
    def test_only_strictly_forward_edges(self):
        model=Relations();model.add([event(1,0),event(2,0,'carts'),event(3,1,'orders')])
        self.assertNotIn(2,model.graph['forward_buy'].get(1,{}))
        self.assertIn(3,model.graph['forward_buy'][1])
        self.assertNotIn(1,model.graph['forward_buy'].get(3,{}))
        self.assertIn(3,model.graph['buy2buy'][2])
        self.assertNotIn(3,model.graph['buy2buy'].get(1,{}))

    def test_window_and_session_dedup(self):
        model=Relations();model.add([event(1,0),event(2,3600000,'orders'),event(2,7200001,'orders')])
        self.assertEqual(model.graph['forward_buy'][1][2],1.0)
        model2=Relations();model2.add([event(1,0),event(2,7200001,'carts')])
        self.assertFalse(model2.graph['forward_buy'])

    def test_training_query_excludes_future_window(self):
        row={'session':42,'events':[event(1,10),event(2,20),event(999,30)]}
        _,prefix,truth=bounded_query(row,10,30)
        self.assertEqual(prefix,[event(1,10)])
        self.assertEqual(truth,{'clicks':2})
        self.assertIsNone(bounded_query(row,11,30))

    def test_new_relation_adds_unseen_candidate_and_features(self):
        base=HybridCovis();base.add([event(1,0),event(2,1)]);base.finish()
        extra=Relations();extra.add([event(1,0),event(99,1,'carts')]);extra.finish()
        builder=CandidateFeatures(base,extra)
        aids,rows,baseline=builder.make([event(1,10)],'orders')
        self.assertIn(99,aids);self.assertNotIn(99,baseline)
        row=rows[aids.index(99)]
        self.assertEqual(len(row),len(FEATURES))
        self.assertEqual(row[FEATURES.index('seen')],0)
        self.assertGreater(row[FEATURES.index('forward_buy_score')],0)
        self.assertEqual(len(aids),len(set(aids)))
