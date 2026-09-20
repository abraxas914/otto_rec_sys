import math,unittest
from otto.pctm import TypedPCTM,PCTM_FEATURES
from otto.pctm_experiment import fresh_query
from otto.prepare import selected


def e(a,t,k='clicks'):return {'aid':a,'ts':t,'type':k}


class PCTMTests(unittest.TestCase):
    def test_log_pooling_matches_probability_ranking(self):
        p=TypedPCTM(tau=3.)
        p.add([e(1,0),e(2,1)]);p.add([e(1,0),e(3,1)]);p.add([e(1,0),e(2,1)])
        p.finish();score,_=p.evidence([e(1,10)])['clicks']
        self.assertAlmostEqual(score[2]-score[3],math.log((2+1)/(1+1)))

    def test_strict_time_and_repeated_target(self):
        p=TypedPCTM();p.add([e(1,0),e(2,0,'orders'),e(1,1,'orders')]);p.finish()
        self.assertNotIn(2,dict(p.graph['orders'][1]))
        self.assertIn(1,dict(p.graph['orders'][1]))
        self.assertGreater(p.evidence([e(1,3)])['orders'][0][1],0)

    def test_feature_shape_and_unknown_item(self):
        p=TypedPCTM();p.add([e(1,0),e(2,1)]);p.finish()
        rows=p.features([999],'carts',p.evidence([e(1,2)]))
        self.assertEqual(len(rows[0]),len(PCTM_FEATURES));self.assertEqual(rows[0][-2:],[0.,0.])

    def test_new_holdout_does_not_reuse_previous_audit(self):
        accepted=[s for s in range(10000) if fresh_query(s)]
        self.assertTrue(accepted)
        self.assertTrue(all(not selected(s,42,.01) and not selected(s,991,.15) for s in accepted))
