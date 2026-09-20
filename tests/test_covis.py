import unittest
from otto.covis import Covis


class CovisTests(unittest.TestCase):
    def test_session_pairs_deduplicated_and_type_specific(self):
        m=Covis()
        m.add([{'aid':1,'ts':0,'type':'clicks'}, {'aid':1,'ts':1,'type':'clicks'},
               {'aid':2,'ts':2,'type':'carts'}, {'aid':3,'ts':86400003,'type':'orders'}])
        self.assertEqual(m.graph['carts'][1][2],1)
        self.assertNotIn(3,m.graph['orders'].get(1,{}))
        self.assertNotIn(1,m.graph['clicks'].get(1,{}))

    def test_recent_candidates_preserved_and_deduplicated(self):
        m=Covis()
        events=[{'aid':1,'ts':0,'type':'clicks'}, {'aid':2,'ts':1,'type':'carts'}]
        m.add(events);m.finish()
        result=m.predict([events[0]],'carts')
        self.assertEqual(result,[1,2])
