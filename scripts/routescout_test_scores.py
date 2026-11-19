import unittest
from routescout_scores import difficulty,frontier

class ScoreRules(unittest.TestCase):
    def test_easier_is_lower(self):
        easy=dict(n=8,passes=8,width=4,speed_span=8,deaths=0,rejoins=0,unresolved=0)
        self.assertEqual(difficulty([easy],0,0),0);self.assertIsNone(difficulty([],1,1));hard=dict(easy,passes=0,width=0,speed_span=0,deaths=8); self.assertEqual(difficulty([hard],8,3),100)
        self.assertGreater(difficulty([hard],4,1),difficulty([dict(hard,deaths=0,rejoins=8)],4,1))
    def test_ties_and_direction(self):
        rs=[dict(id=i,difficulty=d,score=x) for i,d,x in [('a',2,6),('b',2,6),('c',3,6),('d',5,8),('e',None,20)]];self.assertEqual(frontier(rs,'score',-1),['a','b','d']);self.assertEqual(frontier(rs,'score',1),['a','b'])

if __name__=='__main__':unittest.main()
