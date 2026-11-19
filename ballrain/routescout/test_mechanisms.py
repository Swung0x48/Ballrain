import unittest
from unittest.mock import patch
import numpy as np
from ballrain.routescout.mechanisms import Rules,cut
from ballrain.routescout.mech_scores import asset_pin,evidence
from ballrain.reach_mechanisms import level_asset

class Adapter(unittest.TestCase):
    def setUp(self):
        r=Rules.__new__(Rules);r.data=level_asset(2);r.plates=r.data['plates'];r.probe=(1,0,'predicate');r.tr=object();self.r=r
        self.p=np.array([931.,49.,-340.]);self.q=np.array([931.,49.,-327.])
    def test_guard_uses_pose_and_material(self):
        with patch('ballrain.routescout.mechanisms.uses_lowered_plate',return_value=False):self.assertEqual(self.r.gate(self.p,self.q,1),'blocked_when_raised');self.assertIsNone(self.r.gate(self.p,self.q,2))
        with patch('ballrain.routescout.mechanisms.uses_lowered_plate',return_value=True):self.assertIsNone(self.r.gate(self.p,self.q,1));self.assertTrue(self.r.gate(self.p,self.q,1,allow_pose=False))
    def test_touch_bypass_are_not_crossing(self):
        box=self.r.plates[0]['bbox'];self.assertFalse(cut(self.p,self.p,box));self.assertFalse(cut(self.p,self.q-[0,0,10],box));self.assertFalse(cut(self.p+[8,0,0],self.q+[8,0,0],box));self.assertTrue(cut(self.p,self.q,box));self.assertTrue(cut(self.q,self.p,box))
    def test_all_assets_have_per_material_rules(self):
        for n in range(1,14):
            d=level_asset(n)
            for p in d['placements']:
                self.assertEqual(len(d['rules'][p['type']]['rules']),3);self.assertTrue(d['rules'][p['type']]['source'])
        self.assertEqual(self.r.rule('not_in_the_table',1),'unknown')
    def test_no_cross_scope_laundering(self):
        d=dict(level=2,mechanism_pin=asset_pin(),mechanism_observations=[dict(frame=5,gate='blocked_when_raised')]);self.assertTrue(evidence(d,100,200)['blocked']);d['mechanism_pin']='broken'
        with self.assertRaises(ValueError):evidence(d,0,10)

if __name__=='__main__':unittest.main()
