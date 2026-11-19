from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from ballrain.geometry import TriangleSet
from .rail_actions import RailCrowns
from .mechanism_evidence import MechanismEntry, verification


def test_rail_actions_do_not_conflate_floor_air_or_ride():
    t=TriangleSet([[0,0,0]],[[10,0,0]],[[10,0,.1]],owner=[10]);lvl=SimpleNamespace(in_group=lambda i,g:True)
    c=RailCrowns(SimpleNamespace(level=lvl),SimpleNamespace(tris=t));goal=np.array([10.,2.,0.]);v=np.array([2.,-1.,0.]);
    for ball in (0,1,2):
        assert c.propose(np.array([4.,1.,1.]),v,ball,goal,()).phase=='floor mount'
        assert c.propose(np.array([4.,3.,1.]),v,ball,goal,()).phase=='air intercept'
        assert c.propose(np.array([4.,2.,0.]),v,ball,goal,{10}).phase=='ride'


def test_mechanism_proof_requires_same_entry_and_both_exits():
    e=MechanismEntry('map','P_Modul_25_Bridge',3,2,'pin',(1,2,3),(1,0,0),(0,0,0),(1,0,0,0),(0,0,0))
    rows=[(b,e,'past plate',True,True) for b in ('x86soft','2002')]
    assert verification(e,rows)=={'past plate'}
    assert not verification(e,rows[:1])
    assert not verification(replace(e,prefix_pin='another'),rows)
    assert not verification(e,[rows[0],('2002',e,'past plate',False,True)])
    paper=replace(e,ball=1)
    assert not verification(paper,[(b,paper,'past plate',True,True) for b in ('x86soft','2002')])


def test_pose_bins_ignore_quaternion_sign_and_distant_bodies():
    from .mechanism_evidence import pose_bins
    q=np.array([1.,0,0,0]);row=dict(ckid=5,q=q,pos=np.zeros(3));tr=SimpleNamespace(obs=SimpleNamespace(body=[row]),level=SimpleNamespace(name=lambda ck:'P_Box'))
    a=pose_bins(tr,np.zeros(3));row['q']=-q
    assert len(a)==8 and pose_bins(tr,np.zeros(3))==a and not pose_bins(tr,np.ones(3)*20)


def test_vertical_pairs_propose_inner_groove_not_top_of_tube(monkeypatch):
    from . import tube_graph as tg
    theta=np.linspace(0,2*np.pi,50,endpoint=False)
    axis=np.c_[10*np.cos(theta),np.zeros(len(theta)),10*np.sin(theta)]
    high=axis+[0,3.6,0]
    seg=np.vstack([np.c_[v,np.roll(v,-1,axis=0),np.full(len(v),.35)] for v in (axis,high)])
    monkeypatch.setattr(tg,'extract_rails',lambda *unused:(seg,None,None))
    rail=SimpleNamespace(A=[],B=[],C=[],owner=[])
    monkeypatch.setattr(tg,'level_geometry',lambda tr:SimpleNamespace(rails=rail))
    guide=axis+[0,2.35,0];doc={'guide':guide.tolist(),'kinds':['single']*len(guide)}
    result=tg.groove(None,doc);actual=np.array(result['guide']);out=np.array(result['radial'])
    assert np.allclose(actual[:,1],1.8,atol=.04)
    assert np.all(np.linalg.norm(actual[:,[0,2]],axis=1)<9)
    assert np.allclose(np.linalg.norm(out,axis=1),1)
    assert np.array_equal(np.array(doc['guide']),guide)
