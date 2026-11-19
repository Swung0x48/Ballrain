"""Synthetic geometry and triggers: no game assets or replay inputs."""
import numpy as np
from ballrain.geometry import TriangleSet
from ballrain.reach import BALLS, ReachConfig, SupportMap, rollout
from ballrain.routescout.evidence import aggregate_rows
from ballrain.routescout.graph import SurfaceGraph, guide_distance
from ballrain.routescout.map import classify
from ballrain.routescout.search import boundary, cell, contact, search, PUSH


def floor():
    return TriangleSet([[0,0,0], [20,0,0]], [[20,0,0], [20,0,20]], [[0,0,20], [0,0,20]])


def test_welded_mesh_guide():
    g = SurfaceGraph(floor()); line=g.corridor([1,2,1],[19,2,19]); assert len(line)>=2
    assert g.gates(line)==[] and guide_distance(line,np.array(line[-1]))(line[-1])==0
    assert g.corridor([100,2,1],[1,2,1])==[]


def test_drop_proposal_is_directional():
    t=TriangleSet([[0,10,0],[1,0,0]],[[1,10,0],[2,0,0]],[[0,10,1],[1,0,1]])
    g=SurfaceGraph(t); line=g.corridor([0,10,0],[1,0,0]); assert g.gates(line)
    assert not g.corridor([1,0,0],[0,10,0])


def test_swept_gates_and_goal_order():
    path=np.array([[0.,2,0],[12,2,0],[24,2,0]])
    objects=[dict(sector=2,kind='trafo',name='P_Trafo_Paper',pos=[6,2,0])]
    goal=path[-1]; assert boundary(path,0,2,objects,goal)==(1,0,'P_Trafo_Paper',1)
    assert boundary(path,1,2,objects,goal)[2]=='goal region'
    assert boundary(path,0,1,objects,goal)[2]=='goal region'
    assert contact(path,np.array([6,10,0]),4) is None
    assert classify('P_Modul_999')=='unknown module'


def test_rollout_chaining_and_beam():
    t=floor(); field=SupportMap(t.A,t.B,t.C); p=np.array([5.,2.,5.]); v=np.zeros(3)
    cfg=ReachConfig(step_psi=4); u=PUSH[4:5]; first=rollout(field,p,v,BALLS[0],u[:,:4],cfg)
    second=rollout(field,first['traj'][0,-1],first['final_velocity'][0],BALLS[0],u[:,4:],cfg)
    whole=rollout(field,p,v,BALLS[0],u,cfg)
    np.testing.assert_allclose(second['traj'][0,-1],whole['traj'][0,-1],atol=.01)
    nodes,exp=search(field,p,v,0,1,p+[6,0,0],[],depth=3,width=2)
    assert 9<=exp<=45 and nodes[0].status=='goal region'
    assert cell(p,v,0,1)!=cell(p,v,2,1)
    assert search(field,p,v,1,1,p,[])[1]==0


def test_only_aggregate_counts_are_imported():
    text='|2|family|wood|3|P01 P04|2/1/0|1.0|0|123,456|'
    assert aggregate_rows(text)=={'2':[dict(family='family',materials='wood',passed=2,died=1,censored=0)]}


def test_sensitivity_nominal_and_sampling():
    from types import SimpleNamespace
    from ballrain.routescout.sensitivity import section_scan
    t=floor(); field=SupportMap(t.A,t.B,t.C); p=np.array([5.,2.,5.]); v=np.zeros(3)
    goal=np.array([14.,2.,5.]); n=search(field,p,v,0,1,goal,[],depth=3,width=2)[0][0]
    rows=section_scan(field,SimpleNamespace(rails=[]),p,v,0,1,[],goal,n)
    assert rows and rows[0]['nominal_match'] and 0<=rows[0]['model_fragility']<=1
    assert 0<=rows[0]['sampled_width']<=4
