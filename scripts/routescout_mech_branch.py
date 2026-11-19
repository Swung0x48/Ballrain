"""Branch a scout hinge witness at a specified live cut. The engine decides transformer/CP events."""
import argparse,itertools,json,tempfile
from pathlib import Path
import numpy as np
from routescout_atlas import load,state,DEFAULT_DELTA,tas,Ev,world_to_camera_keys
from routescout_pickups import ribbon,drive,take
from ballrain.routescout.mechanisms import Rules


def main(a):
    spec=json.loads(a.plan.read_text());d=json.loads(a.input.read_text());keys=np.repeat([x for x,n in d['spawn_keys']],[n for x,n in d['spawn_keys']]);a.out.parent.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(dir=a.out.parent) as tmp:
        s=load(a,spec['level'],a.bin,a.copy,Path(tmp))
        try:
            cut=spec.get('cut');keys=keys if cut is None else keys[:cut-s.frames_done];seq=[[int(k),len(list(g))] for k,g in itertools.groupby(keys)];s.act(tas.make_frames([DEFAULT_DELTA]*len(keys),keys));r=Rules(s.tracker);p,v,b=state(s)
            if spec['mode']=='follow':
                guide=np.array([r[1:4] for r in json.loads(a.guide.read_text())['rows'] if r[0]>=spec['guide_start']]);ln=ribbon([p,*guide,guide[-1]+guide[-1]-guide[-5]]);res=drive(s,seq,ln,spec['speed'],spec['look'],6,goal_radius=.15,max_blocks=1600)
            else:
                q=min((np.array(x['matrix'][3][:3]) for x in r.rows if x['sector']==spec['sector'] and r.rule(x['type'],b)==spec['avoid_rule']),key=lambda q:np.linalg.norm(q-p));goal=np.array(spec['goal']);axis=(goal-q)[[0,2]];axis/=np.linalg.norm(axis);side=axis[::-1]*[1,-1];res='budget'
                for _ in range(230):
                    p,v,bt=state(s);tar=q[[0,2]]+side*spec['offset']+axis*19 if (p-q)[[0,2]]@axis<5 else goal[[0,2]]+axis*4;u=(tar-p[[0,2]])*1.8;u=u*min(1,11/max(.01,np.linalg.norm(u)))-v[[0,2]];key=(8 if u[0]>.3 else 4 if u[0]<-.3 else 0)|(1 if u[1]>.3 else 2 if u[1]<-.3 else 0);o=take(s,seq,world_to_camera_keys(key,s.obs.cams['cams'][0]['matrix']),6)
                    if o is None or o.has_event(Ev.BallOff) or bt!=b or not s.tracker.flags.ball_nav:break
                    if o.has_event(Ev.PostCheckpoint):res='checkpoint';break
            if res!='checkpoint':raise ValueError('no CP on this branch')
            a.out.write_text(json.dumps(dict(level=spec['level'],sector=spec['sector'],spawn_keys=seq)))
        finally:s.close(kill=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','bin','plan','input','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--guide',type=Path);p.add_argument('--copy',required=True);main(p.parse_args())
