"""Branch a rolling trough witness toward a lateral pickup, then recenter for the exit.

Parameters are per-segment geometry/pin metadata, not level constants. Coordinates
are in a deck with longitudinal +z. Other deck orientations need a local basis.
A proposal is not a family certificate until routescout_payoff verifies both builds.
"""
import argparse,itertools,json,tempfile
from pathlib import Path
import numpy as np
from routescout_atlas import load
from routescout_pickups import bank,take,state,tas,DEFAULT_DELTA,Ev,world_to_camera_keys


def branch(a):
    d=json.loads(a.input.read_text());keys=np.repeat([k for k,n in d['spawn_keys']],[n for k,n in d['spawn_keys']]);a.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.out.parent) as tmp:
        s=load(a,d['level'],a.bin,a.copy,Path(tmp))
        try:
            start=s.frames_done;i=a.cut-start
            if not 0<i<len(keys):raise ValueError('bad cut')
            s.act(tas.make_frames([DEFAULT_DELTA]*i,keys[:i]));seq=[[int(k),len(list(g))] for k,g in itertools.groupby(keys[:i])];got=False;exit=False;last=0;sign=1;life=s.tracker.lives
            for step in range(1200):
                p,v,b=state(s);z=p[2];target=a.exit_x if exit else a.pickup_x;u=np.clip((target-p[0])*2.5,-7,7)-v[0];side=8 if u>.45 else 4 if u<-.45 else 0
                if step-last>8 and sign*v[2]<-.15:sign=-sign;last=step
                if z<a.turn_z and sign==-1:sign=1;last=step
                if z>a.crest_z and v[2]>0 and p[1]>a.crest_y:exit=True
                key=side|(1 if sign>0 else 2);cam=s.obs.cams['cams'][0]['matrix']
                if not exit and s.frames_done<a.hold_until and s.frames_done-start<len(keys):
                    orig=int(keys[s.frames_done-start]);wz=min((0,1,2,4,5,6,8,9,10),key=lambda m:(world_to_camera_keys(m,cam)^orig).bit_count());key=side|(wz&3)
                if exit:key=side|(1 if v[2]<5.5 else 2)
                o=take(s,seq,world_to_camera_keys(key,cam) if s.tracker.flags.ball_nav else 0,6)
                if s.tracker.followers() and not got:got=bank(s,seq)
                got=got or s.tracker.lives>life
                if o is None or o.has_event(Ev.BallOff):break
                if o.has_event(Ev.PostCheckpoint):
                    if got:a.out.write_text(json.dumps(dict(level=d['level'],sector=d['sector'],spawn_keys=seq)));return
                    break
            raise ValueError('no verified payoff branch')
        finally:s.close(kill=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','bin','input','out'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('cut','hold-until'):p.add_argument('--'+k,type=int,required=True)
    for k in ('exit-x','pickup-x','turn-z','crest-z','crest-y'):p.add_argument('--'+k,type=float,required=True)
    p.add_argument('--copy',required=True);branch(p.parse_args())
