"""Replay a single TAS-free archive cell or physical-key prefix on two builds; no promotion to a checkpoint route."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from routescout_atlas import load, state, DEFAULT_DELTA, tas, Ev
from ballrain.features import level_geometry
from ballrain.state import G_RAILS


def observe(a, exe, copy, frames):
    events, rows = [], []
    with tempfile.TemporaryDirectory(dir=a.output.parent) as tmp:
        s=load(a,a.level,exe,copy,Path(tmp));i=0
        try:
            start=s.frames_done;geom=level_geometry(s.tracker).tris
            rail={int(k) for k in np.unique(geom.owner) if s.tracker.level.in_group(k,G_RAILS)
                  or 'geländer' in (s.tracker.level.name(k) or '').lower()}
            while i<len(frames):
                before=s.frames_done;o=s.act(frames[i:i+4],stop=[Ev.BallOff,Ev.BallNavActive,Ev.PostCheckpoint,Ev.LevelFinish]);i+=s.frames_done-before
                if not o or s.frames_done==before or o.has_event(Ev.BallOff):raise ValueError('path ended early')
                if o.has_event(Ev.PostCheckpoint):events.append(s.frames_done)
                p,v,b=state(s)
                if s.tracker.sector!=a.sector or not o.ball['found']:continue
                touches={int(k):s.tracker.level.name(k) or '' for k in o.fric['partner']};poses={};crown=[]
                for k,name in touches.items():
                    if name.startswith(('P_Modul_','P_Ball_','P_Box')):
                        body=next((r for r in o.body if r['ckid']==k),None)
                        if body is not None:poses[name]={field:body[field].tolist() for field in ('pos','q','speed','rot_speed')}
                    if k in rail:
                        h=geom.ray_down((p+[0,.5,0])[None,:],owner_mask=geom.owner==k)[0][0]
                        if np.isfinite(h) and abs(p[1]-h-2)<.15:crown.append(name)
                rows.append(dict(frame=s.frames_done,ball=b,pos=p.tolist(),vel=v.tolist(),contact=sorted(touches.values()),poses=poses,crown=crown))
            return dict(start=start,post=events,rows=rows,terminal_frame=s.frames_done,terminal_ball=int(b),terminal_pos=p.tolist(),terminal_vel=v.tolist())
        finally:s.close(kill=True)


def main(a):
    obj=json.loads(a.input.read_text(encoding='utf8'));case=None
    if a.cell:
        if obj['level']!=a.level:raise ValueError('level differs')
        key=json.loads(a.cell);case=next(c for c in obj['cells'] if c['cell']==key);seq=case['keys']
    else:seq=obj.get('spawn_keys',obj.get('keys'))
    if not seq:raise ValueError('no physical key sequence')
    fr=np.concatenate([np.repeat(tas.make_frames([DEFAULT_DELTA],[k]),n) for k,n in seq]);a.output.parent.mkdir(parents=True,exist_ok=True)
    first=observe(a,a.soft,a.copy,fr);other=observe(a,a.second,a.second_copy,fr)
    if first!=other:raise ValueError('builds disagree')
    if case and (first['terminal_ball']!=case['cell'][1] or not np.allclose(first['terminal_pos'],case['pos'],rtol=0,atol=.0001) or not np.allclose(first['terminal_vel'],case['vel'],rtol=0,atol=.0001)):raise ValueError('terminal state differs')
    first.update(spawn_keys=seq,sha256=hashlib.sha256(fr.tobytes()).hexdigest());a.output.write_text(json.dumps(first),encoding='utf8')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('input','output','soft','second','root'):ap.add_argument('--'+k,type=Path,required=True)
    for k in ('copy','second-copy'):ap.add_argument('--'+k,required=True)
    for k in ('level','sector'):ap.add_argument('--'+k,type=int,required=True)
    ap.add_argument('--cell',help='exact archive cell as JSON array');main(ap.parse_args())
