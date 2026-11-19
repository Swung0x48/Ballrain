"""Evaluation only: perturb the approach to a specified replay section in the engine.

No frame or output from this tool is used by the atlas builder. Keep deltas and all
non-arrow inputs; use legal force pulses rather than teleporting onto the obstacle.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ballrain import tas
from ballrain.client import AgentSession, SessionConfig
from ballrain.headless import HeadlessLauncher
from ballrain.protocol import Ev, EVENT_NAMES


def observe(s):
    b=s.obs.ball
    return dict(found=bool(b['found']),pos=b['pos'].tolist(),vel=(b['speed']+b['speed_change']).tolist(),ball=int(b['type']))


def trial(a, directory, frames, prefix, key, duration):
    start=a.entry-duration
    tas.save_raw(prefix,frames[:start]); session=AgentSession(SessionConfig(a.level,directory,copy=a.copy,prefix=prefix),
        HeadlessLauncher(binary=str(a.bin.resolve()),root=str(a.root.resolve()),socket='tcp',ball_order=a.order))
    try:
        session.start()
        if session.frames_done!=start: raise ValueError('prefix ended early')
        pulse=frames[start:a.entry].copy()
        if key is not None: pulse['keys']=(pulse['keys'] & np.uint32(0xfffffff0)) | key
        session.act(pulse); entry=observe(session)
        session.act(frames[a.entry:a.end],stop=[Ev.BallOff,Ev.LevelFinish]); end=observe(session)
        return dict(key=key,duration=duration,entry=entry,end=end,stop=session.frames_done,
                    events=[EVENT_NAMES.get(e[1],str(e[1])) for e in session.tracker.events if e[0]>start])
    finally: session.close(kill=True)


def main(a):
    with np.load(a.record) as f:
        frames=f['record_frames']; fd=f['frames_done']; i=np.flatnonzero(fd==a.end).item()
        target=f['ball_pos'][i]; ball=int(f['ball_type'][i])
    if not 48<=a.entry<a.end<=len(frames): raise ValueError('invalid frame range')
    with tempfile.TemporaryDirectory(dir=a.out.parent) as tmp:
        d=Path(tmp); rows=[]
        for key,duration in [(None,48),*((k,n) for n in (2,4,12,24,48) for k in (1,2,4,8))]:
            r=trial(a,d/'session',frames,d/'prefix.bin',key,duration); e=r['end']; err=np.linalg.norm(e['pos']-target); r['error']=float(err) if e['found'] else None
            r['pass']=bool(r['stop']==a.end and e['found'] and e['ball']==ball and err<=4 and 'BallOff' not in r['events']); rows.append(r)
    rate=sum(r['pass'] for r in rows[1:])/(len(rows)-1) if rows[0]['pass'] else None
    good=[r['entry'] for r in rows if r['pass'] and rate is not None]
    forward=np.array(rows[0]['entry']['vel'])[[0,2]]; forward/=max(1e-9,np.linalg.norm(forward)); side=forward[::-1]*[1,-1]
    def extent(values): return [float(min(values)),float(max(values))] if values else None
    speeds=extent([np.linalg.norm(np.array(e['vel'])[[0,2]]) for e in good]); offsets=extent([(np.subtract(e['pos'],rows[0]['entry']['pos'])[[0,2]])@side for e in good])
    a.out.write_text(json.dumps(dict(entry=a.entry,end=a.end,order=a.order,pass_rate=rate,hard=(rate<=.5 if rate is not None else None),passing_entry_speed=speeds,passing_lateral_offset=offsets,rows=rows),allow_nan=False),encoding='utf8')
    print(a.level,'baseline',rows[0]['error'],'rate',rate,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('record','bin','root','out'): p.add_argument('--'+k,type=Path,required=True)
    for k in ('level','entry','end'): p.add_argument('--'+k,type=int,required=True)
    p.add_argument('--order',default='ckid'); p.add_argument('--copy',required=True); main(p.parse_args())
