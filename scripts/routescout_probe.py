"""Probe a verified input at a named section; use legal arrow replacements, never state edits."""
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
from routescout_atlas import load, state, tas, Ev, DEFAULT_DELTA


def probe(a):
    d=json.loads(a.measured.read_text()); tape=json.loads(a.input.read_text()); seq=np.repeat([v[0] for v in tape['spawn_keys']], [v[1] for v in tape['spawn_keys']]); start=d['start']['frame']; rows=d['rows']; lookup={r[0]:r for r in rows}; target=lookup[a.end]; entry=a.entry
    last=min((r[0] for r in rows if r[0]>=a.end+300), default=rows[-1][0]); tests=[]
    for key,n in [(0,0)]+[(k,n) for n in (12,48) for k in (1,2,4,8)]:
        with tempfile.TemporaryDirectory(dir=a.out.parent) as tmp:
            s=load(a,d['level'],a.bin,a.copy,Path(tmp)); keys=seq.copy(); i=entry-start
            try:
                assert s.frames_done==start
                if n:keys[i-n:i]=(keys[i-n:i]&~15)|key
                def step(to):
                    while s.frames_done<to:
                        batch=keys[s.frames_done-start:min(to,s.frames_done+256)-start];o=s.act(tas.make_frames([DEFAULT_DELTA]*len(batch),batch),stop=[Ev.BallOff,Ev.LevelFinish]);
                        if o is None or o.has_event(Ev.BallOff) or o.has_event(Ev.LevelFinish):break
                def obs():
                    if not s.obs.ball['found']:return None
                    p,v,b=state(s);return dict(pos=p.tolist(),vel=v.tolist(),ball=int(b))
                def near(t):
                    e=obs();return bool(e and s.frames_done==t and not s.tracker.deaths and e['ball']==lookup[t][7] and np.linalg.norm(np.subtract(e['pos'],lookup[t][1:4]))<=4)
                step(entry); e=obs();step(a.end); ok=near(a.end); end_score=[s.tracker.points,s.tracker.lives,s.tracker.followers()]; recovered=False
                if not ok and not s.tracker.deaths and last>a.end:step(last);recovered=near(last)
                tests.append(dict(key=key,n=n,entry=e,passed=ok,end_score=end_score,recovered=recovered,death=bool(s.tracker.deaths),stop=s.frames_done))
            finally:s.close(kill=True)
    if not tests[0]['passed']: raise ValueError('nominal line did not reproduce')
    a.out.write_text(json.dumps(dict(entry=entry,end=a.end,followup=last,trials=tests)));print(a.out.stem,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('measured','input','root','bin','out'):p.add_argument('--'+arg,type=Path,required=True)
    for arg in ('entry','end'):p.add_argument('--'+arg,type=int,required=True)
    p.add_argument('--copy',required=True);probe(p.parse_args())

