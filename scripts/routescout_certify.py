"""Independently replay a local archive path, export a segment only on exact agreement."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from routescout_atlas import load, state, DEFAULT_DELTA, tas, Ev
from ballrain.path import ReferencePath


def measure(a, binary, copy, frames):
    rows, hits, contacts = [], [], []
    with tempfile.TemporaryDirectory(dir=a.out) as tmp:
        s = load(a, a.level, binary, copy, Path(tmp))
        try:
            map_key=s.tracker.level.static_key()
            p, v, b = state(s)
            rows.append((s.frames_done,p.tolist(),v.tolist(),b))
            i = 0
            while i < len(frames):
                before = s.frames_done
                o = s.act(frames[i:i+4], stop=[Ev.BallOff,Ev.PostCheckpoint,Ev.LevelFinish])
                i += s.frames_done-before
                if o is None or o.has_event(Ev.BallOff) or s.frames_done==before:
                    raise ValueError('path interrupted before expected checkpoint')
                p,v,b = state(s)
                if o.ball['found']: rows.append((s.frames_done,p.tolist(),v.tolist(),b))
                for ck in sorted(set(map(int,o.fric['partner']))):
                    name=s.tracker.level.name(ck) or ''
                    if not name.startswith(('P_Modul_', 'P_Box', 'P_Ball_')): continue
                    body=next((r for r in o.body if r['ckid']==ck),None)
                    contacts.append(dict(frame=s.frames_done,owner=name,pos=p.tolist(),vel=v.tolist(),ball=b,
                        body=({k:body[k].tolist() for k in ('pos','speed','q','rot_speed')} if body is not None else None)))
                if o.has_event(Ev.PostCheckpoint): hits.append(s.frames_done)
        finally: s.close(kill=True)
    return rows,hits,contacts,map_key


def main(a):
    doc=json.loads(a.archive.read_text()); a.level=doc['level']; seq=doc['paths'][str(a.sector)]
    frames=np.concatenate([np.repeat(tas.make_frames([DEFAULT_DELTA],[k]),n) for k,n in seq])
    a.out.mkdir(parents=True,exist_ok=True)
    first=measure(a,a.soft,a.copy,frames); second=measure(a,a.second,a.second_copy,frames)
    if first!=second or len(first[1])!=a.sector or first[1][-1]!=first[0][-1][0]:
        raise ValueError('not a matching dual-build post-checkpoint witness')
    rows,hits,contacts,map_key=first
    start=hits[-2] if len(hits)>1 else rows[0][0]
    points=[r for r in rows if r[0]>=start];t=np.array([r[0] for r in points]);p=np.array([r[1] for r in points]);dest=a.out/f'L{a.level:02}-segment-{a.sector}.npz'
    ReferencePath(p,np.full(len(p),a.sector),source='scout',meta=dict(source='engine-scout',complete_level=False),t_ms=(t-start)*float(DEFAULT_DELTA)).save(dest)
    line=ReferencePath.load(dest);assert len(line.P)==len(t) and np.all(np.diff(line.t_ms)>0)
    info=dict(start=start,post=hits,materials=sorted({r[3] for r in points}),samples=len(t),line_length=line.length,game_ms=float(line.t_ms[-1]),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),contact_rows=[r for r in contacts if r['frame']>=start],exact_rest_strings=[],spawn_keys=seq)
    # Keep exact-entry witnesses, not a claim that all states in a coarse cell pass.
    from ballrain.routescout.mechanism_evidence import MechanismEntry,verification
    from dataclasses import asdict
    episodes={};runs=[]
    for row in info['contact_rows']:
        key=row['owner'];last=episodes.get(key)
        if last and row['frame']-last['last_contact']<=4:
            last['last_contact']=row['frame'];continue
        body=row['body']
        if body is None:continue
        idx=row['frame']-rows[0][0];pin=hashlib.sha256(frames[:idx].tobytes()).hexdigest()
        entry=MechanismEntry(map_key,key,a.sector,row['ball'],pin,tuple(row['pos']),tuple(row['vel']),tuple(body['pos']),tuple(body['q']),tuple(body['speed']))
        label=f'PostCheckpoint {a.sector}'
        exits=verification(entry,[(b,entry,label,True,True) for b in ('2002','x86soft')]) if entry.static_key else set()
        evidence=dict(first_contact=row['frame'],last_contact=row['frame'],entry=asdict(entry),exit_verified=sorted(exits),suffix_pin=hashlib.sha256(frames[idx:].tobytes()).hexdigest())
        episodes[key]=evidence;runs.append(evidence)
    info['observed_continuations']=runs
    dest.with_suffix('.json').write_text(json.dumps(info));print(dest.name,len(t),hits,info['sha256'])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','out','soft','second','root'):ap.add_argument('--'+name,required=True,type=Path)
    for name in ('copy','second-copy'):ap.add_argument('--'+name,required=True)
    ap.add_argument('--sector',type=int,required=True);main(ap.parse_args())
