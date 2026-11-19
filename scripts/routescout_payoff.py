"""Separate measurement from family selection: raw engine clocks/score ledgers, not route forecasts."""
import hashlib
import json
import re
import tempfile
from pathlib import Path
import numpy as np
from routescout_atlas import load, state, tas, Ev, DEFAULT_DELTA
from ballrain.protocol import EVENT_NAMES
from ballrain.routescout.mechanisms import Rules

WATCH = re.compile(r'^(P_Extra_Point_Ball0|P_Extra_Life_MF)(\d*)$')
STOPS = [Ev.BallOff,Ev.PostCheckpoint,Ev.LevelFinish,Ev.ExtraPoint,Ev.PreLifeUp,Ev.PostLifeUp,Ev.PauseLevel,Ev.UnpauseLevel]


def inventory(atlas):
    result={};counts={}
    for ob in atlas['map']['placements']:
        if ob['name'] not in ('P_Extra_Life','P_Extra_Point'):continue
        sec=ob['sector'];kind='life' if ob['name'].endswith('Life') else 'point'
        count=counts.get((sec,kind),0)+1;counts[sec,kind]=count
        result[f'{sec}:{kind}:{count}']=dict(sector=sec,kind=kind,pos=ob['pos'])
    return result


def snapshot(s):
    tr=s.tracker
    return dict(frame=s.frames_done,sr_ms=tr.sr_ms,points=tr.points,lives=tr.lives,life_bonus=tr.life_bonus,
                followers=tr.followers(),potential=tr.hs_partial(),deaths=tr.deaths,level_bonus=tr.level_bonus)


def run(a,binary,copy,doc,atlas):
    ph=inventory(atlas);frames=np.concatenate([np.repeat(tas.make_frames([DEFAULT_DELTA],[k]),n) for k,n in doc['spawn_keys']]);level=doc['level']
    with tempfile.TemporaryDirectory(dir=a.out) as tmp:
        s=load(a,level,binary,copy,Path(tmp));tr=s.tracker
        try:
            if tr.level.static_key()!=atlas['map']['static_key']:raise ValueError('wrong static geometry')
            rules=Rules(tr);mechanisms=[];start=snapshot(s);boundaries=[start];log=[];points=[];picked={};events=[];unknown=[];watched={};i=0;sec=1
            def watch():
                for ck,(visible,position) in tr.watch.items():
                    name=tr.level.name(int(ck)) or ''
                    if WATCH.match(name): watched[name]=(int(visible),np.asarray(position,float).tolist())
            watch()
            def posrow():
                if s.obs.ball is not None and s.obs.ball['found']:
                    p,v,b=state(s);contacts=sorted({tr.level.name(int(ck)) or '' for ck in s.obs.fric['partner']});return [s.frames_done,*p.tolist(),*v.tolist(),int(b),contacts]
            points.append(posrow());last=start
            while i<len(frames):
                before=s.frames_done;wprev=watched.copy();o=s.act(frames[i:i+12],stop=STOPS);n=(min(s.frames_done,tr.levelfinish_frame) if tr.levelfinish_frame is not None else s.frames_done)-before
                if o is None or n<=0:break
                i+=n;now=snapshot(s);watch();row=posrow();
                if row:
                    ev=rules.observe(points[-1],row,sec);points.append(row)
                    if ev['contacts'] or ev['fan'] or ev['gate']:mechanisms.append(ev)
                names=[EVENT_NAMES.get(int(e['code']),str(e['code'])) for e in o.events]
                if names:events.extend([[s.frames_done,name] for name in names])
                dp=now['points']-last['points'];p=np.array(row[1:4]) if row else None
                for kind in ('point','life'):
                    islife=kind=='life'; candidates=[]
                    for key,obj in ph.items():
                        if obj['sector']!=sec or obj['kind']!=kind or key in picked:continue
                        q=np.array(obj['pos']);dist=float(np.linalg.norm(p-q)) if p is not None else 1e4
                        for name,(vis,pold) in wprev.items():
                            if ('Life' in name)!=islife:continue
                            if np.linalg.norm(np.array(pold)-q)>1.0:continue
                            hidden=bool(vis and not watched.get(name,(vis,))[0]);candidates.append((not hidden,dist,key,name,hidden))
                    candidates.sort()
                    # A follower payout alone must not mint a second main. Life uses the pre-hook, not a guessed radius.
                    possible=bool(candidates and candidates[0][1]<6.5)
                    if islife: fired='PreLifeUp' in names
                    else:fired=possible and candidates[0][4] and dp>=98
                    if fired:
                        if not possible:unknown.append([s.frames_done,kind,'no unique nearby placement']);continue
                        _,dist,key,name,hidden=candidates[0];picked[key]=dict(frame=s.frames_done,watched=name,distance=dist,hide=hidden,points_delta=dp)
                if dp or now['lives']!=last['lives'] or now['followers']!=last['followers']:log.append([s.frames_done,now['points'],now['lives'],now['followers'],names])
                if o.has_event(Ev.PostCheckpoint):boundaries.append(now);sec+=1;rules.refresh()
                last=now
                if o.has_event(Ev.LevelFinish):break
                if o.has_event(Ev.BallOff):break
            fin=tr.levelfinish_frame is not None;hs=tr.hs_at_finish if fin else None
            return dict(level=level,start=start,boundaries=boundaries,picked=picked,unresolved=unknown,score_log=log,events=events,
                        mechanism_pin=rules.pin,mechanism_observations=mechanisms,rows=points,end=last,finished=fin,official_hs=hs,finish_sr_ms=s.sr_at_levelfinish if fin else None,
                        frame_sha256=hashlib.sha256(frames[:i].tobytes()).hexdigest(),consumed=i,frame_count=len(frames))
        finally:s.close(kill=True)


def verify(a,file):
    doc=json.loads(file.read_text(encoding='utf8'));atlas=json.loads((a.atlases/f"L{doc['level']:02}.json").read_text(encoding='utf8'))
    first=run(a,a.soft,a.copy,doc,atlas);second=run(a,a.second,a.copy,doc,atlas)
    if first!=second:raise ValueError(f'{file.stem}: builds disagree')
    if first['end']['deaths'] or first['consumed']!=first['frame_count'] or (not first['finished'] and len(first['boundaries'])<=doc['sector']):raise ValueError('unverified endpoint')
    a.out.mkdir(parents=True,exist_ok=True);(a.out/file.name).write_text(json.dumps(first,separators=(',',':'),allow_nan=False),encoding='utf8')
    print(file.stem,first['boundaries'][-1],list(first['picked']),flush=True)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('root','soft','second','atlases','out','input'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--copy',required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);verify(args,args.input)
