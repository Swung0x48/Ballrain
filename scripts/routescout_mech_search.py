"""Receding-horizon route search with the pinned reachmech layers and an engine judge.

The local raster is a cropped version of the reference rollout input. Unknowns trigger
engine validation, not a new guessed ball rule. No TAS inputs or online feature switch.
"""
import argparse,json,tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
from routescout_atlas import load,state,tas,Ev,DEFAULT_DELTA,world_to_camera_keys
from routescout_pickups import ribbon,project,point,append
from ballrain.features import level_geometry
from ballrain.reach import BALLS,SupportMap,ReachConfig,key_push,rollout,KEY_COMBOS
from ballrain.fastreach import fan_dv,FAN_R_NOM
from ballrain.routescout.mechanisms import Rules
from ballrain.reach_mechanisms import uses_lowered_plate


class Local:
    def __init__(self,r):self.r=r;self.cache={};self.centre=None
    def plans(self,p,v,b,H):
        r=self.r;t=level_geometry(r.tr).tris
        if self.centre is None or max(abs((p-self.centre)[[0,2]]))>25:self.centre=p.copy();self.cache.clear()
        k=b
        if r.probe and b==r.probe[0] and uses_lowered_plate(r.tr,p,r.probe[2]):k=r.probe[1]
        if k not in self.cache:
            c=self.centre[[0,2]];mask=np.all((t.hi[:,[0,2]]>=c-55)&(t.lo[:,[0,2]]<=c+55),axis=1);tri=np.stack([t.A[mask],t.B[mask],t.C[mask]],axis=1);extra=r.layers[k]
            if extra is not None:tri=np.concatenate([tri,extra[np.all((extra.max(1)[:,[0,2]]>=c-55)&(extra.min(1)[:,[0,2]]<=c+55),axis=1)]])
            boxes=[(a[-2],a[-1]) for a in r.tr.level.depth if min((a[-1]-a[-2])[[0,2]])>=20];self.cache[k]=SupportMap(*tri.transpose(1,0,2),death_boxes=boxes)
        keys=(0,*KEY_COMBOS);push=np.repeat(key_push(keys,0)[:,None,:],H,axis=1);ball=BALLS[b];dv=fan_dv(ball);fan=(r.fans,dv,FAN_R_NOM) if dv and len(r.fans) else None
        # Paper outside modeled fans/plates is an engine-only proposal, not model-certified.
        result=rollout(self.cache[k],p,v,ball,push,ReachConfig(step_psi=4),fan=fan);return keys,result


def main(a):
    plan=json.loads(a.plan.read_text());level=plan['level'];sector=plan['sector'];rng=np.random.default_rng(a.seed);steps=[];seq=[];out=a.out;out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        s=load(a,level,a.bin,a.copy,Path(tmp));start=s.frames_done
        try:
            if a.input:
                import itertools
                doc=json.loads(a.input.read_text());K=np.repeat([k for k,n in doc['spawn_keys']],[n for k,n in doc['spawn_keys']]);K=K[:plan['cut']-start];assert start+len(K)==plan['cut'];s.act(tas.make_frames([DEFAULT_DELTA]*len(K),K));seq=[[int(k),len(list(g))] for k,g in itertools.groupby(K)]
            r=Rules(s.tracker);model=Local(r);guide=ribbon(plan['guide']);prior=None;reason='budget';blocked=unknown=0
            for j in range(a.blocks):
                if not s.tracker.flags.ball_nav or not s.obs.ball['found']:
                    before=s.frames_done;o=s.act(tas.make_frames([DEFAULT_DELTA],[0]),repeat=12,stop=[Ev.BallOff,Ev.PostCheckpoint,Ev.LevelFinish]);append(seq,0,s.frames_done-before)
                else:
                    p,v,b=state(s);st,err=project(guide,p,prior);prior=st;target=point(guide,st+a.look);future=[point(guide,x) for x in np.linspace(st,min(guide[3][-1],st+max(12,a.look)),9)]
                    if any(r.gate(x,y,b) for x,y in zip(future,future[1:])):blocked+=1;reason='guide intersects raised barrier';break
                    ks,R=model.plans(p,v,b,a.horizon);scores=[];count=0
                    for i,k in enumerate(ks):
                        line=R['traj'][i];fail=any(r.gate(x,y,b) for x,y in zip(line,line[1:]));count+=int(fail);cost=np.linalg.norm(line[-1]-target)+rng.uniform(0,a.noise);scores.append(float(cost) if R['alive'][i] and not fail else float('inf'))
                    blocked+=count
                    fallback=False
                    if not np.isfinite(scores).any():
                        near=r.near(p,sector,b);contacts=[s.tracker.level.name(int(x)) for x in s.obs.fric['partner']];dynamic=any(r.rule(r.parts[n],b) in ('unknown','observed_crossing','pushable','pushable_assumed') for n in contacts if n in r.parts) or any(q in ('unknown','observed_crossing','pushable','pushable_assumed') for _,q in near)
                        if not dynamic:reason='no admissible proposal';break
                        # Finite bounded engine-only expansion near moving support, not a model pass.
                        direction=(target-p)[[0,2]];desired=direction*min(1.5,8/max(.001,np.linalg.norm(direction)))-v[[0,2]];scores=(-key_push(ks,0)@desired).tolist();fallback=True
                    i=int(np.argmin(scores));near=r.near(p,sector,b);unknown+=int(any(q in ('unknown','pushable_assumed','observed_crossing','pushable') for _,q in near));key=world_to_camera_keys(ks[i],s.obs.cams['cams'][0]['matrix']);before=s.frames_done;o=s.act(tas.make_frames([DEFAULT_DELTA],[key]),repeat=a.block,stop=[Ev.BallOff,Ev.PostCheckpoint,Ev.LevelFinish,Ev.BallNavInactive]);append(seq,key,s.frames_done-before);steps.append(dict(engine_only=fallback,frame=before,station=float(st),pos=p.tolist(),ball=b,near=near,fan=r.fan_ids(p),reject=count,key=int(key)))
                    if o is not None and o.ball['found'] and b==state(s)[2] and r.gate(p,state(s)[0],b):reason='table veto';break
                if o is None or o.has_event(Ev.BallOff):reason='fall';break
                if o.has_event(Ev.PostCheckpoint) or o.has_event(Ev.LevelFinish):reason='endpoint';break
                if not s.tracker.flags.ball_nav:model.cache.clear();r.refresh()
            out.write_text(json.dumps(dict(level=level,sector=sector,seed=a.seed,reason=reason,frame=s.frames_done,asset_pin=r.pin,blocked=blocked,uncertain_steps=unknown,steps=steps,spawn_keys=seq)));print(out.stem,reason,len(steps),blocked,flush=True)
        finally:s.close(kill=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','bin','plan','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--input',type=Path);p.add_argument('--copy',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--blocks',type=int,default=80);p.add_argument('--block',type=int,default=16);p.add_argument('--horizon',type=int,default=16);p.add_argument('--look',type=float,default=9);p.add_argument('--noise',type=float,default=.1);main(p.parse_args())
