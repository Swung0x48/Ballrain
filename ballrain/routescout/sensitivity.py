"""One-at-a-time model sensitivity, not a calibrated engine success probability."""
import numpy as np
from ballrain.reach import BALLS, ReachConfig, rollout
from .search import FRAMES, KEYS, PUSH, STEP, boundary, scout_rollout


def section_scan(field, geo, p, v, ball, sector, objects, goal, node):
    sections=[]
    for i,(key,end) in enumerate(zip(node.keys,node.ends)):
        steps=min(FRAMES,node.frames-i*FRAMES)//STEP
        u=PUSH[KEYS.index(key),:steps]; h=v[[0,2]]; speed=np.linalg.norm(h)
        axis=h/max(speed,1e-9) if speed>.1 else np.array(end)[[0,2]]-p[[0,2]]
        axis=axis/max(np.linalg.norm(axis),1e-9); side=np.array([-axis[1],axis[0]])
        # No teleport is applied in the engine. The central trial is propagated exactly.
        trials=[(0,1,0)]+[(x,1,0) for x in (-2,-1,-.5,.5,1,2)]+[(0,x,0) for x in (.8,1.2)]+[(0,1,x) for x in (-10,10)]
        pp=np.tile(p,(len(trials),1)); vv=np.tile(v,(len(trials),1))
        for j,(offset,scale,angle) in enumerate(trials):
            theta=np.deg2rad(angle); pp[j,[0,2]]+=side*offset
            vv[j,[0,2]]=speed*scale*(axis*np.cos(theta)+side*np.sin(theta))
        vv[0]=v
        r=scout_rollout(field,pp,vv,BALLS[ball],np.repeat(u[None,:,:],len(trials),axis=0),ReachConfig(step_psi=STEP))
        status=node.status if i==len(node.keys)-1 else 'frontier'
        good=[]
        for j,line in enumerate(r['traj']):
            hit=boundary(line,ball,sector,objects,goal)
            rules=getattr(field,'scout_rules',None);admissible=not rules or not any(rules.gate(x,y,ball,allow_pose=field.scout_live) for x,y in zip(line,line[1:]))
            good.append(bool(admissible and r['alive'][j] and np.linalg.norm(line[-1]-end)<=2 and (hit[2] if hit else 'frontier')==status))
        nominal=good[0]; offsets=[x for (x,s,a),ok in zip(trials,good) if ok and s==1 and a==0]; speeds=[speed*s for (x,s,a),ok in zip(trials,good) if ok and x==0 and a==0]
        grounded=r['grounded'][0]; tags=[]
        if np.any(grounded[:-1]&~grounded[1:]): tags.append('leaves support')
        if np.any(~grounded[:-1]&grounded[1:]): tags.append('landing')
        if len(geo.rails) and any(geo.rails.closest(q,2.15) for q in r['traj'][0][grounded]): tags.append('possible rail contact')
        sections.append(dict(key=int(key),status=status,techniques=tags,nominal_match=nominal,
                             model_fragility=1-sum(good[1:])/10 if nominal else None,
                             sampled_width=max(offsets)-min(offsets) if nominal else None,
                             sampled_speed_range=[min(speeds),max(speeds)] if nominal else None))
        p=np.array(end); v=r['final_velocity'][0]
    return sections
