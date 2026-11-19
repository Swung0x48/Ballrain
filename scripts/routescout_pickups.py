"""Geometry-guided pickup detours spliced onto a verified scout approach; engine is the sole judge."""
import heapq
import math
import numpy as np
from ballrain.features import level_geometry
from ballrain.env import world_to_camera_keys
from routescout_payoff import state,tas,DEFAULT_DELTA,Ev


def ribbon(pts):
    pts=np.array(pts,float);pts=pts[np.r_[True,np.linalg.norm(np.diff(pts,axis=0),axis=1)>.005]]
    if len(pts)<2:pts=np.vstack([pts,pts[-1]+[.01,0,0]])
    seg=np.diff(pts,axis=0);length=np.linalg.norm(seg,axis=1);return pts,seg,length,np.r_[0.,length.cumsum()]


def project(line,p,prior=None):
    x,v,l,s=line;idx=np.arange(len(l)) if prior is None else np.flatnonzero((s[1:]>=prior-7)&(s[:-1]<=prior+14));idx=idx if len(idx) else np.arange(len(l));f=np.clip(((p-x[idx])*v[idx]).sum(1)/(l[idx]**2),0,1);d=np.linalg.norm(x[idx]+f[:,None]*v[idx]-p,axis=1);j=d.argmin();i=idx[j];return s[i]+f[j]*l[i],d[j]


def point(line,st):
    x,v,l,s=line;i=min(len(l)-1,max(0,np.searchsorted(s,st,side='right')-1));return x[i]+v[i]*np.clip((st-s[i])/l[i],0,1)


def local_line(t,p,q):
    lo=np.floor(np.minimum(p,q)[[0,2]]-11);hi=np.ceil(np.maximum(p,q)[[0,2]]+11);X,Z=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1),indexing='ij');pts=np.c_[X.ravel(),np.full(X.size,max(p[1],q[1])+5.),Z.ravel()];y=np.concatenate([t.ray_down(c,owner_mask=abs(t.N[:,1])>.5)[0] for c in np.array_split(pts,1+len(pts)//512)]).reshape(X.shape)+2;ok=np.isfinite(y);ids=np.argwhere(ok)
    if not len(ids):return None
    def near(a):return tuple(ids[np.argmin((X[ok]-a[0])**2+(Z[ok]-a[2])**2+(y[ok]-a[1])**2)])
    a=near(p);b=near(q);cost=np.zeros(X.shape)
    for dx,dz in ((2,0),(-2,0),(0,2),(0,-2)):cost+=np.where(abs(y-np.roll(y,(dx,dz),(0,1)))<1.7,0.,.85)
    dist={a:0.};par={};queue=[(0.,0.,a)]
    while queue:
        _,w,k=heapq.heappop(queue)
        if k==b:break
        if w!=dist[k]:continue
        for di,dj in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            for jump in (1,2,3,4,5,6):
                m=k[0]+di*jump,k[1]+dj*jump
                if not(0<=m[0]<X.shape[0] and 0<=m[1]<X.shape[1]):break
                if not ok[m]:continue
                dy=y[m]-y[k];ds=math.hypot(di,dj)*jump
                if -12<dy<.85*ds+.3:
                    nw=w+ds*(1+cost[m])+max(0.,dy)*2+4*(jump-1)
                    if nw<dist.get(m,1e99):dist[m]=nw;par[m]=k;heapq.heappush(queue,(nw+math.dist(m,b),nw,m))
                break
    if b not in dist:return None
    path=[b]
    while path[-1]!=a:path.append(par[path[-1]])
    return ribbon([p,*[[X[k],y[k],Z[k]] for k in reversed(path)],q])


def append(seq,key,n):
    if n<=0:return
    if seq and seq[-1][0]==key:seq[-1]=[key,seq[-1][1]+n]
    else:seq.append([key,n])


def take(s,seq,key,n):
    fd=s.frames_done;o=s.act(tas.make_frames([DEFAULT_DELTA],[key]),repeat=n,stop=[Ev.BallOff,Ev.LevelFinish,Ev.PostCheckpoint]);append(seq,key,s.frames_done-fd);return o


def bank(s,seq):
    if not s.tracker.flags.ball_nav or s.tracker.flags.paused:return False
    take(s,seq,tas.KEY_ESC,1);take(s,seq,0,3)
    if not s.tracker.flags.paused:return False
    for _ in range(120):
        if not s.tracker.followers():break
        take(s,seq,0,12)
    ok=not s.tracker.followers();take(s,seq,tas.KEY_ESC,1);take(s,seq,0,3);return ok and not s.tracker.flags.paused


def drive(s,seq,line,cap=8.,look=4.,block=8,goal_radius=2.,stop_at_checkpoint=True,max_blocks=2200,allow_transform=False):
    prior=None;best=-100;stuck=0
    for _ in range(max_blocks):
        p,v,b=state(s);st,err=project(line,p,prior);prior=st
        if s.tracker.flags.ball_nav and np.linalg.norm(p-line[0][-1])<goal_radius:return 'near'
        q=point(line,st+look);e=(q-p)[[0,2]];want=e*min(2.7,cap/max(.01,np.linalg.norm(e)));u=want-v[[0,2]];key=(8 if u[0]>.55 else 4 if u[0]<-.55 else 0)|(1 if u[1]>.55 else 2 if u[1]<-.55 else 0);key=world_to_camera_keys(key,s.obs.cams['cams'][0]['matrix']);o=take(s,seq,key if s.tracker.flags.ball_nav else 0,block)
        if o is None or o.has_event(Ev.BallOff):return 'death'
        if o.has_event(Ev.LevelFinish):return 'finish'
        if o.has_event(Ev.PostCheckpoint) and stop_at_checkpoint:return 'checkpoint'
        if st>best+.3:stuck=0;best=st
        else:stuck+=1
        if stuck>110 or (err>13 and v[1]<-7):return 'off-line'
    return 'timeout'


def banked_suffix(a):
    """Propose a pause-bank at an observed frame; retain only a real checkpoint or finish."""
    import itertools,json,tempfile
    from pathlib import Path
    from routescout_atlas import load
    src=json.loads(a.input.read_text());seq=[];keys=np.repeat([k for k,n in src['spawn_keys']],[n for k,n in src['spawn_keys']]);a.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.out.parent) as tmp:
        s=load(a,src['level'],a.bin,a.copy,Path(tmp))
        try:
            i=a.frame-s.frames_done
            if not 0<i<len(keys):raise ValueError('bank frame outside recorded drive')
            s.act(tas.make_frames([DEFAULT_DELTA]*i,keys[:i]));seq=[[int(k),len(list(g))] for k,g in itertools.groupby(keys[:i])]
            if not bank(s,seq):raise ValueError('pause bank not confirmed')
            for k,g in itertools.groupby(np.r_[keys[i:],np.repeat(keys[-1],a.extension)]):
                o=take(s,seq,int(k),len(list(g)))
                if o is None or o.has_event(Ev.BallOff):break
                if o.has_event(Ev.PostCheckpoint) or o.has_event(Ev.LevelFinish):
                    a.out.write_text(json.dumps(dict(level=src['level'],sector=src['sector'],spawn_keys=seq)));return
            raise ValueError('no checkpoint or finish on modified drive')
        finally:s.close(kill=True)

if __name__=='__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser(description='Insert and physically verify a bank on a checkpoint approach.')
    for k in ('root','bin','input','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--copy',required=True);p.add_argument('--frame',type=int,required=True);p.add_argument('--extension',type=int,default=100);banked_suffix(p.parse_args())
