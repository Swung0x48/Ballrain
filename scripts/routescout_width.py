"""Transverse static contact strip, not ball-centre control tolerance or gap feasibility."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
from routescout_atlas import load
from ballrain.features import level_geometry
from ballrain.geometry import TriangleSet


def strip(t,ids,r):
    v=np.array(r[4:7])[::2];speed=np.linalg.norm(v)
    if speed<1.5:return
    side=v[::-1]*[1,-1]/speed;p=np.array(r[1:4]);rad=6.;mask=np.isin(t.owner,ids)&(abs(t.N[:,1])>.45)&(t.lo[:,0]<p[0]+rad+.1)&(t.hi[:,0]>p[0]-rad-.1)&(t.lo[:,2]<p[2]+rad+.1)&(t.hi[:,2]>p[2]-rad-.1)&(t.hi[:,1]>p[1]-5)&(t.lo[:,1]<p[1]);
    if not mask.any():return
    mesh=TriangleSet(t.A[mask],t.B[mask],t.C[mask],grid_cell=None);rail=any('Rail' in n or 'nder_' in n for n in r[8]);dx=.025 if rail else .1;side3=np.r_[side[0],0,side[1]];w=np.arange(-round(rad/dx),round(rad/dx)+1)*dx;c=len(w)//2;Y,J=mesh.ray_down(p+w[:,None]*side3);j=J[c];
    if j<0 or not .25<p[1]-Y[c]<2.7:return
    normal=mesh.N[j];expected=Y[c]-w*(normal@side3)/normal[1];ok=np.isfinite(Y)&(np.abs(Y-expected)<(.04 if rail else .35));left=np.flatnonzero(~ok[:c]);right=np.flatnonzero(~ok[c+1:]);lo=left[-1]+1 if len(left) else 0;hi=c+right[0] if len(right) else len(w)-1
    return dict(frame=r[0],owners=r[8],width=float(w[hi]-w[lo]),left=float(w[lo]),right=float(w[hi]),grid=dx,capped=bool(lo==0 or hi==len(w)-1),rail=rail)


def main(a):
    m=json.loads(a.manifest.read_text());out=a.out;store={}
    for level in m['levels']:
        files=[c['id'] for c in m['members'] if json.loads((out/'measured'/(c['id']+'.json')).read_text())['level']==level];
        if not files:continue
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            s=load(a,level,a.bin,a.copy,Path(tmp))
            try:
                t=level_geometry(s.tracker).tris;byname={s.tracker.level.name(int(i)):i for i in np.unique(t.owner)}
                for name in files:
                    d=json.loads((out/'measured'/(name+'.json')).read_text());a0,b=d['boundaries'][-2:];rows=[r for r in d['rows'] if a0['frame']<=r[0]<=b['frame']];samples=[]
                    for r in rows[::4]:
                        ids=[byname[n] for n in r[8] if n in byname and not n.startswith('P_')];hit=strip(t,ids,r) if ids else None
                        if hit:samples.append(hit)
                    store[name]=dict(n=len(samples),minimum=min(samples,key=lambda x:x['width']) if samples else None)
            finally:s.close(kill=True)
    (out/'widths.json').write_text(json.dumps(store,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','bin','manifest','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--copy',required=True);main(p.parse_args())
