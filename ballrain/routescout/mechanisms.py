"""Offline scout adapter. Rules, geometries, and the down-pose predicate belong to reachmech."""
from collections import Counter
import numpy as np
from ballrain.reach_mechanisms import asset_pin, level_asset, layers_for_tracker, uses_lowered_plate, fan_layer
from ballrain.fastreach import FAN_R_NOM


def cut(a,b,box):
    """Crossing of a raised panel's centre plane inside its footprint, not proximity."""
    lo,hi=np.asarray(box);j=(0,2)[int(hi[2]-lo[2]<hi[0]-lo[0])];c=(lo[j]+hi[j])/2;v=b[j]-a[j]
    if abs(v)<1e-9 or (a[j]-c)*(b[j]-c)>0:return False
    q=a+(b-a)*((c-a[j])/v);return bool(np.all(q>=lo-1e-5) and np.all(q<=hi+1e-5))


class Rules:
    def __init__(self,tr):
        self.tr=tr;self.data=d=level_asset(tr.level_no);self.pin=asset_pin();self.rows=d['placements'];ar=tr.arrays
        if len(self.rows)!=ar.rows('PH') or not all(ar.get('PH',i,0)==p['sector'] and ar.get('PH',i,1)==p['type'] and np.allclose(ar.get('PH',i,2),p['matrix'],rtol=0,atol=.02) for i,p in enumerate(self.rows)):raise ValueError('retail PH mismatch; do not borrow rules')
        self.centers=np.array([p['matrix'][3][:3] for p in self.rows]);self.bindings={(p['sector'],p.get('object')):p['type'] for p in self.rows if p.get('object')};self.parts={n:k for k,parts in d['physical_parts'].items() for n in parts};self.refresh()

    def refresh(self):
        tr=self.tr;self.fans,self.recovered=fan_layer(tr);self.layers,self.probe=layers_for_tracker(tr);self.plates=[]
        for plate in self.data['plates']:
            k=tr.level.by_name.get(plate['object']);src=self.data['physical_parts'][plate['type']][plate['object']]['bbox'];box=tr.level.mesh_world_bbox(k) if k in tr.level.phys_by_ent else None
            if box is not None and np.allclose(box,src,atol=.15,rtol=0):self.plates.append(plate)

    def rule(self,name,ball):return self.data['rules'].get(name,{'rules':['unknown']*3})['rules'][ball]

    def gate(self,a,b,ball,allow_pose=True):
        for p in self.plates:
            if self.rule(p['type'],ball)=='blocked_when_raised' and cut(a,b,p['bbox']):
                if not allow_pose or self.probe is None or not uses_lowered_plate(self.tr,a,self.probe[2]):return 'blocked_when_raised'
        return None

    def near(self,p,sector,ball):
        # Proximity is a search trigger, never a pass or a contact claim.
        return [(int(i),self.rule(self.rows[i]['type'],ball)) for i in np.flatnonzero(np.linalg.norm(self.centers-p,axis=1)<8) if self.rows[i]['sector'] in (0,sector)]

    def fan_ids(self,p):
        f=self.fans
        if not len(f):return []
        local=np.einsum('nij,nj->ni',f[:,6:15].reshape(-1,3,3),p-f[:,:3]);return np.flatnonzero(np.linalg.norm(np.maximum(abs(local)-f[:,3:6],0),axis=1)<=FAN_R_NOM).tolist()

    def observe(self,prev,row,sector):
        p=np.array(row[1:4]);ball=row[7];contacts=[(n,k,self.rule(k,ball)) for n in row[8] if (k:=self.bindings.get((sector,n),self.parts.get(n)))];hit=self.gate(np.array(prev[1:4]),p,ball) if prev is not None and prev[7]==ball and row[0]-prev[0]<=12 else None
        down=bool(self.probe and uses_lowered_plate(self.tr,p,self.probe[2]));return dict(frame=row[0],ball=ball,sector=sector,contacts=contacts,fan=self.fan_ids(p),gate=hit,aligned_down=down)
