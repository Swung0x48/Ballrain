"""Welded walkable mesh edges, used only as a guide (not a ball-centre line)."""
from heapq import heappop, heappush
import numpy as np


class SurfaceGraph:
    def __init__(self, tris):
        ok = abs(tris.N[:,1]) >= .5
        vertices = np.stack([tris.A[ok],tris.B[ok],tris.C[ok]],axis=1)
        q, idx = np.unique(np.rint(vertices.reshape(-1,3)*1000).astype(np.int64),axis=0,return_inverse=True)
        self.points = q / 1000
        faces = idx.reshape(-1,3)
        pairs = np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]); pairs.sort(axis=1)
        self.edges = [[] for _ in q]
        self.uncertain = set()
        for a,b in np.unique(pairs,axis=0):
            if a!=b:
                w=float(np.linalg.norm(self.points[a]-self.points[b]))
                self.edges[a].append((b,w)); self.edges[b].append((a,w))

        # Attach one representative per spatial patch to nearby patches. Limit horizontal
        # gaps and ascent separately: a one-way drop is not a climb on the reverse graph.
        cells={}
        for i, key in enumerate(np.floor(self.points / [6,4,6]).astype(int)):
            k=tuple(key); rep=cells.setdefault(k,i)
            if i!=rep: self._proposal(i,rep)
        for (x,y,z),i in cells.items():
            for dx in (-1,0,1):
                for dz in (-1,0,1):
                    for dy in range(-3,4):
                        j=cells.get((x+dx,y+dy,z+dz))
                        if j is not None and i>j: self._proposal(i,j)

    def _proposal(self,i,j):
        delta=self.points[j]-self.points[i]
        if np.hypot(delta[0],delta[2])>9: return
        for src,dst,h in ((i,j,delta[1]),(j,i,-delta[1])):
            if -12<=h<=2 and not any(n==src for n,_ in self.edges[dst]):
                # Incoming edges: Dijkstra runs backwards from the goal.
                self.edges[dst].append((src,float(np.linalg.norm(delta))+6))
                self.uncertain.add((src,dst))

    def gates(self, corridor):
        index={tuple(p):i for i,p in enumerate(self.points)}
        ids=[index[tuple(p)] for p in corridor]
        return [i for i,pair in enumerate(zip(ids,ids[1:])) if pair in self.uncertain]

    def corridor(self, start, end, snap_limit=12.):
        if start is None or end is None or not len(self.points): return []
        def nearest(p):
            d=np.linalg.norm(self.points-p,axis=1); i=d.argmin(); return int(i),d[i]
        a,da=nearest(start); b,db=nearest(end)
        if max(da,db)>snap_limit: return []
        dist={b:0.}; after={}; queue=[(0.,b)]
        while queue:
            cost,i=heappop(queue)
            if cost!=dist[i]: continue
            if i==a:
                ids=[a]
                while ids[-1]!=b: ids.append(after[ids[-1]])
                return self.points[ids].tolist()
            for j,w in self.edges[i]:
                c=cost+w
                if c<dist.get(j,float('inf')):
                    dist[j]=c; after[j]=i; heappush(queue,(c,j))
        return []


def guide_distance(corridor, end):
    """Surface suffix length at the nearest guide vertex, smoothed on its incident edges."""
    if len(corridor)<2: return lambda p: np.linalg.norm(p-end)
    pts=np.asarray(corridor); delta=np.diff(pts,axis=0); length=np.linalg.norm(delta,axis=1)
    tail=np.cumsum(length[::-1])[::-1]
    def distance(p):
        t=np.clip(((p-pts[:-1])*delta).sum(1)/length**2,0,1)
        lateral=np.linalg.norm(p-(pts[:-1]+delta*t[:,None]),axis=1); i=lateral.argmin()
        return lateral[i]+tail[i]-length[i]*t[i]
    return distance
