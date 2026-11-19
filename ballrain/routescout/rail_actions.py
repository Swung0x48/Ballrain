"""Rail actions are proposals, never feasibility flags. CK axes: y up."""
from dataclasses import dataclass
import numpy as np
from ballrain.reach import BALLS, mount_speed
from ballrain.state import G_RAILS


@dataclass
class RailAction:
    phase: str
    owner: int
    velocity: np.ndarray


class RailCrowns:
    def __init__(self, tracker, geometry):
        self.t = geometry.tris
        level = tracker.level
        self.owners = {int(i) for i in np.unique(self.t.owner) if level.in_group(i, G_RAILS)
                       or 'geländer' in (level.name(i) or '').lower()}

    def propose(self, p, v, material, goal, friction):
        if material not in BALLS:
            return None
        r = BALLS[material].radius
        t = self.t
        ids = t.candidates(p, 9.)
        ids = ids[np.isin(t.owner[ids], list(self.owners)) & (abs(t.N[ids,1]) > .85)]
        best = None
        for i in ids:
            vertices = np.array([t.A[i], t.B[i], t.C[i]])
            edges = np.roll(vertices, -1, axis=0) - vertices
            lengths = np.linalg.norm(edges[:,[0,2]], axis=1); j = lengths.argmax()
            if lengths[j] < 2: continue
            axis = edges[j,[0,2]] / lengths[j]
            along = np.clip((p-vertices[j])[[0,2]] @ axis, 0, lengths[j])
            crown = vertices[j] + edges[j] * (along / lengths[j])
            lateral = (crown-p)[[0,2]]; gap = np.linalg.norm(lateral)
            height = crown[1] - (p[1]-r)
            if gap > 6 or not -2 < height < 2: continue
            axis *= 1 if (goal-p)[[0,2]] @ axis >= 0 else -1
            owner = int(t.owner[i]); direction = lateral/max(.001,gap)
            if owner in friction and gap < .6 and abs(height)<.3:
                phase = 'ride'; desired = 10*axis + 3*lateral
            elif height < 0 and v[1] < 0:
                phase = 'air intercept'; desired = 6*axis + 2*lateral
            else:
                phase = 'floor mount'; desired = 4*axis + direction*min(13.,max(3.,float(mount_speed(height,r))))
            # Paper is only proposed to the engine; the sphere fit is not a paper feasibility test.
            score = gap + abs(height) - (2 if phase=='ride' else 0)
            if best is None or score < best[0]: best = score,RailAction(phase,owner,desired)
        return best[1] if best else None


def return_graph(tr, sector, modes, d, seed):
    from ballrain.features import level_geometry
    idx, via, feet = modes[sector]
    objs = [o for o in d['map']['placements'] if o['sector'] == sector]
    high = np.array(via[3])
    plate = np.array(via[4])
    w = [o for o in objs if o['name'] == 'P_Modul_30']
    push = [o for o in objs if o['name'] == 'P_Modul_01']
    if not w or not push:
        return
    wp = np.array(min(w, key=lambda o: np.linalg.norm(o['pos'] - high))['pos'])
    pu = np.array(min(push, key=lambda o: np.linalg.norm(o['pos'] - plate))['pos'])
    axis = (pu - wp)[[0, 2]]
    length = np.linalg.norm(axis)
    axis /= max(0.01, length)
    side = np.array([-axis[1], axis[0]])
    t = level_geometry(tr).tris
    owns = {int(i) for i in np.unique(t.owner) if 'nder' in str(tr.level.name(i)) or tr.level.in_group(i, 'Phys_FloorRails')}
    take = np.isin(t.owner, list(owns)) & (abs(t.N[:, 1]) >= 0.8)
    vs = np.stack([t.A[take], t.B[take], t.C[take]], 1)
    ctr = vs.mean(1)
    rel = ctr[:, [0, 2]] - wp[[0, 2]]
    along = rel @ axis
    lat = rel @ side
    keep = (along > 3) & (along < length + 4) & (abs(lat) < 8) & (abs(ctr[:, 1] - wp[1]) < 3)
    bands = {}
    for n in np.flatnonzero(keep):
        proj = (vs[n][:, [0, 2]] - wp[[0, 2]]) @ axis
        perp = (vs[n][:, [0, 2]] - wp[[0, 2]]) @ side
        if np.ptp(proj) < 4 * np.ptp(perp) or np.ptp(proj) < 1:
            continue
        bands.setdefault(round(float(lat[n]) * 2) / 2, []).append((proj.min(), proj.max(), ctr[n, 1], lat[n]))
    lanes = []
    for band, spans in bands.items():
        spans = np.array(spans)
        lo = max(1.0, spans[:, 0].min())
        hi = min(length + 3.0, spans[:, 1].max())
        if hi - lo < 8:
            continue
        lanes.append((hi - lo, abs(band), lo, hi, np.max(spans[:, 2]) + 2, np.mean(spans[:, 3])))
    if not lanes:
        return
    lanes.sort(key=lambda k: -k[0] + k[1])
    lane = lanes[seed % 2 % len(lanes)]
    _, _, lo, hi, y, offset = lane
    # Refine the coarse band from the ball centre's cross section rather than using
    # its triangle centroids as the steering target. The receiver still tests contact.
    stations = np.arange(lo,hi+.01,.25)
    transverse = np.arange(-.8,.81,.025)+offset
    points = wp[[0,2]] + stations[:,None,None]*axis + transverse[None,:,None]*side
    query = np.zeros((len(stations),len(transverse),3)); query[:,:,[0,2]]=points;query[:,:,1]=y+1
    yy=t.ray_down(query.reshape(-1,3),owner_mask=np.isin(t.owner,list(owns)))[0].reshape(len(stations),len(transverse))
    ymax=np.where(np.isfinite(yy),yy,-1e5).max(1)
    offsets=np.array([transverse[row>=top-.004].mean() if top>-9999 else offset for row,top in zip(yy,ymax)])
    profile=(stations,offsets,np.where(ymax>-9999,ymax+2,y))
    return (wp,pu,plate,axis,side,lo,hi,y,offset),profile
