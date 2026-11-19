"""Read only static level exports. Never accept trajectories or reference paths."""
from collections import Counter
import numpy as np
from ballrain.features import level_geometry
from ballrain.fastreach import fan_boxes, fan_obbs
from ballrain.path import level_anchors
from ballrain.state import G_RAILS


MATERIAL = {'P_Trafo_Wood': 0, 'P_Trafo_Paper': 1, 'P_Trafo_Stone': 2}


def classify(name):
    if name in MATERIAL:
        return 'trafo'
    if name in {'P_Box', 'P_Modul_25', 'P_Modul_18'}:
        return {'P_Box': 'box', 'P_Modul_25': 'push plate', 'P_Modul_18': 'fan'}[name]
    for prefix, kind in [('PC_', 'checkpoint'), ('PR_', 'reset'), ('PS_', 'start'),
                         ('PE_', 'finish'), ('P_Ball_', 'prop ball'), ('P_Extra_', 'pickup')]:
        if name.startswith(prefix):
            return kind
    return 'unknown module' if name.startswith('P_Modul_') else 'unknown'


def inventory(tr):
    lv, ar = tr.level, tr.arrays
    g = level_geometry(tr)
    a = level_anchors(tr)
    if len(a['checkpoints']) != ar.rows('Checkpoints'):
        raise ValueError('missing ordered checkpoint position')
    ph = []
    for i in range(ar.rows('PH')):
        sector, name, mat = [ar.get('PH', i, j) for j in range(3)]
        ph.append(dict(sector=int(sector), name=name, kind=classify(name), pos=mat[3,:3].tolist()))
    t = g.tris
    ny = abs(t.N[:,1]); rail = np.isin(t.owner, list(lv.groups[G_RAILS].members) if G_RAILS in lv.groups else [])
    return g, dict(static_key=g.key, triangles=len(t), rail_faces=int(rail.sum()),
        flat=int((ny >= .98).sum()), slope=int(((ny >= .5) & (ny < .98)).sum()), steep=int((ny < .5).sum()),
        boundary_edges=len(g.edges.S0), depth_boxes=len(lv.depth), placements=ph,
        objects=dict(Counter(p['kind'] for p in ph)), fan_obbs=fan_obbs(fan_boxes(lv)).tolist(),
        groups={k:len(v.members) for k,v in lv.groups.items() if k.startswith(('Phys_', 'PC_', 'PR_', 'PE_', 'PS_'))},
        resets=[ar.get('ResetPoints', i, 0)[3,:3].tolist() for i in range(ar.rows('ResetPoints'))],
        anchors=[None if p is None else np.asarray(p)[:3].tolist() for p in [a['start'], *a['checkpoints'], a['airship']]])
