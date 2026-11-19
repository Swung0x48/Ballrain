"""Plot sector-3 geometry and measure the controller's rail and plate contacts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from build_l2_s3_route import MAIN
from probe_l2_s3_route import RECORD
from ballrain import protocol as P
from ballrain.features import level_geometry
from ballrain.path import ReferencePath
from ballrain.state import GameTracker


def geometry(record):
    t = GameTracker(2)
    t.merge_level(P.parse_level_message((record/'level.msg').read_bytes())[0])
    for m in P.parse_level_stream((record/'level.msg.inc').read_bytes())[0]:
        t.merge_level(m)
    return t, level_geometry(t).tris


def main(runtime, base, success, output):
    record = runtime / RECORD
    game, mesh = geometry(record)
    triangles = np.stack((mesh.A, mesh.B, mesh.C), axis=1)
    x, y, z = triangles.mean(axis=1).T
    keep = (mesh.N[:, 1] > .35) & (x > 830) & (x < 1040) & (z > -400) & (z < -310)
    ids = np.flatnonzero(keep)[np.argsort(y[keep])]
    rail_names = {int(k) for k in np.unique(mesh.owner[ids]) if 'A02_Gel' in game.level.name(int(k))}
    zz = np.linspace(-364.4, -363.55, 1701)
    h, face = mesh.ray_down(np.column_stack((np.full_like(zz, 950), np.full_like(zz, 52), zz)))
    is_rail = np.isin(mesh.owner[face.clip(min=0)], list(rail_names)) & (face >= 0)
    print('Rail at x=950: z edges, width, and max height:', zz[is_rail][[0,-1]], np.ptp(zz[is_rail]), h[is_rail].max())
    with np.load(success/'trace.npz') as data:
        a = data['samples']
    events = json.loads((success/'events.json').read_text(encoding='utf8'))
    rail_frames = {f for f, name, b in events['contacts'] if name.startswith('A02_Gel') and b == 2}
    crown = np.isin(a[:,0], list(rail_frames)) & (a[:,1] >= 941) & (a[:,1] <= 958) & (a[:,2]>49.85)
    q = a[crown]
    angle = np.degrees(np.arctan2(q[:,10], -q[:,8]))
    print('Crown endpoints:', q[[0,-1],:4], 'speed', np.min(-q[:,8]), np.max(-q[:,8]),
          'max |z-center|', np.max(abs(q[:,3]+363.98)), 'max |heading| in degrees', np.max(abs(angle)))
    q0, q1 = [np.array(events['poses'][i][-1]) for i in (0,-1)]
    print('Plate rotation (deg)', np.degrees(2*np.arccos(np.clip(abs(q0@q1/np.linalg.norm(q0)/np.linalg.norm(q1)),0,1))))
    for f, code in events['events']:
        if f and code in (P.Ev.PreCheckpoint, P.Ev.PostCheckpoint):
            j = np.searchsorted(a[:,0], f)
            print('CP event', f, code, 'elapsed at next sample (ms)', a[j,7]-a[0,7], 'sample frame', a[j,0])

    main = ReferencePath.load(MAIN)
    alt = ReferencePath.load(base/'east-platform/L02.npz')
    entry = next(e for e in json.loads((base/'branches.json').read_text(encoding='utf8'))['branches'] if e['id']=='east-platform')
    fig, (overview, detail) = plt.subplots(2, 1, figsize=(12, 8.3), layout='constrained',
                                          gridspec_kw={'height_ratios':[2.4,1]})
    for ax in (overview, detail):
        shade = plt.cm.Greys(np.clip((y[ids]-38)/48, .12,.72))
        ax.add_collection(PolyCollection(triangles[ids][:,:,[0,2]], facecolors=shade, edgecolors='none', alpha=.67))
        ax.plot(main.P[main.point_sector==3,0], main.P[main.point_sector==3,2], color='#986120', lw=1.45,
                ls='--', label='TAS wood dodge')
        for n, piece in enumerate(entry['phases']):
            s, e = piece['local_s']
            coords = alt.P[(alt.cum>=s)&(alt.cum<=e)]
            ax.plot(coords[:,0], coords[:,2], color=['#6f7c9f','#a96f90','#b44c7e','#994168'][n],
                    ls=(0,(2,2)) if n<3 else '-', lw=1.65, alpha=.9,
                    label=['v4 paper out','v4 stone to pusher','v4 eastward retrace','v4 south return'][n])
        intended = np.array([(1012,-367),(996,-367),(981,-365),(968,-362.8),(933,-362.8),(932,-358)])
        ax.plot(*intended.T, color='#171717', ls=':', lw=2, label='User sketch (approx.)')
        for material, color, label in [(1,'#007f9e','Script: paper'),(2,'#087940','Script: stone')]:
            p = a[:,[1,3]].copy()
            p[a[:,4]!=material] = np.nan
            ax.plot(*p.T, color=color, lw=2, label=label)
        ax.plot(q[:,1], q[:,3], color='#e97701', lw=3, label='Confirmed rail-top contacts')
        ax.set(aspect='equal', ylabel='z (m), north up')
        ax.grid(alpha=.2, lw=.45)
    for name, point, offset in [('Paper trafo',(916.35,-346.74),(-25,11)),('Stone trafo',(1026.68,-366.74),(-9,-24)),
                                ('Push plate',(931.35,-334.24),(-26,13)),('Fan lifts',(1016.67,-366.74),(-5,13))]:
        overview.scatter(*point, marker='D', c='#252525', s=15, zorder=6)
        overview.annotate(name, point, xytext=offset, textcoords='offset points', fontsize=8)
    overview.set(xlim=(854,1040), ylim=(-395,-310), title='Level 2 sector 3 | v4 outer return vs. verified stone shortcut')
    overview.legend(fontsize=8, loc='upper left', framealpha=.95, ncol=2)
    detail.set(xlim=(928,993), ylim=(-372,-356), xlabel='x (m), east right',
               title='Stone traverses the Wippe, mounts the north guardrail, then crosses to the west ledge')
    detail.annotate('West ledge', (933,-364), (932,-359), fontsize=8,
                    arrowprops={'arrowstyle':'-', 'lw':.7})
    detail.annotate('North rail crown', (949,-364), (944,-360), fontsize=8,
                    arrowprops={'arrowstyle':'-', 'lw':.7})
    detail.annotate('Wippe', (972,-366.7), (972,-361), fontsize=8,
                    arrowprops={'arrowstyle':'-', 'lw':.7})
    fig.savefig(output, dpi=160)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('runtime','base','success','output'):
        p.add_argument(arg, type=Path)
    main(**vars(p.parse_args()))
