"""Build a v2 bundle candidate from a checkpoint-verified controller trace."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from probe_l2_s3_route import MAIN, RECORD
from ballrain.branch_assets import read_bundle, validate
from ballrain.path import ReferencePath, file_sha256
from ballrain.protocol import Ev


MEMBER = 'east-platform'


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf8', newline='\n')


def build(runtime, base, trace, dest, evidence):
    with np.load(trace / 'trace.npz') as z:
        a = z['samples']
        frames = z['frames']
        prefix = int(z['prefix_frame'])
    log = json.loads((trace / 'events.json').read_text(encoding='utf8'))
    assert a[-1, 6] == 4 and (int(a[-1, 0]), Ev.PostCheckpoint) in map(tuple, log['events'])
    assert any('P_Modul_25_Bridge' == name and bt == 2 for _, name, bt in log['contacts'])
    p = a[:, 1:4]
    paper = np.flatnonzero(a[:, 4] == 1)[0]
    stone = np.flatnonzero(a[:, 4] == 2)[0]
    rail_frames = {f for f, name, bt in log['contacts'] if name.startswith('A02_Gel') and bt == 2}
    crown = np.flatnonzero(np.isin(a[:, 0], list(rail_frames)) & (np.arange(len(a)) >= stone)
                          & (p[:, 0] > 940) & (p[:, 0] < 960) & (p[:, 1] > 49.7)
                          & (abs(p[:, 2]+363.98) < .2))
    assert len(crown) > 1 and np.ptp(p[crown, 0]) > 12, 'refuse a non-rail return'
    rail = int(crown[0])
    turn = np.flatnonzero(a[:, 5] == 11)[0]
    end = next(i for i in range(turn, len(a)) if p[i, 2] >= -317)
    cuts = [stone, rail, turn]
    reference = ReferencePath.load(MAIN)
    i, j = [np.linalg.norm(reference.P - p[n], axis=1).argmin() for n in (paper, end)]
    j = int(np.searchsorted(reference.cum, reference.cum[j] + 1.5))
    assert i < j and reference.point_sector[i] == reference.point_sector[j] == 3
    start_points = {
        'paper': ((918, 48.5, -346.5), 2),
        'fan-approach': ((1013.0, 48.1, -366.7), 4),
        'fan-lip': ((1015.0, 48.1, -366.7), 4),
        'fan-upper': ((1007.5, 63.0, -366.7), 5),
        'stone': ((1026.2, 68.7, -366.7), 7),
        'middle-floor': ((984, 48.7, -366.7), 9),
        'rail-top': ((957.7, 49.9, -363.98), 9),
        'before-plate': ((931.4, 48, -342), 12),
    }
    selected = {}
    for key, (point, stage) in start_points.items():
        cand = np.flatnonzero((a[:, 5] == stage) & (np.arange(len(a)) >= paper))
        selected[key] = (int(cand[np.linalg.norm(p[cand] - point, axis=1).argmin()]), stage)
    forced = {*cuts, end, *(n for n, _ in selected.values())}
    ids = [paper]
    for n in range(paper + 1, end + 1):
        if n in forced or np.linalg.norm(p[n] - p[ids[-1]]) > .4:
            ids.append(n)
    measured = p[ids]
    time = (a[ids, 7] - a[paper, 7] + reference.t_ms[i]
            + max(1, np.linalg.norm(measured[0] - reference.P[i]) * 100))
    path = ReferencePath(
        np.vstack([reference.P[:i+1], measured, reference.P[j:]]),
        np.r_[reference.point_sector[:i+1], np.full(len(ids), 3), reference.point_sector[j:]],
        meta=dict(reference.meta, generator='scripts/build_l2_s3_route.py'),
        t_ms=np.r_[reference.t_ms[:i+1], time, reference.t_ms[j:] - reference.t_ms[j]
                   + time[-1] + max(1, np.linalg.norm(reference.P[j]-measured[-1])*100)],
    )
    if dest.exists():
        raise FileExistsError(dest)
    shutil.copytree(base, dest)
    child = dest / MEMBER
    asset = child / 'L02.npz'
    path.save(asset)
    path = ReferencePath.load(asset)
    index = {n: i + 1 + k for k, n in enumerate(ids)}
    arcs = path.cum[[i, *(index[n] for n in cuts), i+1+len(ids)]].tolist()
    ms = reference.cum[[i, j]].tolist()
    pin = file_sha256(asset)
    write_json(child / 'L02.json', dict(level=2, ball_type=1, sector=3,
               main_sha256=file_sha256(MAIN), branch_sha256=pin, branch_s=[arcs[0], arcs[-1]],
               main_s=ms, stations=[[float(np.interp(arcs[1], [arcs[0], arcs[-1]], ms)), arcs[1]]],
               rules={}))
    np.savez_compressed(child / 'drive.npz', frames=frames[:int(a[end, 0])], prefix_frame=prefix,
                        samples=np.column_stack([a[ids, 0], path.cum[i+1:i+1+len(ids)]]),
                        record_sha256=json.loads((runtime / RECORD / 'meta.json').read_text(encoding='utf8'))['record_sha256'],
                        branch_sha256=pin)
    doc = json.loads((dest / 'branches.json').read_text(encoding='utf8'))
    gates = [dict(ball=1, sector=3, box=[[909, 45, -353], [925, 53, -340]])]
    for k, (n, tol) in enumerate(zip(cuts, ([2.5, 3., 2.5], [3., 1.3, .65], [2.5, 2., 2.5])), 1):
        gate = dict(ball=2, sector=3, box=[(p[n]-tol).tolist(), (p[n]+tol).tolist()])
        if k > 1:
            gate['s'] = [arcs[k]-1.25, arcs[k]+.001]
        gates.append(gate)
    phases = [dict(local_s=arcs[k:k+2], when=g) for k, g in enumerate(gates)]
    merge = reference.P[j]
    next(item for item in doc['branches'] if item['id'] == MEMBER).update(
        phases=phases, merge=dict(ball=2, sector=3,
          box=[(merge-[7, 4, 5]).tolist(), (merge+[7, 4, 5]).tolist()], s=[arcs[-1]-.5, arcs[-1]+1]))
    write_json(dest/'branches.json', doc)
    # Remove obsolete station numbers. The rest verifier supplies the replacement menu.
    rest = dest / 'rest-starts.txt'
    kept = [item for item in rest.read_text(encoding='utf8').strip().split(',') if f'@branch:{MEMBER}:' not in item]
    rest.write_text(','.join(kept)+'\n', encoding='ascii', newline='\n')
    menu = {}
    for name, (n, stage) in selected.items():
        station = float(path.cum[index[n]])
        menu[name] = dict(spec=f'{prefix}-{prefix}@branch:{MEMBER}:{station}', stage=stage,
                          sample=p[n].tolist(), offset=int(a[n, 0]))
    write_json(evidence, menu)
    print('bundle', validate(read_bundle(2, dest), 2, file_sha256(MAIN)), 'phases', arcs)
    for name, data in menu.items():
        print(name, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for param in ('runtime', 'base', 'trace', 'dest', 'evidence'):
        parser.add_argument(param, type=Path)
    build(**vars(parser.parse_args()))
