"""Sweep real restored floor-to-guardrail approaches. Separate owner contact, crown contact and ride."""
import json, numpy as np, sys, tempfile, random, math, concurrent.futures as cf
from pathlib import Path
from types import SimpleNamespace
from routescout_atlas import load, state, DEFAULT_DELTA as DT, world_to_camera_keys
from ballrain import tas
from ballrain.features import level_geometry
from ballrain.protocol import Ev
import argparse

def select_cases(t, tr, archives, level, material):
    members = [i for i in np.unique(t.owner) if 'geländer' in (tr.level.name(i) or '').lower()]
    mask = np.isin(t.owner, members)
    idx = np.flatnonzero(mask & (abs(t.N[:, 1]) > 0.85))
    cases = []
    if not len(idx):
        return cases
    verts = np.stack([t.A[idx], t.B[idx], t.C[idx]], 1)
    edge = np.roll(verts, -1, axis=1) - verts
    j = np.linalg.norm(edge[:, :, [0, 2]], axis=2).argmax(1)
    a = verts[np.arange(len(j)), j]
    u = edge[np.arange(len(j)), j]
    den = (u[:, [0, 2]] ** 2).sum(1)
    for src in archives:
        archive = json.loads(src.read_text())
        if archive['level'] != level:
            raise ValueError('archive level differs')
        for c in archive['cells']:
            if c['cell'][1] != material:
                continue
            p = np.array(c['pos'])
            v = np.array(c['vel'])
            f = np.clip(((p - a)[:, [0, 2]] * u[:, [0, 2]]).sum(1) / np.maximum(den, 1e-06), 0, 1)
            q = a + u * f[:, None]
            h = q[:, 1] - p[1] + 2
            dif = np.linalg.norm((q - p)[:, [0, 2]], axis=1)
            ok = (dif >= 2) & (dif < 9) & (h > 0.1) & (h < 2) & (den > 4) & (f > 0.07) & (f < 0.93)
            if not ok.any() or abs(v[1]) > 2:
                continue
            i = np.argmin(np.where(ok, dif, 10000000000.0))
            floor = t.ray_down(p[None, :], owner_mask=~mask)[0][0]
            height = q[i, 1] - floor
            if np.isfinite(floor) and abs(floor + 2 - p[1]) <= 0.5 and (0.1 < height < 2):
                cases.append(dict(cell=c['cell'], pos=c['pos'], keys=c['keys'], q=q[i].tolist(), axis=u[i].tolist(), owner=int(t.owner[idx[i]]), h=height))
    return cases

def run(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    O = args.output.parent
    b = args.material
    with tempfile.TemporaryDirectory(dir=O) as tmp:
        s = load(args, args.level, args.binary, args.copy, Path(tmp))
        try:
            pool = select_cases(level_geometry(s.tracker).tris, s.tracker, args.archives, args.level, b)
        finally:
            s.close(kill=True)
    if not pool:
        raise ValueError('no supported archived approach to a low guardrail')
    cells = [c for c in pool if c['cell'][1] == b]
    random.Random(29).shuffle(cells)
    rows = []
    with tempfile.TemporaryDirectory(dir=O) as tmp:
        for i in range(args.trials):
            c = cells[i % len(cells)]
            s = load(args, args.level, args.binary, args.copy, Path(tmp))
            try:
                frames = np.concatenate([np.repeat(tas.make_frames([DT], [k]), n) for k, n in c['keys']])
                s.act(frames)
                p, v, b0 = state(s)
                assert np.linalg.norm(p - c['pos']) < 0.0001 and b == b0
                t = level_geometry(s.tracker).tris
                mask = t.owner == c['owner']
                axis = np.array(c['axis'])[[0, 2]]
                axis /= np.linalg.norm(axis)
                side = np.array([-axis[1], axis[0]])
                q = np.array(c['q'])
                side *= np.sign((q - p)[[0, 2]] @ side)
                axis *= 1 if v[[0, 2]] @ axis >= 0 else -1
                angle = [15, 45, 75][i // 4 % 3]
                speed = [5, 8, 11, 14][i % 4]
                contact = None
                runlen = 0
                peak = 0
                travel = 0.0
                ridden = False
                near = False
                entry = None
                last = None
                for step in range(55):
                    p, v, _ = state(s)
                    err = (q - p)[[0, 2]] @ side
                    tn = axis * math.cos(math.radians(angle)) * speed
                    vn = side * min(speed * math.sin(math.radians(angle)), max(-5, err * 2))
                    u = tn + vn - v[[0, 2]]
                    key = (8 if u[0] > 0.4 else 4 if u[0] < -0.4 else 0) | (1 if u[1] > 0.4 else 2 if u[1] < -0.4 else 0)
                    key = world_to_camera_keys(key, s.obs.cams['cams'][0]['matrix'])
                    o = s.act(tas.make_frames([DT], [key]), repeat=4, stop=[Ev.BallOff])
                    if o is None or o.has_event(Ev.BallOff) or (not o.ball['found']):
                        break
                    p1, v1, _ = state(s)
                    owner = int(c['owner'])
                    touch = owner in o.fric['partner']
                    rh = t.ray_down((p1 + [0, 1, 0])[None, :], owner_mask=mask)[0][0]
                    floor = t.ray_down(p[None, :], owner_mask=~mask)[0][0]
                    near |= abs(err) < 1.0
                    if touch and contact is None:
                        contact = step
                        entry = dict(vn=float(v[[0, 2]] @ side), vt=float(v[[0, 2]] @ axis), speed=float(np.linalg.norm(v[[0, 2]])), vy=float(v[1]), angle=float(np.degrees(np.arctan2(abs(v[[0, 2]] @ side), abs(v[[0, 2]] @ axis)))), h=c['h'], near_floor=bool(np.isfinite(floor) and abs(floor + 2 - p[1]) < 0.5))
                    crown = bool(touch and np.isfinite(rh) and (abs(p1[1] - rh - 2) < 0.15))
                    if crown:
                        runlen += 1
                        peak = max(peak, runlen)
                        travel += 0 if last is None else abs((p1 - last)[[0, 2]] @ axis)
                        last = p1
                        ridden |= runlen >= 3 and travel >= 1
                    else:
                        runlen = 0
                        travel = 0
                        last = None
                    if ridden:
                        break
                rows.append(dict(material=b, cell=c['cell'], owner=s.tracker.level.name(c['owner']), height=c['h'], set_speed=speed, set_angle=angle, entry=entry, peak_crown_samples=peak, ride=bool(ridden)))
            finally:
                s.close(kill=True)
    args.output.write_text(json.dumps(rows))
    print(b, len(rows), sum((r['entry'] is not None for r in rows)), sum((r['ride'] for r in rows)), flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ('output', 'root', 'binary'):
        ap.add_argument('--' + k, required=True, type=Path)
    ap.add_argument('--archives', nargs='+', required=True, type=Path)
    ap.add_argument('--level', type=int, required=True)
    ap.add_argument('--copy', required=True)
    ap.add_argument('--material', type=int, choices=[0, 2], required=True)
    ap.add_argument('--trials', type=int, default=36)
    run(ap.parse_args())
