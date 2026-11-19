"""Branch the descent search towards (not necessarily onto) intermediate shelf placements. The engine alone decides whether the drop and remaining route actually pass."""
from pathlib import Path
import math, sys, time, json, tempfile, random, concurrent.futures as cf
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from routescout_atlas import load, state
from ballrain.client import DEFAULT_DELTA as DT
from ballrain import tas
from ballrain.reach import BALLS, key_push, _lin_step
from ballrain.features import level_geometry
from ballrain.env import world_to_camera_keys
from ballrain.protocol import Ev
from types import SimpleNamespace
import argparse

def run(args):
    dst = args.out
    dst.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    lev = args.level
    seed = args.seed
    last = args.target_sector
    soft = args.soft
    ret = args.second
    copy = args.copy
    rng = random.Random(seed)
    d = json.loads(args.atlas.read_text(encoding='utf8'))
    guides = []
    for seg in d['segments']:
        sec = seg['sector']
        g = np.array(seg['guide'] or d['map']['anchors'][sec - 1:sec + 1])
        delta = np.diff(g, axis=0)
        length = np.linalg.norm(delta, axis=1)
        guides.append((g, delta, length, np.r_[0, length.cumsum()]))
    gate = next((np.array(o['pos']) for o in d['map']['placements'] if o['sector'] == last and o['kind'] == 'push plate'), None)
    if gate is None:
        raise ValueError('no plate in requested sector')
    endgoal = np.array(d['map']['anchors'][last])
    unit = (endgoal - gate)[[0, 2]]
    unit /= np.linalg.norm(unit)

    def beyond(p):
        return (np.array(p) - gate)[[0, 2]] @ unit

    def centered(tr):
        import heapq
        from ballrain.features import dynamic_triangles
        from ballrain.geometry import TriangleSet
        mesh = level_geometry(tr).tris
        dyn = dynamic_triangles(tr, tr.obs.ball['pos'], 70.0)
        if len(dyn):
            mesh = TriangleSet(*(np.concatenate([getattr(mesh, k), getattr(dyn, k)]) for k in ('A', 'B', 'C', 'owner')))
        for i in [last - 1]:
            src, goal = (guide_start, endgoal)
            lo = np.floor(np.minimum(src, goal)[[0, 2]] - 30)
            hi = np.ceil(np.maximum(src, goal)[[0, 2]] + 30)
            xx, zz = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing='ij')
            sz = xx.shape
            qq = np.c_[xx.ravel(), np.full(xx.size, max(src[1], goal[1]) + 60), zz.ravel()]
            hs = []
            for chunk in np.array_split(qq, max(1, len(qq) // 256)):
                hs.extend(mesh.ray_down(chunk, owner_mask=abs(mesh.N[:, 1]) > 0.5)[0])
            y = np.array(hs).reshape(sz)
            valid = np.isfinite(y)
            cost = np.zeros(sz)
            candidates = np.flatnonzero(valid.ravel())
            for dx, dz in [(2, 0), (-2, 0), (0, 2), (0, -2), (1, 1), (-1, -1), (-1, 1), (1, -1)]:
                ny = np.roll(y, (dx, dz), (0, 1))
                cost += np.where(np.isfinite(ny) & (abs(y - ny) < 1.8), 0.0, 1.0)
            for ob in d['map']['placements']:
                if ob['sector'] != i + 1 or ob['kind'] not in ('trafo', 'box', 'prop ball'):
                    continue
                x0, y0, z0 = ob['pos']
                cost += np.maximum(7 - np.sqrt((xx - x0) ** 2 + (zz - z0) ** 2 + (y - y0) ** 2), 0) * 1.4

            def nearest(p):
                return tuple(map(int, np.unravel_index(candidates[np.argmin((xx.ravel()[candidates] - p[0]) ** 2 + (zz.ravel()[candidates] - p[2]) ** 2 + (y.ravel()[candidates] + 2 - p[1]) ** 2)], sz)))
            a = nearest(src)
            b = nearest(goal)
            todo = [(0.0, 0.0, a)]
            D = {a: 0.0}
            par = {}
            found = False
            while todo:
                _, dist, k = heapq.heappop(todo)
                if k == b:
                    found = True
                    break
                if D[k] != dist:
                    continue
                for dx, dz in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                    for jump in range(1, 51):
                        m = (k[0] + dx * jump, k[1] + dz * jump)
                        if not (0 <= m[0] < sz[0] and 0 <= m[1] < sz[1]):
                            break
                        if not valid[m]:
                            continue
                        dh = y[m] - y[k]
                        length = math.hypot(dx, dz) * jump
                        if -75 < dh < 0.85 * length + 0.2 and (jump < 5 or length <= 4 + 24 * math.sqrt(max(0.0, -dh) * 0.1)):
                            nd = dist + length * (1 + 0.55 * cost[m]) + max(0, dh) * 3 + max(0, -dh - 1) * 2 + (2.5 if dh < -3 else 9) * (jump - 1)
                            if nd < D.get(m, 1e+30):
                                D[m] = nd
                                par[m] = k
                                heapq.heappush(todo, (nd + math.dist(m, b), nd, m))
                        break
            if not found:
                b = min(D, key=lambda k: math.dist(k, b) + abs(y[k] - goal[1] + 2) * 0.2)
            if b != a:
                ids = [b]
                while ids[-1] != a:
                    ids.append(par[ids[-1]])
                ids = ids[::-1]
                gg = np.array([[xx[k], y[k] + 2, zz[k]] for k in ids])
                hh = np.diff(gg, axis=0)
                ll = np.linalg.norm(hh, axis=1)
                guides[i] = (gg, hh, ll, np.r_[0, ll.cumsum()])

    def guide(p, sec, look):
        g, h, l, e = guides[sec - 1]
        f = np.clip(((p - g[:-1]) * h).sum(1) / np.maximum(1e-09, l * l), 0, 1)
        lat = np.linalg.norm(g[:-1] + f[:, None] * h - p, axis=1)
        i = lat.argmin()
        s = e[i] + f[i] * l[i]
        t = min(e[-1] - 1e-05, s + look)
        j = max(0, np.searchsorted(e, t) - 1)
        return (g[j] + (t - e[j]) * h[j] / l[j], e[-1] - s + lat[i])
    A = {}
    growth = []
    paths = {}
    bad = restores = actions = 0
    best = {}

    def act(s, k, n):
        return s.act(tas.make_frames([DT], [k]), repeat=n, stop=[Ev.BallOff, Ev.PostCheckpoint, Ev.LevelFinish])
    with tempfile.TemporaryDirectory(dir=dst) as tmp:
        for src in args.previous:
            archive = json.loads(src.read_text())
            if archive['level'] != lev:
                raise ValueError('seed archive level mismatch')
            for item in archive['cells']:
                k = tuple(item['cell'])
                p0 = np.array(item['pos'])
                _, dist = guide(p0, k[0], 0)
                seq0 = tuple(map(tuple, item['keys']))
                t = item['elapsed']
                if k not in A or t < A[k][5]:
                    A[k] = [dist, seq0, p0, np.array(item['vel']), 0, t]
        roots = [a for k, a in A.items() if k[:2] == (last, 2) and 7 < beyond(a[2]) < 45 and (abs(a[2][1] - gate[1] - 2) < 2) and (abs(a[3][1]) < 1.5)]
        if not roots:
            raise ValueError('no supported post-plate start in archive')
        guide_start = max(roots, key=lambda a: beyond(a[2]))[2]
        prepared = False
        for episode in range(args.episodes):
            s = load(args, lev, soft, copy, Path(tmp) / 'play')
            seq = ()
            stale = 0
            try:
                if A:
                    ks = [k for k, a in A.items() if k[0] == last and beyond(a[2]) > 0 and (a[2][1] > gate[1] - 1 or beyond(a[2]) > 18)]
                    assert ks
                    choice = [k for k in ks if A[k][3][1] > -3.5] or ks
                    if prepared:
                        gg, hh, ll, ss = guides[last - 1]
                        jump = np.flatnonzero((ll > 7) & (hh[:, 1] < -2))[0]
                        lip = gg[jump]
                        delta = rng.uniform(4, 38)
                        c0 = [k for k in choice if abs((lip - A[k][2])[[0, 2]] @ unit - delta) < 6 and abs(A[k][2][1] - lip[1]) < 3 and (abs(A[k][2][0] - lip[0]) < 4)]
                        advanced = [k for k in ks if beyond(A[k][2]) > beyond(lip) + 9 and A[k][3][1] > -4 and (A[k][2][1] > endgoal[1] - 2)]
                        choice = advanced if advanced and episode % 6 else c0 or choice
                    keys = rng.sample(choice, min(80, len(choice)))
                    k = min(keys, key=lambda k: A[k][0] + rng.expovariate(1 / 12) + 3 * A[k][4] - A[k][3][[0, 2]] @ unit * 1.5)
                    a = A[k]
                    a[4] += 1
                    seq = a[1]
                    frames = np.concatenate([np.repeat(tas.make_frames([DT], [key]), n) for key, n in seq])
                    s.act(frames, stop=[Ev.BallOff, Ev.LevelFinish])
                    p, v, b = state(s)
                    restores += 1
                    if not np.allclose(p, a[2], rtol=0, atol=0.0001) or not np.allclose(v, a[3], rtol=0, atol=0.0001) or s.tracker.sector != k[0] or (b != k[1]):
                        bad += 1
                        continue
                if not prepared:
                    centered(s.tracker)
                    prepared = True
                    print('guide', seed, len(guides[last - 1][0]), guides[last - 1][0][-1], flush=True)
                    for k, a in A.items():
                        a[0] = guide(a[2], k[0], 0)[1]
                landing = rng.choice((17, 20, 25, 30))
                pad = rng.choice((0, 0, 1))
                cap = rng.choice((5, 7, 9, 11))
                look = rng.choice((4, 7, 10, 14))
                gain = rng.choice((1.5, 2.5, 4))
                block = rng.choice((6, 10, 14, 18))
                choices = [o for o in d['map']['placements'] if o['sector'] == last and o['kind'] in ('trafo', 'pickup') and (5 < gate[1] - o['pos'][1] < 24) and (beyond(o['pos']) > 15)]
                if not choices:
                    raise ValueError('no lower placement to propose as a landing target')
                target = np.array(min(choices, key=lambda o: beyond(o['pos']))['pos'])
                eligible_lips = [a0 for k0, a0 in A.items() if k0[:2] == (last, 2) and a0[2][1] > target[1] + 8 and (6 < beyond(target) - beyond(a0[2]) < 36) and (abs(a0[3][1]) < 1.8) and (a0[3][[0, 2]] @ unit > 6)]
                if not eligible_lips:
                    raise ValueError('no supported launch candidate near landing target')
                lip = max(eligible_lips, key=lambda a0: beyond(a0[2]) + 0.15 * a0[3][[0, 2]] @ unit)[2].copy()
                side = unit[::-1] * [-1, 1]
                wstart = rng.uniform(4, 18)
                bias = rng.uniform(-1.3, 1.3)
                side_gain = rng.uniform(1.5, 5.5)
                full = rng.choice((0, 1))
                lock = None
                baseblock = block
                for step in range(480):
                    p, v, b = state(s)
                    sec = s.tracker.sector
                    q, dist = guide(p, sec, look)
                    gg, hh, ll, ss = guides[sec - 1]
                    j = np.linalg.norm(gg - p, axis=1).argmin()
                    jx = np.flatnonzero((ss[:-1] >= ss[j] - 5) & (ss[:-1] < ss[j] + 65) & (ll > 5) & (hh[:, 1] < -2))
                    gcap = cap
                    if len(jx):
                        jj = jx[0]
                        gcap = landing
                        if ss[jj] - ss[j] < 5:
                            q = gg[jj + 1]
                        desired = q - p
                    e = (q - p)[[0, 2]]
                    desired = e * gcap / max(0.01, np.linalg.norm(e)) if len(jx) else e * min(gain, gcap / max(0.01, np.linalg.norm(e)))
                    u = desired - v[[0, 2]]
                    key = (8 if u[0] > 0.8 else 4 if u[0] < -0.8 else 0) | (1 if u[1] > 0.8 else 2 if u[1] < -0.8 else 0)
                    if b in (0, 2):
                        ks = np.array([0, 1, 2, 4, 8, 5, 6, 9, 10])
                        par = BALLS[b]
                        v1, dx = _lin_step(v[[0, 2]], key_push(ks, 0) * par.ground_accel, par.ground_lambda, block)
                        key = int(ks[np.argmin(np.linalg.norm(v1 - desired, axis=1))])
                    block = baseblock
                    if len(jx):
                        jj = jx[0]
                        start_edge = gg[jj]
                        end_edge = gg[jj + 1]
                        ad = end_edge[[0, 2]] - start_edge[[0, 2]]
                        ad /= max(0.001, np.linalg.norm(ad))
                        ahead = (p - start_edge)[[0, 2]] @ ad
                        if lock is None and -1 < ahead < 6 and (p[1] < start_edge[1] + 0.7):
                            lock = end_edge.copy()
                    if lock is not None:
                        if p[1] < lock[1] + 0.6:
                            lock = None
                        else:
                            par = BALLS[b]
                            y = p[1]
                            vy = v[1]
                            horizon = 0
                            while y > lock[1] and horizon < 396:
                                y += vy / 66.0
                                vy = vy * par.air_lambda - 20 / 66.0
                                horizon += 1
                            if horizon > 0:
                                block = 4
                                keys9 = np.array([0, 1, 2, 4, 8, 5, 6, 9, 10])
                                acc = key_push(keys9, 0) * par.air_accel
                                n = min(horizon, 8)
                                v1, dx = _lin_step(v[[0, 2]], acc, par.air_lambda, n)
                                v2, tail = _lin_step(v1, np.zeros_like(acc), par.air_lambda, horizon - n)
                                cost = np.linalg.norm(p[[0, 2]] + dx + tail - lock[[0, 2]], axis=1)
                                key = int(keys9[cost.argmin()])
                    al = (p - lip)[[0, 2]] @ unit
                    across = (p - lip)[[0, 2]] @ side
                    land = (target - lip)[[0, 2]] @ unit
                    rest = (target - lip)[[0, 2]] @ side
                    if -60 < al < land + 3 and p[1] > target[1] - 3 and (b != 1):
                        block = 4
                        if al < -wstart:
                            lat_target = 0.0
                        else:
                            lat_target = rest * np.clip((al + wstart) / (wstart + land), 0.0, 1.0) + bias
                        sideways = np.clip(side_gain * (lat_target - across), -10, 10) - v[[0, 2]] @ side
                        vec = unit + side * (1 if sideways > 0.35 else -1 if sideways < -0.35 else 0)
                        key = (8 if vec[0] > 0.45 else 4 if vec[0] < -0.45 else 0) | (1 if vec[1] > 0.45 else 2 if vec[1] < -0.45 else 0)
                    if al > land - 4 and (b == 1 or p[1] < target[1] - 3):
                        block = 6
                        sideerr = (endgoal - p)[[0, 2]] @ side
                        speed = 18 if beyond(endgoal) - beyond(p) > 12 else 8
                        desired = speed * unit + np.clip(sideerr * 1.8, -6, 6) * side
                        u = desired - v[[0, 2]]
                        key = (8 if u[0] > 0.4 else 4 if u[0] < -0.4 else 0) | (1 if u[1] > 0.4 else 2 if u[1] < -0.4 else 0)
                    if episode and step < 3 and (rng.random() < 0.2) or (lock is None and rng.random() < 0.001):
                        key = rng.choice([0, 1, 2, 4, 8, 5, 6, 9, 10])
                    key = world_to_camera_keys(key, s.obs.cams['cams'][0]['matrix'])
                    before = s.frames_done
                    o = act(s, key, block)
                    actions += 1
                    seq += ((key, s.frames_done - before),)
                    if o is None or o.has_event(Ev.BallOff) or s.ended:
                        break
                    if o.has_event(Ev.PostCheckpoint):
                        paths.setdefault(max(1, s.tracker.sector - 1), seq)
                        if last in paths:
                            break
                    if not s.tracker.flags.ball_nav or not o.ball['found']:
                        continue
                    p, v, b = state(s)
                    sec = s.tracker.sector
                    _, dist = guide(p, sec, 0)
                    k = (sec, b, *np.floor(p / 4).astype(int), *np.floor(v / 5).astype(int))
                    old = A.get(k)
                    t = sum((n for _, n in seq))
                    if v[1] < -25 and p[1] < min(q[1], endgoal[1]) - 3:
                        break
                    if old is None or t < old[5]:
                        A[k] = [dist + 0, seq, p.copy(), v.copy(), 0 if old is None else old[4], t]
                    stale = 0 if old is None else stale + 1
                    best[sec] = min(best.get(sec, 1000000000.0), float(np.linalg.norm(p - d['map']['anchors'][sec])))
                    if stale >= 20 or p[1] < min(d['map']['anchors'][sec][1], q[1]) - 30:
                        break
            finally:
                s.close(kill=True)
            growth.append([episode + 1, round(time.monotonic() - start, 1), len(A), actions, dict(best)])
            if last in paths:
                break
        confirmed = {}
        for sec, seq in paths.items():
            s = load(args, lev, ret, copy, Path(tmp) / 'check')
            hits = []
            try:
                for k, n in seq:
                    before = s.frames_done
                    o = act(s, k, n)
                    if o is None or o.has_event(Ev.BallOff) or s.frames_done - before != n:
                        break
                    if o.has_event(Ev.PostCheckpoint):
                        hits.append(s.frames_done)
            finally:
                s.close(kill=True)
            confirmed[sec] = hits
        result = dict(level=lev, seed=seed, growth=growth, restores=restores, restore_mismatch=bad, second_build_event_frames=confirmed, paths=paths, cells=[dict(cell=list(map(int, k)), visits=a[4], elapsed=a[5], keys=a[1], pos=a[2].tolist(), vel=a[3].tolist()) for k, a in A.items()])
        (dst / f'L{lev:02}-{seed}.json').write_text(json.dumps(result, separators=(',', ':')))
        print(lev, len(A), best, confirmed, flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('atlas', 'root', 'out', 'soft', 'second'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--copy', required=True)
    ap.add_argument('--level', type=int, required=True)
    ap.add_argument('--target-sector', type=int, required=True)
    ap.add_argument('--previous', nargs='+', type=Path, required=True)
    ap.add_argument('--episodes', type=int, default=80)
    ap.add_argument('--seed', type=int, required=True)
    run(ap.parse_args())
