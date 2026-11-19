"""Explore after a plate using verified input-prefix restores. Ballistic gap edges are guidance only."""
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
        raise ValueError('sector has no plate placement')
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
            src, goal = (np.array(tr.obs.ball['pos']), endgoal)
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
                raise ValueError('archive map mismatch')
            for item in archive['cells']:
                k = tuple(item['cell'])
                p0 = np.array(item['pos'])
                _, dist = guide(p0, k[0], 0)
                seq0 = tuple(map(tuple, item['keys']))
                t = item['elapsed']
                if k not in A or t < A[k][5]:
                    A[k] = [dist, seq0, p0, np.array(item['vel']), 0, t]
        prepared = False
        for episode in range(args.episodes):
            s = load(args, lev, soft, copy, Path(tmp) / 'play')
            seq = ()
            stale = 0
            try:
                if A:
                    ks = [k for k, a in A.items() if k[:2] == (last, 2) and beyond(a[2]) > 0 and (a[2][1] > gate[1] - 1 or beyond(a[2]) > 18)]
                    if not ks:
                        raise ValueError('no post-gate engine states in supplied archives')
                    keys = rng.sample(ks, min(64, len(ks)))
                    k = min(keys, key=lambda k: A[k][0] + rng.expovariate(1 / 8) + 3 * A[k][4])
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
                landing = rng.choice((12, 15, 18, 22))
                cap = rng.choice((5, 7, 9, 11))
                look = rng.choice((3, 4, 5, 7, 9))
                gain = rng.choice((1.5, 2.5, 4))
                block = rng.choice((6, 10, 14, 18))
                for step in range(240):
                    p, v, b = state(s)
                    sec = s.tracker.sector
                    q, dist = guide(p, sec, look)
                    gg, hh, ll, ss = guides[sec - 1]
                    j = np.linalg.norm(gg - p, axis=1).argmin()
                    jx = np.flatnonzero((ss[:-1] >= ss[j] - 5) & (ss[:-1] < ss[j] + 20) & (ll > 5) & (hh[:, 1] < -2))
                    gcap = cap
                    if len(jx):
                        jj = jx[0]
                        q = gg[jj + 1]
                        gcap = landing
                    e = (q - p)[[0, 2]]
                    desired = e * min(gain, gcap / max(0.01, np.linalg.norm(e)))
                    u = desired - v[[0, 2]]
                    key = (8 if u[0] > 0.8 else 4 if u[0] < -0.8 else 0) | (1 if u[1] > 0.8 else 2 if u[1] < -0.8 else 0)
                    if b in (0, 2):
                        ks = np.array([0, 1, 2, 4, 8, 5, 6, 9, 10])
                        par = BALLS[b]
                        v1, dx = _lin_step(v[[0, 2]], key_push(ks, 0) * par.ground_accel, par.ground_lambda, block)
                        key = int(ks[np.argmin(np.linalg.norm(v1 - desired, axis=1))])
                    if episode and step < 3 and (rng.random() < 0.45) or rng.random() < 0.035:
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
                    if v[1] < -6 and p[1] < q[1] - 2:
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
