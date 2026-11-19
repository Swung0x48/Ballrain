"""Seek the first checkpoint using a topology-respecting tube or flap guide. Allow outward pressure in a winding groove. Only engine-confirmed states enter the archive."""
from pathlib import Path
import sys, time, json, tempfile, random, concurrent.futures as cf
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from routescout_atlas import load, state
from ballrain.client import DEFAULT_DELTA as DT
from ballrain import tas
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
    last = 1
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
    gates = json.loads(args.gate_guide.read_text())
    if gates.get('sector', 1) != 1 or gates.get('static_key', d['map']['static_key']) != d['map']['static_key']:
        raise ValueError('unexpected guide sector or geometry')
    gg = np.array(gates['guide'])
    hh = np.diff(gg, axis=0)
    ll = np.linalg.norm(hh, axis=1)
    guides[0] = (gg, hh, ll, np.r_[0, ll.cumsum()])
    radial = np.array(gates.get('radial', np.zeros((len(gg), 2))))
    kinds = np.array(gates.get('kinds', ['floor'] * len(gg)))
    S = guides[0][3]

    def guide(p, sec, look, prior=None):
        g, h, l, e = guides[sec - 1]
        f = np.clip(((p - g[:-1]) * h).sum(1) / np.maximum(1e-09, l * l), 0, 1)
        lat = np.linalg.norm(g[:-1] + f[:, None] * h - p, axis=1)
        if prior is not None:
            lat = np.where((e[:-1] > prior - 14) & (e[:-1] < prior + 15), lat, np.inf)
        i = lat.argmin()
        s = e[i] + f[i] * l[i]
        t = min(e[-1] - 1e-05, s + look)
        j = max(0, np.searchsorted(e, t) - 1)
        return (g[j] + (t - e[j]) * h[j] / l[j], e[-1] - s + lat[i] * 2, s)
    A = {}
    growth = []
    paths = {}
    bad = restores = actions = 0
    best = {}

    def act(s, k, n):
        return s.act(tas.make_frames([DT], [k]), repeat=n, stop=[Ev.BallOff, Ev.PostCheckpoint, Ev.LevelFinish])
    with tempfile.TemporaryDirectory(dir=dst) as tmp:
        for file in args.previous:
            prev = json.loads(file.read_text())
            if prev['level'] != lev:
                raise ValueError('seed archive level mismatch')
            for c in prev['cells']:
                p0 = np.array(c['pos'])
                v0 = np.array(c['vel'])
                b0 = c['cell'][1]
                _, dist, st = guide(p0, 1, 0)
                k = (1, b0, *np.floor(p0 / 2).astype(int), *np.floor(v0 / 3).astype(int), int(st // 3))
                t = c['elapsed']
                if k not in A or t < A[k][5]:
                    A[k] = [dist, tuple(map(tuple, c['keys'])), p0, v0, 0, t, st]
        for episode in range(args.episodes):
            s = load(args, lev, soft, copy, Path(tmp) / 'play')
            seq = ()
            stale = 0
            station = 0
            try:
                if A:
                    pool = [k for k, a in A.items() if k[1] == 2 and a[6] > 85 and (not (abs(a[3][1]) > 6 or np.linalg.norm(a[2] - gg[np.argmin(abs(S - a[6]))]) > 3.8))]
                    pool = pool or list(A)
                    keys = rng.sample(pool, min(100, len(pool)))
                    k = min(keys, key=lambda k: A[k][0] - 2000 * k[0] + rng.expovariate(0.05) + 5 * A[k][4])
                    a = A[k]
                    a[4] += 1
                    station = a[6]
                    seq = a[1]
                    frames = np.concatenate([np.repeat(tas.make_frames([DT], [key]), n) for key, n in seq])
                    s.act(frames, stop=[Ev.BallOff, Ev.LevelFinish])
                    p, v, b = state(s)
                    restores += 1
                    if not np.allclose(p, a[2], rtol=0, atol=0.0001) or not np.allclose(v, a[3], rtol=0, atol=0.0001) or s.tracker.sector != k[0] or (b != k[1]):
                        bad += 1
                        continue
                cap = rng.choice((5, 6, 8, 10, 13))
                look = rng.choice((2.5, 3.5, 4.5, 6))
                gain = rng.choice((1.5, 2.5, 4))
                block = rng.choice((6, 8, 10, 14))
                cruise = rng.uniform(10.5, 17)
                push = rng.uniform(4, 22)
                rock = rng.choice((0, 1))
                for step in range(700):
                    p, v, b = state(s)
                    sec = s.tracker.sector
                    q, dist, station = guide(p, sec, look, station)
                    e = (q - p)[[0, 2]]
                    idx = min(len(gg) - 1, np.searchsorted(S, station))
                    ri = np.linalg.norm(radial[idx]) > 0.1
                    near = bool(np.any(np.linalg.norm(radial[(S >= station) & (S < station + 24)], axis=1) > 0.1))
                    dcap = cruise if near else cap
                    desired = e * min(gain, dcap / max(0.01, np.linalg.norm(e)))
                    if near:
                        desired = e * dcap / max(0.01, np.linalg.norm(e))
                    if ri:
                        tangent = gg[min(len(gg) - 1, idx + 2), [0, 2]] - gg[max(0, idx - 1), [0, 2]]
                        tangent /= max(0.001, np.linalg.norm(tangent))
                        desired = cruise * tangent + push * radial[idx]
                        block = 4
                    elif near:
                        block = 6
                    u = desired - v[[0, 2]]
                    key = (8 if u[0] > 0.8 else 4 if u[0] < -0.8 else 0) | (1 if u[1] > 0.8 else 2 if u[1] < -0.8 else 0)
                    if episode and step < 3 and (rng.random() < 0.45) or rng.random() < 0.002:
                        key = rng.choice([0, 1, 2, 4, 8, 5, 6, 9, 10])
                    key = world_to_camera_keys(key, s.obs.cams['cams'][0]['matrix'])
                    before = s.frames_done
                    o = act(s, key, block)
                    actions += 1
                    seq += ((key, s.frames_done - before),)
                    if o is None or o.has_event(Ev.BallOff) or s.ended:
                        break
                    if o.has_event(Ev.PostCheckpoint):
                        paths.setdefault(len([ev for ev in s.tracker.events if ev[1] == Ev.PostCheckpoint]), seq)
                        if last in paths:
                            break
                    if not s.tracker.flags.ball_nav or not o.ball['found']:
                        continue
                    p, v, b = state(s)
                    sec = s.tracker.sector
                    _, dist, station = guide(p, sec, 0, station)
                    k = (sec, b, *np.floor(p / 2).astype(int), *np.floor(v / 3).astype(int), int(station // 3))
                    old = A.get(k)
                    t = sum((n for _, n in seq))
                    if p[1] < q[1] - 2.0 and v[1] < -7.0:
                        break
                    if old is None or t < old[5]:
                        A[k] = [dist, seq, p.copy(), v.copy(), 0 if old is None else old[4], t, station]
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
    for name in ('atlas', 'root', 'out', 'soft', 'second', 'gate-guide'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--copy', required=True)
    ap.add_argument('--level', type=int, required=True)
    ap.add_argument('--previous', nargs='*', type=Path, default=[])
    ap.add_argument('--episodes', type=int, default=80)
    ap.add_argument('--seed', type=int, required=True)
    run(ap.parse_args())
