"""TAS-free checkpoint search. Run one process per reserved copy with disjoint seed/output.

Guides only propose targets. Only engine events from an uninterrupted input replay are
checkpoint evidence. Archive keys are physical, not camera-relative world commands.
Previous archives must be trusted local outputs; cached states are checked after replay.
A guide with no connected mesh corridor falls back to the geometric anchor direction.
"""
from pathlib import Path
import math, sys, time, json, tempfile, random, concurrent.futures as cf
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from routescout_atlas import load, state
from ballrain.client import DEFAULT_DELTA as DT
from ballrain import tas
from ballrain.reach import BALLS, key_push, _lin_step
from ballrain.features import level_geometry
from ballrain.routescout.rail_actions import RailCrowns, return_graph
from ballrain.routescout.mechanism_evidence import pose_bins
from ballrain.env import world_to_camera_keys
from ballrain.protocol import Ev
import argparse

from ballrain.routescout.mechanisms import Rules
from ballrain.reach_mechanisms import level_asset,asset_pin,uses_lowered_plate

def run(args):
    dst = args.out
    dst.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    lev = args.level
    seed = args.seed
    rng = random.Random(seed)
    last = args.target_sector
    soft = args.soft
    ret = args.second
    copy = args.copy
    atlas = json.loads(args.atlas.read_text(encoding='utf8'))
    d = {'segments': atlas['segments'], 'map': atlas['map']}
    data=level_asset(lev);table=data['rules'];pin=asset_pin();rules=None;vetoes=0
    rail_choices = {}
    guides = []
    if not 1 <= last <= len(d['segments']):
        raise ValueError('target sector outside map')
    for seg in d['segments']:
        sec = seg['sector']
        g = np.array(seg['guide'] or d['map']['anchors'][sec - 1:sec + 1])
        delta = np.diff(g, axis=0)
        length = np.linalg.norm(delta, axis=1)
        guides.append((g, delta, length, np.r_[0, length.cumsum()]))
    returns={}
    profiles={}
    modes = {}
    base = len(guides)
    pairs = list(zip(d['map']['anchors'], d['map']['anchors'][1:]))
    levels = list(range(1, base + 1))
    for sector in range(1, last + 1) if args.guide in ('gates','moving','return') else []:
        props = [o for o in d['map']['placements'] if o['sector'] == sector]
        plates = [o['pos'] for o in props if table[o['name']]['rules'][1]=='blocked_when_raised']
        paper = [o['pos'] for o in props if 'transform_paper' in table[o['name']]['rules']]
        stone = [o['pos'] for o in props if 'transform_stone' in table[o['name']]['rules']]
        fans = [o['pos'] for o in props if table[o['name']]['rules'][1]=='lift']
        src, dst0 = map(np.array, pairs[sector - 1])
        if not (plates and paper and stone and fans):
            continue
        low = min(paper, key=lambda q: np.linalg.norm(src - q))
        cands = [(q, f) for q in stone for f in fans if 3 < q[1] - f[1] < 35 and np.linalg.norm((np.array(q) - f)[[0, 2]]) < 32]
        if not cands:
            continue
        high, foot = min(cands, key=lambda v: np.linalg.norm(np.array(low) - v[1]) + np.linalg.norm(np.array(v[0]) - v[1]))
        feet = [f for f in fans if np.linalg.norm((np.array(f) - high)[[0, 2]]) < 32 and f[1] <= high[1]]
        bottom = min(feet, key=lambda f: f[1])
        plate = min(plates, key=lambda q: np.linalg.norm(np.array(q) - dst0))
        via = [src, low, bottom, high, plate, dst0]
        modes[sector] = (len(guides), via, feet)
        for a, b in zip(via, via[1:]):
            gg = np.array([a, b])
            hh = np.diff(gg, axis=0)
            ll = np.linalg.norm(hh, axis=1)
            guides.append((gg, hh, ll, np.r_[0, ll.cumsum()]))
            pairs.append((a, b))
            levels.append(sector)

    def raster(tr):
        import heapq
        mesh = level_geometry(tr).tris
        if args.guide in ('moving','return'):
            from ballrain.features import dynamic_triangles
            from ballrain.geometry import TriangleSet
            moving=dynamic_triangles(tr,tr.obs.ball['pos'],70.)
            if len(moving):
                mesh=TriangleSet(*(np.concatenate([getattr(mesh,k),getattr(moving,k)]) for k in ('A','B','C','owner')))
        for i in [j for j in range(len(guides)) if j >= base or j < last]:
            src, goal = np.array(pairs[i])
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
                if ob['sector'] != levels[i] or ob['kind'] not in ('trafo', 'box', 'prop ball'):
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
                    for jump in range(1, 9):
                        m = (k[0] + dx * jump, k[1] + dz * jump)
                        if not (0 <= m[0] < sz[0] and 0 <= m[1] < sz[1]):
                            break
                        if not valid[m]:
                            continue
                        dh = y[m] - y[k]
                        length = math.hypot(dx, dz) * jump
                        if -20 < dh < (.42 * length + .12 if args.guide=='slope' else .85 * length + .2):
                            nd = dist + length * (1 + 0.55 * cost[m]) + max(0, dh) * 3 + max(0, -dh - 1) * 2 + (20 if args.guide=='slope' else 9) * (jump - 1)
                            if nd < D.get(m, 1e+30):
                                D[m] = nd
                                par[m] = k
                                heapq.heappush(todo, (nd + math.dist(m, b), nd, m))
                        break
            if found:
                ids = [b]
                while ids[-1] != a:
                    ids.append(par[ids[-1]])
                ids = ids[::-1]
                gg = np.array([[xx[k], y[k] + 2, zz[k]] for k in ids])
                hh = np.diff(gg, axis=0)
                ll = np.linalg.norm(hh, axis=1)
                guides[i] = (gg, hh, ll, np.r_[0, ll.cumsum()])

    def centered(tr):
        from ballrain.geometry import ray_down_pairs
        mesh = level_geometry(tr).tris
        for i, (g, h, l, e) in enumerate(guides):
            if not d['segments'][i]['guide']:
                continue
            pts = []
            for t in np.arange(0, e[-1], 2):
                j = min(len(h) - 1, max(0, np.searchsorted(e, t) - 1))
                q = g[j] + h[j] * (t - e[j]) / l[j]
                side = h[j, [2, 0]] * [1, -1]
                side /= max(0.001, np.linalg.norm(side))
                w = np.arange(-14, 15, dtype=float)
                ps = q + np.array([0, 5, 0]) + w[:, None] * np.r_[side[0], 0, side[1]]
                heights, _ = mesh.ray_down(ps)
                valid = abs(heights - q[1]) < 4
                groups = np.split(np.flatnonzero(valid), np.where(np.diff(np.flatnonzero(valid)) > 1)[0] + 1)
                groups = [a for a in groups if len(a) and min(abs(w[a])) <= 3]
                if groups:
                    ix = max(groups, key=len)
                    q = q.copy()
                    q[[0, 2]] += side * w[ix].mean()
                    q[1] = np.nanmedian(heights[ix]) + 2
                if not pts or np.linalg.norm(q - pts[-1]) > 0.01:
                    pts.append(q)
            pts.append(g[-1])
            gg = np.array(pts)
            hh = np.diff(gg, axis=0)
            ll = np.linalg.norm(hh, axis=1)
            guides[i] = (gg, hh, ll, np.r_[0, ll.cumsum()])

    def lanes(tr):
        from ballrain.geometry import ray_down_pairs
        mesh = level_geometry(tr).tris
        for i, (g, h, l, e) in enumerate(guides):
            if not d['segments'][i]['guide']:
                continue
            stations = np.arange(0, e[-1], 1.5)
            jj = np.clip(np.searchsorted(e, stations) - 1, 0, len(h) - 1)
            mid = g[jj] + h[jj] * ((stations - e[jj]) / np.maximum(1e-08, l[jj]))[:, None]
            tangent = np.gradient(mid[:, [0, 2]], axis=0)
            side = tangent[:, [1, 0]] * [1.0, -1.0]
            side /= np.maximum(0.001, np.linalg.norm(side, axis=1))[:, None]
            w = np.arange(-14, 15)
            pts = mid[:, None, :] + np.zeros((1, len(w), 1))
            pts[:, :, 0] += side[:, 0, None] * w
            pts[:, :, 2] += side[:, 1, None] * w
            probe = pts.copy()
            probe[:, :, 1] += 5.0
            yy, ti = mesh.ray_down(probe.reshape(-1, 3), owner_mask=abs(mesh.N[:, 1]) > 0.5)
            yy = yy.reshape(pts.shape[:2])
            valid = np.isfinite(yy) & (abs(yy - mid[:, 1, None]) < 7)
            pts[:, :, 1] = np.where(valid, yy + 2, mid[:, 1, None] + 2)
            cost = np.where(valid, 0.0, 30.0) + 0.035 * abs(w) + 0.15 * abs(pts[:, :, 1] - mid[:, 1, None] - 2)
            for offset in [-2, -1, 1, 2]:
                cost += np.where(np.roll(valid, offset, axis=1), 0.0, 1.4)
            for obj in d['map']['placements']:
                if obj['sector'] != i + 1 or obj['kind'] not in ('trafo', 'box', 'prop ball', 'unknown module'):
                    continue
                dist = np.linalg.norm(pts - np.array(obj['pos']), axis=2)
                cost += np.maximum(0.0, 7 - dist) * 2.0
            links = w[:, None] - w[None, :]
            pen = abs(links) * 0.7 + np.maximum(abs(links) - 2, 0) * 8
            D = cost[0] + abs(w)
            parents = []
            for k in range(1, len(pts)):
                dy = pts[k, None, :, 1] - pts[k - 1, :, None, 1]
                trans = D[:, None] + pen + np.maximum(dy - 1.2, 0) * 15 + np.maximum(-dy - 1.6, 0) * 4
                prev = np.argmin(trans, axis=0)
                parents.append(prev)
                D = cost[k] + trans[prev, np.arange(len(w))]
            ix = [int(np.argmin(D + abs(w)))]
            for prev in parents[::-1]:
                ix.append(int(prev[ix[-1]]))
            gg = np.concatenate([pts[np.arange(len(pts)), ix[::-1]], g[-1:]])
            hh = np.diff(gg, axis=0)
            ll = np.linalg.norm(hh, axis=1)
            guides[i] = (gg, hh, np.maximum(ll, 1e-08), np.r_[0, ll.cumsum()])
    prepare = {'centered': centered, 'lanes': lanes, 'raster': raster, 'slope': raster, 'gates': raster, 'moving':raster,'return':raster}[args.guide]

    def active(p, b, sec, look):
        if sec in returns and b == 2:
            wp, pu, plate, axis, side, lo, hi, y, offset = returns[sec]
            rel = p[[0, 2]] - wp[[0, 2]]
            along = rel @ axis
            lat = rel @ side

            def local(u, z, height):
                target = wp.copy()
                target[[0, 2]] += u * axis + z * side
                target[1] = height
                return target
            stations,offsets,heights=profiles[sec]
            offset=float(np.interp(along,stations,offsets)); y=float(np.interp(along,stations,heights))
            edge=(pu-wp)[[0,2]]@axis+1.5
            if along < lo - 12:
                target = local(min(lo, along + max(look, 6)), 0.0, wp[1] + 2)
                return (target, 180 + (lo - along) + abs(lat) * 2)
            if along < edge:
                target = local(along + max(3, look), offset, y)
                return (target, 170 + hi - along + 2 * abs(lat - offset) + (25*max(0.,y-p[1]-.3) if along>=lo else 0.))
            if np.linalg.norm((p - plate)[[0, 2]]) > 5 and (p - plate)[[0, 2]] @ (plate - pu)[[0, 2]] < 0:
                return (plate, 80 + np.linalg.norm(p - plate))
            return guide(p, sec, look)
        if sec not in modes:
            return guide(p, sec, look)
        ix, via, feet = modes[sec]
        low, foot, high, plate = map(np.array, via[1:-1])
        if b == 0:
            k = 0
        elif b == 1 and np.linalg.norm((p - foot)[[0, 2]]) > 15 and (p[1] < foot[1] + 13):
            k = 1
        elif b == 1:
            target = next((f for f in sorted(feet, key=lambda f: f[1]) if f[1] + 16 > p[1] + 1), high)
            target = np.array(target)
            return (target, 200 + np.linalg.norm(p - high))
        elif np.linalg.norm(p - via[-1]) < np.linalg.norm(plate - via[-1]) - 6:
            k = 4
        else:
            k = 3
        q, v = guide(p, ix + k + 1, look)
        return (q, (4 - k) * 130 + v)

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
        nonlocal rules,vetoes
        if rules is None or rules.tr is not s.tracker:rules=Rules(s.tracker)
        remaining=n
        while remaining:
            before=s.frames_done;prev=state(s) if s.obs.ball['found'] else None
            o=s.act(tas.make_frames([DT],[k]),repeat=min(remaining,8),stop=[Ev.BallOff,Ev.PostCheckpoint,Ev.LevelFinish]);remaining-=s.frames_done-before
            if o is None or s.frames_done==before:return None
            if prev and o.ball['found']:
                p,_,b=state(s)
                if b==prev[2] and rules.gate(prev[0],p,b):vetoes+=1;return None
            if o.has_event(Ev.PostCheckpoint):rules.refresh();break
            if o.has_event(Ev.BallOff) or o.has_event(Ev.LevelFinish):break
        return o
    with tempfile.TemporaryDirectory(dir=dst) as tmp:
        for src in args.previous:
            previous = json.loads(src.read_text(encoding='utf8'))
            if previous.get('static_key', d['map']['static_key']) != d['map']['static_key'] or previous['level'] != lev:
                raise ValueError('archive level mismatch')
            for item in previous['cells']:
                k = tuple(item['cell'])
                p0 = np.array(item['pos'])
                _, dist = guide(p0, k[0], 0)
                seq0 = tuple(map(tuple, item['keys']))
                t = item['elapsed']
                if k not in A or t < A[k][5]:
                    A[k] = [dist, seq0, p0, np.array(item['vel']), 0, t]
        dynamic_sectors=set()
        for episode in range(args.episodes):
            s = load(args, lev, soft, copy, Path(tmp) / 'play')
            seq = ()
            stale = 0
            try:
                if episode == 0:
                    if s.tracker.level.static_key() != d['map']['static_key']:
                        raise ValueError('atlas map mismatch')
                    crowns = RailCrowns(s.tracker,level_geometry(s.tracker))
                    prepare(s.tracker)
                    if args.guide=='return':
                        for sector0 in modes:
                            lane=return_graph(s.tracker,sector0,modes,d,seed)
                            if lane is not None: returns[sector0],profiles[sector0]=lane
                    for k, a in A.items():
                        a[0] = active(a[2], k[1], k[0], 0)[1] + (25 if k[1]==2 and args.guide!='slope' else 0)
                if A and (episode % 12 or args.guide=='return'):
                    pool=([k for k in A if k[:2]==(last,2)] if args.guide=='return' else []) or list(A)
                    keys=rng.sample(pool,min(80,len(pool)))
                    k = min(keys, key=lambda k: A[k][0] - 2000 * k[0] + rng.expovariate(0.05) + 5 * A[k][4])
                    a = A[k]
                    a[4] += 1
                    seq = a[1]
                    frames = np.concatenate([np.repeat(tas.make_frames([DT], [key]), n) for key, n in seq])
                    if any(act(s,key,n) is None for key,n in seq):continue
                    p, v, b = state(s)
                    restores += 1
                    if not np.allclose(p, a[2], rtol=0, atol=0.0001) or not np.allclose(v, a[3], rtol=0, atol=0.0001) or s.tracker.sector != k[0] or (b != k[1]) or (args.pose_bins and len(k)>8 and pose_bins(s.tracker,p)!=(k[12:] if k[8]==-1 else k[8:])):
                        bad += 1
                        continue
                sector=s.tracker.sector
                if args.guide=='moving' and sector in modes and sector not in dynamic_sectors and state(s)[2]==1:
                    prepare(s.tracker); dynamic_sectors.add(sector)
                    for k,a in A.items(): a[0]=active(a[2],k[1],k[0],0)[1]
                cap = rng.choice((6, 8, 10, 12, 16))
                look = rng.choice((4, 6, 8, 10))
                gain = rng.choice((1.5, 2.5, 4))
                block = rng.choice((12, 18, 24, 32))
                base_block=block;cruise=rng.choice((7,9,11,13));lateral_gain=rng.choice((1.5,2.4,3.5))
                for step in range(480 if returns else 240):
                    p, v, b = state(s)
                    sec = s.tracker.sector
                    if rules is None or rules.tr is not s.tracker:rules=Rules(s.tracker)
                    q, dist = active(p,b,sec,look)
                    if rules.probe and b==rules.probe[0] and uses_lowered_plate(s.tracker,p,rules.probe[2]):q,dist=guide(p,sec,look)
                    if rules.gate(p,np.asarray(q),b):vetoes+=1;break
                    e = (q - p)[[0, 2]]
                    desired = e * min(gain, cap / max(0.01, np.linalg.norm(e)))
                    block=base_block
                    if b==2 and sec in returns:
                        wp,pu,plate,axis,side,lo,hi,y,off=returns[sec]; along=(p-wp)[[0,2]]@axis
                        if lo-12<=along<(pu-wp)[[0,2]]@axis+1.5:
                            stations,offsets,heights=profiles[sec];err=np.interp(along+1,stations,offsets)-(p-wp)[[0,2]]@side
                            desired=cruise*axis+np.clip(lateral_gain*err,-5,5)*side;block=4
                    if block!=4 and args.rail_probability and rng.random()<args.rail_probability:
                        candidate=crowns.propose(p,v,b,q,set(s.obs.fric['partner']))
                        if candidate is not None:
                            desired=candidate.velocity
                            rail_choices[candidate.phase]=1+rail_choices.get(candidate.phase,0)
                    u = desired - v[[0, 2]]
                    key = (8 if u[0] > 0.8 else 4 if u[0] < -0.8 else 0) | (1 if u[1] > 0.8 else 2 if u[1] < -0.8 else 0)
                    if b in (0, 2):
                        ks = np.array([0, 1, 2, 4, 8, 5, 6, 9, 10])
                        par = BALLS[b]
                        v1, dx = _lin_step(v[[0, 2]], key_push(ks, 0) * par.ground_accel, par.ground_lambda, block)
                        key = int(ks[np.argmin(np.linalg.norm(v1 - desired, axis=1))])
                    if episode and step < 3 and (rng.random() < 0.45) or rng.random() < (.002 if block==4 else .035):
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
                    _, dist = active(p, b, sec, 0)
                    k = (sec, b, *np.floor(p / 4).astype(int), *np.floor(v / 5).astype(int))
                    if b==2 and sec in returns:
                        wp,pu,plate,ax,side,lo,hi,y,off=returns[sec];along=(p-wp)[[0,2]]@ax
                        if lo-12<along<hi+8: k+=(-1,int(along//2),int(np.floor(((p-wp)[[0,2]]@side-off)/.22)),int(np.floor((p[1]-y)/.3)))
                    if args.pose_bins: k+=pose_bins(s.tracker,p)
                    old = A.get(k)
                    t = sum((n for _, n in seq))
                    if v[1] < -6 and p[1] < q[1] - 2:
                        break
                    if old is None or t < old[5]:
                        A[k] = [dist + (25 if b==2 and args.guide!='slope' else 0), seq, p.copy(), v.copy(), 0 if old is None else old[4], t]
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
        result = dict(mechanism_pin=pin,mechanism_vetoes=vetoes,rail_proposals=rail_choices,static_key=d['map']['static_key'], guide=args.guide, level=lev, seed=seed, growth=growth, restores=restores, restore_mismatch=bad, second_build_event_frames=confirmed, paths=paths, cells=[dict(cell=list(map(int, k)), visits=a[4], elapsed=a[5], keys=a[1], pos=a[2].tolist(), vel=a[3].tolist()) for k, a in A.items()])
        (dst / f'L{lev:02}-{seed}.json').write_text(json.dumps(result, separators=(',', ':')))
        print(lev, len(A), best, confirmed, flush=True)
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('atlas', 'root', 'out', 'soft', 'second'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--copy', required=True)
    ap.add_argument('--level', type=int, required=True)
    ap.add_argument('--target-sector', type=int, default=1)
    ap.add_argument('--previous', nargs='*', type=Path, default=[])
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--guide', choices=['centered', 'lanes', 'raster', 'slope', 'gates', 'moving','return'], default='centered')
    ap.add_argument('--rail-probability',type=float,default=.12)
    ap.add_argument('--pose-bins',action='store_true')
    run(ap.parse_args())
