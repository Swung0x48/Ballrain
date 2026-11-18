"""Build the L2 south-loop branch against the recorded collision world, then audit the float32 asset."""
from pathlib import Path
import sys
import json
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ballrain.features import level_geometry
from ballrain import protocol as P, geometry as G
from ballrain.state import GameTracker
from ballrain.path import ReferencePath, file_sha256
from ballrain.reach import BALLS

LIVE = Path("D:/repos/Ballrain/runtime")
REC = LIVE / "datasets/tas-v1/SR02_1_28_992_264"
OUT = LIVE / "viz/l2s4/stone"
ASSET = ROOT / "benchmarks/refpaths-alt/L02-stone/L02.npz"


def geom():
    tr = GameTracker(2)
    tr.merge_level(P.parse_level_message((REC / "level.msg").read_bytes())[0])
    for m in P.parse_level_stream((REC / "level.msg.inc").read_bytes())[0]:
        tr.merge_level(m)
    return tr, level_geometry(tr).tris


def rails(V, F):
    """Collapse short circumferential edges into rings; walk the two longitudinal chains."""
    V, idx = np.unique(V.round(4), axis=0, return_inverse=True)
    f = idx[F]
    a, b = (
        np.concatenate((f[:, 0], f[:, 1], f[:, 2])),
        np.concatenate((f[:, 1], f[:, 2], f[:, 0])),
    )
    keep = np.linalg.norm(V[a] - V[b], axis=1) < 0.34
    lab = np.arange(len(V))
    for i, j in zip(a[keep], b[keep]):
        lab[lab == lab[j]] = lab[i]
    _, lab = np.unique(lab, return_inverse=True)
    n = max(lab) + 1
    pos = np.array([(V[lab == i].max(0) + V[lab == i].min(0)) / 2 for i in range(n)])
    adj = [set() for _ in pos]
    for i, j in zip(lab[a], lab[b]):
        if i != j:
            adj[i].add(j)
            adj[j].add(i)
    paths = []
    used = set()
    for i in range(n):
        if len(adj[i]) != 1 or i in used:
            continue
        inds = []
        while i not in used:
            inds.append(i)
            used.add(i)
            nxt = adj[i] - used
            if not nxt:
                break
            i = next(iter(nxt))
        paths.append(pos[inds])
    return paths




def resample(P, spacing):
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = np.r_[np.arange(0, d[-1] - 0.00001, spacing), d[-1]]
    return np.column_stack([np.interp(s, d, col) for col in P.T])


def seat(T, p, r):
    """Upper envelope of triangle faces, finite edge cylinders and vertex spheres.
    p.y is a layer hint: exclude the other deck where a steel rail crosses itself.
    """
    xz = p[[0, 2]]
    ids = T.grid.query_box(xz - r, xz + r)
    ids = ids[(T.lo[ids, 1] < p[1] + 2.1) & (T.hi[ids, 1] > p[1] - 4)]
    A, B, C = T.A[ids], T.B[ids], T.C[ids]
    N = T.N[ids].copy()
    N *= np.where(N[:, 1:2] < 0, -1.0, 1.0)
    top = -np.inf

    def add(h):
        return np.max(h, initial=-np.inf)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Offset the whole face upward by the sphere radius along its normal.
        ray = np.tile([p[0], 1e5, p[2]], (len(A), 1))
        h = G.ray_down_pairs(ray, A + r * N, B + r * N, C + r * N)
        top = add(h[np.isfinite(h)])
        for U, V in ((A, B), (B, C), (C, A)):
            d = V - U
            L = np.einsum("ij,ij->i", d, d)
            dxz = xz - U[:, [0, 2]]
            l2 = (dxz**2).sum(1)
            dot = (dxz * d[:, [0, 2]]).sum(1)
            a = 1 - d[:, 1] ** 2 / L
            b = -dot * d[:, 1] / L
            c = l2 - dot**2 / L - r * r
            y = (-b + np.sqrt(b * b - a * c)) / a
            t = (dot + y * d[:, 1]) / L
            top = max(top, add((U[:, 1] + y)[(t >= 0) & (t <= 1)]))
            v = U[:, 1] + np.sqrt(r * r - l2)
            top = max(top, add(v[np.isfinite(v)]))
    if not np.isfinite(top):
        raise ValueError(("no support", p))
    return [p[0], top, p[2]]


def measured_clock(main, i, j, Q, sec, ds):
    """Fit a monotone first-passage clock; retain the real TAS tail after rejoin."""
    import re
    run = OUT / "hlw-99-fimyydc1"
    rows = [json.loads(line) for line in (run / "controller.jsonl").read_text().splitlines()
            if line.startswith("{")]
    path = ReferencePath(Q, sec)
    arcs = []
    for row in rows:
        arcs.append(path.project(row["pos"], 4, arcs[-1] if arcs else None)["s"])
    frames = [r["frame"] for r in rows]
    clock = [r["sr"] for r in rows]
    events = [(int(f), name) for f, name in re.findall(r"^(\d+) (\w+)",
              (run / "events.txt").read_text(), re.M) if int(f) > frames[0]]
    entry = next(f for f, n in events if n == "BallNavInactive")
    cp = next(f for f, n in events if n.startswith("PreC"))
    # Frame gaps at the two transforms contain no ball samples. Interpolate IGT,
    # not row count; first passages ignore backtracking. Smooth over 4 m bins.
    frontier, ids = np.unique(np.maximum.accumulate(arcs), return_index=True)
    local = main.cum[i] + np.r_[0, np.cumsum(ds)]
    knots = np.r_[np.arange(local[0], local[-1], 4.), local[-1]]
    assert frontier[0] < knots[0] and frontier[-1] > knots[-1]
    raw = np.interp(knots, frontier, np.asarray(clock)[ids])
    entry_s = np.interp(entry, frames, arcs)
    duration = np.diff(np.interp([entry, cp], frames, clock))[0]
    cp_s = np.interp(cp, frames[-2:], [main.project(r['pos'], 4)['s'] for r in rows[-2:]])
    tail = np.interp(cp_s, main.cum, main.t_ms) - main.t_ms[j]
    scale = (duration - tail) / (raw[-1] - np.interp(entry_s, knots, raw))
    return main.t_ms[i] + (np.interp(local, knots, raw) - raw[0]) * scale


def build():
    tr, T = geom()
    mainfile = ROOT / "benchmarks/refpaths/L02.npz"
    main = ReferencePath.load(mainfile)
    r = BALLS[2].radius
    tracks = {}
    if main.t_ms is None:
        from ballrain.path import build_tas_path

        raw = np.load(REC / "raw.npz")
        lo, hi = main.meta["frames"]
        sl = slice(lo, hi + 1)
        pos = raw["ball_pos"][sl].astype(float)
        pos[~raw["ball_found"][sl].astype(bool)] = np.nan
        rebuilt = build_tas_path(pos, np.ones(len(pos)), raw["delta_ms"][sl])
        main.t_ms = np.interp(main.cum, rebuilt.cum, rebuilt.t_ms)
    for ck in np.unique(T.owner):
        name = tr.level.name(ck)
        if name in ("A03_Rail_01", "A03_Rail_02", "A03_Rail_03"):
            m = list(tr.level.meshes_of(ck))[0]
            a, b = rails(G.ck_to_world(m.vertices, m.matrix), m.faces)
            tracks[name[-2:]] = (a + b) / 2

    def splice(mask):
        ids = np.flatnonzero(mask & (main.point_sector == 4))
        return min(ids, key=lambda i: abs(T.closest(main.P[i], 5)["d"] - r))

    x, y, z = main.P.T
    i = splice((z < -62) & (z > -77) & (x < 936))
    j = splice((z > 16) & (z < 37))
    a, b = main.P[[i, j]]
    rail = tracks["02"]
    rail = rail[rail[:, 2] < -59]
    guides = np.vstack(
        [
            a,
            [933.8, 2.7, -57],
            [948.74, 2.7, -56.51],
            tracks["01"],
            [1050, 6, -56.5],
            [1055.43, 4.72, -57.5],
            rail,
            tracks["03"][::-1],
            [1040.6, 13, -209.4],
            [1040.6, 13, -204.4],
        ]
    )
    # The board is a wave, not a ramp: use its vertical intersection as layer hint.
    board = np.c_[
        np.full(362, 1040.6), np.full(362, 30.0), np.linspace(-204.4, -24, 362)
    ]
    h, _ = T.ray_down(board)
    board[:, 1] = h
    guides = np.vstack(
        [
            guides,
            board,
            [1040.55, 5.55, -23],
            [1040.55, 5.55, 9.22],
            [1040.61, 4.76, 12.35],
            b,
        ]
    )
    dense = resample(guides, 0.2)
    surface = np.asarray([seat(T, p, r) for p in dense])
    surface[0], surface[-1] = a, b
    # Find the next supported centre half a unit away. Simply lifting a regularly
    # sampled chord changes its length at the lip where the rail meets the floor.
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(surface, axis=0), axis=1))]

    def supported(u):
        if u >= arc[-1]:
            return b
        idx = np.searchsorted(arc, u, side="right") - 1
        pt = surface[idx] + (surface[idx + 1] - surface[idx]) * (
            (u - arc[idx]) / (arc[idx + 1] - arc[idx])
        )
        pt = seat(T, pt - [0, 1.5, 0], r)
        pt[1] += 0.00015
        return np.array(pt)

    pts = [a]
    u = 0.0
    while u < arc[-1]:
        low = u
        high = min(arc[-1], u + 0.52)
        while high < arc[-1] and np.linalg.norm(supported(high) - pts[-1]) < 0.5:
            high = min(high + 0.1, arc[-1])
        if high == arc[-1] and np.linalg.norm(b - pts[-1]) <= 0.5:
            pts.append(b)
            break
        for _ in range(15):
            mid = (low + high) / 2
            if np.linalg.norm(supported(mid) - pts[-1]) < 0.5:
                low = mid
            else:
                high = mid
        u = high
        pts.append(supported(u))
    P = np.array(pts)
    Q = np.vstack([main.P[:i], P, main.P[j + 1 :]])
    sec = np.r_[main.point_sector[:i], np.full(len(P), 4), main.point_sector[j + 1 :]]
    ds = np.linalg.norm(np.diff(P, axis=0), axis=1)
    t = measured_clock(main, i, j, Q, sec, ds)
    times = np.r_[main.t_ms[:i], t, main.t_ms[j + 1 :] - main.t_ms[j] + t[-1]]
    out = ReferencePath(Q, sec, meta=main.meta, t_ms=times)
    out.meta.update(
        points=len(Q),
        length=out.length,
        generator="scripts/build_l2_stone.py",
        sector_range=out.sector_range,
    )
    out.save(ASSET)
    out = ReferencePath.load(ASSET)
    k = i + len(P) - 1
    desc = dict(
        level=2,
        sector=4,
        ball_type=2,
        main_sha256=file_sha256(mainfile),
        branch_sha256=file_sha256(ASSET),
        main_s=main.cum[[i, j]].tolist(),
        branch_s=out.cum[[i, k]].tolist(),
        endpoints=main.P[[i, j]].tolist(),
        timing="TAS before splice; measured hlw-99-fimyydc1 first passages (trafo to PreC); shifted TAS after merge.",
    )
    desc["stations"] = [
        [float(main.project(pt, 4)["s"]), float(out.project(pt, 4)["s"])]
        for pt in (
            [948.74, 2.7, -56.51],
            [1040.61, 6.76, 12.35],
        )
    ]
    desc["main_span_t_ms"] = main.t_ms[i : j + 1].tolist()
    ASSET.with_suffix(".json").write_text(json.dumps(desc, indent=2) + "\n")
    owners = {}
    gaps = []
    for pt in out.P[i : k + 1]:
        c = T.closest(pt, 4)
        name = tr.level.name(c["owner"])
        owners[name] = owners.get(name, 0) + 1
        gaps.append(c["d"] - r)
    assert min(gaps) >= 0 and max(gaps[1:-1]) < 0.001, (min(gaps), max(gaps[1:-1]))
    assert (
        np.all(ds > 0)
        and np.all(np.diff(times) > 0)
        and np.array_equal(out.P[[i, k]], main.P[[i, j]])
    )
    print(
        json.dumps(
            dict(
                interior_gap_range=[min(gaps[1:-1]), max(gaps[1:-1])],
                points=len(P),
                **{key: v for key, v in desc.items() if key != "main_span_t_ms"},
                owners=owners,
                gap_range=[min(gaps), max(gaps)],
                worst_step=max(ds),
                min_step=min(ds),
                length=out.length,
                monotone_time=bool(np.all(np.diff(times) > 0)),
                splice_errors=np.linalg.norm(
                    out.P[[i, k]] - main.P[[i, j]], axis=1
                ).tolist(),
            ),
            indent=2,
        )
    )
    os.environ["MPLCONFIGDIR"] = str(OUT / "mplcache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    for top in (True, False):
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        for line, l, h, label in [
            (out, i, k, "钢双轨南环石线"),
            (main, i, j, "TAS 纸球线"),
        ]:
            pts = line.P[l : h + 1]
            ax.plot(
                pts[:, 0] if top else line.cum[l : h + 1] - line.cum[l],
                pts[:, 2 if top else 1],
                label=label,
            )
        ax.set(
            xlabel="东西位置 x" if top else "各线距分岔点弧长",
            ylabel="南北位置 z" if top else "球心高度 y",
            title="小节 4：石变至纸变",
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        if top:
            ax.set_aspect("equal")
        fig.savefig(
            OUT / ("stone_line_top.png" if top else "stone_line_profile.png"), dpi=160
        )
        plt.close(fig)


if __name__ == "__main__":
    build()
