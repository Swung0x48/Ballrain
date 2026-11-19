"""Offline check of the f8 contact / rail blocks (ballrain.features, docs/observations.md f8) on recorded TAS dataset
records (runtime/datasets/tas-v1): no game, no engine.

    python scripts/check_contact_features.py SR01_1_14_457_264 SR02_1_28_992_264 [--baseline f7.npz] [--out DIR]
        [--crown]

Per record, two extractors with the r8 configuration (f7: world camera, reach, look-ahead) and the same with
version "f8" read the same replay:
  * prefix: the f8 vector's first entries equal the f7 vector exactly (and, with --baseline, the f7 vectors equal the
    ones stored by an older checkout: key "<record>__r8");
  * bounds: every f8 entry is in [-1, 1];
  * cost: compute() time per frame of each extractor, and of the f8 blocks alone;
  * double rails: frames where two rail tubes lie on opposite sides of the ball (tube axes from
    fastreach.extract_rails, an independent geometric reading: the nearest axis and another one more than 2 apart
    across it, both surfaces within 0.5 of the ball). On those frames: how often rails_lr shows both (left and right
    present, more than 1.5 apart along the route's right axis), how often the contacts block holds two rail contacts,
    and the side flips per frame of the legacy single-contact features (normal_cam, rail_dir: side = sign along the
    route's right axis) against the f8 slots. A plot of the longest stretch goes to <out>/<record>_double_rail.png.
  * --crown (L2): rails_lr along the east-platform member's measured stone-ball trace on the sector-3 north guardrail
    (benchmarks/refpaths-alt/L02-multi-s3route/east-platform/L02.npz), with the level geometry of the SR02 record.
Results: <out>/summary.json (default runtime/viz/contacts).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ballrain import dataset, features as FX, instances  # noqa: E402
from ballrain.state import G_RAILS  # noqa: E402

R8 = dict(version="f7", camera="world", path_dir="benchmarks/refpaths-v2", far_waypoints=(20.0, 30.0, 45.0),
          ref_speed_at=(0.0, 15.0, 30.0), hs_block=False, reach=True)
CROWN = REPO / "benchmarks" / "refpaths-alt" / "L02-multi-s3route" / "east-platform" / "L02.npz"


def tube_axes(geo):
    from ballrain.fastreach import extract_rails
    seg, _, _ = extract_rails(geo.rails.A, geo.rails.B, geo.rails.C, geo.rails.owner)
    return seg


def double_rail(seg, pos, radius) -> bool:
    """Two tube axes on opposite sides of the ball: the nearest axis point and another one more than 2 away from it
    across the nearest segment's direction (horizontally), both tube surfaces within 0.5 of the ball."""
    if not len(seg):
        return False
    a0, a1, R = seg[:, :3], seg[:, 3:6], seg[:, 6]
    dv = a1 - a0
    t = np.clip(((pos - a0) * dv).sum(1) / np.maximum((dv * dv).sum(1), 1e-12), 0, 1)
    q = a0 + t[:, None] * dv
    gap = np.linalg.norm(pos - q, axis=1) - R - radius
    near = np.flatnonzero(gap < 0.5)
    if len(near) < 2:
        return False
    k = near[np.argmin(gap[near])]
    u = dv[k] / max(np.linalg.norm(dv[k]), 1e-12)
    across = np.cross(u, [0.0, 1.0, 0.0])
    h = np.linalg.norm(across)
    if h < 1e-6:
        return False
    across /= h
    sep = (q[near] - q[k]) @ across
    side = (q[near] - pos) @ across * ((q[k] - pos) @ across)
    return bool(np.any((np.abs(sep) > 2.0) & (side < 0)))


def flips(x) -> int:
    s = np.sign(np.asarray(x, float))
    s = s[(s != 0) & ~np.isnan(s)]
    return int(np.sum(s[1:] != s[:-1]))


def slot_swaps(st) -> int:
    """Consecutive frames where a contacts slot holds a rail contact in both and its normal changes side (the slot
    went from one tube to the other)."""
    n = 0
    for a, b in zip(st, st[1:]):
        for x, y in zip(a["slots"], b["slots"]):
            if not (np.isnan(x) or np.isnan(y)) and np.sign(x) != np.sign(y) and x and y:
                n += 1
    return n


def runs(mask):
    out, start = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    return out


def check_record(rec: Path, out: Path, baseline=None, plot=True) -> dict:
    f7 = FX.FeatureExtractor(FX.FeatureConfig(**R8))
    f8 = FX.FeatureExtractor(FX.FeatureConfig(**dict(R8, version="f8")))
    n7 = f7.size
    s8 = f8.slices()
    new = slice(n7, f8.size)
    t7, t8, tb = [], [], []
    orig = f8._contact_block

    def timed(*a, **k):
        t0 = time.perf_counter()
        orig(*a, **k)
        tb.append(time.perf_counter() - t0)
    f8._contact_block = timed
    V7, V8, rows = [], [], []
    seg = None
    for i, o, t in dataset.OfflineReplay(rec):
        a = time.perf_counter()
        v7, _ = f7.compute(t)
        b = time.perf_counter()
        v8, info = f8.compute(t)
        c = time.perf_counter()
        t7.append(b - a)
        t8.append(c - b)
        V7.append(v7)
        V8.append(v8)
        if "pos" not in info or not v8[0]:
            rows.append(None)
            continue
        if seg is None:
            seg = tube_axes(FX.level_geometry(t))
        radius = float(t.obs.ball["extra_radius"]) or 2.0
        lat = info["lateral_axis"]
        lat_f = FX.to_frame(lat, *f8.frame(t))
        rr = v8[s8["rails_lr"]].reshape(2, 6)
        cs = v8[s8["contacts"]].reshape(FX.CONTACT_SLOTS, 8)
        rail_contacts = int(np.sum((cs[:, 0] > 0) & (cs[:, 6] > 0) & (cs[:, 1] > 1 - 0.3 / FX.NO_CONTACT_GAP)))
        rd = v8[f8.slices()["rail_dir"]]
        n_cl = len(info.get("contacts", ()))
        rows.append(dict(i=i, dr=double_rail(seg, info["pos"], radius), n_cl=n_cl,
                         dropped=n_cl - sum(e is not None for e in info.get("contact_slots", ())),
                         both=bool(rr[0, 0] and rr[1, 0] and rr[1, 5] - rr[0, 5] > 1.5 / f8.cfg.rail_radius),
                         rail_contacts=rail_contacts, lat_l=rr[0, 5] if rr[0, 0] else np.nan,
                         lat_r=rr[1, 5] if rr[1, 0] else np.nan,
                         legacy_n=float(info["normal"] @ lat) if "normal" in info else np.nan,
                         legacy_rail=float(rd[0] * lat_f[0] + rd[1] * lat_f[2]),
                         slots=[float(cs[k, 5]) if cs[k, 0] and cs[k, 6] else np.nan for k in range(len(cs))]))
    V7, V8 = np.stack(V7), np.stack(V8)
    res = {"frames": len(V7), "n_f7": n7, "n_f8": f8.size,
           "prefix_equal": bool(np.array_equal(V8[:, :n7], V7)),
           "new_abs_max": float(np.abs(V8[:, new]).max()),
           "ms_f7": {"median": 1e3 * float(np.median(t7)), "mean": 1e3 * float(np.mean(t7))},
           "ms_f8": {"median": 1e3 * float(np.median(t8)), "mean": 1e3 * float(np.mean(t8))},
           "us_blocks": {"median": 1e6 * float(np.median(tb)), "mean": 1e6 * float(np.mean(tb)),
                         "p99": 1e6 * float(np.percentile(tb, 99))}}
    if baseline is not None:
        key = f"{rec.name}__r8"
        res["baseline_equal"] = bool(key in baseline.files and np.array_equal(baseline[key], V7))
    live = [r for r in rows if r is not None]
    drm = np.array([r["dr"] for r in live])
    dr = [r for r in live if r["dr"]]
    res["valid_frames"] = len(live)
    # contacts within NO_CONTACT_GAP per frame, and frames where a contact has no slot (a nearer one in its direction)
    res["contacts_per_frame"] = {str(k): int(v) for k, v in zip(*np.unique([r["n_cl"] for r in live], return_counts=True))}
    res["frames_with_unslotted_contact"] = int(sum(r["dropped"] > 0 for r in live))
    res["double_rail_frames"] = len(dr)
    if dr:
        res["double_rail"] = {
            "rails_lr_both": float(np.mean([r["both"] for r in dr])),
            "two_rail_contacts": float(np.mean([r["rail_contacts"] >= 2 for r in dr])),
            "one_plus_rail_contact": float(np.mean([r["rail_contacts"] >= 1 for r in dr])),
        }
        # side flips per frame inside the stretches (consecutive frames only)
        fl = {"legacy_normal_cam": 0, "legacy_rail_dir": 0, "f8_rail_slot_swaps": 0, "f8_rails_left": 0,
              "f8_rails_right": 0}
        n = 0
        for lo, hi in runs(drm):
            st = live[lo:hi]
            n += len(st)
            fl["legacy_normal_cam"] += flips([r["legacy_n"] for r in st])
            fl["legacy_rail_dir"] += flips([r["legacy_rail"] for r in st])
            fl["f8_rail_slot_swaps"] += slot_swaps(st)
            fl["f8_rails_left"] += flips([r["lat_l"] for r in st if not np.isnan(r["lat_l"])])
            fl["f8_rails_right"] += flips([r["lat_r"] for r in st if not np.isnan(r["lat_r"])])
        res["side_flips_per_100_frames"] = {k: 100.0 * v / max(n, 1) for k, v in fl.items()}
        res["stretches"] = [(live[lo]["i"], live[hi - 1]["i"]) for lo, hi in runs(drm) if hi - lo >= 10]
        if plot:
            lo, hi = max(runs(drm), key=lambda r: r[1] - r[0])
            lo, hi = max(lo - 30, 0), min(hi + 30, len(live))
            plot_stretch(live[lo:hi], out / f"{rec.name}_double_rail.png", rec.name)
    return res


def plot_stretch(st, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    i = np.array([r["i"] for r in st])
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    shade = np.array([r["dr"] for r in st])
    for a in ax:
        for lo, hi in runs(shade):
            a.axvspan(i[lo], i[hi - 1], color="0.9", lw=0)
    ax[0].plot(i, [r["legacy_n"] for r in st], ".", ms=3, label="legacy normal_cam . route right")
    ax[0].plot(i, [r["legacy_rail"] for r in st], ".", ms=3, label="legacy rail_dir . route right")
    ax[0].set_ylim(-1.1, 1.1)
    ax[0].legend(loc="upper right", fontsize=8)
    for k in range(FX.CONTACT_SLOTS):
        ax[1].plot(i, [r["slots"][k] for r in st], ".", ms=3, label=f"f8 contacts[{k}] n_lat (rail contacts)")
    ax[1].plot(i, [r["rail_contacts"] / 2 for r in st], "-", lw=1, label="rail contacts (gap < 0.3) / 2")
    ax[1].set_ylim(-1.1, 1.1)
    ax[1].legend(loc="upper right", fontsize=8)
    ax[2].plot(i, np.array([r["lat_l"] for r in st]) * 6, ".", ms=3, label="f8 rails_lr left: offset along route right")
    ax[2].plot(i, np.array([r["lat_r"] for r in st]) * 6, ".", ms=3, label="f8 rails_lr right")
    ax[2].set_ylabel("units")
    ax[2].set_xlabel("dataset frame (grey: two tubes on opposite sides, tube-axis reading)")
    ax[2].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{title}: double-rail stretch")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def check_crown(rec: Path) -> dict:
    """rails_lr along the measured east-platform guardrail trace (stone ball centres on the crown)."""
    from ballrain.path import ReferencePath
    rep = dataset.OfflineReplay(rec)
    for i, o, t in rep:
        if t.obs is not None and t.obs.ball is not None and t.obs.ball["found"]:
            break
    geo = FX.level_geometry(t)
    line = ReferencePath.load(CROWN)
    P = line.P.astype(np.float64)
    m = (P[:, 0] > 940) & (P[:, 0] < 960) & (P[:, 1] > 49.7) & (np.abs(P[:, 2] + 363.98) < 0.2)
    k = np.flatnonzero(m)
    right, fwd = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    rows = []
    for j in k:
        tng = P[min(j + 1, len(P) - 1)] - P[max(j - 1, 0)]
        h = np.hypot(tng[0], tng[2])
        lat = np.array([tng[2] / h, 0.0, -tng[0] / h])
        v, sides = FX.rail_sides(geo.rails, P[j], 2.0, lat, right, fwd, 6.0)
        rows.append((v.reshape(2, 6), sides))
    near = geo.tris.candidates(P[k[len(k) // 2]], 3.0)
    owners = sorted({str(t.level.name(int(o))) for o in geo.tris.owner[near]})
    both = [r[0][0, 0] > 0 and r[0][1, 0] > 0 for r in rows]
    gaps = [min(s[0] for s in r[1] if s is not None) for r in rows if any(s is not None for s in r[1])]
    return {"points": len(k), "x_range": [float(P[k, 0].min()), float(P[k, 0].max())],
            "rails_lr_any": float(np.mean([r[0][0, 0] > 0 or r[0][1, 0] > 0 for r in rows])),
            "rails_lr_both": float(np.mean(both)),
            "nearest_rail_gap": {"median": float(np.median(gaps)) if gaps else None,
                                 "max": float(np.max(gaps)) if gaps else None},
            "lateral_left_units": [float(np.nanmin([r[0][0, 5] * 6 if r[0][0, 0] else np.nan for r in rows])),
                                   float(np.nanmax([r[0][0, 5] * 6 if r[0][0, 0] else np.nan for r in rows]))],
            "lateral_right_units": [float(np.nanmin([r[0][1, 5] * 6 if r[0][1, 0] else np.nan for r in rows])),
                                    float(np.nanmax([r[0][1, 5] * 6 if r[0][1, 0] else np.nan for r in rows]))],
            "rail_owners_near": sorted({str(t.level.name(s[2])) for r in rows for s in r[1] if s is not None}),
            "static_owners_near_mid": owners[:12]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+")
    ap.add_argument("--baseline", default=None, help="npz of f7 vectors from an older checkout (<record>__r8)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--crown", action="store_true")
    a = ap.parse_args(argv)
    root = instances.runtime_dir() / "datasets" / f"tas-v{dataset.DATASET_VERSION}"
    out = Path(a.out) if a.out else instances.runtime_dir() / "viz" / "contacts"
    out.mkdir(parents=True, exist_ok=True)
    base = np.load(a.baseline) if a.baseline else None
    summary = {}
    for name in a.records:
        rec = root / name
        summary[name] = check_record(rec, out, base)
        print(name, json.dumps(summary[name], indent=1), flush=True)
    if a.crown:
        rec = next((root / n for n in a.records if n.startswith("SR02_")), None)
        if rec is not None:
            summary["crown_east_platform"] = check_crown(rec)
            print("crown", json.dumps(summary["crown_east_platform"], indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    ok = all(v.get("prefix_equal", True) and v.get("baseline_equal", True) and v.get("new_abs_max", 0) <= 1.0
             for v in summary.values() if isinstance(v, dict) and "frames" in v)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
