"""Edge-fan rule comparison on the TAS dataset (docs/observations.md section 2, f4 edge fan).

    uv run python tools/analysis/edge_fan_rule.py [RECORD ...] [--stride 3] [--out DIR]
    uv run python tools/analysis/edge_fan_rule.py --references [RECORD ...] [--stride 4]

Replays each record offline (ballrain.dataset.OfflineReplay), and on every stride-th frame with geometric
floor contact (min mindist gap <= 0.05, control, not paused) samples the ray fan exactly like
FeatureExtractor._terrain (8 directions from the contact point, 0.25 steps up to 8, rays from the ball
centre + fan_up). Then:
  - the histogram of height differences between consecutive samples (the contact height before the first),
    which separates slopes (|dy| <= 0.25 per 0.25 step) from drops (>= 1);
  - for the f3 rule ("contact": no hit or > 1 below the contact) and the f4 rule ("step": no hit or a drop
    > fan_step_drop between consecutive samples): fan-min distribution, fraction of directions / frames
    without an edge, correlation with edge_mesh, "false" (fan min <= 1 while edge_mesh > 3) and
    "miss" (edge_mesh < 0.3 while fan min >= 2) rates, split rail / non-rail contact.
Writes <out>/edge_fan_<record>.json (default <runtime>/analysis).

--references compares the fan reference on every stride-th control frame (not only floor contact): f4 (base = the
nearest mindist contact point, reference height = its height when its gap <= 1, else the ball bottom) against f5
(FeatureExtractor: the floor support = the nearest floor-facing mindist (n_y >= 0.5) within gap 1, else the floor
under the ball centre, else none), split into floor support / airborne / side-or-overhead nearest contact / rail /
moving partner. Writes <out>/edge_fan_ref_<record>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("BALLRAIN_REPO") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from ballrain import dataset, features as FX, geometry as G, instances  # noqa: E402
from ballrain.state import G_RAILS  # noqa: E402

DS = instances.runtime_dir() / "datasets" / f"tas-v{dataset.DATASET_VERSION}"
HIST_EDGES = [-50, -10, -3, -2, -1.5, -1, -0.75, -0.5, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.5, 1, 2, 50]


def sample(record: str, stride: int):
    fx = FX.FeatureExtractor(FX.FeatureConfig())
    c = fx.cfg
    steps = np.arange(c.fan_step, c.fan_max + 1e-9, c.fan_step)
    rows = []
    for i, o, t in dataset.OfflineReplay(DS / record):
        if i % stride or not t.control_valid() or t.flags.paused or not len(o.mind):
            continue
        k = int(np.argmin(o.mind["gap"]))
        gap = float(o.mind["gap"][k])
        if gap > FX.GEOM_EPS:
            continue
        pos = np.asarray(o.ball["pos"], np.float64)
        n = FX.mind_normals(t, np.asarray(o.ball["pos"], np.float64), o.mind)[k]  # paper ball: oriented (f5)
        partner = int(o.mind["partner"][k])
        r = float(o.ball["extra_radius"]) or 2.0
        cp = pos - n * (r + gap)
        right, fwd = fx.frame(t)
        geo = FX.level_geometry(t)
        dyn = FX.dynamic_triangles(t, pos, c.body_radius)
        ang = np.arange(c.fan_dirs) * (2 * math.pi / c.fan_dirs)
        dirs = np.cos(ang)[:, None] * fwd + np.sin(ang)[:, None] * right
        P = (cp + steps[None, :, None] * dirs[:, None, :]).reshape(-1, 3)
        P[:, 1] = pos[1] + c.fan_up
        y, tt = geo.tris.ray_down(P)
        if len(dyn):
            y2, t2 = dyn.ray_down(P)
            y, _, _ = G.merge_ray(y, tt, y2, t2)
        rows.append({"fd": o.frames_done, "h": cp[1], "y": y.reshape(c.fan_dirs, len(steps)),
                     "rail": t.level.in_group(partner, G_RAILS), "em": geo.edges.distance(cp, 12.0)})
    return rows, steps


def evaluate(rows, steps, rule: str) -> dict:
    cfg = FX.FeatureConfig(fan_rule=rule)
    out = {}
    for name, sel in (("non_rail", [r for r in rows if not r["rail"]]), ("rail", [r for r in rows if r["rail"]])):
        if not sel:
            continue
        F = np.array([FX.fan_edges(r["y"], r["h"], steps, cfg) for r in sel])
        mn = F.min(1)
        em = np.array([r["em"] for r in sel])
        fin = np.isfinite(em)
        d = {"frames": len(sel), "fan_min_le_0.25": float(np.mean(mn <= 0.25)),
             "fan_min_pct": [float(x) for x in np.percentile(mn, [10, 25, 50, 75, 90])],
             "dirs_capped": float(np.mean(F == cfg.fan_max)), "frames_all_capped": float(np.mean((F == cfg.fan_max).all(1))),
             "dir_pct": [float(x) for x in np.percentile(F, [10, 25, 50, 75, 90])]}
        if fin.sum() > 10:
            d["corr_fan_min_edge_mesh"] = float(np.corrcoef(mn[fin], np.minimum(em[fin], 12))[0, 1])
            d["false_fan_le_1_mesh_gt_3"] = float(np.mean((mn <= 1) & (em > 3)))
            d["miss_mesh_lt_0.3_fan_ge_2"] = float(np.mean((em < 0.3) & (mn >= 2)))
        out[name] = d
    return out


def histogram(rows) -> dict:
    out = {}
    for name, sel in (("non_rail", [r for r in rows if not r["rail"]]), ("rail", [r for r in rows if r["rail"]])):
        D = []
        for r in sel:
            y = np.c_[np.full(len(r["y"]), r["h"]), r["y"]]
            d = np.diff(y, axis=1)
            D.append(d[~np.isnan(d)])
        if D:
            h, _ = np.histogram(np.concatenate(D), HIST_EDGES)
            out[name] = {f"[{a},{b})": int(c) for a, b, c in zip(HIST_EDGES[:-1], HIST_EDGES[1:], h)}
    return out


def references(record: str, stride: int) -> dict:
    fx = FX.FeatureExtractor(FX.FeatureConfig())
    c = fx.cfg
    steps = np.arange(c.fan_step, c.fan_max + 1e-9, c.fan_step)
    ang = np.arange(c.fan_dirs) * (2 * math.pi / c.fan_dirs)
    rows = []
    for i, o, t in dataset.OfflineReplay(DS / record):
        if o.ball is None or not t.flags.started:
            continue
        _, info = fx.compute(t)  # every frame: keeps the path projection continuous
        if i % stride or not t.control_valid() or t.flags.paused or "edge_fan" not in info:
            continue
        pos = np.asarray(o.ball["pos"], np.float64)
        r = float(o.ball["extra_radius"]) or 2.0
        cp, gap = info.get("contact_point"), info.get("gap")
        base = cp if cp is not None else pos
        h4 = cp[1] if cp is not None and gap <= 1.0 else pos[1] - r
        right, fwd = fx.frame(t)
        dirs = np.cos(ang)[:, None] * fwd + np.sin(ang)[:, None] * right
        P = (base + steps[None, :, None] * dirs[:, None, :]).reshape(-1, 3)
        P[:, 1] = pos[1] + c.fan_up
        geo = FX.level_geometry(t)
        xz = P[:, [0, 2]]
        dyn = FX.dynamic_triangles(t, pos, c.body_radius, (xz.min(0), xz.max(0)))
        y, tt = geo.tris.ray_down(P)
        if len(dyn):
            y2, t2 = dyn.ray_down(P)
            y, _, _ = G.merge_ray(y, tt, y2, t2)
        f4 = FX.fan_edges(y.reshape(c.fan_dirs, len(steps)), h4, steps, c)
        partner = info.get("partner")
        n = info.get("normal")
        rows.append({"f4": f4, "f5": info["edge_fan"], "ref": info["fan_ref"], "em": info["edge_mesh"],
                     "air": gap is None or gap > 1.0, "side": gap is not None and gap <= 1.0 and n[1] < 0.5,
                     "rail": partner is not None and t.level.in_group(partner, G_RAILS),
                     "moving": partner is not None and not t.level.is_fixed(partner)})
    out = {"record": record, "stride": stride, "frames": len(rows)}
    cats = {"all": lambda r: True, "floor_support_nearest": lambda r: not r["air"] and not r["side"],
            "airborne": lambda r: r["air"], "airborne_floor_below": lambda r: r["air"] and r["ref"] == "below",
            "airborne_no_floor": lambda r: r["air"] and r["ref"] == "none",
            "side_or_overhead_nearest": lambda r: r["side"], "side_non_rail": lambda r: r["side"] and not r["rail"],
            "rail": lambda r: r["rail"], "moving_non_rail": lambda r: r["moving"] and not r["rail"]}
    for name, f in cats.items():
        sel = [r for r in rows if f(r)]
        if not sel:
            continue
        d = {"frames": len(sel)}
        em = np.array([r["em"] for r in sel])
        for k in ("f4", "f5"):
            F = np.array([r[k] for r in sel])
            d[k] = {"all_dirs_le_0.25": float(np.mean((F <= 0.25).all(1))), "min_le_0.25": float(np.mean(F.min(1) <= 0.25)),
                    "false_min_le_1_mesh_gt_3": float(np.mean((F.min(1) <= 1) & (em > 3)))}
        d["changed"] = int(sum(not np.array_equal(r["f4"], r["f5"]) for r in sel))
        out[name] = d
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="*", default=["SR01_1_14_457_264"])
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--references", action="store_true", help="f4 vs f5 fan reference on all control frames")
    ap.add_argument("--out", default=str(instances.runtime_dir() / "analysis"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.references:
        for rec in a.records:
            res = references(rec, a.stride)
            (out / f"edge_fan_ref_{rec}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
            print(json.dumps(res), flush=True)
        return 0
    for rec in a.records:
        rows, steps = sample(rec, a.stride)
        res = {"record": rec, "stride": a.stride, "samples": len(rows), "step_dy_histogram": histogram(rows),
               "contact_f3": evaluate(rows, steps, "contact"), "step_f4": evaluate(rows, steps, "step")}
        (out / f"edge_fan_{rec}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps({"record": rec, "samples": len(rows), "contact_f3": res["contact_f3"],
                          "step_f4": res["step_f4"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
