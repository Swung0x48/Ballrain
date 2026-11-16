"""Build the TAS reference lines benchmarks/refpaths/L<nn>.npz from the TAS dataset (docs/observations.md,
"参考路径"). Offline: reads <runtime>/datasets/tas-v1 (built by scripts/m1_gates.py --gates D), no game.

    uv run python scripts/build_refpaths.py [--levels 1,2] [--record 1=SR01_1_14_457_264] [--spacing 0.5]
                                            [--dataset DIR] [--out DIR] [--check]

Per level the fastest completing record (end reason LevelFinish, smallest official SR) is the source unless
--record overrides it. The trajectory runs from the first frame with control (BallNavActive, ball found, not
paused) to LevelFinish; frames without an IVP ball (transformations) are bridged. ballrain.path.build_tas_path
smooths it in time and resamples it by arc length. The asset stores the points (float32), the sector label of
every point (from the tracker's sector, i.e. what the projection is constrained by at runtime), and in `meta`
the record name and sha256, spacing, smoothing, the level's start / checkpoint / airship positions (used to
check that an asset belongs to the running level) and per-sector arc ranges. No paths.

--check projects every completing record of the level onto the new line with ballrain.path.PathFollower and
prints lateral distance per sector, monotonicity on the source record, and how often the continuity
projection disagrees with a plain nearest-point search over the sector range.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from ballrain import dataset, instances  # noqa: E402
from ballrain.path import REFPATH_DIR, PathFollower, ReferencePath, asset_path, build_tas_path  # noqa: E402

DATASET = instances.runtime_dir() / "datasets" / f"tas-v{dataset.DATASET_VERSION}"


def completing_records(index: list, level: int) -> list:
    rs = [m for m in index if m.get("level") == level and m.get("end_reason") == "LevelFinish" and m.get("ok")
          and m.get("sr_ms") is not None and not m["record"].endswith("_esc.tas")]
    return sorted(rs, key=lambda m: m["sr_ms"])


def control_span(tr: dict) -> tuple[int, int]:
    """[first observation with control, observation carrying LevelFinish]."""
    i0 = int(np.argmax(tr["control"]))
    lf = tr["levelfinish_frame"]
    i1 = int(np.searchsorted(tr["frames_done"], lf)) if lf is not None else len(tr["frames_done"]) - 1
    return i0, i1


def build_level(level: int, record: str, ds: Path, spacing: float) -> ReferencePath:
    d = ds / record
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    tr = dataset.record_trajectory(d, level)
    i0, i1 = control_span(tr)
    sl = slice(i0, i1 + 1)
    keep = ~tr["paused"][sl]
    a = tr["anchors"]
    m = {"level": level, "record": record, "record_sha256": meta["record_sha256"], "sr_ms": meta["sr_ms"],
         "frames": [int(tr["frames_done"][i0]), int(tr["frames_done"][i1])],
         "start": a["start"].tolist() if a["start"] is not None else None,
         "checkpoints": [c.tolist() for c in a["checkpoints"]],
         "airship": a["airship"].tolist() if a["airship"] is not None else None,
         "generator": "scripts/build_refpaths.py"}
    path = build_tas_path(tr["pos"][sl][keep], tr["sector"][sl][keep], tr["delta_ms"][sl][keep], spacing=spacing,
                          meta=m)
    path.meta["length"] = path.length
    path.meta["sector_range"] = {str(k): list(v) for k, v in path.sector_range.items()}
    path.meta["points"] = len(path.P)
    return ReferencePath(path.P, path.point_sector, source="tas", meta=path.meta, t_ms=path.t_ms)


def lateral_report(path: ReferencePath, ds: Path, records: list, level: int, global_check: bool = True) -> dict:
    out = {}
    for rec in records:
        tr = dataset.record_trajectory(ds / rec, level)
        i0, i1 = control_span(tr)
        f = PathFollower(path)
        rows = []
        for i in range(i0, i1 + 1):
            if not tr["control"][i]:
                continue
            p, sec = tr["pos"][i], int(tr["sector"][i])
            r = f.update(p, sec)
            g = path.project(p, sec, None) if global_check else None  # nearest point in the sector range
            u = path.project(p, None, None) if global_check else None  # nearest point of the whole line
            rows.append((sec, r["dist"], r["s"], r["jump"] and r["mode"] != "init",
                         g is not None and abs(g["s"] - r["s"]) > 5.0, u is not None and abs(u["s"] - r["s"]) > 5.0))
        a = np.array(rows, dtype=float)
        per = {}
        for s in np.unique(a[:, 0]):
            d = a[a[:, 0] == s, 1]
            per[int(s)] = {"n": int(len(d)), "median": round(float(np.median(d)), 2),
                           "p95": round(float(np.percentile(d, 95)), 2), "max": round(float(d.max()), 2)}
        ds_ = np.diff(a[:, 2])
        out[rec] = {"per_sector": per, "all_median": round(float(np.median(a[:, 1])), 2),
                    "all_max": round(float(a[:, 1].max()), 2), "reacquire_jumps": int(a[:, 3].sum()),
                    "backward_steps_gt_0.5": int((ds_ < -0.5).sum()), "s_end": round(float(a[-1, 2]), 1),
                    "disagree_with_sector_nearest": int(a[:, 4].sum()),
                    "disagree_with_global_nearest": int(a[:, 5].sum()), "frames": int(len(a))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1-13")
    ap.add_argument("--record", action="append", default=[], help="LEVEL=RECORD (override the source record)")
    ap.add_argument("--spacing", type=float, default=0.5)
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--out", default=str(REFPATH_DIR))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    ds = Path(a.dataset)
    index = json.loads((ds / "index.json").read_text(encoding="utf-8"))
    lv = a.levels.split("-") if "-" in a.levels else None
    levels = list(range(int(lv[0]), int(lv[1]) + 1)) if lv else [int(x) for x in a.levels.split(",")]
    override = {int(k): v for k, v in (r.split("=", 1) for r in a.record)}
    summary = {}
    for level in levels:
        recs = completing_records(index, level)
        src = override.get(level) or (Path(recs[0]["record"]).stem if recs else None)
        if src is None:
            print(f"L{level:02d}: no completing record in {ds.name}; no asset (the checkpoint polyline is used)")
            summary[level] = None
            continue
        path = build_level(level, src, ds, a.spacing)
        out = asset_path(level, a.out)
        path.save(out)
        summary[level] = {"record": src, "points": len(path.P), "length": round(path.length, 1),
                          "sectors": path.n_sectors, "bytes": out.stat().st_size}
        print(f"L{level:02d}: {src} -> {out.name}: {len(path.P)} points, {path.length:.1f} units, "
              f"{path.n_sectors} sectors, {out.stat().st_size} bytes", flush=True)
        if a.check:
            rep = lateral_report(ReferencePath.load(out), ds, [Path(m["record"]).stem for m in recs], level)
            for rec, r in rep.items():
                print(f"   {rec}: {json.dumps(r)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
