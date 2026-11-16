"""SR reward checks on the TAS dataset (docs/observations.md section 4).

    uv run python tools/analysis/sr_reward_check.py records RECORD [RECORD ...]
    uv run python tools/analysis/sr_reward_check.py probes [--level-record SR01_1_14_457_264]

records  replays each dataset record offline (ballrain.dataset.OfflineReplay) with env-like steps (every 4 / 132 s
         of SR time, plus the frames with BallOff / LevelFinish), one episode per life, and runs the SR reward with
         the old configuration (hold_unsupported=False, death_refund=0) and the defaults side by side. Prints per
         life: start / end frame, end reason, steps, unsupported steps, arc gained, progress paid by each, the death
         refund, the largest supported single-step arc gain and the episode totals.
probes   void probes on the record's level: from every 5 units of arc of the reference line, fly along the line's
         horizontal tangent with a fall's per-step displacement (34 u/s horizontal, -47 u/s vertical, unsupported)
         until inside a death box (DepthTestCubes), and report the progress paid (old vs defaults).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("BALLRAIN_REPO") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from ballrain import dataset, instances  # noqa: E402
from ballrain.features import FeatureConfig, FeatureExtractor  # noqa: E402
from ballrain.path import PathFollower  # noqa: E402
from ballrain.protocol import Ev  # noqa: E402
from ballrain.rewards import Reward, RewardConfig  # noqa: E402

DS = instances.runtime_dir() / "datasets" / f"tas-v{dataset.DATASET_VERSION}"
STEP_MS = 4 * 1000 / 132
OLD = dict(hold_unsupported=False, death_refund=0.0)


def records(rec: str) -> dict:
    cfgs = {"old": RewardConfig(mode="sr", **OLD), "new": RewardConfig(mode="sr")}
    fx = FeatureExtractor(FeatureConfig())
    rws = {k: Reward(c) for k, c in cfgs.items()}
    eps, cur, nev = [], None, 0
    for _i, o, t in dataset.OfflineReplay(DS / rec):
        new = t.events[nev:]
        nev = len(t.events)
        codes = {c for _, c, _ in new}
        died, fin = Ev.BallOff in codes, Ev.LevelFinish in codes
        if cur is None:
            if t.control_valid() and not t.flags.paused:
                fx.reset_episode()
                _, info = fx.compute(t)
                s = info["path"]["s"] if info.get("path") else None
                for r in rws.values():
                    r.begin(t, s)
                cur = {"start_frame": o.frames_done, "s0": s, "last_sr": t.sr_ms or 0.0, "steps": 0, "unsupported": 0,
                       "max_supported_ds": 0.0, "last_s": s}
            continue
        if not (died or fin or (t.sr_ms or 0.0) - cur["last_sr"] >= STEP_MS - 1e-6):
            continue
        _, info = fx.compute(t)
        p = info.get("path")
        s = p["s"] if p else None
        sup = bool(info.get("supported", False))
        for r in rws.values():
            r.step(t, s, died, fin, died or fin, rebase=bool(p and p.get("jump")), supported=sup)
        cur["steps"] += 1
        cur["unsupported"] += not sup
        cur["last_sr"] = t.sr_ms or cur["last_sr"]
        if s is not None and cur["last_s"] is not None and sup and not (p and p.get("jump")):
            cur["max_supported_ds"] = max(cur["max_supported_ds"], s - cur["last_s"])
        if s is not None:
            cur["last_s"] = s
        if died or fin:
            eps.append(_close(cur, o.frames_done, "BallOff" if died else "LevelFinish", rws))
            cur = None
    if cur is not None:
        eps.append(_close(cur, None, "record_end", rws))
    return {"record": rec, "episodes": eps}


def _close(cur, end_frame, reason, rws) -> dict:
    out = {k: cur[k] for k in ("start_frame", "steps", "unsupported", "max_supported_ds")}
    out.update(end_frame=end_frame, end=reason, arc=(cur["last_s"] or 0.0) - (cur["s0"] or 0.0))
    for k, r in rws.items():
        out[k] = {"progress_units": r.terms.get("progress", 0.0) / r.cfg.progress_weight,
                  "death_refund": r.terms.get("death_refund", 0.0), "total": r.total}
    return out


class _Tracker:
    sr_ms = 0.0
    checkpoints = 0

    def hs_partial(self):
        return 0

    def followers(self):
        return 0


def probes(rec: str) -> dict:
    rp = dataset.OfflineReplay(DS / rec)
    t = None
    for _i, _o, t in rp:
        if rp.loaded and t.flags.ball_nav:
            break
    fx = FeatureExtractor(FeatureConfig())
    path = fx.reference_path(t)
    dt = STEP_MS / 1000
    out = {"record": rec, "path_source": path.source, "old": [], "new": []}
    for s0 in np.arange(0.0, path.length - 5, 5.0):
        seg = int(min(np.searchsorted(path.cum, s0), len(path.seg) - 1))
        pr0 = path.project(path.point_at(s0), int(path.seg_sector[seg]))
        for name, cfg in (("old", RewardConfig(**OLD)), ("new", RewardConfig())):
            fol = PathFollower(path, back=10, ahead=30, margin=20, reacquire=5)
            q, ft, r = np.array(pr0["point"]), _Tracker(), Reward(cfg)
            r.begin(ft, fol.update(q, pr0["sector"])["s"])
            for _ in range(400):
                q = q + pr0["tangent"] * 34 * dt + np.array([0.0, -47 * dt, 0.0])
                ft.sr_ms += STEP_MS
                died = any((lo <= q).all() and (q <= hi).all() for _, lo, hi in t.level.depth)
                pr = fol.update(q, pr0["sector"])
                r.step(ft, pr["s"], died, False, died, rebase=pr["jump"], supported=False)
                if died:
                    out[name].append((r.terms.get("progress", 0.0) / cfg.progress_weight, r.total))
                    break
    summ = {"record": rec, "path_source": path.source}
    for name in ("old", "new"):
        a = np.array(out[name]).reshape(-1, 2)
        summ[name] = {"probes_reaching_death": len(a),
                      "paid_units_p50_p90_max": np.percentile(a[:, 0], [50, 90, 100]).round(1).tolist() if len(a) else None,
                      "paid_over_50_units": int((a[:, 0] > 50).sum()),
                      "episode_total_max": float(a[:, 1].max()) if len(a) else None}
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["records", "probes"])
    ap.add_argument("records", nargs="*", default=["SR01_1_14_457_264"])
    a = ap.parse_args(argv)
    for rec in a.records:
        res = records(rec) if a.what == "records" else probes(rec)
        print(json.dumps(res, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
