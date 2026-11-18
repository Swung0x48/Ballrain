"""Pool tasks against the learner: the start distribution of the gateway's tasks (server_tasks.PolicySource.pick_start)
compared with BallanceEnv._draw_start built from the same checkpoint (its env_config, curriculum state, exclude_starts):
the share of each start kind (rehearsal, at-rest drive / brake, curriculum, level start), the k quantiles per kind, and
the valid start-frame sets of both sides.

    python tools/native/start_dist.py --ckpt runtime/nc/ckpt/r7_latest.pt --datasets <tas-v1 dir> --path-dirs <dir> [-n 20000]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ballrain.pool import weights as W  # noqa: E402
from ballrain.pool.server_tasks import PolicySource  # noqa: E402


def env_draws(state, datasets: Path, n: int, seed: int):
    """n draws of BallanceEnv._draw_start (no process launched), with the kind of each."""
    from ballrain.curriculum import CurriculumConfig, ReverseCurriculum
    from ballrain.env import BallanceEnv, EnvConfig
    from dataclasses import fields
    ec = dict(state["env_config"])
    known = {f.name for f in fields(EnvConfig)}
    cs = state.get("curriculum") or {}
    cfg = dict(cs.get("config") or {})
    cfg["seed"] = seed
    kw = {k: v for k, v in ec.items() if k in known and k not in ("features", "reward", "headless", "curriculum")}
    kw.update(start_record=str(datasets / ec["start_record"]), curriculum=None, start_frame=None)
    c = EnvConfig(**kw)
    env = object.__new__(BallanceEnv)  # only what _draw_start reads
    env.cfg = c
    from ballrain.curriculum import TasStarts
    env.starts = TasStarts(c.start_record, camera_tol=c.camera_quarter_tol, camera_grace_ms=c.camera_turn_grace_ms)
    if c.exclude_starts:
        v = env.starts.valid
        keep = np.ones(len(v), bool)
        for lo, hi in c.exclude_starts:
            keep &= ~((v >= int(lo)) & (v < int(hi)))
        env.starts.exclude(keep)
    env.curriculum = ReverseCurriculum(env.starts, CurriculumConfig(**{k: v for k, v in cfg.items()
                                                                        if k in CurriculumConfig.__dataclass_fields__}))
    env.curriculum.load_state_dict({k: v for k, v in cs.items() if k in ("frontier", "history", "bin_fail")})
    env._start_rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = env._draw_start()
        r = env._rest_next
        out.append((int(k), "drive" if isinstance(r, tuple) else "brake" if r else None))
    return out, env


def kind_of(k, rest, frontier):
    if rest:
        return "rest_" + rest
    if k == 0:
        return "level"
    return "tas>=frontier" if k >= frontier else "tas<frontier"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--path-dirs", nargs="+", required=True)
    ap.add_argument("-n", type=int, default=20000)
    a = ap.parse_args(argv)
    state = W.load_trusted_checkpoint(a.ckpt)
    with tempfile.TemporaryDirectory() as tmp:
        ps = PolicySource(Path(tmp) / "pool", "dist", a.datasets, a.path_dirs)
        pub = ps.publish_state(state)
        fr = int(pub.curriculum.frontier)
        srv = []
        for _ in range(a.n):
            k, rest = ps.pick_start(pub)
            srv.append((k, None if rest is None else ("drive" if rest["target"] is not None else "brake")))
        envd, env = env_draws(state, Path(a.datasets), a.n, 12345)
        sv, ev = set(pub.valid.tolist()) - {0}, set(env.starts.valid.tolist())
        print(f"frontier {fr}; valid start frames: server {len(sv)}, env {len(ev)}, only server {len(sv - ev)}, "
              f"only env {len(ev - sv)}")
        cs = Counter(kind_of(k, r, fr) for k, r in srv)
        ce = Counter(kind_of(k, r, fr) for k, r in envd)
        print(f"{'kind':16s} {'server':>8s} {'env':>8s}   k quantiles 5/50/95 (server | env)")
        for kd in sorted(set(cs) | set(ce)):
            ks = np.array([k for k, r in srv if kind_of(k, r, fr) == kd])
            ke = np.array([k for k, r in envd if kind_of(k, r, fr) == kd])
            q = lambda x: "-" if not len(x) else "/".join(str(int(v)) for v in np.percentile(x, [5, 50, 95]))
            print(f"{kd:16s} {cs[kd] / a.n:8.4f} {ce[kd] / a.n:8.4f}   {q(ks)} | {q(ke)}")
        # a two-sample check on k over all draws (binned by 250 frames)
        bins = np.arange(0, max(max(k for k, _ in srv), max(k for k, _ in envd)) + 500, 250)
        hs, _ = np.histogram([k for k, _ in srv], bins)
        he, _ = np.histogram([k for k, _ in envd], bins)
        m = (hs + he) > 0
        chi2 = float(np.sum((hs[m] - he[m]) ** 2 / (hs[m] + he[m])))
        print(f"k histogram (250-frame bins): chi2 {chi2:.1f} over {int(m.sum())} bins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
