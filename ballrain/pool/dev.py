"""Development / test fixtures of the pool: a complete "pool world" on ballrain.fakegame (no game, no torch).

fake_world(root) builds, under root:
- a FakeGame dataset record (the toy level, up-arrow from frame 40: ballrain.dataset through FakeLauncher);
- a synthetic actor-critic (random numpy weights) exported like a learner checkpoint (weights.export_state), with the
  rollout semantics of r6 (split bonus, stall_terminal / stall_compensate, a gate, route probes) and a curriculum
  state, so the gateway's task picker and the verifier run the same code paths as for a real run;
- the starts pack and a dummy reference-line asset (the toy level is synthetic: its line comes from the level itself,
  FeatureConfig.path_sha256 only has to name the asset the task ships).

The gateway's --dev-fake mode and tests/test_pool_*.py use it; nothing here touches runtime/ outside root.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import task as T, weights as W

RECORD = "SR01_0_00_000_132"
D = np.float32(1000 / 132)


class _Tensor:
    """What weights.export_state reads from a torch tensor (detach / cpu / numpy / shape)."""

    def __init__(self, a):
        self.a = np.asarray(a, np.float32)
        self.shape = self.a.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def fake_record(root) -> Path:
    from ballrain import dataset, fakegame, tas
    root = Path(root)
    d = root / "ds" / RECORD
    if (d / "starts.npz").is_file():
        return d
    keys = np.zeros(600, np.uint32)
    keys[40:] = tas.KEY_UP
    rec = root / f"{RECORD}.tas"
    rec.parent.mkdir(parents=True, exist_ok=True)
    tas.save(rec, tas.make_frames(np.full(600, D), keys))
    res = dataset.build_dataset([rec], root / "ds", copies=["pool-dev"], launcher_factory=fakegame.FakeLauncher)
    if not res or not res[0].get("ok"):
        raise RuntimeError(f"fake record: {res}")
    return d


def path_asset(root) -> Path:
    p = Path(root) / "refpaths" / "L01.npz"
    if not p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        n = 64
        z = np.linspace(-5, -200, n)
        T.save_npz_deterministic(p, {"points": np.stack([np.zeros(n), np.full(n, 2.0), z], 1).astype("<f4"),
                                     "sector": np.ones(n, np.uint8), "t_ms": np.linspace(0, 5000, n),
                                     "meta": np.array(json.dumps({"level": 1, "source": "dev"}))})
    return p


def fake_state(record_dir, path_sha: str, hidden=(32, 32), seed=0, rollout: dict | None = None,
               curriculum: dict | None = None) -> dict:
    """A learner-checkpoint-like dict (scripts/train_m2.py save_ckpt layout) for the toy level."""
    from ballrain.env import BallanceEnv, EnvConfig
    from ballrain.fakegame import FakeLauncher
    from ballrain.features import FeatureConfig
    cfg = EnvConfig(level=1, world_actions=True, action_repeat=4, start_record=str(record_dir),
                    features=FeatureConfig(path_sha256=path_sha), max_process_frames=20000)
    probe = BallanceEnv(cfg, launcher=FakeLauncher(backend="headless"), backend="headless")
    n_obs, n_act = probe.observation_space.shape[0], probe.action_space.n
    probe.close()
    rng = np.random.default_rng(seed)
    dims = [n_obs, *hidden]
    model = {}
    for head, n_out in (("pi", n_act), ("v", 1)):
        for i, (a, b) in enumerate(zip(dims, dims[1:] + [n_out])):
            model[f"{head}.{2 * i}.weight"] = _Tensor(rng.normal(size=(b, a)) / np.sqrt(a))
            model[f"{head}.{2 * i}.bias"] = _Tensor(rng.normal(size=b) * 0.1)
    ec = asdict(cfg)
    ec["start_record"] = Path(record_dir).name
    ec["out_root"] = None
    args = {"activation": "tanh", "chunk": 64, "gamma": 0.9985, "cap_mult": 2.5, "stall_s": 2.0, "respawn_stall_s": 2.0,
            "stall_terminal": True, "stall_compensate": True, "stall_speed": 3.0, "stall_max_s": 6.0, "split_k": 5.0,
            "split_min_ref_s": 1.0, "route_probes": json.dumps({"mid": [-5, 5, -80, -40]}),
            "gates": json.dumps({"g1": {"center": [0, 2, -30], "radius": 4.0, "bonus": 0.5}}), "split_extra": "[]"}
    args.update(rollout or {})
    from ballrain.curriculum import TasStarts
    st = TasStarts(str(record_dir))
    cur = curriculum or {"frontier": int(st.valid[len(st.valid) // 2]),
                         "config": {"start_back": 100, "step": 50, "band": 100, "recent": 0.7, "advance_at": 0.5,
                                    "min_episodes": 20, "p_level_start": 0.1, "weighted": 0.0, "bin_frames": 100,
                                    "bin_floor": 0.05, "seed": 1}}
    return {"model": model, "rms": {"mean": rng.normal(size=n_obs) * 0.1, "var": rng.random(n_obs) + 0.5,
                                    "count": 1e5},
            "n_obs": n_obs, "n_act": n_act, "args": args, "env_config": ec, "curriculum": cur,
            "counters": {"update": 7, "steps": 1234}}


def fake_world(root, **state_kw) -> dict:
    """Everything a dev gateway / test needs for the toy level (see the module docstring)."""
    root = Path(root)
    rd = fake_record(root)
    pa = path_asset(root)
    state = fake_state(rd, T.file_sha256(pa), **state_kw)
    return {"root": root, "record_dir": rd, "path_asset": pa, "state": state, "datasets": rd.parent,
            "path_dirs": [pa.parent]}
