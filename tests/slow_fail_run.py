"""Helper of tests/test_slow_fail.py (not a test module), a copy of tests/worker_identity_run.py whose WorkerSpec takes
extra fields from a JSON argument: run the learner's rollout worker
(ballrain.train.worker.Worker) of the checkout at ROOT for a fixed number of chunks on the fake game, with a TAS start
record and r8's start distribution, and print a JSON digest of everything it produced.

    python tests/slow_fail_run.py ROOT RECORD_DIR OUT_DIR N_CHUNKS [SPEC_JSON]

Only ROOT's ballrain is imported (it goes first on sys.path), so the same script measures main and this branch.
Deterministic: fixed seeds, a fixed model / observation statistics / curriculum state handed over before the first
chunk, and the worker stopped by its data queue after N_CHUNKS (no timing involved).
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT, RECORD, OUT, N = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(), int(sys.argv[4])
EXTRA = __import__("json").loads(sys.argv[5]) if len(sys.argv) > 5 else {}
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ballrain  # noqa: E402

assert Path(ballrain.__file__).resolve().parents[1] == ROOT, ballrain.__file__
from ballrain.curriculum import TasStarts  # noqa: E402
from ballrain.env import EnvConfig  # noqa: E402
from ballrain.fakegame import FakeLauncher  # noqa: E402
from ballrain.features import FeatureConfig  # noqa: E402
from ballrain.train.ppo import ActorCritic, RunningMeanStd  # noqa: E402
from ballrain.train.worker import Worker, WorkerSpec  # noqa: E402

v = TasStarts(str(RECORD)).valid
lo, hi = int(v[1]), int(v[-2])
mid = int(v[len(v) // 2])
cfg = EnvConfig(level=1, world_actions=True, action_repeat=4, start_record=str(RECORD), backend="headless",
                features=FeatureConfig(), max_process_frames=20000, copy="ident-00", out_root=str(OUT), seed=1234,
                curriculum={"start_back": 100, "step": 50, "band": 100, "recent": 0.4, "advance_at": 0.6,
                            "min_episodes": 20, "p_level_start": 0.0, "weighted": 0.5, "bin_frames": 100,
                            "bin_floor": 0.01, "seed": 1234},
                # r8: level-load 0-0 entries and section ranges, drawn behind the frontier
                rehearse_starts=[[0, 0], [0, 0], [lo, mid], [mid, hi]], p_rehearse=0.3, rehearse_behind_frontier=True,
                # r7-style at-rest starts, brake only and with drive targets
                rest_starts=[[int(v[2]), int(v[5])], [int(v[3]), int(v[6]), 0.0, -30.0, 1.0, 120]], p_rest=0.2)
spec = WorkerSpec(wid=0, copy="ident-00", env=cfg, seed=1234, chunk=64, hidden=(32, 32), activation="tanh",
                  stall_terminal=True, stall_compensate=True, stall_s=2.0, respawn_stall_s=2.0, stall_speed=3.0,
                  stall_max_s=6.0, split_k=5.0, split_min_ref_s=1.0, gamma=0.9985, prune="none",
                  p_early_respawn=0.0, respawn_level_starts=True,
                  counter_rand={"lives": {"values": [0.0, 0.1, 0.2, 0.3], "p": [0.25] * 4}, "sr_s": [0.0, 0.3],
                                "points": [-0.05, 0.45]},
                  launcher_factory=lambda: FakeLauncher(backend="headless"), **EXTRA)


class Ctrl:
    """The learner's control queue: one parameter message, then nothing."""

    def __init__(self, msgs):
        self.msgs = list(msgs)

    def get(self, timeout=None):
        if self.msgs:
            return self.msgs.pop(0)
        import queue
        raise queue.Empty

    def get_nowait(self):
        return self.get()


class Data:
    def __init__(self):
        self.items, self.worker = [], None

    def put(self, m):
        self.items.append(m)
        if sum("obs" in x for x in self.items) >= N:
            self.worker.stop = True

    def cancel_join_thread(self):
        pass


torch.manual_seed(0)
probe = __import__("ballrain.env", fromlist=["BallanceEnv"]).BallanceEnv(cfg, launcher=FakeLauncher(backend="headless"))
n_obs = probe.observation_space.shape[0]
probe.close()
model = ActorCritic(n_obs, 9, (32, 32), "tanh")
rms = RunningMeanStd(n_obs)
rng = np.random.default_rng(0)
rms.mean, rms.var = rng.normal(size=n_obs) * 0.1, rng.random(n_obs) + 0.5
st = TasStarts(str(RECORD))
cur = {"frontier": mid, "config": dict(cfg.curriculum), "outcomes": [], "history": [], "bin_fail": {}}
msg = {"version": 3, "rms": rms.state_dict(), "curriculum": cur,
       "model": {k: t.detach().numpy() for k, t in model.state_dict().items()}}
data = Data()
w = Worker(spec, Ctrl([msg]), data)
data.worker = w
w.parent_alive = lambda: True
w.run()

h = hashlib.sha256()
chunks = []
for m in data.items:
    if "obs" not in m:
        continue
    d = {}
    for k in ("raw", "actions", "logp", "values", "rewards", "terminated", "truncated", "trunc_values"):
        d[k] = hashlib.sha256(np.ascontiguousarray(m[k]).tobytes()).hexdigest()[:16]
    d["last_value"] = float(m["last_value"])
    d["frames"] = int(m["frames"])
    keep = ("kind", "start_frame", "start_sector", "end", "end_sector", "ret", "steps", "frames", "finish",
            "success_cp", "respawn_skipped", "new_process", "splits", "terms")
    d["episodes"] = [{k: e.get(k) for k in keep} for e in m["episodes"]]
    chunks.append(d)
print(json.dumps({"chunks": chunks}, sort_keys=True, default=str))
