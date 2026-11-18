"""The gateway's task source: the exported policy version tasks name, and how a task's start is picked
(docs/pool-prototype.md 5).

PolicySource.refresh() takes the newest readable learner checkpoint of the run (read-only, a local copy such as
runtime/m2-train-mac/r6/ckpt_*.pt; the newest file may be mid-copy, so it falls back to older ones), exports its weights
(weights.export_state -> <pool>/weights/<sha>.{npz,json}) and the task assets (the start record's starts pack, the
reference line the policy was trained on -> <pool>/assets/<sha>.npz), and keeps the curriculum state OF THE SAME
checkpoint. A task's start frame is drawn like the trainer draws it: ballrain.curriculum.ReverseCurriculum.sample
over the record's valid start frames after the run's exclude_starts, with that state (frontier band, failure-weighted
bins, k = 0 at the start of the level), from an unpredictable server-side generator; the ball-order seed is a fresh 32-bit
draw. Format 3: the trainer's rehearsal and at-rest draws too (the checkpoint's env_config rehearse_starts / p_rehearse,
rest_starts / p_rest and rehearse_behind_frontier, drawn in BallanceEnv._draw_start's order: rehearsal, at-rest,
curriculum), the at-rest drive target and limit going into the task ("rest"); a life_seed for the per-life draws
(counter_rand, the early-respawn draw) and, when the rollout skips early respawns (p_early_respawn < 1), the current
curriculum frontier (respawn_frontier). The client runs no random draw of its own.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import task as T, weights as W


@dataclass
class Published:
    manifest: dict
    policy_sha: str
    run: str
    record: str
    record_sha256: str
    starts_sha: str
    path_sha: str
    level: int
    valid: np.ndarray  # valid start frames after exclude_starts (with 0)
    curriculum: object  # ReverseCurriculum or None
    source: str  # checkpoint file name (or "state")
    published_at: int
    starts: object = None  # TasStarts of the record with exclude_starts applied (nearest_valid / nearest_any)
    draws: dict | None = None  # the trainer's start draws: rehearse_starts, p_rehearse, rest_starts, p_rest

    @property
    def n_obs(self) -> int:
        return int(self.manifest["arch"]["n_obs"])

    @property
    def n_act(self) -> int:
        return int(self.manifest["arch"]["n_act"])


class PolicySource:
    def __init__(self, pool_dir, run: str, datasets_dir, path_dirs, ckpt_dir=None, ckpt_glob: str = "ckpt_*.pt",
                 clock=time.time):
        self.pool_dir = Path(pool_dir)
        self.run = run
        self.datasets_dir = Path(datasets_dir)
        self.path_dirs = [Path(p) for p in path_dirs]
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.ckpt_glob = ckpt_glob
        self.clock = clock
        self.lock = threading.Lock()
        self.current: Published | None = None
        self.history: dict = {}  # policy_sha -> Published
        self.rng = np.random.Generator(np.random.PCG64(secrets.randbits(128)))
        self.last_error: str | None = None

    @property
    def weights_dir(self) -> Path:
        return self.pool_dir / "weights"

    @property
    def assets_dir(self) -> Path:
        return self.pool_dir / "assets"

    # ------------------------------------------------------------------------------------------ publishing
    def publish_state(self, state: dict, source: str = "state") -> Published:
        """Export a checkpoint dict and its assets; make it the version new tasks name."""
        from ballrain.curriculum import CurriculumConfig, ReverseCurriculum, TasStarts
        m = W.export_state(state, self.weights_dir, self.run)
        record = m["env"]["start_record"]
        if not record or "/" in record or "\\" in record:
            raise ValueError(f"the policy's start record {record!r} is not a dataset record name")
        rd = self.datasets_dir / record
        meta = json.loads((rd / "meta.json").read_text(encoding="utf-8"))
        ssha, _ = T.export_starts_pack(rd, self.assets_dir)
        psha_want = m["features"].get("path_sha256")
        if not psha_want:
            raise ValueError("the policy does not pin its reference line (features.path_sha256)")
        line = T.find_path_asset(int(m["env"]["level"]), psha_want, self.path_dirs)
        psha, _ = T.export_path_asset(line, self.assets_dir)
        excl = m["env"].get("exclude_starts")
        st = TasStarts(str(rd))
        if excl:
            v = st.valid
            keep = np.ones(len(v), bool)
            for lo, hi in excl:
                keep &= ~((v >= int(lo)) & (v < int(hi)))
            st.exclude(keep)  # excluded frames stay hand-over states for at-rest starts with a drive target
        ec = dict(state.get("env_config") or {})
        draws = {k: ec.get(k) for k in ("rehearse_starts", "rehearse_always", "p_rehearse", "rest_starts", "p_rest",
                                         "rehearse_behind_frontier")}
        cu = None
        cs = state.get("curriculum")
        if cs:
            cfg = dict(cs.get("config") or {})
            cfg["seed"] = None
            cu = ReverseCurriculum(st, CurriculumConfig(**{k: v for k, v in cfg.items()
                                                           if k in CurriculumConfig.__dataclass_fields__}))
            cu.load_state_dict({k: v for k, v in cs.items() if k in ("frontier", "history", "bin_fail")})
        pub = Published(manifest=m, policy_sha=m["policy_sha"], run=self.run, record=record,
                        record_sha256=meta["record_sha256"], starts_sha=ssha, path_sha=psha,
                        level=int(m["env"]["level"]), valid=T.valid_start_frames(rd, excl), curriculum=cu,
                        source=source, published_at=int(self.clock()), starts=st, draws=draws)
        with self.lock:
            self.current = pub
            self.history[pub.policy_sha] = pub
        return pub

    def refresh(self, max_tries: int = 3) -> Published | None:
        """Publish the newest readable checkpoint of ckpt_dir if it is not the current one. Returns the current."""
        if self.ckpt_dir is None:
            return self.current
        files = sorted(self.ckpt_dir.glob(self.ckpt_glob))
        for p in reversed(files[-max_tries:]):
            if self.current is not None and self.current.source == p.name:
                return self.current
            try:
                state = W.load_trusted_checkpoint(p)
                return self.publish_state(state, p.name)
            except Exception as e:  # noqa: BLE001  (mid-copy file, missing asset, ...): try the previous one
                self.last_error = f"{p.name}: {type(e).__name__}: {str(e)[:200]}"
        return self.current

    # ------------------------------------------------------------------------------------------ tasks
    def pick_start(self, pub: Published) -> tuple[int, dict | None]:
        """(k, rest): the start frame and the at-rest start spec (protocol TASK "rest", None = a plain start), drawn as
        BallanceEnv._draw_start draws them (same order and distributions, the server's generator)."""
        d, st, rng = pub.draws or {}, pub.starts, self.rng
        rest = None
        with self.lock:
            k = None
            # rehearse_always entries join the rehearsal draw but are never clipped to the frontier (env._draw_start)
            n_rh = len(d.get("rehearse_starts") or [])
            rh = list(d.get("rehearse_starts") or []) + list(d.get("rehearse_always") or [])
            rs = d.get("rest_starts")
            if st is not None and rh and d.get("p_rehearse") and rng.random() < float(d["p_rehearse"]):
                j = int(rng.integers(len(rh)))
                lo, hi = (int(v) for v in rh[j][:2])
                if d.get("rehearse_behind_frontier") and j < n_rh and pub.curriculum is not None and lo > 0:
                    lo = max(lo, int(pub.curriculum.frontier))  # only the part the frontier has passed
                if lo <= hi:
                    k = st.nearest_valid(int(rng.integers(lo, hi + 1)))
            # (a rehearsal range the frontier has not reached falls through, as in the env)
            if k is None and st is not None and rs and d.get("p_rest") and rng.random() < float(d["p_rest"]):
                e = rs[int(rng.integers(len(rs)))]
                lo, hi = e[0], e[1]
                rest = {"target": None}
                if len(e) >= 5:
                    j = float(e[4])
                    rest["target"] = [float(e[2]) + float(rng.uniform(-j, j)), float(e[3]) + float(rng.uniform(-j, j))]
                    if len(e) >= 6:
                        rest["max_frames"] = int(e[5])
                    k = st.nearest_any(int(rng.integers(int(lo), int(hi) + 1)))
                else:
                    k = st.nearest_valid(int(rng.integers(int(lo), int(hi) + 1)))
            elif k is None and pub.curriculum is not None:
                k = int(pub.curriculum.sample(rng))
            elif k is None:
                k = int(pub.valid[int(rng.integers(len(pub.valid)))])
        k = int(k)
        if rest is not None and rest["target"] is not None:
            if k <= 0 or st.expected(k) is None:
                raise ValueError(f"at-rest start frame {k} is not a hand-over state")
        else:
            i = int(np.searchsorted(pub.valid, k))
            if not (i < len(pub.valid) and int(pub.valid[i]) == k):  # never hand out a start the env would snap
                raise ValueError(f"start frame {k} is not valid")
        if rest is not None and k <= 0:
            rest = None
        return k, rest

    def new_task(self, engine: str, now: int, max_steps: int, max_frames: int, deadline_s: int,
                 kind: str = "sample") -> T.Task:
        pub = self.current
        if pub is None:
            raise LookupError("no policy published")
        chunk = int(pub.manifest["rollout"].get("chunk") or 256)
        k, rest = self.pick_start(pub)
        ro = pub.manifest["rollout"]
        # the early-respawn rule needs the frontier the learner compares with (Worker._skip_respawn: off without one)
        rf = int(pub.curriculum.frontier) if pub.curriculum is not None and float(ro.get("p_early_respawn", 1.0)) < 1.0             else None
        return T.Task(task_id=T.new_task_id(), run=pub.run, policy_sha=pub.policy_sha, level=pub.level,
                      record=pub.record, record_sha256=pub.record_sha256, start_frame=k, rest=rest, life_seed=T.new_sample_seed(),
                      respawn_frontier=rf,
                      ball_order_seed=T.new_ball_order_seed(), engine=engine,
                      assets={"starts": pub.starts_sha, "path": pub.path_sha}, chunk_steps=min(chunk, 4096),
                      max_steps=int(max_steps), max_frames=int(max_frames), deadline_s=int(deadline_s),
                      issued_at=int(now), kind=kind, greedy=kind == "eval", sample_seed=T.new_sample_seed())
