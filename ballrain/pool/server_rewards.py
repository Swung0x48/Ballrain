"""The training reward, recomputed by the gateway from the chunk's reward inputs (reward_inputs_v1).

The client uploads what the reward and its shaping read, per step (chunk array "rin", RIN_* columns below; header
"episodes": each episode's start), instead of a reward the gateway would have to trust or replay. The gateway runs
the trainer's own code over them: ballrain.rewards.Reward (env reward: progress, time, finish, death and its refund,
the sector-3 terms) and ballrain.train.worker.Worker's split bonus, plus the stall and camera penalties of
Worker.run, so a change of the reward or its shaping needs no client update. The verifier no longer compares a
client-reported reward; it checks the inputs (they come from the engine state and the features) when it replays.

Supported: SR rewards; rollout gates / route_probes / zones need the client's reward (refuse_reason()).
One TaskRewards per task carries the state from chunk to chunk (chunks arrive in order: seq).
"""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np

RIN_COLS = 17
RIN_SR, RIN_S, RIN_CHECKPOINTS, RIN_SECTOR, RIN_BALL_TYPE, RIN_FLAGS = 0, 1, 2, 3, 4, 5
RIN_POS, RIN_POINT, RIN_TANGENT, RIN_REF_ELAPSED, RIN_STALL_WIN = 6, 9, 12, 15, 16
DIED, FINISHED, TERMINATED, REBASE, SUPPORTED, CAM_BAD, STALL_PEN, ROUTE_BRANCH, LEVEL_FINISH, HAS_POS, HAS_POINT = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def refuse_reason(manifest: dict) -> str | None:
    """Why this policy's reward cannot be recomputed from the inputs (None: it can)."""
    if (manifest.get("reward") or {}).get("mode", "sr") != "sr":
        return "HS rewards"
    ro = manifest.get("rollout") or {}
    for k in ("gates", "route_probes", "zones"):
        if ro.get(k):
            return f"rollout {k}"
    return None


class _Tracker:
    """What ballrain.rewards.Reward reads from a GameTracker (SR mode)."""

    def __init__(self):
        self.sr_ms, self.checkpoints, self.sector, self._ball_type = None, 0, 0, 0

    def ball_type(self):
        return self._ball_type

    def hs_partial(self):  # (HS rewards only; Reward.begin reads it in every mode)
        return 0.0

    def followers(self):
        return 0


class TaskRewards:
    """The reward state of one task across its chunks."""

    def __init__(self, manifest: dict, starts):
        from ballrain.rewards import Reward, RewardConfig
        from ballrain.train import worker as W
        from dataclasses import fields

        rc = manifest.get("reward") or {}
        self.reward = Reward(RewardConfig(**{f.name: rc[f.name] for f in fields(RewardConfig) if f.name in rc}))
        ro = manifest.get("rollout") or {}
        names = {f.name for f in fields(W.WorkerSpec)}
        env = manifest.get("env") or {}
        self.spec = SimpleNamespace(**{f.name: f.default for f in fields(W.WorkerSpec) if f.name in names
                                       and not callable(f.default)})
        for k, v in ro.items():
            setattr(self.spec, k, v)
        self.action_repeat = max(int(env.get("action_repeat") or 1), 1)
        # Worker's split methods bound to a stand-in with the task's start record (rollout._Shaping does the same)
        self.sh = SimpleNamespace(env=SimpleNamespace(starts=starts), spec=self.spec)
        st = starts
        if st is not None:
            self.sh.lf = st.levelfinish_frame or int(st.valid[-1])
            self.sh.rec_ms = (st.sr_final / max(self.sh.lf - int(st.valid[0]), 1) if np.isfinite(st.sr_final)
                              else 1000 / 264)
        for name in ("_tas_igt", "_tas_cross", "_tas_cross_frame", "_split_start", "_split_bonus"):
            setattr(self.sh, name, getattr(W.Worker, name).__get__(self.sh))
        self.t = _Tracker()
        self.ep = None
        self.next_step = 0
        self.seconds = 0.0  # time spent in recompute (the gateway's cost)

    def _begin(self, e: dict):
        t = self.t
        t.sr_ms, t.checkpoints, t.sector = e["sr_ms"], int(e["checkpoints"]), int(e["sector"])
        self.reward.begin(t, e.get("s"))
        self.ep = SimpleNamespace(d={"kind": e["kind"], "start_frame": int(e["frame"]), "start_sector": int(e["sector"]),
                                     "start_sr_ms": float(e["sr_ms"])}, seg=None, splits=[],
                                  max_sector_seen=int(e["sector"]))
        info = {"sector": int(e["sector"]), "sr_ms": e["sr_ms"]}
        if e.get("ref_elapsed_ms") is not None:
            info["ref_elapsed_ms"] = e["ref_elapsed_ms"]
        self.sh._split_start(self.ep, info)

    def chunk(self, step0: int, rin: np.ndarray, episodes: list) -> np.ndarray:
        """The training rewards of a chunk's steps (float32, as the client's rewards array)."""
        t0 = time.perf_counter()
        if step0 != self.next_step:
            raise ValueError(f"reward inputs out of order: chunk at step {step0}, expected {self.next_step}")
        rin = np.asarray(rin, np.float64)
        if rin.ndim != 2 or rin.shape[1] != RIN_COLS:
            raise ValueError("reward inputs: shape")
        starts = {int(e["step"]): e for e in episodes or ()}
        out = np.empty(len(rin), np.float32)
        sp, R, t = self.spec, self.reward, self.t
        for i, q in enumerate(rin):
            step = step0 + i
            if step in starts:
                self._begin(starts[step])
            if self.ep is None:
                raise ValueError("reward inputs: a step before its episode's start")
            f = int(q[RIN_FLAGS])
            t.sr_ms = None if math.isnan(q[RIN_SR]) else float(q[RIN_SR])
            t.checkpoints, t.sector, t._ball_type = int(q[RIN_CHECKPOINTS]), int(q[RIN_SECTOR]), int(q[RIN_BALL_TYPE])
            s = None if math.isnan(q[RIN_S]) else float(q[RIN_S])
            s3 = None
            if f & HAS_POS:
                s3 = {"pos": [float(v) for v in q[RIN_POS:RIN_POS + 3]],
                      "path": ({"point": [float(v) for v in q[RIN_POINT:RIN_POINT + 3]],
                                "tangent": [float(v) for v in q[RIN_TANGENT:RIN_TANGENT + 3]],
                                "branch": bool(f & ROUTE_BRANCH)} if f & HAS_POINT else {})}
            r, _ = R.step(t, s, bool(f & DIED), bool(f & FINISHED), bool(f & TERMINATED), bool(f & REBASE),
                          bool(f & SUPPORTED), s3)
            if f & CAM_BAD:  # env.step: no credit for the episode, its return becomes -death_penalty
                pen = -R.total - R.cfg.death_penalty
                r += pen
                R.total += pen
            sector = t.sector
            lf = bool(f & LEVEL_FINISH)
            if sp.split_k and (sector > self.ep.max_sector_seen or lf):
                info = {"sector": sector, "sr_ms": t.sr_ms}
                if not math.isnan(q[RIN_REF_ELAPSED]):
                    info["ref_elapsed_ms"] = float(q[RIN_REF_ELAPSED])
                r += self.sh._split_bonus(self.ep, info, lf)
            self.ep.max_sector_seen = max(self.ep.max_sector_seen, sector)
            if f & STALL_PEN:  # Worker.run: the stall ends the episode as a death
                rc = R.cfg
                pen = -(rc.death_penalty + rc.death_refund * max(R.sector_paid, 0.0) + sp.stall_extra)
                if sp.stall_compensate:
                    pen /= sp.gamma ** (float(q[RIN_STALL_WIN]) / self.action_repeat)
                r += pen
            out[i] = np.float32(r)
        self.next_step = step0 + len(rin)
        self.seconds += time.perf_counter() - t0
        return out


def rewards_path(chunk_file):
    """<task>/<seq>.rewards.npy next to the chunk file <task>/<seq>.brpc."""
    from pathlib import Path
    p = Path(chunk_file)
    return p.with_name(p.stem + ".rewards.npy")


def write_rewards(chunk_file, rewards: np.ndarray) -> None:
    import secrets
    dst = rewards_path(chunk_file)
    tmp = dst.with_name(f".{dst.name}.{secrets.token_hex(4)}.tmp.npy")
    np.save(tmp, np.asarray(rewards, "<f4"), allow_pickle=False)
    tmp.replace(dst)


def read_rewards(chunk_file, steps: int):
    """The gateway's rewards of a chunk (float32 (steps,)), or None."""
    p = rewards_path(chunk_file)
    if not p.is_file():
        return None
    a = np.load(p, allow_pickle=False)
    if a.dtype != np.dtype("<f4") or a.shape != (steps,) or not np.isfinite(a).all():
        raise ValueError("server rewards file")
    return a
