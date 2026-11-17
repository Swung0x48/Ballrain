"""Pool task rollout, shared by the player's sampler and the gateway's verifier (docs/pool-prototype.md 3.4).

A task is ONE game process. TaskRunner plays it step by step in the BallanceEnv that task.build_env made, with the
training reward of ballrain.train.worker.Worker.run (split bonus, sub-goal gates, frame cap, stall limit with
stall_terminal / stall_compensate), driven either by the policy (sample_task: the player's machine) or by the actions a
chunk carries (replay_task: the verifier). The worker's helpers are reused as they are (_Episode and the Worker methods
bound to a small stand-in object), so the semantics cannot drift apart; ballrain/train/worker.py is not modified.

Rules that differ from the worker, because a task is one process that the verifier must be able to replay:
- the task ends whenever the worker would start a new process: LevelFinish, GameOver, an env truncation, a camera
  guard violation, the worker's cap / stall truncation (including stall_terminal), the task's max_steps / max_frames
  (game frames since the hand-over, respawn fast-forwards included); end names as end_name();
- after a BallOff the respawn runs in the same process (env.reset(options={"relaunch": False})); when the process
  ends during the respawn (e.g. GameOver) the task ends there, nothing is relaunched;
- the observation rows of a task are the observation before each step plus the one after the last step; after a
  death in the middle of a task the next row is the respawn observation (the terminal observation of the dead life is
  not a row: a terminated step needs no bootstrap);
- no client-side random draws apart from the action sampling (task.build_env_config switches them off);
- ends the client declares (protocol.CLIENT_ENDS) cannot be reproduced and are accepted only on the last step of the
  final chunk: "stopped" / "deadline" are decided after a step, before a respawn (the replay is told to stop there
  too); "error" means the engine failed on the step after the last one (the replay just runs the last step);
- these task-level decisions follow ballrain.pool.client_sampler.TaskSampler.run_task (the controller's sampler) in
  the same order, so the verifier reproduces the controller's chunks; tests/test_pool_verifier.py checks the two
  against each other.

Per step the chunk carries: action, training reward (float32, bit-exact on replay), behaviour log-prob and value
(float32, checked with a tolerance: different CPUs / BLAS round differently), flags, and the optional env reward,
sector, frames, IGT and arc length (bit-exact on replay).
"""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from . import protocol as P
from .reasons import Reason
from .sampling import action_consistent, sample_u

LOGP_TOL = 1e-3  # |logp - replayed logp| allowed (float32 MLP on another CPU / BLAS)
VALUE_TOL = 1e-3  # |v - replayed v| <= VALUE_TOL * max(1, |v|)


class TaskEnded(RuntimeError):
    pass


class EngineFailure(RuntimeError):
    """The local engine failed (launch, hand-over, crash): the client ends the task with "error", the verifier
    reports replay_failed (not the player's fault)."""


# ============================================================================================== worker semantics

def _worker():
    from ballrain.train import worker as W  # module level of worker.py imports no torch
    return W


class _Shaping:
    """Stand-in for ballrain.train.worker.Worker: the attributes its reward helpers read (env, spec, lf, rec_ms)."""

    def __init__(self, env, spec):
        W = _worker()
        self.env, self.spec = env, spec
        st = env.starts
        if st is not None:  # as Worker._make_env
            self.lf = st.levelfinish_frame or int(st.valid[-1])
            self.rec_ms = st.sr_final / max(self.lf - int(st.valid[0]), 1) if np.isfinite(st.sr_final) else 1000 / 264
        for name in ("_tas_igt", "_tas_cross", "_tas_cross_frame", "_split_start", "_split_bonus", "_cap_frames"):
            setattr(self, name, getattr(W.Worker, name).__get__(self))


def worker_spec(env, manifest: dict):
    """WorkerSpec with the manifest's rollout fields (the run's sampling semantics) for this env."""
    W = _worker()
    names = {f.name for f in fields(W.WorkerSpec)}
    ro = {k: v for k, v in (manifest.get("rollout") or {}).items() if k in names}
    unknown = set(manifest.get("rollout") or {}) - names
    if unknown:
        raise ValueError(f"rollout fields {sorted(unknown)} unknown to WorkerSpec")
    return W.WorkerSpec(wid=0, copy=env.cfg.copy, env=env.cfg, seed=0, prune="none", **ro)


def end_name(info: dict, worker_end: str | None) -> str:
    """The task's end name when the process ends with a step (client_sampler._end_name): the worker's cap / stall,
    LevelFinish, GameOver, else "truncated" (env truncations, the camera guard, a death that ends the process)."""
    if worker_end in ("cap", "stall"):
        return worker_end
    names = {n for _, n in info.get("events") or ()}
    ee = info.get("episode_end")
    if ee == "LevelFinish" or "LevelFinish" in names:
        return "LevelFinish"
    if ee == "GameOver" or "GameOver" in names:
        return "GameOver"
    return "truncated"


@dataclass
class StepResult:
    action: int
    reward: np.float32  # training reward (worker semantics)
    env_reward: np.float32
    flags: int  # FLAG_TERMINATED | FLAG_TRUNCATED
    frames: int  # game frames the step advanced
    sector: int
    sr_ms: float
    arc: float  # nan: no projection
    next_obs: np.ndarray  # the next observation row (respawn observation after a death)
    end: str | None  # set when the task ended with this step (protocol.END_REASONS)


class TaskRunner:
    """One task in an env from task.build_env. start() -> first observation row; step(a) -> StepResult until
    .end is set. The owner closes the env (close())."""

    def __init__(self, env, task, manifest: dict):
        W = _worker()
        self.W = W
        self.env, self.task, self.manifest = env, task, manifest
        self.spec = worker_spec(env, manifest)
        self.sh = _Shaping(env, self.spec)
        self.fx_info: dict = {}
        orig = env.fx.compute

        def compute(tracker):  # as Worker._make_env: the step's feature info (path jumps, arc length)
            o, fi = orig(tracker)
            self.fx_info = fi
            return o, fi
        env.fx.compute = compute
        sp = self.spec
        self.stall_all = int(1000 * sp.stall_s / sp.env.delta_ms)
        self.stall_rsp = int(1000 * (sp.respawn_stall_s or sp.stall_s) / sp.env.delta_ms)
        self.stall_max = int(1000 * sp.stall_max_s / sp.env.delta_ms)
        self.steps = 0
        self.frames = 0  # game frames since the hand-over (respawn fast-forwards included)
        self.f_hand = 0
        self.end: str | None = None
        self.ep = None
        self.obs = None
        self.info = None
        self.episodes = 0

    # ------------------------------------------------------------------------------------------ episodes
    def _new_episode(self, info):
        self.ep = self.W._Episode(info, self.env, 0, self.sh._cap_frames(info))
        self.sh._split_start(self.ep, info)
        self.episodes += 1

    def start(self) -> np.ndarray:
        try:
            obs, info = self.env.reset()
        except Exception as e:  # noqa: BLE001  launch failure, hand-over mismatch, crashed engine
            raise EngineFailure(f"start: {type(e).__name__}: {str(e)[:300]}") from e
        self.obs, self.info = np.asarray(obs, np.float32), info
        self.f_hand = int(info.get("frames_done") or 0)
        self._new_episode(info)
        return self.obs

    # ------------------------------------------------------------------------------------------ one step
    def step(self, a: int, stop=None) -> StepResult:
        """Worker.run's step with its reward shaping (same order of operations, same float arithmetic), then the
        task-level end. stop: None, an end name or a callable returning one ("stopped" / "deadline"), consulted
        after the step when nothing else ended the task (before a respawn, so a stopped task has no respawn row)."""
        if self.end is not None:
            raise TaskEnded(self.end)
        env, sp, ep = self.env, self.spec, self.ep
        a = int(a)
        f0 = env.session.frames_done if env.session is not None else 0
        try:
            obs2, r, term, trunc, info2 = env.step(a)
        except Exception as e:  # noqa: BLE001
            raise EngineFailure(f"step: {type(e).__name__}: {str(e)[:300]}") from e
        if info2.get("error"):  # BallanceEnv.step caught PeerGone / Timeout and truncated: an engine failure, not a step
            raise EngineFailure(f"step: {str(info2['error'])[:300]}")
        env_r = r
        frames = int(info2.get("frames_done", f0)) - f0
        ep.step(a, r, info2, self.fx_info)
        if sp.split_k and (int(info2.get("sector") or 0) > ep.max_sector_seen
                           or info2.get("episode_end") == "LevelFinish"):
            sb = self.sh._split_bonus(ep, info2, info2.get("episode_end") == "LevelFinish")
            if sb:
                r += sb
                ep.ret += sb
        ep.max_sector_seen = max(ep.max_sector_seen, int(info2.get("sector") or 0))
        if (sp.route_probes or sp.gates) and env.session is not None:
            o = env.session.tracker.obs
            if o is not None and o.ball is not None and bool(o.ball["found"]):
                bpos = np.asarray(o.ball["pos"], np.float64)
                if sp.route_probes:
                    ep.probe(bpos, sp.route_probes)
                if sp.gates and not term:
                    gb = ep.gate(bpos, np.asarray(o.ball["speed"], np.float64)
                                 + np.asarray(o.ball["speed_change"], np.float64), sp.gates)
                    if gb:
                        r += gb
                        ep.ret += gb
        end = info2.get("episode_end") if (term or trunc) else None
        if not (term or trunc):
            ef = env._ep_frames
            s = env.reward.s
            if s is not None and s > ep.mark_s + sp.stall_progress:
                ep.mark_s, ep.mark_frame = s, ef
            win = self.stall_rsp if ep.d["kind"] == "respawn" else self.stall_all
            since = ef - ep.mark_frame
            if sp.stall_speed is not None:
                o = env.session.tracker.obs if env.session is not None else None
                if o is not None and o.ball is not None and float(np.linalg.norm(o.ball["speed"])) > sp.stall_speed:
                    ep.move_frame = ef
                since = min(since, ef - ep.move_frame)
                if ef - ep.mark_frame >= self.stall_max:
                    since = win
            if ep.d["cap_frames"] is not None and ef >= ep.d["cap_frames"]:
                trunc, end = True, "cap"
            elif s is not None and since >= win:
                trunc, end = True, "stall"
                if sp.stall_terminal:
                    rc = env.reward.cfg
                    pen = -(rc.death_penalty + rc.death_refund * max(env.reward.sector_paid, 0.0) + sp.stall_extra)
                    if sp.stall_compensate:
                        win = self.stall_rsp if ep.d["kind"] == "respawn" else self.stall_all
                        pen /= sp.gamma ** (win / max(sp.env.action_repeat, 1))
                    r += pen
                    ep.ret += pen
                    term, trunc = True, False
        # ---- task-level end (the same decisions, in the same order, as client_sampler.TaskSampler.run_task)
        self.steps += 1
        respawn = bool(term and end != "stall" and getattr(env, "_after_death", False))
        flags = P.FLAG_TERMINATED if term else (P.FLAG_TRUNCATED if trunc else 0)
        task_end = None
        if trunc or (term and not respawn):
            task_end = end_name(info2, end)
        else:
            if self.steps >= int(self.task.max_steps):
                task_end = "max_steps"
            elif int(info2.get("frames_done") or 0) - self.f_hand >= int(self.task.max_frames):
                task_end = "max_frames"
            elif stop is not None:
                task_end = stop() if callable(stop) else stop
            if task_end and not term:
                flags = P.FLAG_TRUNCATED
        self.frames = int(info2.get("frames_done") or 0) - self.f_hand
        nxt = np.asarray(obs2, np.float32)
        if task_end is None and term:  # a respawn in the same process (or the process ends during it)
            try:
                o_n, i_n = env.reset(options={"relaunch": False})
            except Exception as e:  # noqa: BLE001
                raise EngineFailure(f"respawn: {type(e).__name__}: {str(e)[:300]}") from e
            if o_n is None:
                task_end = "GameOver"
            else:
                nxt = np.asarray(o_n, np.float32)
                self.info = i_n
                self._new_episode(i_n)
        # arc length: the feature info of the next row (after a respawn: the respawn's), as the client records it
        p = (self.fx_info or {}).get("path") or {}
        arc = float(p["s"]) if p.get("s") is not None else float("nan")
        self.end = task_end or None
        self.obs = nxt
        return StepResult(action=a, reward=np.float32(r), env_reward=np.float32(env_r), flags=int(flags),
                          frames=int(frames), sector=int(info2.get("sector") or 0),
                          sr_ms=float(info2.get("sr_ms") or 0.0), arc=arc, next_obs=nxt, end=self.end)

    def close(self):
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass


def remove_run_dirs(env) -> None:
    """Delete the run directories the env wrote (a pool task keeps nothing locally: about 1.8 MB each)."""
    for meta in list(getattr(env, "finished_runs", ())):
        shutil.rmtree(Path(meta).parent, ignore_errors=True)


# ============================================================================================== chunks

class ChunkAssembler:
    """Collects a task's steps and cuts them into encoded chunks (protocol.encode_chunk) of task.chunk_steps steps:
    rows step0 .. step0 + steps, chain_prev = h(step0 - 1), the final chunk with its end reason."""

    OPTIONAL = ("env_rewards", "sector", "frames", "sr_ms", "arc")

    def __init__(self, task, n_obs: int, n_act: int, policy_sha: str):
        self.task, self.n_obs, self.n_act, self.policy_sha = task, int(n_obs), int(n_act), policy_sha
        self.prev = P.chain_seed(task.task_id)
        self.seq = 0
        self.step0 = 0
        self.rows: list = []
        self.values: list = []
        self.cols = {k: [] for k in ("actions", "rewards", "logp", "flags") + self.OPTIONAL}
        self.wall0 = None

    def begin(self, obs, value: float):
        self.rows = [np.asarray(obs, "<f4")]
        self.values = [float(value)]

    def add(self, logp: float, st: StepResult, next_value: float):
        c = self.cols
        c["actions"].append(st.action)
        c["rewards"].append(st.reward)
        c["logp"].append(float(np.clip(logp, P.MIN_LOGP, 0.0)))
        c["flags"].append(st.flags)
        c["env_rewards"].append(st.env_reward)
        c["sector"].append(min(max(st.sector, 0), 255))
        c["frames"].append(min(max(st.frames, 0), 65535))
        c["sr_ms"].append(st.sr_ms)
        c["arc"].append(st.arc)
        self.rows.append(np.asarray(st.next_obs, "<f4"))
        self.values.append(float(next_value))

    @property
    def pending(self) -> int:
        return len(self.cols["actions"])

    def ready(self) -> bool:
        return self.pending >= int(self.task.chunk_steps)

    def cut(self, end: str | None = None, stats: dict | None = None) -> bytes:
        """Encode the collected steps as the next chunk (final when end is set)."""
        T = self.pending
        if T == 0:
            raise ValueError("no steps to cut")
        rows = np.stack(self.rows).astype("<f4")
        hashes = P.chain_hashes(rows, self.prev)
        c = self.cols
        flags = np.asarray(c["flags"], np.uint8)
        if end is not None and not flags[-1]:
            flags[-1] = P.FLAG_TRUNCATED
        arrays = {"actions": np.asarray(c["actions"], np.uint8), "rewards": np.asarray(c["rewards"], "<f4"),
                  "logp": np.asarray(c["logp"], "<f4"), "values": np.asarray(self.values, "<f4"), "flags": flags,
                  "obs": rows.astype("<f2"), "obs_hash": hashes,
                  "env_rewards": np.asarray(c["env_rewards"], "<f4"), "sector": np.asarray(c["sector"], np.uint8),
                  "frames": np.asarray(c["frames"], "<u2"), "sr_ms": np.asarray(c["sr_ms"], "<f8"),
                  "arc": np.asarray(c["arc"], "<f4")}
        header = {"task_id": self.task.task_id, "seq": self.seq, "step0": self.step0, "steps": T,
                  "n_obs": self.n_obs, "n_act": self.n_act, "policy_sha": self.policy_sha,
                  "chain_prev": self.prev.hex(), "final": end is not None, "end": end}
        if stats:
            header["stats"] = stats
        data = P.encode_chunk(header, arrays)
        self.prev = bytes(hashes[-2])
        self.seq += 1
        self.step0 += T
        self.rows, self.values = [self.rows[-1]], [self.values[-1]]
        self.cols = {k: [] for k in self.cols}
        return data


def sample_task(runner: TaskRunner, policy, rng: np.random.Generator | None, on_chunk, stop=lambda: False,
                deadline=None, clock=None) -> str:
    """The player's sampling loop: play the task with the policy (the task's deterministic draws when rng is None --
    what a gateway accepts --, rng draws otherwise; argmax for greedy tasks), hand
    every encoded chunk to on_chunk(bytes) as soon as it is complete; returns the end reason. stop() -> True ends the
    task with "stopped"; deadline (a clock() value) with "deadline"; an engine failure with "error" (the steps so
    far are sent)."""
    import time
    clock = clock or time.time
    task = runner.task
    asm = ChunkAssembler(task, policy.n_obs, policy.n_act, policy.sha)
    obs = runner.start()
    x = policy.normalize(obs)
    asm.begin(obs, float(policy.value(x)))
    end = None
    def _stop():
        if stop():
            return "stopped"
        if deadline is not None and clock() >= deadline:
            return "deadline"
        return None

    from .sampling import sample_u
    while end is None:
        u = None if rng is not None else sample_u(task.sample_seed, runner.steps)
        a, logp, _ = policy.act(x, rng, greedy=bool(task.greedy), u=u)
        try:
            st = runner.step(a, stop=_stop)
        except EngineFailure:
            end = "error"
            break
        x = policy.normalize(st.next_obs)
        asm.add(logp, st, float(policy.value(x)))
        end = st.end
        if end is None and asm.ready():
            on_chunk(asm.cut())
    if asm.pending:
        on_chunk(asm.cut(end))
    return end


# ============================================================================================== replay

@dataclass
class Mismatch:
    reason: Reason
    seq: int  # chunk
    step: int  # task step (row index for an observation)
    detail: str = ""

    def params(self, task_id: str) -> dict:
        return {"task_id": task_id, "chunk": int(self.seq), "step": int(self.step)}


def _bits_equal(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def replay_task(runner: TaskRunner, policy, chunks: list, upto: int | None = None) -> Mismatch | None:
    """Verifier: replay a task's chunks (decoded protocol.Chunk, seq 0, 1, ... consecutive) in runner's env and compare
    step by step; returns the first Mismatch (earliest step; within a step: observation, log-prob, value, then the
    step's reward, flags and optional arrays) or None. upto: replay chunks 0..upto only. Raises EngineFailure when the
    local engine fails (replay_failed, not the player's fault)."""
    if upto is not None:
        chunks = chunks[:upto + 1]
    task = runner.task
    prev = P.chain_seed(task.task_id)
    row = runner.start()
    greedy = bool(task.greedy)
    for c in chunks:
        h, A = c.header, c.arrays
        s, T, step0 = int(h["seq"]), int(h["steps"]), int(h["step0"])
        last_chunk = bool(h["final"])
        if h["chain_prev"] != prev.hex() or step0 != runner.steps:
            return Mismatch(Reason.REPLAY_MISMATCH, s, step0, "chunk does not continue the replay")
        for j in range(T + 1):
            i = step0 + j
            # ---- observation row i (chain value and float16 copy)
            r32 = np.asarray(row, "<f4")
            hv = hashlib.sha256(prev + r32.tobytes()).digest()
            if hv != bytes(A["obs_hash"][j]) or not _bits_equal(r32.astype("<f2"), A["obs"][j]):
                return Mismatch(Reason.REPLAY_MISMATCH, s, i, "observation")
            x = policy.normalize(r32)
            v = float(policy.value(x))
            if abs(v - float(A["values"][j])) > VALUE_TOL * max(1.0, abs(v)):
                return Mismatch(Reason.REPLAY_VALUE_MISMATCH, s, i, "value")
            if j == T:  # the row after the chunk's last step (= the next chunk's first row)
                break
            prev = hv
            a = int(A["actions"][j])
            lp = policy.log_probs(x)
            if abs(float(np.clip(lp[a], P.MIN_LOGP, 0.0)) - float(A["logp"][j])) > LOGP_TOL:
                return Mismatch(Reason.REPLAY_LOGP_MISMATCH, s, i, "logp")
            if greedy and float(lp[a]) < float(lp.max()) - LOGP_TOL:
                return Mismatch(Reason.REPLAY_LOGP_MISMATCH, s, i, "greedy task, not the argmax action")
            if not greedy and not action_consistent(lp, sample_u(task.sample_seed, i), a):
                return Mismatch(Reason.REPLAY_ACTION_MISMATCH, s, i, "not the action of the task's sample_seed")
            if runner.end is not None:  # the submitted task goes on after the replay ended
                return Mismatch(Reason.REPLAY_MISMATCH, s, i, f"the replay ended ({runner.end})")
            is_last = last_chunk and j == T - 1
            # a stop / deadline the client declared after this step: the runner ends the task there too (no respawn)
            st = runner.step(a, stop=h["end"] if is_last and h["end"] in ("stopped", "deadline") else None)
            # ---- the step's results
            if not _bits_equal(np.float32(st.reward), A["rewards"][j]):
                return Mismatch(Reason.REPLAY_REWARD_MISMATCH, s, i, "reward")
            sub_flag = int(A["flags"][j])
            if is_last and h["end"] == "error":  # the client's engine failed on the NEXT step: this one went on
                if st.end is not None or not (st.flags == sub_flag or (st.flags == 0 and sub_flag == P.FLAG_TRUNCATED)):
                    return Mismatch(Reason.REPLAY_MISMATCH, s, i, "flags")
            else:
                if st.flags != sub_flag:
                    return Mismatch(Reason.REPLAY_MISMATCH, s, i, "flags")
                if is_last and st.end != h["end"]:
                    return Mismatch(Reason.REPLAY_MISMATCH, s, i, f"end {h['end']} != replay {st.end}")
            if not is_last and st.end is not None:
                return Mismatch(Reason.REPLAY_MISMATCH, s, i, f"the replay ended ({st.end})")
            for name, got in (("env_rewards", np.float32(st.env_reward)), ("sector", np.uint8(min(max(st.sector, 0), 255))),
                              ("frames", np.dtype("<u2").type(min(max(st.frames, 0), 65535))),
                              ("sr_ms", np.float64(st.sr_ms)), ("arc", np.float32(st.arc))):
                if name in A and not _bits_equal(np.asarray(got, A[name].dtype), A[name][j]):
                    return Mismatch(Reason.REPLAY_MISMATCH, s, i, name)
            row = st.next_obs
    return None
