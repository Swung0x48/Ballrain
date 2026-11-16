"""Rollout worker: one OS process per game copy, collecting fixed-length chunks with its own policy snapshot.

Why not a vector env: gymnasium / SB3 vector envs step all copies in lockstep, so every new game process (3-5 s:
launch, level load, prefix replay) stalls all the others. Measured on L1 curriculum starts with a random policy,
8 copies: 80 decision steps/s in lockstep (AsyncVectorEnv) against 222 steps/s for 8 independent processes
(docs/observations.md 7.1). Each worker here steps its own BallanceEnv (including its resets) and sends chunks of
`chunk` steps to the learner; after each update the learner broadcasts the new weights, the observation
statistics and the curriculum state, which the worker picks up at its next chunk boundary (at most one update
stale; PPO's ratio uses the behaviour log-probs stored with the chunk).

Per-episode limits (worker-side truncation, bootstrapped like a time limit, logged as end "cap" / "stall"):
- frame cap: cap_mult x the start record's remaining time from the episode's start (TAS start k, or the start of the
  sector for a respawn) + cap_extra_s;
- stall: the projection's arc length has not grown by stall_progress units for stall_s seconds of game time.
A forced truncation ends the game process (env.reset() in the middle of a life starts a new process).

Run directories: after a process ends its meta (.json) is read for the run statistics; with prune="finish"
non-finishing runs are deleted and finishing runs keep only the .json, .tas and events.txt (disk: a run
directory is about 1.8 MB, mostly level.msg); prune="none" keeps everything (evaluation).
"""
from __future__ import annotations

import json
import queue
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KEEP_FILES_SUFFIX = (".json", ".tas")
KEEP_FILES = ("events.txt", "trace.txt")


@dataclass
class WorkerSpec:
    wid: int
    copy: str
    env: object  # ballrain.env.EnvConfig (copy / seed set by the learner)
    seed: int
    chunk: int = 128
    hidden: tuple = (256, 256)
    activation: str = "tanh"
    cap_mult: float = 2.5
    cap_extra_s: float = 10.0
    stall_s: float = 10.0
    stall_terminal: bool = False  # a stall ends the episode as a failure (terminal, death penalty + refund) instead of a truncation
    respawn_stall_s: float | None = None  # stall limit of respawn lives (ball at rest at the checkpoint); None = stall_s
    stall_progress: float = 5.0
    prune: str = "finish"  # "finish" | "none"
    trace_sample: float = 0.0  # fraction of game processes run with an agent-mode trace and kept whatever the outcome
    #                            (export spot-checks of an approximate backend: scripts/sim2real_m2.py spotcheck)
    launcher_factory: object = None  # tests (ballrain.fakegame.FakeLauncher)
    max_reset_failures: int = 6
    extra: dict = field(default_factory=dict)


class _Episode:
    def __init__(self, info: dict, env, version: int, cap_frames: int | None):
        st = info.get("start") or {}
        self.d = {"kind": st.get("kind"), "start_frame": int(st.get("frame") or 0), "copy": st.get("copy"),
                  "start_sector": int(info.get("sector") or 0), "start_sr_ms": float(info.get("sr_ms") or 0.0),
                  "version_start": version, "run_id": info.get("run_id"), "new_process": bool(st.get("new_process")),
                  "cap_frames": cap_frames, "t_wall_start": time.time()}
        self.ret = 0.0
        self.steps = 0
        self.max_sector = self.d["start_sector"]
        self.terms: dict = {}
        self.jumps = 0
        self.s0 = env.reward.s
        self.best_s = env.reward.s if env.reward.s is not None else -np.inf
        self.mark_s, self.mark_frame = self.best_s, 0
        self.actions = np.zeros(env.action_space.n, np.int64)

    def step(self, a, r, info, finfo):
        self.ret += r
        self.steps += 1
        self.actions[a] += 1
        self.max_sector = max(self.max_sector, int(info.get("sector") or 0))
        for k, v in (info.get("reward_terms") or {}).items():
            self.terms[k] = self.terms.get(k, 0.0) + float(v)
        p = (finfo or {}).get("path")
        if p and p.get("jump"):
            self.jumps += 1

    def finish(self, info: dict, end: str, env, frames: int) -> dict:
        d = self.d
        d.update(end=end, ret=self.ret, steps=self.steps, frames=frames, max_sector=self.max_sector,
                 end_sector=int(info.get("sector") or 0), sr_ms_end=float(info.get("sr_ms") or 0.0),
                 finish=end == "LevelFinish", terms=self.terms, jumps=self.jumps,
                 s_start=self.s0, s_end=env.reward.s, t_wall=time.time() - d.pop("t_wall_start"),
                 actions=self.actions.tolist())
        d["success_cp"] = bool(d["finish"] or self.max_sector > d["start_sector"])
        return d


class Worker:
    def __init__(self, spec: WorkerSpec, ctrl_q, data_q):
        import torch
        from .ppo import ActorCritic
        torch.set_num_threads(1)
        self.torch = torch
        self.spec = spec
        self.ctrl_q, self.data_q = ctrl_q, data_q
        self.gen = torch.Generator().manual_seed(spec.seed)
        self._trace_rng = np.random.default_rng(spec.seed + 99991)
        self.env = None
        self.fx_info = {}
        self.curriculum_state = None
        self.errors: list = []
        self.pending_runs: list = []
        self._make_env()
        self.model = ActorCritic(self.env.observation_space.shape[0], self.env.action_space.n, spec.hidden,
                                 spec.activation)
        self.model.eval()
        self.version = -1
        self.rms = None
        self.stop = False

    # ---------------------------------------------------------------------------------------- env
    def _make_env(self):
        from ballrain.env import BallanceEnv
        launcher = self.spec.launcher_factory() if self.spec.launcher_factory is not None else None
        self.env = BallanceEnv(self.spec.env, launcher=launcher)
        self.runs_seen = 0
        orig = self.env.fx.compute

        def compute(tracker):
            o, fi = orig(tracker)
            self.fx_info = fi
            return o, fi
        self.env.fx.compute = compute
        st = self.env.starts
        if st is not None:
            self.lf = st.levelfinish_frame or int(st.valid[-1])
            self.rec_ms = st.sr_final / max(self.lf - int(st.valid[0]), 1) if np.isfinite(st.sr_final) else 1000 / 264
        if self.curriculum_state is not None and self.env.curriculum is not None:
            self.env.set_start_state(self.curriculum_state)

    def _cap_frames(self, info) -> int | None:
        st = self.env.starts
        if st is None:
            return None
        s = info.get("start") or {}
        if s.get("kind") == "respawn":
            m = st.sector == int(info.get("sector") or 0)
            k = int(st.valid[m][0]) if m.any() else int(st.valid[0])
        else:
            k = int(s.get("frame") or 0)
        rem_ms = max(self.lf - k, 0) * self.rec_ms
        return int((self.spec.cap_mult * rem_ms + 1000 * self.spec.cap_extra_s) / self.spec.env.delta_ms)

    def _collect_runs(self):
        fr = self.env.finished_runs
        while self.runs_seen < len(fr):
            p = Path(fr[self.runs_seen])
            self.runs_seen += 1
            try:
                m = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"meta {p}: {e}")
                continue
            ss = m.get("start_state") or {}
            self.pending_runs.append({"run_id": m.get("run_id"), "copy": m.get("copy"), "end_reason": m.get("end_reason"),
                                      "sr_ms": m.get("sr_ms"), "start_frame": ss.get("frame", 0),
                                      "frames": m.get("frames"), "lives": len(m.get("lives") or []),
                                      "deaths": sum(1 for x in m.get("lives") or [] if x.get("end_reason") == "BallOff"),
                                      "framelog_match": m.get("framelog_match"),
                                      "handover_mismatches": ss.get("handover_mismatches"),
                                      "invalid": m.get("invalid"),
                                      "dir": str(p.parent)})
            self._prune(p.parent, m.get("end_reason") == "LevelFinish" and not m.get("invalid"))

    def _prune(self, d: Path, finished: bool):
        if self.spec.prune == "none":
            return
        if (d / "trace.txt").is_file():  # a spot-check sample: keep it whatever the outcome (without the level files)
            finished = True
        try:
            if not finished:
                shutil.rmtree(d, ignore_errors=True)
                return
            for f in d.iterdir():
                if f.is_file() and not (f.suffix in KEEP_FILES_SUFFIX or f.name in KEEP_FILES):
                    f.unlink(missing_ok=True)
        except OSError as e:
            self.errors.append(f"prune {d}: {e}")

    def _reset(self):
        fails = 0
        if self.spec.trace_sample:  # read by the env when it launches the next process
            self.env.cfg.trace = bool(self._trace_rng.random() < self.spec.trace_sample)
        while True:
            try:
                obs, info = self.env.reset()
                self._collect_runs()
                return obs, info
            except Exception as e:  # noqa: BLE001  launch failure, hand-over mismatches, crashed game
                fails += 1
                self.errors.append(f"reset: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
                try:
                    self.env.close()
                except Exception as e2:  # noqa: BLE001
                    self.errors.append(f"close: {e2}")
                self._collect_runs()
                if fails >= self.spec.max_reset_failures:
                    raise
                time.sleep(2.0 * fails)
                self._make_env()

    # ---------------------------------------------------------------------------------------- control
    def _apply(self, msg):
        if msg.get("cmd") == "stop":
            self.stop = True
            return
        if "model" in msg:
            self.model.load_state_dict({k: self.torch.as_tensor(v) for k, v in msg["model"].items()})
            self.version = int(msg["version"])
        if "rms" in msg:
            self.rms = msg["rms"]
        if msg.get("curriculum") is not None:
            self.curriculum_state = msg["curriculum"]
            if self.env.curriculum is not None:
                self.env.set_start_state(self.curriculum_state)

    def _drain(self, block=False):
        try:
            if block:
                self._apply(self.ctrl_q.get())
            while True:
                self._apply(self.ctrl_q.get_nowait())
        except queue.Empty:
            pass

    # ---------------------------------------------------------------------------------------- loop
    def run(self):
        from .ppo import normalize
        sp = self.spec
        self._drain(block=True)
        n_obs = self.env.observation_space.shape[0]
        obs, info = self._reset()
        ep = _Episode(info, self.env, self.version, self._cap_frames(info))
        stall_all = int(1000 * sp.stall_s / sp.env.delta_ms)
        stall_rsp = int(1000 * (sp.respawn_stall_s or sp.stall_s) / sp.env.delta_ms)
        while not self.stop:
            T = sp.chunk
            buf = {"obs": np.zeros((T, n_obs), np.float32), "raw": np.zeros((T, n_obs), np.float32),
                   "actions": np.zeros(T, np.int64), "logp": np.zeros(T, np.float32), "values": np.zeros(T, np.float32),
                   "rewards": np.zeros(T, np.float32), "terminated": np.zeros(T, bool), "truncated": np.zeros(T, bool),
                   "trunc_values": np.zeros(T, np.float32)}
            episodes = []
            version = self.version
            t0 = time.perf_counter()
            t_reset = 0.0
            frames = 0
            t = 0
            while t < T:
                x = normalize(obs, self.rms)
                a, logp, v = self.model.act(x, gen=self.gen)
                a = int(a)
                f0 = self.env.session.frames_done if self.env.session is not None else 0
                try:
                    obs2, r, term, trunc, info2 = self.env.step(a)
                except Exception as e:  # noqa: BLE001  should not happen (the env truncates on PeerGone / Timeout)
                    self.errors.append(f"step: {type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
                    if t > 0:
                        buf["truncated"][t - 1] = True
                        buf["trunc_values"][t - 1] = float(v)
                    ep_d = ep.finish(info, "error", self.env, self.env._ep_frames)
                    episodes.append(ep_d)
                    try:
                        self.env.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self._make_env()
                    tr = time.perf_counter()
                    obs, info = self._reset()
                    t_reset += time.perf_counter() - tr
                    ep = _Episode(info, self.env, self.version, self._cap_frames(info))
                    continue
                frames += int(info2.get("frames_done", f0)) - f0
                ep.step(a, r, info2, self.fx_info)
                end = info2.get("episode_end") if (term or trunc) else None
                if not (term or trunc):
                    ef = self.env._ep_frames
                    s = self.env.reward.s
                    if s is not None and s > ep.mark_s + sp.stall_progress:
                        ep.mark_s, ep.mark_frame = s, ef
                    if ep.d["cap_frames"] is not None and ef >= ep.d["cap_frames"]:
                        trunc, end = True, "cap"
                    elif s is not None and ef - ep.mark_frame >= (stall_rsp if ep.d["kind"] == "respawn" else stall_all):
                        trunc, end = True, "stall"
                        if sp.stall_terminal:  # stuck = a failure like a death: penalty + the sector's progress back
                            rc = self.env.reward.cfg
                            pen = -(rc.death_penalty + rc.death_refund * max(self.env.reward.sector_paid, 0.0))
                            r += pen
                            ep.ret += pen
                            ep.terms["stall_penalty"] = ep.terms.get("stall_penalty", 0.0) + pen
                            term, trunc = True, False
                buf["obs"][t], buf["raw"][t] = x, obs
                buf["actions"][t], buf["logp"][t], buf["values"][t] = a, logp, v
                buf["rewards"][t], buf["terminated"][t] = r, term
                buf["truncated"][t] = trunc and not term
                if trunc and not term:
                    buf["trunc_values"][t] = self.model.value(normalize(obs2, self.rms))
                t += 1
                if term or trunc:
                    episodes.append(ep.finish(info2, end or "truncated", self.env, self.env._ep_frames))
                    tr = time.perf_counter()
                    obs, info = self._reset()
                    t_reset += time.perf_counter() - tr
                    ep = _Episode(info, self.env, self.version, self._cap_frames(info))
                else:
                    obs, info = obs2, info2
            buf["last_value"] = self.model.value(normalize(obs, self.rms))
            buf.update(wid=sp.wid, copy=sp.copy, version=version, episodes=episodes, runs=self.pending_runs,
                       backend=getattr(sp.env, "backend", "retail"),
                       errors=self.errors, t_chunk=time.perf_counter() - t0, t_reset=t_reset, frames=frames)
            self.pending_runs, self.errors = [], []
            self.data_q.put(buf)
            self._drain()
        self.env.close()
        self._collect_runs()
        self.data_q.put({"wid": sp.wid, "closed": True, "runs": self.pending_runs, "errors": self.errors})


def worker_main(spec: WorkerSpec, ctrl_q, data_q):
    w = None
    try:
        w = Worker(spec, ctrl_q, data_q)
        w.run()
    except BaseException as e:  # noqa: BLE001
        data_q.put({"wid": spec.wid, "fatal": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"})
        if w is not None and w.env is not None:
            try:
                w.env.close()
            except Exception:  # noqa: BLE001
                pass
