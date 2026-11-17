"""The pool sampling loop: one task = one game process, played with the task's policy version, cut into protocol chunks
(docs/pool-prototype.md section 3 and 9).

Training reward semantics are the learner's own: TaskSampler subclasses ballrain.train.worker.Worker only to reuse its
per-episode helpers unchanged (_Episode, frame cap, split bonus, TAS reference times, run pruning), and run_task() is a
port of Worker.run's per-step block (split bonus, gates, route probes, cap / stall truncation, stall_terminal penalty)
with the WorkerSpec fields taken from the policy manifest's "rollout". No torch: the policy is weights.NumpyPolicy.

Task-level rules on top of the worker's (decided after every step, before the step's chunk is closed, so the last step
of any chunk can become the final one):
- a death with a life left respawns in the same process (env.reset(options={"relaunch": False}): never a second
  process); if the respawn fast-forward ends the process (GameOver), the task ends there;
- any truncation (the env's, or the worker's cap / stall) and any terminal end of the process (LevelFinish, GameOver,
  stall_terminal, the camera guard) ends the task;
- max_steps, max_frames (game frames since the hand-over), the local deadline and the player's pause / stop end the
  task with a truncated last step (end "max_steps" / "max_frames" / "deadline" / "stopped").
End names (protocol.END_REASONS): LevelFinish, GameOver, cap, stall, max_steps, max_frames, deadline, stopped, error;
every other way the environment ends the process (PeerGone / timeout truncation, max_episode_frames, the
camera_off_quarter guard) is "truncated" -- the step's flag still says whether it was terminal.

The action source is pluggable (PolicyActions samples at temperature 1 or argmax for eval tasks; ReplayActions feeds
recorded actions), so a verifier can replay a task through the same code.
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

from . import protocol as P
from .weights import ROLLOUT_FIELDS


class TaskAborted(Exception):
    """The task cannot go on (a refusal, a stop before the first step, an engine failure without a step to end on)."""

    def __init__(self, msg: str, refusal=None):
        super().__init__(msg)
        self.refusal = refusal


# ============================================================================================== action sources

class PolicyActions:
    """The task's policy: raw observation -> (action, behaviour log-prob, value); temperature 1 sampling, argmax for
    greedy (eval) tasks. With seed (the task's sample_seed) step t's action is the deterministic draw
    sampling.sample_u(seed, t) that the verifier checks; without it, rng draws (tests of the reward semantics only:
    a gateway refuses such actions)."""

    def __init__(self, policy, rng: np.random.Generator | None = None, greedy: bool = False, seed: int | None = None):
        self.policy, self.rng, self.greedy, self.seed = policy, rng, greedy, seed
        self.t = 0  # the task step of the next call

    @classmethod
    def for_task(cls, policy, task):
        return cls(policy, None, bool(task.greedy), int(task.sample_seed))

    def __call__(self, raw):
        u = None
        if self.seed is not None:
            from .sampling import sample_u
            u = sample_u(self.seed, self.t)
        self.t += 1
        return self.policy.act(self.policy.normalize(raw), self.rng, self.greedy, u=u)

    def value(self, raw) -> float:
        return float(self.policy.value(self.policy.normalize(raw)))


class ReplayActions(PolicyActions):
    """Recorded actions in order, with the policy's log-prob and value of each (verifier, tests)."""

    def __init__(self, policy, actions):
        super().__init__(policy, np.random.default_rng(0), True)
        self.actions = [int(a) for a in actions]
        self.i = 0

    def __call__(self, raw):
        if self.i >= len(self.actions):
            raise TaskAborted("replay: no recorded action left")
        a = self.actions[self.i]
        self.i += 1
        x = self.policy.normalize(raw)
        lp = self.policy.log_probs(x)
        return a, float(lp[a]), float(self.policy.value(x))


# ============================================================================================== chunks

_OPTIONAL = ("env_rewards", "sector", "frames", "sr_ms", "arc")


class ChunkBuilder:
    """Steps of one task -> encoded chunks with the observation hash chain carried across chunks."""

    def __init__(self, task_id: str, policy_sha: str, n_obs: int, n_act: int):
        self.task_id, self.policy_sha, self.n_obs, self.n_act = task_id, policy_sha, int(n_obs), int(n_act)
        self.prev = P.chain_seed(task_id)
        self.seq = 0
        self.step0 = 0
        self._clear()

    def _clear(self):
        self.rows, self.values = [], []
        self.cols = {k: [] for k in ("actions", "logp", "rewards", "flags") + _OPTIONAL}
        self.frames = self.episodes = 0
        self.t0 = time.perf_counter()

    def __len__(self):
        return len(self.rows)

    def add(self, obs, a, logp, v, r, flag, env_r, sector, frames, sr_ms, arc, episode_end: bool):
        self.rows.append(np.asarray(obs, np.float32))
        self.values.append(float(v))
        c = self.cols
        c["actions"].append(int(a))
        c["logp"].append(float(np.clip(logp, P.MIN_LOGP, 0.0)))
        c["rewards"].append(float(r))
        c["flags"].append(int(flag))
        c["env_rewards"].append(float(env_r))
        c["sector"].append(int(np.clip(sector, 0, 255)))
        c["frames"].append(int(np.clip(frames, 0, 65535)))
        c["sr_ms"].append(float(sr_ms))
        c["arc"].append(float(arc))
        self.frames += int(frames)
        self.episodes += int(episode_end)

    def build(self, next_obs, next_value: float, final: bool, end: str | None) -> tuple[bytes, dict]:
        T = len(self.rows)
        if T == 0:
            raise ValueError("empty chunk")
        obs = np.stack(self.rows + [np.asarray(next_obs, np.float32)]).astype("<f4")
        h = P.chain_hashes(obs, self.prev)
        c = self.cols
        arrays = {"actions": np.asarray(c["actions"], np.uint8), "logp": np.asarray(c["logp"], np.float32),
                  "rewards": np.asarray(c["rewards"], np.float32), "flags": np.asarray(c["flags"], np.uint8),
                  "values": np.asarray(self.values + [float(next_value)], np.float32), "obs": obs, "obs_hash": h,
                  "env_rewards": np.asarray(c["env_rewards"], np.float32), "sector": np.asarray(c["sector"], np.uint8),
                  "frames": np.asarray(c["frames"], np.uint16), "sr_ms": np.asarray(c["sr_ms"], np.float64),
                  "arc": np.asarray(c["arc"], np.float32)}
        header = {"task_id": self.task_id, "seq": self.seq, "step0": self.step0, "steps": T, "n_obs": self.n_obs,
                  "n_act": self.n_act, "policy_sha": self.policy_sha, "chain_prev": self.prev.hex(),
                  "final": bool(final), "end": end if final else None,
                  "stats": {"frames": int(self.frames), "wall_s": round(time.perf_counter() - self.t0, 3),
                            "episodes": int(self.episodes)}}
        data = P.encode_chunk(header, arrays)
        self.prev = bytes(h[-2])
        self.step0 += T
        self.seq += 1
        self._clear()
        return data, header


class Uploader:
    """Posts a task's chunks in order from a background thread (the engine keeps running meanwhile). A network failure
    is retried with backoff; a refusal stops everything (.refusal); chunk_duplicate after a retry counts as accepted."""

    def __init__(self, post, on_result=None, retries: int = 6, max_queue: int = 8):
        self.post, self.on_result, self.retries = post, on_result, retries
        self.q: queue.Queue = queue.Queue(max_queue)
        self.refusal = None
        self.error = None
        self.accepted_steps = 0
        self._pending = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True, name="chunk-upload")
        self.t.start()

    @property
    def failed(self):
        return self.refusal is not None or self.error is not None

    def submit(self, data: bytes, header: dict):
        with self._lock:
            self._pending += 1
        while not self._stop.is_set() and not self.failed:
            try:
                self.q.put((data, header), timeout=0.5)
                return
            except queue.Full:
                continue
        with self._lock:
            self._pending -= 1

    def _run(self):
        while not self._stop.is_set():
            try:
                item = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                return
            try:
                if not self.failed:
                    self._post_one(*item)
            finally:
                with self._lock:
                    self._pending -= 1

    def _post_one(self, data: bytes, header: dict):
        from .client_api import GatewayRefusal, GatewayUnavailable
        res, delay, tried = None, 2.0, 0
        while not self._stop.is_set():
            try:
                res = self.post(data)
                break
            except GatewayUnavailable as e:
                tried += 1
                if tried > self.retries:
                    self.error = str(e)
                    return
                self._stop.wait(delay)
                delay = min(delay * 2, 30.0)
            except GatewayRefusal as e:
                if tried and e.code == "chunk_duplicate":  # our own earlier attempt arrived
                    res = {"status": "accepted", "verify": "pending", "steps": header["steps"], "seq": header["seq"],
                           "task_id": header["task_id"]}
                    break
                self.refusal = e
                return
        if res is None:
            return
        if res.get("status") == "rejected" or res.get("token_revoked"):
            reason = res.get("reason") or {"code": "bad_chunk", "message": "数据段被拒收。", "params": {}}
            self.refusal = GatewayRefusal(reason, None, "chunk")
            if res.get("token_revoked"):
                self.refusal.fatal = True
        else:
            self.accepted_steps += int(res.get("steps", 0))
        if self.on_result is not None:
            self.on_result(res, header)

    def close(self, timeout_s: float = 120.0) -> bool:
        """Wait until every submitted chunk was posted (or a refusal / error ended the queue). True when drained."""
        t = time.monotonic() + timeout_s
        while time.monotonic() < t and self._pending > 0 and not self.failed:
            time.sleep(0.05)
        drained = self._pending == 0
        self._stop.set()
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        self.t.join(2.0)
        return drained


# ============================================================================================== the sampler

def _end_name(info: dict, worker_end: str | None) -> str:
    if worker_end in ("cap", "stall"):
        return worker_end
    names = {n for _, n in info.get("events") or ()}
    ee = info.get("episode_end")
    if ee == "LevelFinish" or "LevelFinish" in names:
        return "LevelFinish"
    if ee == "GameOver" or "GameOver" in names:
        return "GameOver"
    return "truncated"


def _worker_base():
    from ballrain.train.worker import Worker
    return Worker


class TaskSampler(_worker_base()):
    """Worker's helpers on a task's env (no torch, no queues). spec: WorkerSpec from the manifest's rollout."""

    def __init__(self, env, rollout: dict):  # noqa: D107  (Worker.__init__ is not called: it builds torch models)
        from ballrain.train.worker import WorkerSpec
        kw = {k: rollout[k] for k in ROLLOUT_FIELDS if k in rollout}
        for k in ("route_probes", "gates"):
            kw[k] = dict(kw.get(k) or {})
        kw["split_extra"] = [list(x) for x in kw.get("split_extra") or []]
        self.spec = WorkerSpec(wid=0, copy=env.cfg.copy, env=env.cfg, seed=0, prune="finish", **kw)
        self.env = env
        self.errors: list = []
        self.pending_runs: list = []
        self.runs_seen = 0
        self.fx_info: dict = {}
        self.curriculum_state = None
        self.version = 0
        self.stop = False
        orig = env.fx.compute

        def compute(tracker):
            o, fi = orig(tracker)
            self.fx_info = fi
            return o, fi
        env.fx.compute = compute
        st = env.starts
        if st is not None:
            self.lf = st.levelfinish_frame or int(st.valid[-1])
            self.rec_ms = st.sr_final / max(self.lf - int(st.valid[0]), 1) if np.isfinite(st.sr_final) else 1000 / 264

    def _make_env(self):  # pragma: no cover - a task never rebuilds its env
        raise TaskAborted("the task's game process ended")

    # ------------------------------------------------------------------------------------------ the loop
    def run_task(self, task, actions, chunk_out, should_stop=lambda: None, on_start=None, on_step=None,
                 n_act: int | None = None) -> dict:
        """Play one task. chunk_out(data, header) receives each encoded chunk in order; should_stop() returns None or an
        end name ("stopped" / "deadline") and is asked after every step; on_start(env) after the hand-over. Returns a
        summary {steps, chunks, end, frames, episodes}."""
        from ballrain.train.worker import _Episode
        sp, env = self.spec, self.env
        n_obs = env.observation_space.shape[0]
        b = ChunkBuilder(task.task_id, task.policy_sha, n_obs, n_act or env.action_space.n)
        obs, info = env.reset()
        self._collect_runs()
        if on_start is not None:
            on_start(env)
        f_hand = int(info.get("frames_done") or 0)
        ep = _Episode(info, env, self.version, self._cap_frames(info))
        self._split_start(ep, info)
        stall_all = int(1000 * sp.stall_s / sp.env.delta_ms)
        stall_rsp = int(1000 * (sp.respawn_stall_s or sp.stall_s) / sp.env.delta_ms)
        stall_max = int(1000 * sp.stall_max_s / sp.env.delta_ms)
        steps = episodes = 0
        final_end = None
        def engine_failed(what: str):
            """The engine failed during this step (or its respawn): the step is dropped and the task ends with "error"
            on the previous step (the verifier replays that one normally; docs/pool-prototype.md 10.6)."""
            self.errors.append(what)
            if len(b) == 0:
                raise TaskAborted(f"第 {steps} 步引擎出错：{what}")
            b.cols["flags"][-1] |= P.FLAG_TRUNCATED if not b.cols["flags"][-1] & P.FLAG_TERMINATED else 0
            chunk_out(*b.build(obs, actions.value(obs), True, "error"))
            return "error"

        while final_end is None:
            a, logp, v = actions(obs)
            a = int(a)
            f0 = env.session.frames_done if env.session is not None else 0
            try:
                obs2, r, term, trunc, info2 = env.step(a)
            except Exception as e:  # noqa: BLE001
                final_end = engine_failed(f"step: {type(e).__name__}: {e}")
                break
            if info2.get("error"):  # BallanceEnv.step caught PeerGone / Timeout: a half-run step, not reproducible
                final_end = engine_failed(f"step: {str(info2['error'])[:300]}")
                break
            env_r = float(r)
            step_frames = int(info2.get("frames_done", f0)) - f0
            ep.step(a, r, info2, self.fx_info)
            # ---- Worker.run's per-step shaping and limits, unchanged
            if sp.split_k and (int(info2.get("sector") or 0) > ep.max_sector_seen
                               or info2.get("episode_end") == "LevelFinish"):
                sb = self._split_bonus(ep, info2, info2.get("episode_end") == "LevelFinish")
                if sb:
                    r += sb
                    ep.ret += sb
                    ep.terms["split_bonus"] = ep.terms.get("split_bonus", 0.0) + sb
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
                            ep.terms["gate_bonus"] = ep.terms.get("gate_bonus", 0.0) + gb
            end = info2.get("episode_end") if (term or trunc) else None
            if not (term or trunc):
                ef = env._ep_frames
                s = env.reward.s
                if s is not None and s > ep.mark_s + sp.stall_progress:
                    ep.mark_s, ep.mark_frame = s, ef
                win = stall_rsp if ep.d["kind"] == "respawn" else stall_all
                since = ef - ep.mark_frame
                if sp.stall_speed is not None:
                    o = env.session.tracker.obs if env.session is not None else None
                    if o is not None and o.ball is not None and float(np.linalg.norm(o.ball["speed"])) > sp.stall_speed:
                        ep.move_frame = ef
                    since = min(since, ef - ep.move_frame)
                    if ef - ep.mark_frame >= stall_max:
                        since = win
                if ep.d["cap_frames"] is not None and ef >= ep.d["cap_frames"]:
                    trunc, end = True, "cap"
                elif s is not None and since >= win:
                    trunc, end = True, "stall"
                    if sp.stall_terminal:
                        rc = env.reward.cfg
                        pen = -(rc.death_penalty + rc.death_refund * max(env.reward.sector_paid, 0.0) + sp.stall_extra)
                        if sp.stall_compensate:
                            win = stall_rsp if ep.d["kind"] == "respawn" else stall_all
                            pen /= sp.gamma ** (win / max(sp.env.action_repeat, 1))
                        r += pen
                        ep.ret += pen
                        ep.terms["stall_penalty"] = ep.terms.get("stall_penalty", 0.0) + pen
                        term, trunc = True, False
            # ---- task-level end
            steps += 1
            respawn = bool(term and end not in ("stall",) and getattr(env, "_after_death", False))
            flag = P.FLAG_TERMINATED if term else (P.FLAG_TRUNCATED if trunc else 0)
            if trunc or (term and not respawn):
                final_end = _end_name(info2, end)
            else:
                lim = None
                if steps >= int(task.max_steps):
                    lim = "max_steps"
                elif int(info2.get("frames_done") or 0) - f_hand >= int(task.max_frames):
                    lim = "max_frames"
                else:
                    lim = should_stop()
                if lim:
                    final_end = lim
                    if not term:
                        flag = P.FLAG_TRUNCATED
            next_obs, next_info = obs2, info2
            if term or trunc:
                episodes += 1
                ep.finish(info2, end or "truncated", env, env._ep_frames)
            if final_end is None and term:  # a respawn in the same process (or the process ends during it)
                try:
                    o3, i3 = env.reset(options={"relaunch": False})
                except Exception as e:  # noqa: BLE001  (PeerGone / Timeout during the respawn fast-forward)
                    final_end = engine_failed(f"respawn: {type(e).__name__}: {e}")
                    break
                self._collect_runs()
                if o3 is None:
                    final_end = "GameOver"
                else:
                    next_obs, next_info = o3, i3
                    ep = _Episode(i3, env, self.version, self._cap_frames(i3))
                    self._split_start(ep, i3)
            arc = (self.fx_info.get("path") or {}).get("s")
            b.add(obs, a, logp, v, r, flag, env_r, int(info2.get("sector") or 0), step_frames,
                  float(info2.get("sr_ms") or 0.0), np.nan if arc is None else float(arc), bool(term or trunc))
            if on_step is not None:
                on_step(steps)
            if final_end is not None or len(b) >= int(task.chunk_steps):
                nv = actions.value(next_obs)
                chunk_out(*b.build(next_obs, nv, final_end is not None, final_end))
            obs, info = next_obs, next_info
        frames = 0
        try:
            frames = int(info2.get("frames_done") or 0) - f_hand
        except Exception:  # noqa: BLE001
            pass
        return {"steps": steps, "chunks": b.seq, "end": final_end, "frames": frames, "episodes": episodes}
