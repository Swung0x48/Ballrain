"""The gateway's link to the PPO learner (docs/pool-prototype.md 11): the pool as one more remote host of the learner.

The gateway host runs this next to the gateway and the verifier. It connects to the learner's RemoteHub
(scripts/train_m2.py --remote-listen, ballrain.train.remote) exactly like a remote worker bridge does -- LAN only, the
hub's shared key from an environment variable, never in a file -- with host label "pool" and no workers of its own:

- from the hub it gets the worker template (the learner's LIVE EnvConfig: start record, rehearsal ranges, at-rest
  starts, exclude_starts, features, reward; and the WorkerSpec fields: shaping, counter_rand, p_early_respawn,
  respawn_level_starts) and, after every update, the parameter message (weights, observation statistics, the
  curriculum state the learner's own workers draw from). Every publish_every updates (at most every publish_min_s
  seconds) it publishes that state as the policy new tasks name (server_tasks.PolicySource.publish_state): tasks then
  follow the same start distribution as the learner's workers, from the same curriculum state, and the policy lag of
  pool data is about publish_every updates plus the task's own duration and its verification;
- back to the hub it sends every RELEASED chunk (verified, or released by the spot-check rule; never held, rejected or
  quarantined; server_store.to_ingest) converted to the learner's chunk message (chunk_message): raw observations
  (the float16 copy, which the verifier checked against its own features), actions, behaviour log-probs, values,
  rewards, terminated / truncated, bootstrap values, version = the learner update the task's policy was taken at, plus
  "pool": True and the observation row after the chunk (raw_last, for the learner's V-trace correction). The learner
  drops pool chunks unless it runs with --pool-ingest. Each fed chunk is marked in the store with the learner update it
  was offered at (chunks.ingested), so a later revocation can be traced to the updates it may have reached.

Nothing from a volunteer is unpickled anywhere: chunks are decoded by protocol.decode_chunk (fixed dtypes, checked),
and only this trusted process pickles the converted arrays onto the existing LAN connection.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, is_dataclass
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

from . import protocol as P, weights as W

PROTOCOL = 1  # ballrain.train.remote.PROTOCOL
HOST_LABEL = "pool"  # the learner's host / backend label of pool data (no names, no hosts)


def chunk_message(c: "P.Chunk", version: int, reach_patch=None) -> dict:
    """A decoded pool chunk -> the learner's chunk message (ballrain.train.worker.Worker.run's buffer layout).
    reach_patch (server_verify.read_reach_patch): f7 rows whose reach block the verifier computed otherwise get the
    verifier's values (docs/f7-native-port.md 7)."""
    A = c.arrays
    T = int(c.header["steps"])
    obs = np.array(A["obs"], np.float32)  # (T + 1, n_obs): the rows before each step and the one after the last
    if reach_patch is not None:
        rows, reach, start = reach_patch
        obs[rows, start:start + reach.shape[1]] = reach
    if not np.isfinite(obs).all():  # guard: never hand the learner a non-finite row (a bad patch, a decoder slip)
        raise ValueError("non-finite observation after the reach patch")
    values = np.asarray(A["values"], np.float32)
    flags = np.asarray(A["flags"], np.uint8)
    term = (flags & P.FLAG_TERMINATED) != 0
    trunc = ((flags & P.FLAG_TRUNCATED) != 0) & ~term
    tv = np.zeros(T, np.float32)
    tv[trunc] = values[1:][trunc]  # a truncated step bootstraps the value of the row after it (only the last step)
    frames = int(np.asarray(A["frames"], np.int64).sum()) if "frames" in A else 0
    raw = np.ascontiguousarray(obs[:T])
    return {"obs": raw, "raw": raw, "raw_last": np.ascontiguousarray(obs[T]),
            "actions": np.asarray(A["actions"], np.int64), "logp": np.asarray(A["logp"], np.float32),
            "values": np.ascontiguousarray(values[:T]), "rewards": np.asarray(A["rewards"], np.float32),
            "terminated": term, "truncated": trunc, "trunc_values": tv, "last_value": float(values[T]),
            "version": int(version), "episodes": [], "runs": [], "errors": [], "t_chunk": 0.0, "t_reset": 0.0,
            "frames": frames, "wid": 0, "backend": HOST_LABEL, "copy": HOST_LABEL, "pool": True,
            "pool_task": str(c.header["task_id"]), "pool_seq": int(c.header["seq"])}


def state_from_params(template: dict, msg: dict) -> dict:
    """A checkpoint-like dict (weights.export_state / PolicySource.publish_state) from the hub's worker template and
    one parameter message: the learner's live configuration and curriculum at update msg["version"]."""
    model = {k: np.asarray(v) for k, v in msg["model"].items()}
    n_layers = sum(1 for k in model if k.startswith("pi.") and k.endswith(".weight"))
    env = template["env"]
    ec = asdict(env) if is_dataclass(env) else dict(env)
    branch_assets = ec.pop("branch_assets", None)
    args = dict(template.get("worker") or {})
    args.setdefault("activation", "tanh")
    state = {"model": model, "rms": msg["rms"], "curriculum": msg.get("curriculum"),
             "counters": {"update": int(msg["version"]), "steps": int(msg.get("steps", 0) or 0)},
             "args": args, "n_obs": int(model["pi.0.weight"].shape[1]),
             "n_act": int(model[f"pi.{2 * (n_layers - 1)}.weight"].shape[0]), "env_config": ec}
    # the learner's route-branch bundle (level, line_npz, desc_json, drive_npz raw bytes), carried as the learner sent it:
    # the gateway never reads branch files of its own (the export pins features.branch from these bytes). A parameter
    # message's own copy wins over the template's (a learner that changed its branch without a restart)
    ba = msg.get("branch_assets") if msg.get("branch_assets") is not None else template.get("branch_assets")
    if ba is None:
        ba = branch_assets  # the EnvConfig copy (popped above so the exported env_config holds no raw bytes)
    if ba is not None:
        state["branch_assets"] = ba
    return state


# what makes a chunk's observations and rewards mean the same to the learner: the feature version, the reference
# line, the route bundle (branch pin, v1 or multi-branch v2) and the reach block version. A chunk whose policy pinned
# other values (a re-pinned learner: --branch-update) is never fed.
IDENTITY_KEYS = ("version", "path_sha256", "branch_sha256", "reach_version")


def feature_identity(features) -> tuple | None:
    """IDENTITY_KEYS of a manifest's features dict or the template's FeatureConfig; None when there is none."""
    if features is None:
        return None
    get = features.get if isinstance(features, dict) else (lambda k: getattr(features, k, None))
    return tuple(get(k) or None for k in IDENTITY_KEYS)


DEFAULT_MAX_LAG = 300  # train_m2 --pool-max-lag's default (the template's pool_max_lag when the learner sends it)


class LearnerLink:
    def __init__(self, addr, authkey: bytes, policies, store, pool_dir, publish_every: int = 10,
                 publish_min_s: float = 20.0, ingest: bool = True, log=None, clock=time.time, batch: int = 64):
        self.addr, self.authkey = addr, authkey
        self.policies, self.store, self.pool_dir = policies, store, Path(pool_dir)
        self.publish_every, self.publish_min_s, self.ingest = int(publish_every), float(publish_min_s), bool(ingest)
        self.log = log or (lambda m: None)
        self.clock, self.batch = clock, int(batch)
        self.conn = None
        self.template = None
        self.run = None
        self.session = 0
        self._send_lock = threading.Lock()
        self._latest = None  # newest parameter message
        self._published_version = None
        self._published_t = -1e18
        self.version = None  # newest learner update seen
        self._versions: dict = {}  # policy_sha -> learner update of its weights (manifest source.update)
        self._identity: dict = {}  # policy_sha -> feature_identity of its manifest
        self.stop_event = threading.Event()
        self.learner_ingests = False  # the hub's template says the learner runs with --pool-ingest
        self._refused_logged = False
        self.counts = {"published": 0, "fed": 0, "fed_steps": 0, "other_run": 0, "errors": 0, "reconnects": 0,
                       "stale": 0}
        self._stale_pass_done = False
        self.last_error: str | None = None

    # ------------------------------------------------------------------------------------------ connection
    def connect(self, timeout_s: float = 30.0) -> bool:
        try:
            c = Client(self.addr, authkey=self.authkey)
            # branch_assets_version 1: the link carries the learner's route-branch bundle into the export state (the
            # hub refuses a hello without it when the learner distributes one). branch_rules_version 1: the pool
            # reproduces the bundle's branch rules (stall clock, rest drive / brake): the native client 0.4.0 and the
            # verifier's replay (rollout.TaskRunner via Worker.stall_age and the env) apply them, and a policy with
            # non-default rules requires the branch_rules_v1 capability, so only clients that apply them get its tasks.
            # multi_branch_v2: the link carries a multi-branch envelope too (docs/pool-prototype.md 15); its tasks go
            # only to clients with the multi_branch_v2 capability, and the verifier replays with the Python env
            c.send({"type": "hello", "protocol": PROTOCOL, "host": HOST_LABEL, "session": self.session + 1,
                    "backend": HOST_LABEL, "n_workers": 0, "branch_assets_version": 1, "branch_rules_version": 1,
                    "multi_branch_v2": True, "actor_ball_type_v1": True})
            if not c.poll(timeout_s):
                raise ConnectionError(f"no config within {timeout_s:.0f} s")
            m = c.recv()
            if m.get("type") != "config":
                raise ConnectionError(f"hub refused: {str(m)[:200]}")
        except Exception as e:  # noqa: BLE001
            self.last_error = f"connect: {type(e).__name__}: {str(e)[:200]}"
            return False
        tpl = m["template"]
        if "pool_ingest" not in tpl:
            # a learner without pool support (older code) would train pool chunks as ordinary remote chunks, with no
            # lag bound and no correction: never connect to it
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
            self.last_error = "connect: the learner does not advertise pool support (pool_ingest); refused"
            if not self._refused_logged:
                self.log("learner link: the learner's hub does not advertise pool support (older train_m2); "
                         "not connecting (retrying in the background)")
                self._refused_logged = True
            return False
        self._refused_logged = False
        self.session += 1
        self.template = tpl
        self.run = str(self.template.get("run"))
        self.learner_ingests = bool(tpl.get("pool_ingest"))
        # a reconnect may be a restarted / rolled-back learner: publish its parameters again from scratch, and only
        # what this session's learner sends (never the previous session's last parameters: after the r8 NaN stop on
        # 2026-11-18 the old learner's non-finite last send was published at the reconnect, caught by the export)
        self._published_version, self._published_t = None, -1e18
        self._forget_params()
        if getattr(self.policies, "run", self.run) != self.run:
            # the policies this link publishes carry the learner's run name (manifest source.run), which feed_once
            # compares with the template's: a differing [policy] run would mark every chunk "not for this learner"
            self.log(f"learner link: [policy] run {self.policies.run!r} differs from the learner's run {self.run!r}; "
                     f"publishing as {self.run!r}")
            self.policies.run = self.run
        self.conn = c
        self.log(f"learner link: connected (session {self.session}, run {self.run}, learner "
                 f"{'ingests pool chunks' if self.learner_ingests else 'runs without --pool-ingest: no chunks sent'})")
        return True

    def _forget_params(self):
        """Drop the cached parameters and the learner version (nothing is published or fed until the current
        session's learner sends its own); the next feed starts with a stale pass."""
        self._latest = None
        self.version = None
        self._stale_pass_done = False

    def max_lag(self) -> int:
        """The learner's --pool-max-lag (its template's pool_max_lag; 300, the learner's default, when absent)."""
        try:
            return int((self.template or {}).get("pool_max_lag", DEFAULT_MAX_LAG))
        except (TypeError, ValueError):
            return DEFAULT_MAX_LAG

    def _stale_pass(self, version: int) -> int:
        """Once per ingesting session, at its first parameters: every released, unfed chunk whose policy is more than
        max_lag updates behind the learner is marked stale in one pass (the learner would drop it: pool_max_lag),
        instead of being decoded, sent and dropped. Returns the number marked."""
        self._stale_pass_done = True
        lim = version - self.max_lag()
        old, other_pin = [], []
        for sha in self.store.releasable_policies():
            try:
                run, v = self._version_of(sha)
                same = self._same_identity(sha)
            except Exception:  # noqa: BLE001  (an unreadable manifest: feed_once marks those -1)
                continue
            if run != self.run:
                continue
            if not same:
                other_pin.append(sha)
            elif v < lim:
                old.append(sha)
        n, st = self.store.mark_stale_policies(old)
        n2, st2 = self.store.mark_stale_policies(other_pin)
        if n:
            self.log(f"learner link: {n} released chunks ({st} steps) more than {self.max_lag()} updates behind the "
                     f"learner (update {version}) marked stale, not fed")
        if n2:
            self.log(f"learner link: {n2} released chunks ({st2} steps) of {len(other_pin)} policies with other "
                     f"features than the learner's {self._learner_identity()} (a re-pinned route or reference line) "
                     "marked stale, not fed")
        self.counts["stale"] = self.counts.get("stale", 0) + n + n2
        return n + n2

    def _learner_identity(self):
        env = (self.template or {}).get("env")
        feats = env.get("features") if isinstance(env, dict) else getattr(env, "features", None)
        return feature_identity(feats)

    def _same_identity(self, sha: str) -> bool:
        """The chunk policy's features (manifest) pin what the learner's template pins (IDENTITY_KEYS); True when
        the template carries no features (nothing to compare)."""
        want = self._learner_identity()
        if want is None:
            return True
        if sha not in self._identity:
            m = W.read_manifest(self.pool_dir / "weights" / f"{sha}.json")
            self._identity[sha] = feature_identity(m.get("features"))
        return self._identity[sha] == want

    def _lost(self):
        c, self.conn = self.conn, None
        self._forget_params()
        if c is not None:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
            self.counts["reconnects"] += 1
            self.log("learner link: disconnected; reconnecting")

    def _recv_loop(self):
        while not self.stop_event.is_set():
            c = self.conn
            if c is None:
                time.sleep(0.5)
                continue
            try:
                if not c.poll(1.0):
                    continue
                m = c.recv()
            except (EOFError, OSError, ConnectionError):
                self._lost()
                continue
            if m.get("type") == "params" and "model" in m.get("msg", {}) and c is self.conn:  # this session's only
                self._latest = m["msg"]
                self.version = int(m["msg"]["version"])

    # ------------------------------------------------------------------------------------------ publishing
    def maybe_publish(self, force: bool = False) -> bool:
        msg = self._latest
        if msg is None or self.template is None:
            return False
        v = int(msg["version"])
        now = self.clock()
        if self._published_version is not None and v < self._published_version:
            # the learner went back (auto-rollback, a resume from an older checkpoint): its numbering restarted
            self.log(f"learner link: learner update went back {self._published_version} -> {v}; republishing")
            self._published_version, self._published_t = None, -1e18
        due = (self._published_version is None or v - self._published_version >= self.publish_every) \
            and now - self._published_t >= self.publish_min_s
        if not (force or due) or v == self._published_version:
            return False
        pub = self.policies.publish_state(state_from_params(self.template, msg), source=f"learner-u{v}")
        self._published_version, self._published_t = v, now
        self._versions[pub.policy_sha] = v
        self.counts["published"] += 1
        self.log(f"learner link: policy {pub.policy_sha[:12]} from update {v} (frontier "
                 f"{pub.curriculum.frontier if pub.curriculum is not None else '-'})")
        return True

    # ------------------------------------------------------------------------------------------ ingest
    def _version_of(self, sha: str):
        """(run, learner update) of a policy version (its manifest's source), cached."""
        if sha not in self._versions:
            m = W.read_manifest(self.pool_dir / "weights" / f"{sha}.json")
            self._versions[sha] = (str(m["source"]["run"]), int(m["source"]["update"]))
        v = self._versions[sha]
        return v if isinstance(v, tuple) else (self.run, v)

    def feed_once(self) -> int:
        """Send up to `batch` released chunks to the learner; returns the number sent."""
        version = self.version  # this call's learner version (a drop below clears self.version)
        if not self.ingest or not self.learner_ingests or self.conn is None or version is None:
            return 0
        version = int(version)
        if not getattr(self, "_stale_pass_done", False):
            self._stale_pass(version)
        rows = self.store.to_ingest(self.batch)
        sent, other, stale = [], [], []
        for r in rows:
            try:
                run, ver = self._version_of(r["policy_sha"])
            except Exception as e:  # noqa: BLE001  (a manifest that is gone: nothing to tell the learner)
                self.last_error = f"manifest {r['policy_sha'][:12]}: {type(e).__name__}"
                other.append((r["task_id"], r["seq"]))
                continue
            if run != self.run:  # a chunk of another run (an older gateway policy): never this learner's data
                other.append((r["task_id"], r["seq"]))
                continue
            if version - ver > self.max_lag() or not self._same_identity(r["policy_sha"]):
                stale.append((r["task_id"], r["seq"]))  # the learner would drop it (lag), or another route: not sent
                continue
            try:
                from .server_verify import read_reach_patch
                cf = self.pool_dir / "chunks" / r["file"]
                c = P.decode_chunk(cf.read_bytes())
                m = chunk_message(c, ver, read_reach_patch(cf))
                from .server_rewards import read_rewards, rewards_path
                if rewards_path(cf).is_file():  # reward_inputs_v1: the gateway's rewards
                    m["rewards"] = read_rewards(cf, int(c.header["steps"]))
            except Exception as e:  # noqa: BLE001
                self.counts["errors"] += 1
                self.last_error = f"chunk {r['task_id'][:8]}/{r['seq']}: {type(e).__name__}: {str(e)[:120]}"
                other.append((r["task_id"], r["seq"]))
                continue
            try:
                with self._send_lock:
                    self.conn.send({"type": "data", "msg": m})
            except (OSError, EOFError, ConnectionError, ValueError, AttributeError):
                self._lost()
                break
            sent.append((r["task_id"], r["seq"]))
            self.counts["fed"] += 1
            self.counts["fed_steps"] += int(r["steps"])
        if sent:
            self.store.mark_ingested(sent, version)
        if stale:
            from .server_store import INGEST_STALE
            self.counts["stale"] = self.counts.get("stale", 0) + len(stale)
            self.store.mark_ingested(stale, INGEST_STALE)
        if other:
            self.counts["other_run"] += len(other)
            self.store.mark_ingested(other, -1)  # -1: not for this learner (never sent)
            self.log(f"learner link: {len(other)} released chunks not for run {self.run!r} (other run or unreadable; "
                     f"marked ingested = -1, not sent){': ' + self.last_error if self.last_error else ''}")
        return len(sent)

    # ------------------------------------------------------------------------------------------ loop
    def start_recv(self):
        if not getattr(self, "_recv_started", False):
            self._recv_started = True
            threading.Thread(target=self._recv_loop, name="learner-link-recv", daemon=True).start()

    def run_forever(self, idle_s: float = 2.0):
        self.start_recv()
        backoff = 3.0
        while not self.stop_event.is_set():
            if self.conn is None:
                if not self.connect():
                    self.stop_event.wait(backoff)
                    backoff = min(backoff * 2, 60.0)
                    continue
                backoff = 3.0
            try:
                self.maybe_publish()
                n = self.feed_once()
            except Exception as e:  # noqa: BLE001
                self.counts["errors"] += 1
                self.last_error = f"{type(e).__name__}: {str(e)[:200]}"
                self.log(f"learner link: {self.last_error}")
                n = 0
            if n == 0:
                self.stop_event.wait(idle_s)

    def start_thread(self) -> threading.Thread:
        th = threading.Thread(target=self.run_forever, name="learner-link", daemon=True)
        th.start()
        return th

    def close(self):
        self.stop_event.set()
        c, self.conn = self.conn, None
        if c is not None:
            try:
                c.send({"type": "bye"})
                c.close()
            except Exception:  # noqa: BLE001
                pass


def from_config(cfg, policies, store, log=None, environ=None):
    """A LearnerLink for PoolConfig's [learner] section, or None (no hub configured). The key is read from the
    environment variable cfg.learner_key_env only."""
    if not cfg.learner_hub:
        return None
    from ballrain.train.remote import parse_addr
    environ = os.environ if environ is None else environ
    key = (environ.get(cfg.learner_key_env) or "").encode()
    if not key:
        raise ValueError(f"[learner] hub is set but ${cfg.learner_key_env} is empty (the hub's shared key)")
    return LearnerLink(parse_addr(cfg.learner_hub), key, policies, store, cfg.pool_dir,
                       publish_every=cfg.learner_publish_every, publish_min_s=cfg.learner_publish_min_s,
                       ingest=cfg.learner_ingest, log=log)
