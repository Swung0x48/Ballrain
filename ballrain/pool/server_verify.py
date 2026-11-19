"""The verifier: replays closed tasks with the same engine kind the client ran and compares bit for bit
(docs/volunteer-compute.md 4, docs/pool-prototype.md 3.4).

Which tasks are replayed is decided by the gateway, never told to the client (2026-11-18: per task, not per chunk):
- when a task is issued (tasks.replay, select_task): every task during the account's probation (probation_chunks x
  chunk_steps steps replayed or waiting), every task while the account is escalated, else a task_sample draw
  (secrets); a replayed task is replayed in full, so its chunks are all selected;
- when a chunk arrives it gets the cheap checks: the structural ones (protocol.decode_chunk, chain link, sizes) and
  the policy-consistency check (policy_consistency: the task's policy recomputed on the chunk's float16 observations
  must give the submitted log-probs and values and draw the submitted actions from the task's sample_seed, within
  tolerances 5-25x the measured float16 noise). A chunk that fails it is flagged (chunks.flag): its task becomes a
  full replay and the account's next escalate_tasks tasks too. The flag itself revokes nothing (float16 copies are
  approximate); the replay decides.
A task is replayed once it is closed (final chunk received, or the deadline passed): from the task's start (record
frame k, ball-order seed) with the submitted actions, up to its highest selected chunk. Chunks up to there pass or the
first mismatching one is rejected; unselected tasks are released unreplayed (verify "skipped").

Any mismatch (rollout.replay_task: observation chain / float16 copy, reward, flags, end, value, log-prob; earliest
step first) rejects that chunk with its reason (replay_mismatch / replay_reward_mismatch / replay_value_mismatch /
replay_logp_mismatch, with task, chunk and step: the player sees it in me.stats.last_rejection). What happens then
(2026-11-18, the user's rule: the system rejects bad data by itself, revocations and rollbacks are the user's call):
- the task's data never reaches the learner: the failing chunk is rejected, the task's other chunks are quarantined
  (Store.reject_task);
- the account is flagged suspect, NOT revoked (Store.flag_suspect): while flagged every task of it is replayed in full
  (select_task, and here every chunk of its tasks) and only replay-passed chunks of it are fed to the learner
  (Store.to_ingest);
- an alert for the coordinator: <pool>/alerts/<time>-suspect-<task>.json with the account (forum id and a pseudonym),
  task, chunk, step, reason, the mismatching values against their tolerance (rollout.Mismatch.values) and the learner
  updates the account's earlier sampled-but-not-replayed data already reached (chunks.ingested of verify 'skipped'),
  plus one line in the gateway log;
- revoke / unflag are manual (tools/pool/gateway.py admin revoke / unflag); learner updates are never undone here. A failure of the verifier's own engine (launch,
hand-over, crash) is replay_failed: the task is retried later, never held against the player.

workers replay in parallel (Verifier.start_threads(n), [verify] workers), one engine process each; a task is claimed
by one worker at a time. The replay's run directories are deleted afterwards.
"""
from __future__ import annotations

import json
import secrets
import shutil

import numpy as np
import threading
import traceback
from pathlib import Path

from . import protocol as P, rollout as R, task as T, weights as W
from .reasons import Reason, payload
from .server_store import Store


class Verifier:
    def __init__(self, store: Store, pool_dir, engines: dict, launcher_factory=None, log=None,
                 retry_s: int = 300, reach_max_row_frac: float = R.REACH_MAX_ROW_FRAC,
                 reach_max_logp_shift: float = R.REACH_MAX_LOGP_SHIFT):
        """engines: kind -> object with .binary / .root (server_config.EngineRef); launcher_factory: tests
        (ballrain.fakegame.FakeLauncher), None = the headless engine of HeadlessConfig."""
        self.store = store
        self.pool_dir = Path(pool_dir)
        self.engines = engines
        self.launcher_factory = launcher_factory
        self.log = log or (lambda msg: None)
        self.retry_s = retry_s
        # f7 reach block (docs/f7-native-port.md 7): its tolerance bounds; reach_slice overrides the env's (tests)
        self.reach_max_row_frac, self.reach_max_logp_shift = float(reach_max_row_frac), float(reach_max_logp_shift)
        self.reach_slice = None
        self._policies: dict = {}
        self._plock = threading.Lock()
        self._claim_lock = threading.Lock()
        self._claimed: set = set()  # task ids a worker is replaying
        self.stop_event = threading.Event()
        self.replays = 0

    # ------------------------------------------------------------------------------------------ inputs
    def _policy(self, sha: str):
        with self._plock:
            if sha not in self._policies:
                wd = self.pool_dir / "weights"
                m = W.read_manifest(wd / f"{sha}.json")
                if len(self._policies) >= 8:  # keep a few (parallel workers on tasks of recent versions)
                    self._policies.pop(next(iter(self._policies)))
                self._policies[sha] = (m, W.load_policy(wd / f"{sha}.npz", m, expect_sha=sha))
            return self._policies[sha]

    def _assets(self, task: T.Task, manifest: dict):
        cache = self.pool_dir / "cache"
        ad = self.pool_dir / "assets"
        bdir = None
        with self._plock:  # parallel workers: one installs a pack at a time (Windows refuses a replace of an open file)
            sdir = T.install_starts_pack(ad / f"{task.assets['starts']}.npz", cache, task.assets["starts"])
            pdir = T.install_path_asset(ad / f"{task.assets['path']}.npz", cache, task.level, task.assets["path"])
            f = manifest["features"]
            bsha = task.assets.get("branch")
            if f.get("branch_sha256") and bsha:  # the route branch: the pack the gateway exported from the learner's files
                bdir = T.install_branch_pack(ad / f"{bsha}.npz", cache, task.level, bsha, f["branch_sha256"],
                                             f.get("path_sha256"))
        return sdir, pdir, bdir

    def _decoded(self, task_id: str, rows: list, upto: int, n_obs: int, n_act: int) -> list:
        out = []
        for c in rows:
            if c["seq"] > upto:
                break
            data = (self.pool_dir / "chunks" / c["file"]).read_bytes()
            out.append(P.decode_chunk(data, n_act=n_act, n_obs=n_obs))
        return out

    # ------------------------------------------------------------------------------------------ one task
    def verify_task(self, row) -> str:
        """Verify one closed task; returns "passed", "failed", "skipped", "retry" or "empty"."""
        s = self.store
        task_id = row["task_id"]
        acct = s.account(row["forum_id"])
        chunks = s.chunks(task_id)
        held = [c for c in chunks if c["status"] == "held"]
        if acct is not None and acct["revoked"]:  # its unverified chunks are quarantined (any late one too)
            s.q("UPDATE chunks SET status='quarantined' WHERE task_id=? AND status='held'", (task_id,))
            s.set_task_state(task_id, "done")
            return "skipped"
        top = max((c["seq"] for c in chunks), default=-1)  # the snapshot: a later arrival is not decided here
        if not held:
            s.skip_task(task_id, top)
            return "empty"
        selected = [c["seq"] for c in chunks if c["selected"]]
        if acct is not None and acct["suspect"]:  # a suspect account: every task in full, whatever was drawn
            selected = [c["seq"] for c in chunks]
        if not selected:
            s.skip_task(task_id, top)
            return "skipped"
        upto = max(selected)
        task = T.Task.from_dict(json.loads(row["body"]))
        out_root = self.pool_dir / "replay" / task_id
        runner = None
        try:
            manifest, policy = self._policy(task.policy_sha)
            decoded = self._decoded(task_id, chunks, upto, policy.n_obs, policy.n_act)
            eng = self.engines.get(task.engine)
            if eng is None or (self.launcher_factory is None and not eng.binary):
                raise R.EngineFailure(f"no {task.engine} engine configured for the verifier")
            sdir, pdir, bdir = self._assets(task, manifest)
            if task.assets.get("reach"):  # f7: replay from the same field files the client got (byte-identical)
                from .reach_assets import check_cache
                why = check_cache(task.assets["reach"], self.pool_dir / "assets")
                if why is not None:
                    raise R.EngineFailure(f"reach fields: {why}")
            lg = T.LocalGame(engine=task.engine, binary=eng.binary, root=eng.root)
            launcher = self.launcher_factory() if self.launcher_factory else None
            env = T.build_env(task, manifest, lg, sdir, pdir, out_root, copy="verify-00", launcher=launcher,
                              branch_dir=bdir)
            runner = R.TaskRunner(env, task, manifest)
            self.replays += 1
            reach = {}
            mm = R.replay_task(runner, policy, decoded, reach_slice=self.reach_slice,
                               reach_max_row_frac=self.reach_max_row_frac,
                               reach_max_logp_shift=self.reach_max_logp_shift, report=reach)
        except Exception as e:  # noqa: BLE001  the verifier's own failure: replay_failed, retried later
            self.log(f"verify {task_id[:8]}: replay_failed: {type(e).__name__}: {str(e)[:300]}")
            if not isinstance(e, R.EngineFailure):
                self.log(traceback.format_exc(limit=4))
            s.verify_later(task_id, self.retry_s)
            return "retry"
        finally:
            if runner is not None:
                runner.close()
            shutil.rmtree(out_root, ignore_errors=True)
        if mm is None:
            # f7: chunks with too many reach-deviant rows are rejected alone (no suspect flag, the task's other chunks
            # pass); the passed chunks' deviant rows get this replay's reach values for the learner (a patch file)
            for q, r in reach.items():  # guard: the verifier's own reach values must be finite (else: this chunk
                if not r["excess"] and not np.isfinite(np.asarray(r["reach"])).all():  # out, never patched in)
                    r["excess"] = "the verifier's reach values are not finite"
            excess = sorted(q for q, r in reach.items() if r["excess"])
            step0_of = {int(x.header["seq"]): int(x.header["step0"]) for x in decoded}
            for q in excess:
                s.reject_chunk(task_id, q, payload(Reason.REPLAY_MISMATCH, task_id=task_id, chunk=q,
                                                   step=step0_of[q] + int(reach[q]["rows"][0])))
            files = {c["seq"]: c["file"] for c in chunks}
            for q, r in reach.items():
                if not r["excess"] and q in files:
                    write_reach_patch(self.pool_dir / "chunks" / files[q], r)
            s.finish_verify(task_id, upto, True, None, None, max_seq=top)
            self.log(f"verify {task_id[:8]}: passed up to chunk {upto}"
                     + (f"; reach-deviant rows in chunks {sorted(reach)} (patched)" if reach else "")
                     + (f"; rejected for reach (not flagged): {[(q, reach[q]['excess']) for q in excess]}" if excess else ""))
            return "passed"
        params = mm.params(task_id)
        s.reject_task(task_id, mm.seq, payload(mm.reason, **params))
        fid = int(row["forum_id"])
        span = s.record_ingest_span(fid, unreplayed_only=True)
        record = {"t": s.now(), "kind": "replay_mismatch", "account": fid, "pseudonym": pseudonym(fid),
                  "task_id": task_id, "reason": mm.reason.value, "chunk": int(mm.seq), "step": int(mm.step),
                  "detail": mm.detail, "values": mm.values or {}, "learner_updates_unreplayed": span,
                  "learner_updates_all": dict(zip(("first", "last", "steps"), s.ingested_updates(fid))),
                  "action": "task rejected / quarantined; account flagged suspect (every task replayed, only verified "
                            "data fed); revoke or unflag by hand: tools/pool/gateway.py admin revoke|unflag --forum-id"}
        s.flag_suspect(fid, record)
        path = write_alert(self.pool_dir, f"suspect-{task_id[:12]}", record)
        self.log(f"verify {task_id[:8]}: {mm.reason.value} at chunk {mm.seq} step {mm.step} ({mm.detail}); task "
                 f"rejected, account {pseudonym(fid)} flagged suspect (not revoked); its unreplayed data reached "
                 f"learner updates {span['first']}..{span['last']} ({span['steps']} steps); alert {path.name if path else '-'}")
        return "failed"

    # ------------------------------------------------------------------------------------------ loop
    def run_once(self, limit: int = 10) -> int:
        self.store.close_expired()
        n = 0
        for row in self.store.tasks_to_verify(limit + len(self._claimed)):
            if self.stop_event.is_set() or n >= limit:
                break
            with self._claim_lock:
                if row["task_id"] in self._claimed:
                    continue
                self._claimed.add(row["task_id"])
            try:
                cur = self.store.one("SELECT state, verify_after FROM tasks WHERE task_id=?", (row["task_id"],))
                if cur is None or cur["state"] != "closed":  # another worker finished it since the listing
                    continue
                self.verify_task(row)
                n += 1
            finally:
                with self._claim_lock:
                    self._claimed.discard(row["task_id"])
        return n

    def run_forever(self, idle_s: float = 5.0):
        while not self.stop_event.is_set():
            try:
                n = self.run_once()
            except Exception as e:  # noqa: BLE001
                self.log(f"verifier: {type(e).__name__}: {str(e)[:300]}")
                n = 0
            if n == 0:
                self.stop_event.wait(idle_s)

    def start_thread(self) -> threading.Thread:
        return self.start_threads(1)[0]

    def start_threads(self, n: int = 1) -> list:
        out = []
        for i in range(max(1, int(n))):
            th = threading.Thread(target=self.run_forever, name=f"pool-verifier-{i}", daemon=True)
            th.start()
            out.append(th)
        return out


def reach_patch_path(chunk_file) -> Path:
    """<task>/<seq>.reach.npz next to the chunk file <task>/<seq>.brpc."""
    p = Path(chunk_file)
    return p.with_name(p.stem + ".reach.npz")


def write_reach_patch(chunk_file, r: dict) -> None:
    """The verifier's reach values of a passed chunk's deviant rows (rows j of the chunk, row T = the row after the last
    step), for the learner link (learner_link.chunk_message applies it)."""
    from .task import save_npz_deterministic
    if not np.isfinite(np.asarray(r["reach"], np.float64)).all():
        raise ValueError("reach patch: non-finite values")
    dst = reach_patch_path(chunk_file)
    tmp = dst.with_name(f".{dst.name}.{secrets.token_hex(4)}.tmp.npz")
    save_npz_deterministic(tmp, {"rows": np.asarray(r["rows"], "<i4"), "reach": np.asarray(r["reach"], "<f4"),
                                 "start": np.asarray(int(r["start"]), "<i4")})
    tmp.replace(dst)


def read_reach_patch(chunk_file):
    """(rows, reach float32 (n, width), start) of a chunk's patch, or None."""
    p = reach_patch_path(chunk_file)
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        return np.asarray(z["rows"], np.int64), np.asarray(z["reach"], np.float32), int(z["start"])


def pseudonym(forum_id: int) -> str:
    """A stable short name of an account for alerts and logs (no forum name / nickname)."""
    import hashlib
    return "acct-" + hashlib.sha256(f"ballrain-pool/account/{int(forum_id)}".encode()).hexdigest()[:10]


def write_alert(pool_dir, name: str, record: dict):
    """<pool>/alerts/<utc time>-<name>.json (one file per alert, written atomically; the coordinator polls the dir)."""
    import time as _time
    d = Path(pool_dir) / "alerts"
    try:
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{_time.strftime('%Y%m%dT%H%M%SZ', _time.gmtime())}-{name}.json"
        tmp = p.with_name(f".{p.name}.tmp")
        tmp.write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(p)
        return p
    except OSError:
        return None


def select_task(store: Store, forum_id: int, probation_steps: int, p: float) -> bool:
    """Whether a task about to be issued is replayed in full: every task until the account has probation_steps steps
    waiting for or passed by the replay (probation_chunks x chunk_steps: counted in steps, so a flood of 1-step tasks
    does not end it), every task while it is escalated, then a draw with probability p (secrets: unpredictable to the
    client, fixed before it plays)."""
    if store.probation_steps(forum_id) < probation_steps:
        return True
    if store.is_suspect(forum_id):  # flagged by a replay mismatch: every task in full until unflagged by hand
        return True
    if store.take_escalation(forum_id):
        return True
    return secrets.randbelow(1_000_000) < int(round(p * 1_000_000))


# the cheap check's tolerances: 5-25x the float16-copy noise measured on honest native-client chunks (L1 6563 and
# L2 5423 steps, 2026-11-18: |logp| <= 0.011, value <= 0.3 % relative, action outside its cdf interval by <= 4e-4)
CONSIST_LOGP_TOL = 0.05
CONSIST_VALUE_TOL = 0.02
CONSIST_CDF_TOL = 0.01


def policy_consistency(chunk, policy, task: dict, step0: int):
    """The cheap per-chunk check: the task's policy on the chunk's float16 observations must reproduce the submitted
    behaviour log-probs and values, and the submitted actions must be the task's sample_seed draws (argmax for greedy
    tasks), within loose tolerances. Returns None or a short reason (the chunk is then flagged for a full replay).
    It cannot see a fabricated trajectory whose observations are self-consistent with the policy: only the replay can."""
    from .sampling import cdf, sample_u
    A = chunk.arrays
    T = int(chunk.header["steps"])
    x = policy.normalize(np.asarray(A["obs"], np.float32))
    lp = policy.log_probs(x)
    v = policy.value(x)
    if np.any(np.abs(v - A["values"]) > CONSIST_VALUE_TOL * np.maximum(1.0, np.abs(v))):
        return "value"
    a = A["actions"].astype(np.int64)
    got = np.clip(lp[np.arange(T), a], P.MIN_LOGP, 0.0)
    if np.any(np.abs(got - A["logp"]) > CONSIST_LOGP_TOL):
        return "logp"
    greedy = bool(task.get("greedy"))
    for j in range(T):
        if greedy:
            if lp[j, a[j]] < lp[j].max() - CONSIST_LOGP_TOL:
                return "action"
            continue
        c = cdf(lp[j])
        lo = float(c[a[j] - 1]) if a[j] > 0 else 0.0
        hi = float(c[a[j]]) if a[j] < len(c) - 1 else 1.0
        u = sample_u(int(task["sample_seed"]), int(step0) + j)
        if not lo - CONSIST_CDF_TOL <= u < hi + CONSIST_CDF_TOL:
            return "action"
    return None
