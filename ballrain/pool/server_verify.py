"""The verifier: replays closed tasks with the same engine kind the client ran and compares bit for bit
(docs/volunteer-compute.md 4, docs/pool-prototype.md 3.4).

Which chunks are replayed is decided by the gateway when a chunk arrives (chunks.selected: every chunk of an
account's first probation_chunks, then a sample_rate draw), never told to the client. A task is replayed once it is
closed (final chunk received, or the deadline passed): from the task's start (record frame k, ball-order seed) with
the submitted actions, up to its highest selected chunk (a chunk can only be replayed together with all chunks before
it). Chunks up to there pass or the first mismatching one is rejected; later chunks of the task are released
unreplayed (verify "skipped").

Any mismatch (rollout.replay_task: observation chain / float16 copy, reward, flags, end, value, log-prob; earliest
step first) rejects that chunk with its reason (replay_mismatch / replay_reward_mismatch / replay_value_mismatch /
replay_logp_mismatch, with task, chunk and step), quarantines every unverified chunk of the account and revokes the
account (token_revoked_replay with the same task / chunk / step). A failure of the verifier's own engine (launch,
hand-over, crash) is replay_failed: the task is retried later, never held against the player.

One engine process at a time; the replay's run directories are deleted afterwards.
"""
from __future__ import annotations

import json
import secrets
import shutil
import threading
import traceback
from pathlib import Path

from . import protocol as P, rollout as R, task as T, weights as W
from .reasons import Reason, payload
from .server_store import Store


class Verifier:
    def __init__(self, store: Store, pool_dir, engines: dict, launcher_factory=None, log=None,
                 retry_s: int = 300):
        """engines: kind -> object with .binary / .root (server_config.EngineRef); launcher_factory: tests
        (ballrain.fakegame.FakeLauncher), None = the headless engine of HeadlessConfig."""
        self.store = store
        self.pool_dir = Path(pool_dir)
        self.engines = engines
        self.launcher_factory = launcher_factory
        self.log = log or (lambda msg: None)
        self.retry_s = retry_s
        self._policies: dict = {}
        self.stop_event = threading.Event()
        self.replays = 0

    # ------------------------------------------------------------------------------------------ inputs
    def _policy(self, sha: str):
        if sha not in self._policies:
            wd = self.pool_dir / "weights"
            m = W.read_manifest(wd / f"{sha}.json")
            self._policies = {sha: (m, W.load_policy(wd / f"{sha}.npz", m, expect_sha=sha))}  # keep one
        return self._policies[sha]

    def _assets(self, task: T.Task):
        cache = self.pool_dir / "cache"
        ad = self.pool_dir / "assets"
        sdir = T.install_starts_pack(ad / f"{task.assets['starts']}.npz", cache, task.assets["starts"])
        pdir = T.install_path_asset(ad / f"{task.assets['path']}.npz", cache, task.level, task.assets["path"])
        return sdir, pdir

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
            sdir, pdir = self._assets(task)
            lg = T.LocalGame(engine=task.engine, binary=eng.binary, root=eng.root)
            launcher = self.launcher_factory() if self.launcher_factory else None
            env = T.build_env(task, manifest, lg, sdir, pdir, out_root, copy="verify-00", launcher=launcher)
            runner = R.TaskRunner(env, task, manifest)
            self.replays += 1
            mm = R.replay_task(runner, policy, decoded)
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
            s.finish_verify(task_id, upto, True, None, None, max_seq=top)
            self.log(f"verify {task_id[:8]}: passed up to chunk {upto}")
            return "passed"
        params = mm.params(task_id)
        s.finish_verify(task_id, upto, False, payload(mm.reason, **params), mm.seq)
        s.revoke(row["forum_id"], payload(Reason.TOKEN_REVOKED_REPLAY, **params))
        self.log(f"verify {task_id[:8]}: {mm.reason.value} at chunk {mm.seq} step {mm.step} ({mm.detail}); "
                 f"account revoked")
        return "failed"

    # ------------------------------------------------------------------------------------------ loop
    def run_once(self, limit: int = 10) -> int:
        self.store.close_expired()
        n = 0
        for row in self.store.tasks_to_verify(limit):
            if self.stop_event.is_set():
                break
            self.verify_task(row)
            n += 1
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
        th = threading.Thread(target=self.run_forever, name="pool-verifier", daemon=True)
        th.start()
        return th


def select_chunk(store: Store, forum_id: int, probation_steps: int, rate: float) -> bool:
    """Whether a newly accepted chunk is replayed: every chunk until the account has probation_steps steps waiting for
    or passed by the replay (probation_chunks x the task's chunk_steps: counted in steps, so a flood of 1-step tasks
    does not end it), then a draw (secrets-based: unpredictable to the client, decided before it could know the
    outcome)."""
    if store.probation_steps(forum_id) < probation_steps:
        return True
    return secrets.randbelow(1_000_000) < int(rate * 1_000_000)
