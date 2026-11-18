"""Pool task -> chunks -> verifier, end to end on ballrain.fakegame (no game process): the shared rollout
(ballrain.pool.rollout) samples a task in the client's env and replays it bit-exactly in the verifier's; the gateway
accepts the honest chunks (and refuses duplicates, gaps, other policies, other accounts' tasks, broken chains); the
verifier passes them, and catches FORGED chunks (tampered reward / observation / action / log-prob / value) with the
right reason code and step, quarantining the account's unverified chunks and revoking it."""
import json
import sys
import urllib.parse
from pathlib import Path

import numpy as np
import pytest

from ballrain import fakegame
from ballrain.pool import dev, protocol as P, rollout as R, task as T, weights as W
from ballrain.pool.reasons import Reason
from ballrain.pool.server_verify import Verifier

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="ballrain.fakegame's shared memory needs Windows")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from test_pool_gateway import ENGINE_SHA, GAME, PUBLIC as PUBLIC_, Gw  # noqa: E402

FORGERS = {f"forger_{i}": {"id": 2000 + i, "username": f"forger{i}", "nickname": f"伪造者{i}", "joined_days_ago": 400,
                            "is_email_confirmed": True, "comment_count": 9} for i in range(10)}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return dev.fake_world(tmp_path_factory.mktemp("pool-world"))


@pytest.fixture
def pool(tmp_path, world):
    g = Gw(tmp_path, users=FORGERS, max_steps=180, probation_chunks=200, sample_rate=0.1)
    g.policies.datasets_dir = world["datasets"]
    g.policies.path_dirs = world["path_dirs"]
    g.pub = g.policies.publish_state(world["state"])
    g.verifier = Verifier(g.store, g.cfg.pool_dir, g.cfg.engines,
                          launcher_factory=lambda: fakegame.FakeLauncher(backend="headless"), log=g.logs.append)
    yield g
    g.close()


class Player:
    """The controller's side for one account: login, hello, tasks, the sampler, chunk upload."""

    def __init__(self, g, user, tmp_path):
        self.g = g
        self.token = g.login(user)[0]["token"]
        self.sid = P.parse_message("hello_ok", g.hello(self.token).body)["session_id"]
        self.cache = tmp_path / f"client-{user}"

    def task(self) -> T.Task:
        r = self.g.call("GET", "/pool/task", token=self.token, session=self.sid)
        assert r.status == 200, r.body
        return T.Task.from_dict(P.parse_message("task", r.body)["task"])

    def fetch(self, path, dst):
        r = self.g.call("GET", path)
        assert r.status == 200 and "immutable" in dict(r.headers)["Cache-Control"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(r.body)
        return dst

    def play(self, task: T.Task, seed=None, sampler="rollout", stop_after=None, on_start=None, actions=None):
        """Download what the task names (weights, assets) and run it: sampler "rollout" = ballrain.pool.rollout's
        loop, "client" = the controller's (ballrain.pool.client_sampler). stop_after: the player stops after that
        many steps. seed None: the task's own deterministic draws (what an honest controller does); an int: a
        client choosing its actions with its own generator (refused by the replay). Returns (chunks, end, policy)."""
        c = self.cache
        mj = self.fetch(f"/pool/weights/{task.policy_sha}.json", c / "w" / f"{task.policy_sha}.json")
        npz = self.fetch(f"/pool/weights/{task.policy_sha}.npz", c / "w" / f"{task.policy_sha}.npz")
        m = W.read_manifest(mj)
        pol = W.load_policy(npz, m, expect_sha=task.policy_sha)
        sp = self.fetch(f"/pool/assets/{task.assets['starts']}.npz", c / "dl" / "s.npz")
        pp = self.fetch(f"/pool/assets/{task.assets['path']}.npz", c / "dl" / "p.npz")
        sdir = T.install_starts_pack(sp, c, task.assets["starts"])
        pdir = T.install_path_asset(pp, c, task.level, task.assets["path"])
        env = T.build_env(task, m, T.LocalGame("x87soft", "SimTool", "Ballance"), sdir, pdir, c / "out",
                          launcher=fakegame.FakeLauncher(backend="headless"))
        chunks = []
        n = [0]
        if sampler == "client":
            cs = pytest.importorskip("ballrain.pool.client_sampler")

            def should_stop():
                return "stopped" if stop_after is not None and n[0] >= stop_after else None
            try:
                res = cs.TaskSampler(env, m["rollout"]).run_task(
                    task, (actions(pol) if actions is not None else cs.PolicyActions.for_task(pol, task) if seed is None
                           else cs.PolicyActions(pol, np.random.default_rng(seed), task.greedy)),
                    lambda data, header: chunks.append(data), should_stop=should_stop,
                    on_step=lambda k: n.__setitem__(0, k), n_act=pol.n_act,
                    on_start=None if on_start is None else (lambda e: on_start(e, n)))
                end = res["end"]
            finally:
                env.close()
                R.remove_run_dirs(env)
            return chunks, end, pol
        runner = R.TaskRunner(env, task, m)

        def stop():
            n[0] += 1
            return stop_after is not None and n[0] >= stop_after
        try:
            end = R.sample_task(runner, pol, None if seed is None else np.random.default_rng(seed), chunks.append,
                                stop=stop)
        finally:
            runner.close()
            R.remove_run_dirs(env)
        return chunks, end, pol

    def send(self, data: bytes):
        return self.g.call("POST", "/pool/chunk", data, token=self.token, session=self.sid)

    def me(self):
        return P.parse_message("me", self.g.call("GET", "/pool/me", token=self.token).body)


def reseal(data: bytes, **changes) -> bytes:
    """A forged chunk that still passes every structural check: arrays replaced, section sha256 and the trailing
    digest recomputed (protocol.encode_chunk)."""
    c = P.decode_chunk(data)
    h = {k: v for k, v in c.header.items() if k not in ("arrays", "protocol", "type")}
    arrays = {k: np.array(v) for k, v in c.arrays.items()}
    for k, fn in changes.items():
        arrays[k] = fn(arrays[k])
    return P.encode_chunk(h, arrays)


def test_rollout_replays_bit_exactly(pool, tmp_path):
    pl = Player(pool, "eligible", tmp_path)
    t = pl.task()
    assert t.engine == "x87soft" and t.policy_sha == pool.pub.policy_sha and t.chunk_steps == 64
    assert t.start_frame == 0 or t.start_frame in set(pool.pub.valid.tolist())
    chunks, end, pol = pl.play(t)
    dec = [P.decode_chunk(c, n_act=pol.n_act, n_obs=pol.n_obs) for c in chunks]
    assert end in P.END_REASONS and dec[-1].header["final"] and dec[-1].header["end"] == end
    assert sum(d.steps for d in dec) <= t.max_steps
    assert all(d.steps == t.chunk_steps for d in dec[:-1])
    for a, b in zip(dec, dec[1:]):
        assert P.check_chain_link(a, b) is None
    # the replay in a fresh env reproduces everything (rewards bit for bit)
    m = W.read_manifest(pool.cfg.pool_dir / "weights" / f"{t.policy_sha}.json")
    sdir, pdir, _ = pool.verifier._assets(t, m)
    env = T.build_env(t, m, T.LocalGame("x87soft", "b", "r"), sdir, pdir, tmp_path / "rp",
                      launcher=fakegame.FakeLauncher(backend="headless"))
    runner = R.TaskRunner(env, t, m)
    try:
        assert R.replay_task(runner, pol, dec) is None
    finally:
        runner.close()


def test_honest_chunks_are_accepted_and_verified(pool, tmp_path):
    pl = Player(pool, "eligible", tmp_path)
    t = pl.task()
    chunks, end, _ = pl.play(t)
    first = pl.send(chunks[0])
    cr = P.parse_message("chunk_result", first.body)
    assert (cr["status"], cr["verify"], cr["seq"], cr["steps"]) == ("accepted", "pending", 0, 64)
    dup = pl.send(chunks[0])
    assert dup.status == 409 and pool.js(dup)["reason"]["code"] == "chunk_duplicate"
    if len(chunks) > 2:
        gap = pl.send(chunks[2])
        assert gap.status == 409 and pool.js(gap)["reason"]["params"] == {"task_id": t.task_id, "chunk": 2, "expected": 1}
    for c in chunks[1:]:
        assert pl.send(c).status == 200
    assert (pool.cfg.pool_dir / "chunks" / t.task_id / "000000.brpc").read_bytes() == chunks[0]
    steps = sum(P.decode_chunk(c).steps for c in chunks)
    st = pl.me()["stats"]
    assert st["steps_accepted"] == steps and st["steps_pending"] == steps and st["steps_verified"] == 0
    assert pool.verifier.run_once() == 1
    st = pl.me()["stats"]
    assert st["steps_verified"] == steps and st["steps_pending"] == 0 and st["chunks_rejected"] == 0
    assert pl.me()["token_state"] == "active"
    lb = P.parse_message("leaderboard", pool.call("GET", "/pool/leaderboard").body)
    assert lb["entries"] == [{"rank": 1, "nickname": "合格玩家", "steps_verified": steps}]
    assert not list((pool.cfg.pool_dir / "replay").glob("*/*"))  # the replay's run directories are gone
    # a chunk after the final one / another account's task
    late = pl.send(chunks[-1])
    assert late.status == 409
    other = Player(pool, "forger_7", tmp_path)
    r = other.send(chunks[0])
    assert r.status == 403 and pool.js(r)["reason"]["code"] == "task_not_yours"


def test_gateway_refuses_inconsistent_chunks(pool, tmp_path):
    pl = Player(pool, "eligible", tmp_path)
    t = pl.task()
    chunks, _, _ = pl.play(t)
    # another policy version
    r = pl.send(_rehead(chunks[0], policy_sha="ab" * 32))
    assert r.status == 400 and pool.js(r)["reason"]["code"] == "policy_mismatch"
    # seq 0 must start the task's chain
    r = pl.send(_rehead(chunks[0], chain_prev="cd" * 32))
    assert pool.js(r)["reason"]["code"] == "chain_broken"
    assert pl.send(chunks[0]).status == 200
    if len(chunks) > 2:
        # chunk 1 whose first row is not chunk 0's last: chain_broken (resealed, so only the link check sees it)
        r = pl.send(reseal(chunks[1], obs=lambda o: _bump(o, 0)))
        assert r.status == 400 and pool.js(r)["reason"]["code"] == "chain_broken"
        r = pl.send(_rehead(chunks[1], step0=5))
        assert pool.js(r)["reason"]["code"] == "chain_broken"
    # an unknown task
    r = pl.send(_rehead(chunks[0], task_id="0" * 32))
    assert r.status == 404 and pool.js(r)["reason"]["code"] == "task_unknown"
    # eight open tasks at most
    for _ in range(7):
        pl.task()
    r = pool.call("GET", "/pool/task", token=pl.token, session=pl.sid)
    assert r.status == 400 and pool.js(r)["reason"]["code"] == "too_many_instances"
    # the controller was hard-killed (no final chunks): once its session is quiet, a restarted one gets tasks again
    pool.store.q("UPDATE sessions SET last_seen=last_seen-120 WHERE session_id=?", (pl.sid,))
    sid2 = P.parse_message("hello_ok", pool.hello(pl.token).body)["session_id"]
    r = pool.call("GET", "/pool/task", token=pl.token, session=sid2)
    assert r.status == 200, r.body


def _rehead(data: bytes, **hdr) -> bytes:
    c = P.decode_chunk(data)
    h = {k: v for k, v in c.header.items() if k not in ("arrays", "protocol", "type")}
    h.update(hdr)
    return P.encode_chunk(h, {k: np.array(v) for k, v in c.arrays.items()})


def _bump(a, row, col=0, by=0.5):
    a = np.array(a)
    a[row, col] = a[row, col] + np.float16(by)
    return a


FORGERIES = {
    # name: (array changes on the forged chunk given j (local step), expected reason, step offset from j)
    "reward": (lambda j: {"rewards": lambda a: _set(a, j, a[j] + 1.0)}, Reason.REPLAY_REWARD_MISMATCH, 0),
    "logp": (lambda j: {"logp": lambda a: _set(a, j, a[j] - 0.5)}, Reason.REPLAY_LOGP_MISMATCH, 0),
    "value": (lambda j: {"values": lambda a: _set(a, j, a[j] + 3.0)}, Reason.REPLAY_VALUE_MISMATCH, 0),
    "action": (lambda j: {"actions": lambda a: _set(a, j, (int(a[j]) + 1) % 9)}, Reason.REPLAY_LOGP_MISMATCH, 0),
    # the float16 copy of one row changed (the engine-state chain does not cover it: the gateway's checks all pass);
    # the verifier compares it with its own features
    "observation": (lambda j: "obs", Reason.REPLAY_MISMATCH, 1),
    # the chain recomputed from a forged row digest on (it no longer links to the honest next chunk)
    "engine_state": (lambda j: "chain", Reason.REPLAY_MISMATCH, 1),
    "flags": (lambda j: {"flags": lambda a: _set(a, j, 0 if a[j] else P.FLAG_TERMINATED)}, Reason.REPLAY_MISMATCH, 0),
}


def test_float_noise_of_another_cpu_is_tolerated(pool, tmp_path):
    """Chain v2: a client on another CPU / compiler (the native C++ client) rounds its features, policy outputs and
    rewards slightly differently. Such noise in every float array of every chunk passes the verifier (the
    engine-state chain is untouched); the forgery tests below show what does not."""
    pl = Player(pool, "forger_9", tmp_path)
    t = pl.task()
    chunks, _, _ = pl.play(t)

    def noisy(data, k):
        rng = np.random.default_rng(k)
        o = lambda a: (np.asarray(a, np.float32) + rng.uniform(-3e-5, 3e-5, np.shape(a)).astype(np.float32)).astype(np.float16)  # noqa: E731
        f = lambda rel: (lambda a: (np.asarray(a, np.float64) * (1 + rng.uniform(-rel, rel, np.shape(a)))).astype(np.float32))  # noqa: E731
        lp = lambda a: np.minimum(np.asarray(a, np.float32) + np.float32(2e-5), np.float32(0))  # noqa: E731
        return reseal(data, obs=o, rewards=f(5e-5), env_rewards=f(5e-5), values=f(5e-5), logp=lp, arc=f(1e-6))
    # the first row of chunk s + 1 must still equal the last row of chunk s (the gateway's link check)
    out, last = [], None
    for k, c in enumerate(chunks):
        n = P.decode_chunk(noisy(c, k))
        if last is not None:
            h = {kk: v for kk, v in n.header.items() if kk not in ("arrays", "protocol", "type")}
            arrays = {kk: np.array(v) for kk, v in n.arrays.items()}
            arrays["obs"][0] = last
            out.append(P.encode_chunk(h, arrays))
        else:
            out.append(P.encode_chunk({kk: v for kk, v in n.header.items() if kk not in ("arrays", "protocol", "type")},
                                      {kk: np.array(v) for kk, v in n.arrays.items()}))
        last = P.decode_chunk(out[-1]).arrays["obs"][-1].copy()
    assert any(P.decode_chunk(a).arrays["obs"].tobytes() != P.decode_chunk(b).arrays["obs"].tobytes()
               for a, b in zip(out, chunks))
    for c in out:
        assert pl.send(c).status == 200
    assert pool.verifier.run_once() == 1
    rows = pool.store.chunks(t.task_id)
    assert all(r["verify"] == "passed" for r in rows), [(r["seq"], r["verify"], r["reason"]) for r in rows]
    assert pl.me()["token_state"] == "active"


def _set(a, j, v):
    a = np.array(a)
    a[j] = v
    return a


def _forge(data: bytes, kind: str, j: int) -> bytes:
    spec = FORGERIES[kind][0](j)
    if spec not in ("obs", "chain"):
        return reseal(data, **spec)
    # row j + 1 (the observation after step j; not the chunk's first or last row, which the link check compares)
    c = P.decode_chunk(data)
    h = {k: v for k, v in c.header.items() if k not in ("arrays", "protocol", "type")}
    arrays = {k: np.array(v) for k, v in c.arrays.items()}
    if spec == "obs":  # the float16 copy changed far beyond the tolerance
        o32 = arrays["obs"].astype("<f4")
        o32[j + 1, 3] += 0.75
        arrays["obs"] = o32.astype("<f2")
    else:  # another engine state from row j + 1 on: the chain recomputed over forged row digests
        n = len(arrays["obs_hash"]) - (j + 1)
        arrays["obs_hash"][j + 1:] = P.chain_from_digests([b"\x01" * 32] * n, bytes(c.arrays["obs_hash"][j]))
    return P.encode_chunk(h, arrays)


@pytest.mark.parametrize("kind", list(FORGERIES))
def test_forged_chunks_are_caught_and_the_account_flagged(pool, tmp_path, kind):
    """A forged chunk is caught by the replay with the right reason and step; the task's data is rejected /
    quarantined (none of it reaches the learner); the account is flagged suspect, NOT revoked (the user decides):
    its other tasks are then replayed in full and fed only once verified; an alert with the mismatching values goes to
    <pool>/alerts/."""
    user = f"forger_{list(FORGERIES).index(kind)}"
    pl = Player(pool, user, tmp_path)
    t = pl.task()
    chunks, end, _ = pl.play(t)
    # forge chunk 1 when it has room (chunk 0 must then pass), else chunk 0; step j inside the chunk
    s = 1 if len(chunks) > 1 and P.decode_chunk(chunks[1]).steps >= 8 else 0
    c = P.decode_chunk(chunks[s])
    assert c.steps >= 8, "the fake task is too short for this test"
    j = 5
    forged = _forge(chunks[s], kind, j)
    later = chunks[s + 1:]
    if kind == "engine_state" and later:
        # a recomputed chain no longer links to the honest next chunk: the gateway refuses that one already
        for x in chunks[:s] + [forged]:
            assert pl.send(x).status == 200
        r = pl.send(later[0])
        assert r.status == 400 and pool.js(r)["reason"]["code"] == "chain_broken"
        pool.store.q("UPDATE tasks SET deadline=0 WHERE task_id=?", (t.task_id,))  # the verifier takes the prefix
    else:
        for x in chunks[:s] + [forged] + later:
            r = pl.send(x)
            assert r.status == 200, (kind, r.body)
    # a second task of the same account, in flight (not closed) when the mismatch is found
    t2 = pl.task()
    c2, _, _ = pl.play(t2)
    assert pl.send(c2[0]).status == 200
    pool.store.q("UPDATE tasks SET replay=0 WHERE task_id=?", (t2.task_id,))  # as if the spot check had not drawn it
    assert pool.verifier.run_once() == 1
    rows = pool.store.chunks(t.task_id)
    bad = [r for r in rows if r["verify"] == "failed"]
    assert len(bad) == 1 and bad[0]["seq"] == s and bad[0]["status"] == "rejected"
    reason = json.loads(bad[0]["reason"])
    _, want, off = FORGERIES[kind]
    step = c.header["step0"] + j + off
    if kind == "action":  # the honest log-prob does not fit the forged action (or, if it happens to, the replay differs)
        assert reason["code"] in (Reason.REPLAY_LOGP_MISMATCH.value, Reason.REPLAY_MISMATCH.value,
                                  Reason.REPLAY_REWARD_MISMATCH.value, Reason.REPLAY_ACTION_MISMATCH.value)
        assert reason["params"]["step"] >= step
        step = reason["params"]["step"]
        s = reason["params"]["chunk"]
    else:
        assert reason["code"] == want.value, (kind, reason)
    assert reason["params"] == {"task_id": t.task_id, "chunk": s, "step": step}, (kind, reason)
    # the task's data never reaches the learner: the replay-passed earlier chunks are quarantined too
    assert all(r["verify"] == "passed" and r["status"] == "quarantined" for r in rows if r["seq"] < s)
    assert all(r["status"] in ("quarantined", "rejected") for r in rows)
    assert not [r for r in pool.store.to_ingest(1000) if r["task_id"] == t.task_id]
    # flagged, not revoked: the player keeps working, sees the rejected chunk's reason verbatim
    acct = pool.store.account(rows[0]["forum_id"])
    assert acct["suspect"] == 1 and acct["revoked"] == 0
    me = pl.me()
    assert me["token_state"] == "active"
    assert me["stats"]["chunks_rejected"] == 1 and me["stats"]["last_rejection"]["code"] == reason["code"]
    assert f"第 {s} 段数据在第 {step} 步" in me["stats"]["last_rejection"]["message"]
    assert pool.call("GET", "/pool/task", token=pl.token, session=pl.sid).status == 200
    # the coordinator's alert: account, task, the mismatching values against their tolerance, the learner updates
    alerts = sorted((pool.cfg.pool_dir / "alerts").glob("*-suspect-*.json"))
    assert len(alerts) == 1
    rec = json.loads(alerts[0].read_text(encoding="utf-8"))
    assert rec["task_id"] == t.task_id and rec["chunk"] == s and rec["step"] == step and rec["reason"] == reason["code"]
    assert rec["pseudonym"].startswith("acct-") and "learner_updates_unreplayed" in rec
    if kind in ("reward", "value", "logp", "observation", "flags"):
        assert {"submitted", "replayed", "tolerance"} <= set(rec["values"]), rec
    assert any("flagged suspect (not revoked)" in m and alerts[0].name in m for m in pool.logs)
    # its in-flight task: replayed in full although not drawn, fed only after it passed
    assert [r["status"] for r in pool.store.chunks(t2.task_id)] == ["held"]
    assert not [r for r in pool.store.to_ingest(1000) if r["task_id"] == t2.task_id]
    pool.store.q("UPDATE tasks SET deadline=0 WHERE task_id=?", (t2.task_id,))
    assert pool.verifier.run_once() == 1
    r2 = pool.store.chunks(t2.task_id)
    assert [r["verify"] for r in r2] == ["passed"] and [r["status"] for r in r2] == ["released"]
    assert [r["task_id"] for r in pool.store.to_ingest(1000)] == [t2.task_id]
    # every new task of a suspect account is a full replay
    t3 = pl.task()
    assert pool.store.task(t3.task_id)["replay"] == 1


def test_sampling_after_probation_and_verifier_faults(pool, tmp_path):
    """Per-task spot checks: a probation task is replayed in full; after the probation (task_sample 0) a task is
    released unreplayed, every chunk of it still passing the cheap checks."""
    pool.cfg.probation_chunks = 1
    pool.cfg.task_sample = 0.0
    pl = Player(pool, "eligible", tmp_path)
    t = pl.task()
    chunks, _, _ = pl.play(t)
    for c in chunks:
        assert pl.send(c).status == 200
    assert all(r["selected"] == 1 for r in pool.store.chunks(t.task_id))  # the probation task: in full
    assert pool.verifier.run_once() == 1
    assert all(r["verify"] == "passed" for r in pool.store.chunks(t.task_id))
    t1 = pl.task()  # probation over (one chunk's steps replayed): no replay for this one
    c1, _, _ = pl.play(t1)
    for c in c1:
        assert pl.send(c).status == 200
    rows = pool.store.chunks(t1.task_id)
    assert all(r["selected"] == 0 and r["flag"] is None for r in rows)
    assert pool.verifier.run_once() == 1
    rows = pool.store.chunks(t1.task_id)
    assert all(r["verify"] == "skipped" and r["status"] == "released" for r in rows)
    assert pl.me()["stats"]["steps_verified"] == sum(r["steps"] for r in pool.store.q("SELECT steps FROM chunks"))
    # the verifier's own engine failing is replay_failed: retried, never a revocation
    t2 = pl.task()
    c2, _, _ = pl.play(t2)
    for c in c2:
        assert pl.send(c).status == 200
    pool.store.q("UPDATE chunks SET selected=1 WHERE task_id=?", (t2.task_id,))

    class Broken:
        backend = "headless"

        def make_channel(self, cfg):
            raise OSError("engine missing")

        def start(self, env):
            raise OSError("engine missing")

    pool.verifier.launcher_factory = Broken
    pool.verifier.retry_s = 0
    for _ in range(5):
        pool.verifier.run_once()
    rows = pool.store.chunks(t2.task_id)
    assert all(r["verify"] == "replay_failed" and r["status"] == "held" for r in rows)
    assert pl.me()["token_state"] == "active"


def test_deadline_closes_a_task_and_its_prefix_is_verified(pool, tmp_path):
    pl = Player(pool, "eligible", tmp_path)
    t = pl.task()
    chunks, _, _ = pl.play(t)
    if len(chunks) < 2:
        pytest.skip("the task ended within one chunk")
    assert pl.send(chunks[0]).status == 200  # the player vanished after chunk 0
    assert pool.verifier.run_once() == 0  # still open
    pool.store.q("UPDATE tasks SET deadline=0 WHERE task_id=?", (t.task_id,))
    r = pl.send(chunks[1])
    assert r.status == 410 and pool.js(r)["reason"]["code"] == "task_expired"
    assert pool.verifier.run_once() == 1
    assert [r["verify"] for r in pool.store.chunks(t.task_id)] == ["passed"]


@pytest.mark.parametrize("sampler,stop_after", [("client", None), ("client", 70), ("rollout", 70), ("client", 3)])
def test_both_samplers_pass_the_verifier(pool, tmp_path, sampler, stop_after):
    """The controller's sampler (client_sampler.TaskSampler) and rollout.sample_task produce chunks the verifier
    (rollout.replay_task) reproduces bit for bit, including a player's stop and respawns in the same process."""
    pl = Player(pool, "eligible", tmp_path)
    ends = set()
    for seed in range(3):
        t = pl.task()
        chunks, end, _ = pl.play(t, sampler=sampler, stop_after=stop_after)
        ends.add(end)
        for c in chunks:
            r = pl.send(c)
            assert r.status == 200, r.body
        if stop_after is not None:
            assert end == "stopped" and P.decode_chunk(chunks[-1]).header["end"] == "stopped"
        assert pool.verifier.run_once() == 1
        rows = pool.store.chunks(t.task_id)
        assert rows and all(r["verify"] == "passed" for r in rows), (sampler, seed, end, pool.logs[-3:])
    assert pl.me()["token_state"] == "active"


def test_controller_gateway_client_against_the_real_gateway(pool, tmp_path):
    """The controller's HTTP client (client_api.GatewayClient + Cache) against this gateway over real HTTP: login
    poll, hello, task, weights / assets downloads, chunk upload, verification, me, leaderboard, and a refusal."""
    import threading
    api_mod = pytest.importorskip("ballrain.pool.client_api")
    cs = pytest.importorskip("ballrain.pool.client_sampler")
    from ballrain.pool.server_app import make_server
    srv = make_server(pool.app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        api = api_mod.GatewayClient(base)
        ls = api.login_start()
        assert ls["browser_url"].startswith(f"{PUBLIC_}/pool/login?login=")
        # the player's browser (the forum is the mock; the gateway pages are served through the app)
        q = urllib.parse.urlsplit(ls["browser_url"]).query
        page, _, _ = pool.browser("/pool/login?" + q, "eligible")
        assert page.status == 200
        st = api.login_poll(ls["login_id"], ls["poll_secret"])
        assert st["status"] == "ok"
        api.token = st["token"]
        ok = api.hello("windows", {"kind": "x87soft", "sha256": ENGINE_SHA, "version": "1"},
                       {n: GAME[n] for n in P.GAME_FILE_LIST}, 2)
        assert ok["leaderboard"] is True and api.session_id == ok["session_id"]
        t = T.Task.from_dict(api.task()["task"])
        cache = api_mod.Cache(tmp_path / "cc", api)
        pol, m = cache.policy(t.policy_sha)
        env = T.build_env(t, m, T.LocalGame("x87soft", "SimTool", "Ballance"), cache.starts(t.assets["starts"]),
                          cache.path(t.assets["path"], t.level), tmp_path / "cc-out",
                          launcher=fakegame.FakeLauncher(backend="headless"))
        results = []
        try:
            cs.TaskSampler(env, m["rollout"]).run_task(t, cs.PolicyActions.for_task(pol, t),
                                                       lambda data, header: results.append(api.chunk(data)),
                                                       n_act=pol.n_act)
        finally:
            env.close()
        assert results and all(r["status"] == "accepted" for r in results)
        assert pool.verifier.run_once() == 1
        me = api.me()
        assert me["stats"]["steps_verified"] == sum(r["steps"] for r in results) and me["token_state"] == "active"
        assert api.leaderboard()["entries"][0]["nickname"] == "合格玩家"
        hb = api.heartbeat(True, [], 1.0)
        assert hb["max_instances"] == 8
        with pytest.raises(api_mod.GatewayRefusal) as e:  # a refusal comes back with the reason, verbatim
            api.chunk(b"\x00" * 100)
        assert e.value.code == "bad_chunk" and not e.value.fatal
    finally:
        srv.shutdown()
        srv.server_close()


# ============================================================================================== action draws / engine faults

def _replay_reason(pool, task_id):
    bad = [r for r in pool.store.chunks(task_id) if r["verify"] == "failed"]
    return json.loads(bad[0]["reason"]) if bad else None


@pytest.mark.parametrize("sampler", ["client", "rollout"])
def test_actions_not_drawn_from_the_task_seed_are_caught(pool, tmp_path, sampler):
    """A client choosing its own actions (its own generator) while reporting the honest log-probs and values: every
    other check passes, the action draw does not (replay_action_mismatch), and the account is flagged suspect."""
    pl = Player(pool, "forger_0" if sampler == "client" else "forger_1", tmp_path)
    t = pl.task()
    chunks, _, _ = pl.play(t, seed=12345, sampler=sampler)
    for c in chunks:
        assert pl.send(c).status == 200
    assert pool.verifier.run_once() == 1
    reason = _replay_reason(pool, t.task_id)
    assert reason is not None and reason["code"] == Reason.REPLAY_ACTION_MISMATCH.value, reason
    assert pl.me()["token_state"] == "active" and pool.store.is_suspect(pool.store.task(t.task_id)["forum_id"])


def test_a_trajectory_resigned_onto_another_task_is_caught(pool, tmp_path, monkeypatch):
    """The cached-trajectory attack: task B gets task A's start (same k; the fake game has no ball order), A's honest
    chunks are re-signed onto B's task_id (chain recomputed). B's sample_seed differs, so its action draws differ."""
    pl = Player(pool, "forger_2", tmp_path)
    ta = pl.task()
    k = ta.start_frame
    monkeypatch.setattr(pool.policies, "pick_start", lambda pub: (k, None))
    tb = pl.task()
    assert tb.start_frame == k and tb.sample_seed != ta.sample_seed
    rows = []  # the attacker keeps the engine row digests of its honest run (the chunk only has the chain values)
    real = P.chain_from_digests

    def keep(digests, prev):
        rows.append([bytes(d) for d in digests])
        return real(digests, prev)
    monkeypatch.setattr(P, "chain_from_digests", keep)
    chunks, _, _ = pl.play(ta)
    monkeypatch.setattr(P, "chain_from_digests", real)
    assert len(rows) == len(chunks)
    prev = P.chain_seed(tb.task_id)
    for data, dg in zip(chunks, rows):
        c = P.decode_chunk(data)
        h = {kk: v for kk, v in c.header.items() if kk not in ("arrays", "protocol", "type")}
        arrays = {kk: np.array(v) for kk, v in c.arrays.items()}
        arrays["obs_hash"] = P.chain_from_digests(dg, prev)
        h.update(task_id=tb.task_id, chain_prev=prev.hex())
        prev = bytes(arrays["obs_hash"][-2])
        assert pl.send(P.encode_chunk(h, arrays)).status == 200
    pool.store.q("UPDATE tasks SET deadline=0 WHERE task_id=?", (ta.task_id,))  # A itself was never sent
    pool.verifier.run_once()
    pool.verifier.run_once()
    reason = _replay_reason(pool, tb.task_id)
    assert reason is not None and reason["code"] == Reason.REPLAY_ACTION_MISMATCH.value, (reason, pool.logs[-3:])
    assert pl.me()["token_state"] == "active" and pool.store.is_suspect(pool.store.task(tb.task_id)["forum_id"])


def _peergone_after(n_steps, steps_of):
    from ballrain.transport import PeerGone

    def inject(sess, owner):
        orig = sess._next_obs

        def nxt(*a, **k):
            if steps_of(owner) >= n_steps:
                raise PeerGone("engine died (test)")
            return orig(*a, **k)
        sess._next_obs = nxt
    return inject


def test_client_engine_dying_mid_task_is_not_a_mismatch(pool, tmp_path):
    """BallanceEnv.step catches PeerGone / Timeout and truncates; the controller's sampler must treat that as an engine
    failure (drop the half-run step, end "error" on the previous one), so the honest chunks still pass the replay."""
    pl = Player(pool, "eligible", tmp_path)
    t = pl.task()
    inj = _peergone_after(20, lambda n: n[0])
    chunks, end, _ = pl.play(t, sampler="client", on_start=lambda e, n: inj(e.session, n))
    assert end == "error"
    last = P.decode_chunk(chunks[-1])
    assert last.header["end"] == "error" and last.header["step0"] + last.steps == 20
    for c in chunks:
        assert pl.send(c).status == 200
    assert pool.verifier.run_once() == 1
    assert all(r["verify"] == "passed" for r in pool.store.chunks(t.task_id)), pool.logs[-3:]
    assert pl.me()["token_state"] == "active"


def test_client_engine_dying_in_a_respawn_is_not_a_mismatch(pool, tmp_path, monkeypatch):
    """The engine dies during the respawn fast-forward after a death: the death step is dropped, the task ends
    "error" on the step before it, and the replay passes. (Forced actions off the fake road make the deaths; the
    replay's action-draw check is switched off for that.)"""
    from ballrain.pool import client_sampler as cs
    from ballrain.transport import PeerGone
    monkeypatch.setattr(R, "action_consistent", lambda *a, **k: True)
    pl = Player(pool, "eligible", tmp_path)
    t = pl.task()
    hit = []

    def on_start(e, n):
        orig = e.reset

        def reset(*a, **k):
            if (k.get("options") or {}).get("relaunch") is False:
                hit.append(n[0])
                raise PeerGone("engine died in the respawn (test)")
            return orig(*a, **k)
        e.reset = reset
    seq = [0, 0, 1, 1, 1, 0] * 3 + [3] * 400  # a few steps on the road, then off it (LEFT)
    chunks, end, _ = pl.play(t, sampler="client", on_start=on_start, actions=lambda pol: cs.ReplayActions(pol, seq))
    assert hit and end == "error"
    last = P.decode_chunk(chunks[-1])
    assert last.header["end"] == "error" and last.header["step0"] + last.steps == hit[0]  # (steps before the death)
    assert not last.arrays["flags"][-1] & P.FLAG_TERMINATED  # the death step itself is not in the chunk
    for c in chunks:
        assert pl.send(c).status == 200
    assert pool.verifier.run_once() == 1
    assert all(r["verify"] == "passed" for r in pool.store.chunks(t.task_id)), pool.logs[-3:]
    assert pl.me()["token_state"] == "active"


def test_verifier_engine_dying_mid_replay_is_replay_failed(pool, tmp_path, monkeypatch):
    """The verifier's own engine dies in the middle of a replay: replay_failed (retried later), never a revocation."""
    pl = Player(pool, "eligible", tmp_path)
    t = pl.task()
    chunks, _, _ = pl.play(t)
    for c in chunks:
        assert pl.send(c).status == 200
    orig = R.TaskRunner.start
    inj = _peergone_after(30, lambda runner: runner.steps)

    def start(self):
        o = orig(self)
        inj(self.env.session, self)
        return o
    monkeypatch.setattr(R.TaskRunner, "start", start)
    pool.verifier.retry_s = 0
    assert pool.verifier.run_once() == 1
    assert any("replay_failed" in x for x in pool.logs), pool.logs[-3:]
    rows = pool.store.chunks(t.task_id)
    assert all(r["status"] == "held" and r["verify"] == "pending" for r in rows)
    assert pool.store.task(t.task_id)["verify_attempts"] == 1
    assert pl.me()["token_state"] == "active"
    monkeypatch.setattr(R.TaskRunner, "start", orig)  # the engine works again: the retry passes
    pool.verifier.run_once()
    assert all(r["verify"] == "passed" for r in pool.store.chunks(t.task_id))


# ---------------------------------------------------------------------------------------------- format 3 starts
COUNTER_RAND = {"lives": {"values": [0.0, 0.1, 0.2, 0.3]}, "sr_s": [0.0, 0.3], "points": [-0.05, 0.45]}


@pytest.fixture(scope="module")
def world3(tmp_path_factory):
    """The toy run with the trainer's at-rest starts (a drive target and a brake-only entry) and counter_rand."""
    w = dev.fake_world(tmp_path_factory.mktemp("pool-world3"),
                       rollout={"counter_rand": json.dumps(COUNTER_RAND), "p_early_respawn": 0.5})
    ec = w["state"]["env_config"]
    # [100, 200] lies before the frontier (the middle of the record): rehearse_behind_frontier never draws it
    ec.update(rest_starts=[[300, 330, 1.0, -42.0, 0.5, 400], [200, 230]], p_rest=0.7,
              rehearse_starts=[[500, 560], [100, 200]], p_rehearse=0.3, rehearse_behind_frontier=True)
    return w


@pytest.fixture
def pool3(tmp_path, world3):
    g = Gw(tmp_path, users=FORGERS, max_steps=120, probation_chunks=200, sample_rate=0.1)
    g.policies.datasets_dir = world3["datasets"]
    g.policies.path_dirs = world3["path_dirs"]
    g.pub = g.policies.publish_state(world3["state"])
    g.verifier = Verifier(g.store, g.cfg.pool_dir, g.cfg.engines,
                          launcher_factory=lambda: fakegame.FakeLauncher(backend="headless"), log=g.logs.append)
    yield g
    g.close()


def test_counter_draws_are_the_tasks():
    from ballrain.pool import counters as CR
    sl = {"lives": slice(10, 11), "points": slice(11, 12), "sr_s": slice(12, 13)}
    assert not CR.active(COUNTER_RAND, 0) and CR.active(COUNTER_RAND, 5) and not CR.active({}, 5)
    a = [CR.draw(COUNTER_RAND, 1234, i, sl) for i in range(400)]
    assert a == [CR.draw(COUNTER_RAND, 1234, i, sl) for i in range(400)]  # replayable from the seed
    assert a[0] != CR.draw(COUNTER_RAND, 1235, 0, sl)
    lives = [d[sl["lives"]] for d in a]
    assert {op for op, _ in lives} == {"set"} and {v for _, v in lives} == {0.0, 0.1, 0.2, 0.3}
    sr = [v for op, v in (d[sl["sr_s"]] for d in a)]
    pts = [v for op, v in (d[sl["points"]] for d in a)]
    assert 0.0 <= min(sr) and max(sr) <= 0.3 and -0.05 <= min(pts) and max(pts) <= 0.45
    p = CR.draw({"lives": {"values": [1.0, 2.0], "p": [0.0, 1.0]}}, 9, 3, sl)
    assert p == {sl["lives"]: ("set", 2.0)}
    o = np.zeros(16, np.float32)
    o[11], o[12] = 0.5, 0.25
    r = CR.apply(o, a[0])
    assert r.dtype == np.float32 and o[10] == 0 and r[10] == np.float32(a[0][sl["lives"]][1])
    assert r[11] == np.float32(0.5) + a[0][sl["points"]][1]


def test_format3_tasks_follow_the_trainers_start_draws(pool3):
    ts = [pool3.policies.new_task("x87soft", 1000, 120, 60000, 3600) for _ in range(300)]
    for t in ts:
        P.check(P.TASK, t.to_dict())
    rest = [t for t in ts if t.rest is not None]
    drive = [t for t in rest if t.rest["target"] is not None]
    brake = [t for t in rest if t.rest["target"] is None]
    assert drive and brake and len(rest) < len(ts)
    assert all(300 <= t.start_frame <= 330 and t.rest["max_frames"] == 400 for t in drive)
    assert all(abs(t.rest["target"][0] - 1.0) <= 0.5 and abs(t.rest["target"][1] + 42.0) <= 0.5 for t in drive)
    assert all(200 <= t.start_frame <= 230 and "max_frames" not in t.rest for t in brake)
    assert any(500 <= t.start_frame <= 560 and t.rest is None for t in ts)  # rehearsal draws
    assert not any(100 <= t.start_frame < 200 for t in ts)  # behind the frontier only
    assert all(t.life_seed and t.respawn_frontier == pool3.pub.curriculum.frontier for t in ts)
    assert T.Task.from_dict(drive[0].to_dict()) == drive[0]


@pytest.mark.parametrize("sampler", ["rollout", "client"])
def test_format3_tasks_pass_the_verifier(pool3, tmp_path, sampler):
    """At-rest starts (driven / braked) and per-life counter values: what the client samples, the verifier replays."""
    pl = Player(pool3, "eligible", tmp_path)
    kinds = set()
    for _ in range(4):
        t = pl.task()
        chunks, end, pol = pl.play(t, sampler=sampler)
        dec = [P.decode_chunk(c, n_act=pol.n_act, n_obs=pol.n_obs) for c in chunks]
        kinds.add("drive" if t.rest and t.rest["target"] else "brake" if t.rest else "tas" if t.start_frame else "level")
        if t.start_frame > 0:  # the lives feature carries the drawn value, not the true 3 lives
            li = _slices(pool3.pub.manifest)["lives"].start
            seen = sorted({round(float(v), 3) for d in dec for v in d.arrays["obs"][:, li]})
            assert set(seen) <= {0.0, 0.1, 0.2, 0.3}, seen
        for c in chunks:
            assert pl.send(c).status == 200
        assert pool3.verifier.run_once() == 1
        rows = pool3.store.chunks(t.task_id)
        assert rows and all(r["verify"] == "passed" for r in rows), (sampler, t.rest, end, pool3.logs[-3:])
    assert kinds - {"level"}, kinds


def test_early_respawns_are_skipped_as_the_learner_does(pool3, tmp_path):
    """p_early_respawn 0.5: a death whose respawn sector the record enters before the frontier ends the task
    (respawn_skipped) for about half the deaths; the verifier replays the decision."""
    from ballrain.pool import counters as CR
    pl = Player(pool3, "eligible", tmp_path)
    ends = []
    for i in range(40):  # the policy's own deaths (the task's draws): until a skipped respawn and a played one
        t = pl.task()
        chunks, end, _ = pl.play(t, sampler="client" if i % 2 else "rollout")
        ends.append(end)
        for c in chunks:
            assert pl.send(c).status == 200
        assert pool3.verifier.run_once() == 1
        assert all(r["verify"] == "passed" for r in pool3.store.chunks(t.task_id)), (end, pool3.logs[-3:])
        if "respawn_skipped" in ends and i >= 3:
            break
    assert "respawn_skipped" in ends, ends
    t = T.Task.from_dict(dict(pl.task().to_dict(), start_frame=0))
    ro = {"p_early_respawn": 0.0, "respawn_level_starts": True}
    assert not CR.skip_respawn(ro, t, 0, 1, lambda s: 0)  # level starts play every respawn
    assert CR.skip_respawn(dict(ro, respawn_level_starts=False), t, 0, 1, lambda s: 0)
    assert not CR.skip_respawn({"p_early_respawn": 0.0}, t, 0, 1, lambda s: t.respawn_frontier)  # at the frontier
    assert not CR.skip_respawn({"p_early_respawn": 1.0}, t, 0, 1, lambda s: 0)


def _slices(m) -> dict:
    from ballrain.features import FeatureConfig, FeatureExtractor
    f = dict(m["features"])
    for k in ("far_waypoints", "ref_speed_at"):
        f[k] = tuple(f.get(k) or ())
    return FeatureExtractor(FeatureConfig(**f)).slices()


def test_wrong_counter_values_are_caught(pool3, tmp_path, monkeypatch):
    """A client that leaves the counters true (or draws its own) sends observations the replay does not reproduce."""
    from ballrain.pool import counters as CR
    pl = Player(pool3, "forger_3", tmp_path)
    for _ in range(20):
        t = pl.task()
        if t.start_frame > 0:
            break
    else:
        pytest.skip("no TAS start drawn")
    monkeypatch.setattr(CR, "draw", lambda spec, seed, life, slices: None)
    chunks, end, _ = pl.play(t)
    monkeypatch.undo()
    for c in chunks:
        pl.send(c)
    pool3.verifier.run_once()
    rows = pool3.store.chunks(t.task_id)
    assert any(r["verify"] == "failed" for r in rows), rows


def test_task_end_closes_a_task_without_a_final_chunk(pool, tmp_path):
    """A task whose engine failed at the start (no chunks) or right after a cut (no final chunk): the client's task_end
    closes it at once (instance slot free, the verifier takes what arrived); a wrong chunk count is refused."""
    pl = Player(pool, "eligible", tmp_path)

    def end(tid, n, why="failed: engine"):
        return pool.call("POST", "/pool/task", P.encode_message("task_end", task_id=tid, chunks=n, reason=why),
                         token=pl.token, session=pl.sid)
    t = pl.task()
    assert pool.store.task(t.task_id)["state"] == "open"
    r = end(t.task_id, 1)
    assert r.status == 409 and pool.js(r)["reason"]["code"] == "chunk_out_of_order"
    r = end(t.task_id, 0)
    assert r.status == 200 and P.parse_message("task_end_ok", r.body)["state"] == "closed"
    assert pool.store.task(t.task_id)["state"] == "closed"
    assert pool.store.open_tasks(pool.store.task(t.task_id)["forum_id"], 3600) == 0
    assert P.parse_message("task_end_ok", end(t.task_id, 0).body)["state"] == "closed"  # idempotent
    assert pool.verifier.run_once() == 1 and pool.store.task(t.task_id)["state"] == "done"  # "empty"
    # some chunks, then the engine died right after a cut: the prefix is verified as at a deadline
    t2 = pl.task()
    chunks, _, _ = pl.play(t2)
    assert len(chunks) >= 2
    assert pl.send(chunks[0]).status == 200
    other = Player(pool, "forger_1", tmp_path)
    r = pool.call("POST", "/pool/task", P.encode_message("task_end", task_id=t2.task_id, chunks=1, reason="x"),
                  token=other.token, session=other.sid)
    assert r.status == 403 and pool.js(r)["reason"]["code"] == "task_not_yours"
    assert end(t2.task_id, 1).status == 200
    assert pl.send(chunks[1]).status == 409  # closed: nothing more is accepted
    assert pool.verifier.run_once() == 1
    rows = pool.store.chunks(t2.task_id)
    assert [r["verify"] for r in rows] == ["passed"]


def test_revoke_and_unflag_are_manual_admin_actions(pool, tmp_path, capsys):
    """tools/pool/gateway.py admin: suspects lists the flagged account; unflag clears it; revoke (manual) shows the
    player the stored replay reason and can list / mark the learner updates its data reached (nothing is undone)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("gw_cli", Path(__file__).resolve().parents[1] / "tools" / "pool" / "gateway.py")
    gw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gw)
    pl = Player(pool, "forger_3", tmp_path)
    t = pl.task()
    chunks, _, _ = pl.play(t)
    for c in chunks[:-1] + [reseal(chunks[-1], rewards=lambda r: r + np.float32(0.25))]:
        assert pl.send(c).status == 200
    assert pool.verifier.run_once() == 1
    fid = pool.store.task(t.task_id)["forum_id"]
    pool.store.q("UPDATE chunks SET ingested=12 WHERE task_id=? AND seq=0 AND status!='rejected'", (t.task_id,))
    base = ["--pool-dir", str(pool.cfg.pool_dir)]
    gw.main(base + ["admin", "suspects"])
    out = capsys.readouterr().out
    assert str(fid) in out and "replay_reward_mismatch" in out
    gw.main(base + ["admin", "unflag", "--forum-id", str(fid)])
    assert not pool.store.is_suspect(fid) and pl.me()["token_state"] == "active"
    pool.store.flag_suspect(fid, json.loads(sorted((pool.cfg.pool_dir / "alerts").glob("*.json"))[-1].read_text("utf-8")))
    gw.main(base + ["admin", "revoke", "--forum-id", str(fid), "--list-updates", "--mark-updates"])
    out = capsys.readouterr().out
    assert f"revoked {fid}" in out and "learner updates" in out and "marked:" in out
    me = pl.me()
    assert me["token_state"] == "revoked" and me["revoke_reason"]["code"] == "token_revoked_replay"
    assert me["revoke_reason"]["params"]["task_id"] == t.task_id
    assert list((pool.cfg.pool_dir / "alerts").glob("*-revoked-acct-*.json"))


def test_prefetch_sessions_may_hold_three_tasks_per_instance(pool, tmp_path):
    """task_prefetch_v1 (0.5.1): a session may hold 3 open tasks per hello instance (playing, prepared, the previous
    one's upload tail), the account 3 x max_instances; a session without it keeps the account-wide max_instances."""
    pl = Player(pool, "eligible", tmp_path)

    def hello(**over):
        return P.parse_message("hello_ok", pool.hello(pl.token, **over).body)["session_id"]

    def get(sid):
        r = pool.call("GET", "/pool/task", token=pl.token, session=sid)
        return r.status, pool.js(r)
    mi = int(pool.cfg.max_instances)
    pre = hello(instances=2, capabilities=["task_prefetch_v1"])
    st, js = get(pre)
    assert st == 200
    fid = pool.store.task(js["task"]["task_id"])["forum_id"]
    for _ in range(5):
        assert get(pre)[0] == 200
    st, js = get(pre)
    assert st == 400 and js["reason"]["code"] == "too_many_instances" and js["reason"]["params"]["open_limit"] == 6
    # the old rule on the same account: the account's open tasks (6 so far) against max_instances
    old = hello(instances=2)
    while pool.store.open_tasks(fid, 90) < mi:
        assert get(old)[0] == 200
    st, js = get(old)
    assert st == 400 and js["reason"]["code"] == "too_many_instances" and "open_limit" not in js["reason"]["params"]
    # a big prefetch session: its own limit is 3 x instances, the account's 3 x max_instances
    big = hello(instances=mi, capabilities=["task_prefetch_v1"])
    while pool.store.open_tasks(fid, 90) < 3 * mi:
        assert get(big)[0] == 200
    st, js = get(big)
    assert st == 400 and js["reason"]["params"]["open_limit"] == 3 * mi
