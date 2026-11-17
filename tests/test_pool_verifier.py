"""Pool task -> chunks -> verifier, end to end on ballrain.fakegame (no game process): the shared rollout
(ballrain.pool.rollout) samples a task in the client's env and replays it bit-exactly in the verifier's; the gateway
accepts the honest chunks (and refuses duplicates, gaps, other policies, other accounts' tasks, broken chains); the
verifier passes them, and catches FORGED chunks (tampered reward / observation / action / log-prob / value) with the
right reason code and step, quarantining the account's unverified chunks and revoking it."""
import json
import sys
import urllib.parse

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
                            "is_email_confirmed": True, "comment_count": 9} for i in range(8)}


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
    sdir, pdir = pool.verifier._assets(t)
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
    # the float16 copy and the chain recomputed from the forged row on (the gateway's checks all pass)
    "observation": (lambda j: "obs", Reason.REPLAY_MISMATCH, 1),
    "flags": (lambda j: {"flags": lambda a: _set(a, j, 0 if a[j] else P.FLAG_TERMINATED)}, Reason.REPLAY_MISMATCH, 0),
}


def _set(a, j, v):
    a = np.array(a)
    a[j] = v
    return a


def _forge(data: bytes, kind: str, j: int) -> bytes:
    spec = FORGERIES[kind][0](j)
    if spec != "obs":
        return reseal(data, **spec)
    # row j + 1 (the observation after step j; not the chunk's first or last row, which the link check compares):
    # the float16 copy changed and the chain recomputed from that row on, so every gateway check still passes
    c = P.decode_chunk(data)
    h = {k: v for k, v in c.header.items() if k not in ("arrays", "protocol", "type")}
    arrays = {k: np.array(v) for k, v in c.arrays.items()}
    o32 = arrays["obs"].astype("<f4")
    o32[j + 1, 3] += 0.75
    arrays["obs"] = o32.astype("<f2")
    arrays["obs_hash"][j + 1:] = P.chain_hashes(o32[j + 1:], bytes(c.arrays["obs_hash"][j]))
    return P.encode_chunk(h, arrays)


@pytest.mark.parametrize("kind", list(FORGERIES))
def test_forged_chunks_are_caught_and_the_account_revoked(pool, tmp_path, kind):
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
    if kind == "observation" and later:
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
    # a second task of the same account, in flight (not closed): its chunks must end up quarantined
    t2 = pl.task()
    c2, _, _ = pl.play(t2)
    assert pl.send(c2[0]).status == 200
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
    assert all(r["verify"] == "passed" for r in rows if r["seq"] < s)
    assert [r["status"] for r in pool.store.chunks(t2.task_id)] == ["quarantined"]
    # the player sees why, verbatim, on every request
    r = pool.call("GET", "/pool/task", token=pl.token, session=pl.sid)
    assert r.status == 403
    rr = pool.js(r)["reason"]
    assert rr["code"] == "token_revoked_replay" and rr["params"] == {"task_id": t.task_id, "chunk": s, "step": step}
    assert f"第 {s} 段数据在第 {step} 步" in rr["message"]
    me = pl.me()
    assert me["token_state"] == "revoked" and me["revoke_reason"] == rr
    assert me["stats"]["chunks_rejected"] == 1 and me["stats"]["last_rejection"]["code"] == reason["code"]
    assert pool.js(pool.call("POST", "/pool/hello", b"{}", token=pl.token))["reason"]["code"] == "token_revoked_replay"
    lb = P.parse_message("leaderboard", pool.call("GET", "/pool/leaderboard").body)
    assert all(e["nickname"] != FORGERS[user]["nickname"] for e in lb["entries"])


def test_sampling_after_probation_and_verifier_faults(pool, tmp_path):
    pool.cfg.probation_chunks = 1
    pool.cfg.sample_rate = 0.0
    pl = Player(pool, "eligible", tmp_path)
    t = pl.task()
    chunks, _, _ = pl.play(t)
    for c in chunks:
        assert pl.send(c).status == 200
    sel = [r["selected"] for r in pool.store.chunks(t.task_id)]
    assert sel == [1] + [0] * (len(chunks) - 1)  # only the probation chunk is replayed
    assert pool.verifier.run_once() == 1
    rows = pool.store.chunks(t.task_id)
    assert rows[0]["verify"] == "passed" and all(r["verify"] == "skipped" for r in rows[1:])
    assert pl.me()["stats"]["steps_verified"] == sum(r["steps"] for r in rows)  # sampled-out chunks count
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
    other check passes, the action draw does not (replay_action_mismatch), and the account is revoked."""
    pl = Player(pool, "forger_0" if sampler == "client" else "forger_1", tmp_path)
    t = pl.task()
    chunks, _, _ = pl.play(t, seed=12345, sampler=sampler)
    for c in chunks:
        assert pl.send(c).status == 200
    assert pool.verifier.run_once() == 1
    reason = _replay_reason(pool, t.task_id)
    assert reason is not None and reason["code"] == Reason.REPLAY_ACTION_MISMATCH.value, reason
    assert pl.me()["token_state"] == "revoked"


def test_a_trajectory_resigned_onto_another_task_is_caught(pool, tmp_path, monkeypatch):
    """The cached-trajectory attack: task B gets task A's start (same k; the fake game has no ball order), A's honest
    chunks are re-signed onto B's task_id (chain recomputed). B's sample_seed differs, so its action draws differ."""
    pl = Player(pool, "forger_2", tmp_path)
    ta = pl.task()
    k = ta.start_frame
    monkeypatch.setattr(pool.policies, "pick_start", lambda pub: k)
    tb = pl.task()
    assert tb.start_frame == k and tb.sample_seed != ta.sample_seed
    rows32 = []  # the attacker keeps the float32 rows of its honest run (the chunk only has float16 copies)
    real = P.chain_hashes

    def keep(obs32, prev):
        rows32.append(np.array(obs32, "<f4"))
        return real(obs32, prev)
    monkeypatch.setattr(P, "chain_hashes", keep)
    chunks, _, _ = pl.play(ta)
    monkeypatch.setattr(P, "chain_hashes", real)
    assert len(rows32) == len(chunks)
    prev = P.chain_seed(tb.task_id)
    for data, o32 in zip(chunks, rows32):
        c = P.decode_chunk(data)
        h = {kk: v for kk, v in c.header.items() if kk not in ("arrays", "protocol", "type")}
        arrays = {kk: np.array(v) for kk, v in c.arrays.items()}
        arrays["obs_hash"] = P.chain_hashes(o32, prev)
        h.update(task_id=tb.task_id, chain_prev=prev.hex())
        prev = bytes(arrays["obs_hash"][-2])
        assert pl.send(P.encode_chunk(h, arrays)).status == 200
    pool.store.q("UPDATE tasks SET deadline=0 WHERE task_id=?", (ta.task_id,))  # A itself was never sent
    pool.verifier.run_once()
    pool.verifier.run_once()
    reason = _replay_reason(pool, tb.task_id)
    assert reason is not None and reason["code"] == Reason.REPLAY_ACTION_MISMATCH.value, (reason, pool.logs[-3:])
    assert pl.me()["token_state"] == "revoked"


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
