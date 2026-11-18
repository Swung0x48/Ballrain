"""The pool connected to the learner (docs/pool-prototype.md 11-13), on top of task format 3 (the native branch's):
- the weights export refuses shaping the pool cannot reproduce and carries the format-3 rollout fields;
- the learner link: hub -> live policy published with the hub's EnvConfig / curriculum; released chunks -> the
  learner's chunk messages (only released ones), marked as ingested;
- V-trace (ppo.vtrace) and the learner's --pool-ingest flag (off: pool chunks dropped and counted; on: trained on);
- per-task spot checks: unsampled tasks released after the cheap per-chunk policy-consistency check; a flagged chunk
  escalates to a full replay; a mismatch rejects the task's data and flags the account suspect (never an automatic
  revocation) with an alert listing the learner updates its unreplayed data reached."""
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ballrain.pool import dev, learner_link as LL, protocol as P, rollout as R, sampling as S, task as T, weights as W

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

win_only = pytest.mark.skipif(sys.platform != "win32", reason="ballrain.fakegame's shared memory needs Windows")

COUNTERS = {"lives": {"values": [0.0, 0.1, 0.2, 0.3], "p": [0.1, 0.2, 0.3, 0.4]}, "sr_s": [0.0, 0.3],
            "points": [-0.05, 0.45]}




# ============================================================================================== export

@pytest.fixture(scope="module")
def world(tmp_path_factory):
    if sys.platform != "win32":
        pytest.skip("ballrain.fakegame needs Windows")
    w = dev.fake_world(tmp_path_factory.mktemp("pool-learner-world"),
                       rollout={"counter_rand": json.dumps(COUNTERS), "p_early_respawn": 0.0,
                                "respawn_level_starts": True})
    st = w["state"]
    from ballrain.curriculum import TasStarts
    v = TasStarts(str(w["record_dir"])).valid
    ec = st["env_config"]
    ec["rehearse_starts"] = [[0, 0], [int(v[1]), int(v[-2])]]
    ec["p_rehearse"] = 0.3
    ec["rehearse_behind_frontier"] = True
    ec["rest_starts"] = [[int(v[2]), int(v[5])], [int(v[3]), int(v[6]), 0.0, -30.0, 1.0, 120]]
    ec["p_rest"] = 0.3
    return w


def test_export_carries_the_new_rollout_fields_and_refuses_unsupported_shaping(world, tmp_path):
    m = W.export_state(world["state"], tmp_path, "r")
    ro = m["rollout"]
    assert ro["counter_rand"] == COUNTERS and ro["p_early_respawn"] == 0.0 and ro["respawn_level_starts"] is True
    assert "rest_starts" not in m["env"] and "rehearse_starts" not in m["env"]  # server-side only
    for k, v in (("zones", json.dumps({"z": {"box": [0, 1, 0, 1], "penalty": 1}})), ("wall_push_rate", 0.5),
                 ("rail_gate", {"s": 3, "below": 4})):
        st = dict(world["state"], args=dict(world["state"]["args"], **{k: v}))
        with pytest.raises(ValueError, match=k):
            W.export_state(st, tmp_path / "x", "r")


# ============================================================================================== gateway + verifier

@pytest.fixture
def pool(tmp_path, world):
    from ballrain import fakegame
    from ballrain.pool.server_verify import Verifier
    from test_pool_gateway import Gw
    g = Gw(tmp_path, max_steps=160, probation_chunks=200, task_sample=0.15)
    g.policies.datasets_dir = world["datasets"]
    g.policies.path_dirs = world["path_dirs"]
    g.pub = g.policies.publish_state(world["state"])
    g.verifier = Verifier(g.store, g.cfg.pool_dir, g.cfg.engines,
                          launcher_factory=lambda: fakegame.FakeLauncher(backend="headless"), log=g.logs.append)
    yield g
    g.close()


# ============================================================================================== learner link

def _chunk(T_=5, n_obs=4, trunc_last=False, term_at=None, seed=0):
    rng = np.random.default_rng(seed)
    flags = np.zeros(T_, np.uint8)
    if term_at is not None:
        flags[term_at] = P.FLAG_TERMINATED
    if trunc_last:
        flags[-1] = P.FLAG_TRUNCATED
    arrays = {"obs": rng.normal(size=(T_ + 1, n_obs)).astype(np.float16), "values": rng.normal(size=T_ + 1).astype("<f4"),
              "actions": rng.integers(0, 9, T_).astype(np.uint8), "logp": -rng.random(T_).astype("<f4"),
              "rewards": rng.normal(size=T_).astype("<f4"), "flags": flags, "frames": np.full(T_, 2, "<u2")}
    return SimpleNamespace(header={"steps": T_, "task_id": "x" * 32, "seq": 3}, arrays=arrays)


def test_chunk_message_is_the_workers_layout():
    from ballrain.train.ppo import gae
    c = _chunk(trunc_last=True, term_at=1)
    m = LL.chunk_message(c, 42)
    assert m["pool"] and m["version"] == 42 and m["backend"] == "pool" and m["frames"] == 10
    assert m["raw"].dtype == np.float32 and m["raw"].shape == (5, 4) and m["raw_last"].shape == (4,)
    assert m["terminated"].tolist() == [False, True, False, False, False]
    assert m["truncated"].tolist() == [False, False, False, False, True]
    v = c.arrays["values"]
    assert m["trunc_values"][-1] == v[-1] and m["last_value"] == float(v[-1]) and len(m["values"]) == 5
    adv, ret = gae(m, 0.99, 0.95)  # the learner's own GAE runs on it
    assert np.isfinite(adv).all() and adv.shape == (5,)
    for k in ("episodes", "runs", "errors", "t_chunk", "t_reset", "wid", "obs"):
        assert k in m


def test_vtrace_on_policy_value_targets_are_gae_returns_and_weights_truncate():
    from ballrain.train.ppo import gae, vtrace
    c = LL.chunk_message(_chunk(T_=40, trunc_last=True, term_at=13, seed=2), 0)
    v = c["values"].astype(np.float32)
    adv, vs, rho = vtrace(c, c["logp"], v, c["last_value"], 0.99, 0.95)
    _, ret = gae(c, 0.99, 0.95)
    np.testing.assert_allclose(vs, ret, rtol=1e-5, atol=1e-5)
    assert np.all(rho == 1.0)
    # one-step advantage at a terminated step: r - V
    assert adv[13] == pytest.approx(c["rewards"][13] - v[13], abs=1e-5)
    # a far more likely action now: rho truncated at rho_bar, the trace cut at c_bar
    adv2, vs2, rho2 = vtrace(c, c["logp"] + 2.0, v, c["last_value"], 0.99, 0.95, rho_bar=1.0, c_bar=1.0)
    assert np.all(rho2 == 1.0)
    adv3, _, rho3 = vtrace(c, c["logp"] - 2.0, v, c["last_value"], 0.99, 0.95)
    assert np.allclose(rho3, np.exp(-2.0), rtol=1e-5) and np.abs(adv3).sum() < np.abs(adv).sum()


@win_only
def test_learner_link_publishes_live_policies_and_feeds_only_released_chunks(pool, world, tmp_path):
    """A RemoteHub (the learner's) with the world's template: the link connects as host "pool", publishes the hub's
    parameters as a task policy with the hub's EnvConfig and curriculum, and feeds back exactly the released chunks
    as learner chunk messages; they are marked ingested and never fed twice."""
    import queue
    from ballrain.env import EnvConfig
    from ballrain.features import FeatureConfig
    from ballrain.train.remote import RemoteHub
    from test_pool_verifier import Player
    st = world["state"]
    ec = dict(st["env_config"])
    feats = FeatureConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ec.pop("features").items()})
    from ballrain.rewards import RewardConfig
    rw = ec.pop("reward")
    env = EnvConfig(**ec, features=feats, reward=RewardConfig(**rw) if rw else None)
    worker = {k: (json.loads(v) if k in ("route_probes", "gates", "split_extra", "counter_rand") else v)
              for k, v in st["args"].items()}
    template = {"env": env, "seed": 1, "run": "test", "pool_ingest": True, "worker": worker}
    data_q = queue.Queue()
    hub = RemoteHub(("127.0.0.1", 0), b"k" * 16, data_q, template, log=lambda m: None)
    addr = hub._listener.address
    pool.policies.run = "test"
    link = LL.LearnerLink(addr, b"k" * 16, pool.policies, pool.store, pool.cfg.pool_dir, publish_every=5,
                          publish_min_s=0.0)
    try:
        assert link.connect() and link.run == "test"
        link.start_recv()
        msg = {"version": 20, "rms": st["rms"], "curriculum": dict(st["curriculum"], frontier=st["curriculum"]["frontier"]),
               "model": {k: v.a for k, v in st["model"].items()}}
        hub.publish(msg)
        t0 = time.time()
        while link.version != 20 and time.time() - t0 < 10:
            time.sleep(0.05)
        assert link.maybe_publish()
        pub = pool.policies.current
        assert pub.source == "learner-u20" and pub.manifest["source"]["update"] == 20
        assert pub.manifest["rollout"]["counter_rand"] == COUNTERS and pub.draws["p_rest"] == 0.3
        assert not link.maybe_publish()  # fewer than publish_every updates since
        # a played, verified task: its chunks are released and fed; a held chunk is not
        pl = Player(pool, "eligible", tmp_path)
        t = pl.task()
        assert t.policy_sha == pub.policy_sha
        chunks, end, _ = pl.play(t)
        for c in chunks:
            assert pl.send(c).status == 200
        assert link.feed_once() == 0  # held: not yet verified
        assert pool.verifier.run_once() == 1
        n = link.feed_once()
        assert n == len(chunks) and link.feed_once() == 0
        got = [data_q.get(timeout=10) for _ in range(n)]
        assert all(m["pool"] and m["remote"] and m["version"] == 20 and m["host"] == "pool" for m in got)
        assert sum(len(m["rewards"]) for m in got) == sum(P.decode_chunk(c).steps for c in chunks)
        rows = pool.store.chunks(t.task_id)
        assert all(r["ingested"] == 20 for r in rows)
        lo, hi, steps = pool.store.ingested_updates(rows[0]["forum_id"])
        assert lo == hi == 20 and steps == sum(r["steps"] for r in rows)
    finally:
        link.close()
        hub.close()


# ============================================================================================== learner flag

def _pool_feeder(addr, key, stop, sent, n_obs_box):
    """Connects to the learner's hub as host "pool" and sends synthetic pool chunks of the current version."""
    from multiprocessing.connection import Client
    c = Client(addr, authkey=key)
    c.send({"type": "hello", "protocol": 1, "host": "pool", "session": 1, "backend": "pool", "n_workers": 0})
    c.recv()
    version, i = None, 0
    while not stop.is_set():
        try:
            ready = c.poll(0.05)
            m = c.recv() if ready else None
        except (OSError, EOFError):  # the learner finished and closed its hub
            break
        while m is not None:
            if m.get("type") == "params":
                version = int(m["msg"]["version"])
                n_obs_box[0] = int(m["msg"]["model"]["pi.0.weight"].shape[1])
            try:
                m = c.recv() if c.poll(0) else None
            except (OSError, EOFError):
                m = None
                stop.set()
        if version is None or n_obs_box[0] is None:
            continue
        ch = _chunk(T_=64, n_obs=n_obs_box[0], seed=i)
        ch.arrays["obs"] = np.clip(ch.arrays["obs"], -3, 3)
        msg = LL.chunk_message(ch, version)
        try:
            c.send({"type": "data", "msg": msg})
        except OSError:
            break
        sent[0] += 1
        i += 1
        time.sleep(0.05)
    try:
        c.close()
    except Exception:  # noqa: BLE001
        pass


def _train(tmp_path, ingest: bool, record) -> dict:
    import socket
    import torch
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        port = so.getsockname()[1]
    key = "test-key-" + "x" * 16
    envv = dict(os.environ, BALLRAIN_RUNTIME=str(tmp_path / "rt"), BALLRAIN_REMOTE_KEY=key)
    args = [sys.executable, str(REPO / "scripts" / "train_m2.py"), "--run", "pool", "--fake", "--workers", "1",
            "--total-steps", "3000", "--batch-steps", "512", "--chunk", "64", "--minibatch", "256", "--epochs", "1",
            "--remote-listen", f"127.0.0.1:{port}", "--log-every", "1", "--ckpt-every", "1000",
            "--record", str(record)]  # the curriculum's record (the toy level's; --fake plays from level load)
    if ingest:
        args += ["--pool-ingest", "--pool-max-lag", "50"]
    p = subprocess.Popen(args, env=envv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    stop, sent, nob = threading.Event(), [0], [None]
    th = None
    t0 = time.time()
    while th is None and time.time() - t0 < 120:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                pass
            th = threading.Thread(target=_pool_feeder, args=(("127.0.0.1", port), key.encode(), stop, sent, nob),
                                  daemon=True)
            th.start()
        except OSError:
            time.sleep(0.5)
    try:
        out, _ = p.communicate(timeout=600)
    finally:
        stop.set()
        if p.poll() is None:
            p.kill()
    assert p.returncode == 0, out[-3000:]
    ck = torch.load(tmp_path / "rt" / "m2-train" / "pool" / "latest.pt", weights_only=False)
    return dict(ck["counters"], sent=sent[0], log=out)


@win_only
@pytest.mark.parametrize("ingest", [False, True])
def test_learner_pool_flag(tmp_path, world, ingest):
    c = _train(tmp_path, ingest, world["record_dir"])
    assert c["sent"] > 0
    if ingest:
        assert c.get("steps_pool", 0) > 0 and not c.get("pool_refused")
    else:
        assert c.get("pool_refused", 0) > 0 and "--pool-ingest is off" in c["log"]


def test_reference_line_without_t_ms_is_a_valid_asset(tmp_path):
    """A line without t_ms (like the original benchmarks/refpaths/L02.npz) is accepted by the env (ref_speed 0), so it
    must be by the pool too. A fixture line, so the test does not depend on what the repo's lines carry."""
    n = 64
    src = tmp_path / "L02.npz"
    T.save_npz_deterministic(src, {"points": np.stack([np.zeros(n), np.full(n, 2.0), np.linspace(-5, -200, n)],
                                                      1).astype("<f4"),
                                   "sector": np.ones(n, np.uint8), "meta": np.array(json.dumps({"level": 2}))})
    sha, dst = T.export_path_asset(src, tmp_path / "assets")
    d = T.install_path_asset(dst, tmp_path / "cache", 2, sha)
    assert (d / "L02.npz").is_file()
    # a line with t_ms of the wrong length is still refused
    bad = tmp_path / "bad.npz"
    with np.load(src, allow_pickle=False) as z:
        arrs = {k: z[k] for k in z.files}
    arrs["t_ms"] = np.zeros(3)
    T.save_npz_deterministic(bad, arrs)
    with pytest.raises(ValueError):
        T.install_path_asset(bad, tmp_path / "cache2", 2, T.file_sha256(bad))



def test_min_client_version_is_never_below_the_task_format(tmp_path):
    from ballrain.pool.server_config import MIN_CLIENT_VERSION, load_config
    (tmp_path / "config.toml").write_text('[pool]' + chr(10) + 'min_client_version = "0.2.0"' + chr(10), encoding="utf-8")
    assert load_config(tmp_path / "config.toml", tmp_path).min_client_version == MIN_CLIENT_VERSION == "0.3.0"
    (tmp_path / "config.toml").write_text('[pool]' + chr(10) + 'min_client_version = "0.5.1"' + chr(10), encoding="utf-8")
    assert load_config(tmp_path / "config.toml", tmp_path).min_client_version == "0.5.1"

# ============================================================================================== per-task spot checks

@win_only
def test_unsampled_forgery_is_flagged_escalated_replayed_and_the_learner_updates_listed(pool, tmp_path):
    """After the probation (task_sample 0: no task drawn for a replay) an honest task is released unreplayed and its
    chunks pass the cheap check; chunks of it are fed to the learner (ingested at updates 40 / 41). A forged chunk
    (actions not the task's sample_seed draws) of a later unsampled task fails the cheap policy-consistency check: the
    task goes to a full replay, the account is escalated, the replay rejects the task and flags the account suspect
    (not revoked), and the alert lists the learner updates its earlier unreplayed data reached."""
    from test_pool_verifier import Player, reseal
    pool.cfg.probation_chunks = 1
    pool.cfg.task_sample = 0.0
    pl = Player(pool, "eligible", tmp_path)
    t0 = pl.task()  # the probation task
    for c in pl.play(t0)[0]:
        assert pl.send(c).status == 200
    assert pool.verifier.run_once() == 1
    t1 = pl.task()
    c1, _, _ = pl.play(t1)
    for c in c1:
        assert pl.send(c).status == 200
    rows = pool.store.chunks(t1.task_id)
    assert all(not r["selected"] and r["flag"] is None for r in rows)
    assert pool.verifier.run_once() == 1  # released unreplayed
    pool.store.mark_ingested([(t1.task_id, rows[0]["seq"])], 40)
    if len(rows) > 1:
        pool.store.mark_ingested([(t1.task_id, rows[1]["seq"])], 41)
    t2 = pl.task()
    c2, _, pol = pl.play(t2)
    forged = reseal(c2[0], actions=lambda a: ((a.astype(np.int64) + 1) % pol.n_act).astype(np.uint8))
    assert pl.send(forged).status == 200  # accepted (the client never learns about the flag) ...
    r0 = pool.store.chunks(t2.task_id)[0]
    assert r0["flag"] in ("action", "logp") and r0["selected"] == 1  # ... but flagged for the replay
    assert pool.store.task(t2.task_id)["replay"] == 1
    assert pool.store.account(r0["forum_id"])["escalate"] == pool.cfg.escalate_tasks
    pool.store.q("UPDATE tasks SET state='closed' WHERE task_id=?", (t2.task_id,))
    assert pool.verifier.run_once() == 1
    acct = pool.store.account(r0["forum_id"])
    assert not acct["revoked"] and acct["suspect"] and pl.me()["token_state"] == "active"
    alert = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((pool.cfg.pool_dir / "alerts").glob("*-suspect-*.json"))]
    assert alert[-1]["task_id"] == t2.task_id and alert[-1]["learner_updates_unreplayed"]["first"] == 40
    assert alert[-1]["learner_updates_unreplayed"]["last"] == (41 if len(rows) > 1 else 40)
    assert any("learner updates 40.." in m and "flagged suspect" in m for m in pool.logs)
    assert not [r for r in pool.store.to_ingest(1000) if r["task_id"] == t2.task_id]


def test_policy_consistency_tolerances_cover_float16_noise_and_catch_gross_changes():
    """The cheap check on a synthetic policy: honest rows (computed on the float32 observations, stored as float16)
    pass; changed log-probs, values or actions do not."""
    from ballrain.pool.server_verify import policy_consistency
    from ballrain.pool import sampling as S2
    rng = np.random.default_rng(0)
    n_obs, n_act, T = 12, 9, 64
    dims = [n_obs, 32, 32]
    arr = {}
    for head, n_out in (("pi", n_act), ("v", 1)):
        for i, (a, b) in enumerate(zip(dims, dims[1:] + [n_out])):
            arr[f"{head}.{2 * i}.weight"] = (rng.normal(size=(b, a)) / np.sqrt(a)).astype("<f4")
            arr[f"{head}.{2 * i}.bias"] = (rng.normal(size=b) * 0.1).astype("<f4")
    arr["rms_mean"], arr["rms_var"] = np.zeros(n_obs), np.ones(n_obs)
    pol = W.NumpyPolicy(arr, {"arch": {"hidden": [32, 32], "activation": "tanh", "n_obs": n_obs, "n_act": n_act},
                              "obs_norm": {"eps": 1e-8, "clip": 10.0}})
    raw = rng.normal(size=(T + 1, n_obs)).astype(np.float32) * 3
    x = pol.normalize(raw)
    lp, v = pol.log_probs(x), pol.value(x)
    task = {"sample_seed": 99, "greedy": False}
    acts = np.array([S2.sample_action(lp[j], S2.sample_u(99, 10 + j)) for j in range(T)], np.uint8)
    good = {"obs": raw.astype(np.float16), "values": v.astype(np.float32), "actions": acts,
            "logp": np.clip(lp[np.arange(T), acts], -60, 0).astype(np.float32)}
    ch = SimpleNamespace(header={"steps": T}, arrays=good)
    assert policy_consistency(ch, pol, task, 10) is None
    assert policy_consistency(ch, pol, task, 11) == "action"  # another step offset: other draws
    for k, f, why in (("logp", lambda a: a - 0.2, "logp"), ("values", lambda a: a + 0.5, "value"),
                      ("actions", lambda a: (a + 1) % n_act, "action")):
        bad = dict(good, **{k: f(good[k]).astype(good[k].dtype)})
        if k == "actions":
            bad["logp"] = np.clip(lp[np.arange(T), bad["actions"].astype(int)], -60, 0).astype(np.float32)
        assert policy_consistency(SimpleNamespace(header={"steps": T}, arrays=bad), pol, task, 10) == why


# ============================================================================================== link vs learner versions

def _hub(template):
    import queue
    from ballrain.train.remote import RemoteHub
    hub = RemoteHub(("127.0.0.1", 0), b"k" * 16, queue.Queue(), template, log=lambda m: None)
    return hub, hub._listener.address


def test_link_refuses_a_learner_without_pool_support_and_feeds_none_without_ingest(tmp_path):
    """An older learner (no pool_ingest in its template) would train pool chunks as plain remote chunks: the link does
    not connect. A learner advertising pool_ingest false is connected (policies) but gets no chunks."""
    from ballrain.pool.server_store import Store
    st = Store(tmp_path / "s.sqlite3")
    try:
        for tpl, connects, ingests in (({"env": {}, "run": "r", "worker": {}}, False, None),
                                       ({"env": {}, "run": "r", "worker": {}, "pool_ingest": False}, True, False),
                                       ({"env": {}, "run": "r", "worker": {}, "pool_ingest": True}, True, True)):
            hub, addr = _hub(tpl)
            logs = []
            link = LL.LearnerLink(addr, b"k" * 16, SimpleNamespace(run="r"), st, tmp_path, log=logs.append)
            try:
                assert link.connect() is connects, (tpl, link.last_error)
                if not connects:
                    assert "pool support" in link.last_error and any("does not advertise" in m for m in logs)
                    continue
                assert link.learner_ingests is ingests
                link.version = 5
                if not ingests:  # without --pool-ingest on the learner the link never even looks for chunks
                    link.store = SimpleNamespace(to_ingest=lambda n: (_ for _ in ()).throw(AssertionError("queried")))
                assert link.feed_once() == 0
            finally:
                link.close()
                hub.close()
    finally:
        st.close()


def test_link_republishes_after_a_learner_rollback_and_on_reconnect():
    published = []

    class Pol:
        run = "r"

        def publish_state(self, state, source):
            published.append(source)
            return SimpleNamespace(policy_sha=f"{len(published):064x}", curriculum=None)
    link = LL.LearnerLink(("127.0.0.1", 1), b"k", Pol(), None, ".", publish_every=10, publish_min_s=0.0)
    link.template = {"env": {}, "run": "r", "worker": {}, "pool_ingest": True}
    from unittest import mock
    with mock.patch.object(LL, "state_from_params", lambda t, m: {}):
        def at(v):
            link._latest = {"version": v}
            return link.maybe_publish()
        assert at(100) and not at(105) and at(110)
        assert at(60)  # auto-rollback / resume from an older checkpoint: published at once, not at 120
        assert not at(65) and at(70)
        link._published_version, link._published_t = 70, 0.0
        hub, addr = _hub({"env": {}, "run": "r", "worker": {}, "pool_ingest": True})
        link.addr, link.authkey = addr, b"k" * 16
        try:
            assert link.connect()  # a reconnect: the next parameters are published whatever their number
            assert link._published_version is None and at(71)
        finally:
            link.close()
            hub.close()
    assert published == ["learner-u100", "learner-u110", "learner-u60", "learner-u70", "learner-u71"]


def test_reconnect_never_publishes_the_previous_sessions_parameters():
    """The r8 NaN stop (2026-11-18): the old learner's last (non-finite) parameters were still cached at the reconnect
    and published before the resumed learner sent anything. A drop and a new session both forget them."""
    published = []

    class Pol:
        run = "r"

        def publish_state(self, state, source):
            published.append(source)
            return SimpleNamespace(policy_sha=f"{len(published):064x}", curriculum=None)
    link = LL.LearnerLink(("127.0.0.1", 1), b"k", Pol(), None, ".", publish_every=10, publish_min_s=0.0)
    link.template = {"env": {}, "run": "r", "worker": {}, "pool_ingest": True}
    from unittest import mock
    with mock.patch.object(LL, "state_from_params", lambda t, m: {}):
        link._latest, link.version = {"version": 16870}, 16870  # the old session's last send
        link.conn = SimpleNamespace(close=lambda: None)
        link._lost()
        assert link._latest is None and link.version is None and not link.maybe_publish()
        link._latest, link.version = {"version": 16870}, 16870  # cached again (a race), then a new session
        hub, addr = _hub({"env": {}, "run": "r", "worker": {}, "pool_ingest": True})
        link.addr, link.authkey = addr, b"k" * 16
        try:
            assert link.connect()
            assert link._latest is None and link.version is None and not link.maybe_publish()
            assert link.feed_once() == 0  # no learner version yet: nothing fed either
            link._latest = {"version": 16877}  # the new learner's own first parameters
            assert link.maybe_publish()
        finally:
            link.close()
            hub.close()
    assert published == ["learner-u16877"]


def test_state_from_params_carries_the_learners_branch_bundle():
    """The route-branch files come from the learner (its hub template, or a parameter message), never from the
    gateway's disk: state_from_params passes them through to the export state unchanged."""
    model = {"pi.0.weight": np.zeros((4, 3)), "pi.0.bias": np.zeros(4), "pi.2.weight": np.zeros((9, 4)),
             "pi.2.bias": np.zeros(9), "v.0.weight": np.zeros((4, 3)), "v.0.bias": np.zeros(4),
             "v.2.weight": np.zeros((1, 4)), "v.2.bias": np.zeros(1)}
    msg = {"version": 7, "rms": {"mean": np.zeros(3), "var": np.ones(3), "count": 1.0}, "model": model}
    tpl = {"env": {"level": 2}, "run": "r", "worker": {}, "pool_ingest": True}
    assert "branch_assets" not in LL.state_from_params(tpl, msg)
    ba = {"level": 2, "line_npz": b"L", "desc_json": b"{}", "drive_npz": b"D"}
    st = LL.state_from_params(dict(tpl, branch_assets=ba), msg)
    assert st["branch_assets"] == ba and st["n_obs"] == 3 and st["n_act"] == 9
    ba2 = dict(ba, line_npz=b"L2")
    assert LL.state_from_params(dict(tpl, branch_assets=ba), dict(msg, branch_assets=ba2))["branch_assets"] == ba2
# ============================================================================================== capability gate

def test_capability_refusal_explains_what_changed_with_an_existing_code():
    """A client lacking a capability gets no_task with the existing code client_too_old (old clients know it), and a
    message that names what was added, why this program cannot take part, and what to update to."""
    from ballrain.pool.capabilities import missing, refusal
    assert missing(None, None) == [] and missing(["branch_v1"], ["task_format3", "branch_v1", "x"]) == []
    assert missing(["task_format3", "branch_v1"], ["task_format3"]) == ["branch_v1"]
    r = refusal("0.3.0", ["branch_v1"])
    assert r["code"] == "client_too_old" and r["params"]["missing"] == "branch_v1" and r["params"]["min_version"] == "0.4.0"
    m = r["message"]
    assert "0.3.0" in m and "石头分支" in m and "验证不会通过" in m and "0.4.0" in m and len(m) <= 1000
    # what an old client receives parses with today's schema (no new code, no new field)
    msg = P.parse_message("no_task", P.encode_message("no_task", reason=r, retry_s=600))
    assert msg["reason"]["message"] == m
    assert "新功能 future_x" in refusal("0.4.0", ["future_x"])["message"]
    f7 = refusal("0.3.0", ["reach_fastreach5_2-block1"])
    assert "可达性" in f7["message"] and "fastreach5.2-block1" in f7["message"] and f7["params"]["min_version"] == "0.4.0"
    br = refusal("0.3.0", ["branch_v1", "branch_rules_v1"])
    assert "规则" in br["message"] and br["params"]["missing"] == "branch_v1,branch_rules_v1"


@win_only
def test_policies_needing_a_capability_go_only_to_clients_that_have_it(pool, tmp_path):
    from test_pool_verifier import Player
    pl = Player(pool, "eligible", tmp_path)
    assert pl.task().policy_sha == pool.pub.policy_sha  # no requires: every client
    pool.policies.current.manifest["requires"] = ["branch_v1"]
    r = pool.call("GET", "/pool/task", token=pl.token, session=pl.sid)
    nt = P.parse_message("no_task", r.body)
    assert r.status == 200 and nt["reason"]["code"] == "client_too_old" and "石头分支" in nt["reason"]["message"]
    pool.store.q("UPDATE sessions SET capabilities=? WHERE session_id=?", (json.dumps(["branch_v1"]), pl.sid))
    assert pl.task().policy_sha == pool.pub.policy_sha


def test_gateway_start_draw_equals_the_env_draw_with_rehearse_always(tmp_path):
    """server_tasks.PolicySource.pick_start draws exactly what BallanceEnv._draw_start draws for the same generator:
    rehearsal entries clipped to the frontier, rehearse_always entries never clipped, brake-only and driven at-rest
    starts, then the curriculum."""
    from ballrain.env import BallanceEnv, EnvConfig
    from ballrain.pool.server_tasks import PolicySource
    starts = SimpleNamespace(nearest_valid=lambda k: int(k), nearest_any=lambda k: int(k),
                             expected=lambda k: {"frame": k})
    cu = SimpleNamespace(frontier=5000, sample=lambda rng: int(rng.integers(5000, 6000)))
    kw = dict(rehearse_starts=[[0, 0], [1000, 9000]], rehearse_always=[[2000, 2100]], p_rehearse=0.4,
              rehearse_behind_frontier=True, rest_starts=[[3000, 3100], [4000, 4010, 1.0, -2.0, 0.5, 77]], p_rest=0.3)
    env = object.__new__(BallanceEnv)
    env.cfg, env.starts, env.curriculum = EnvConfig(**kw), starts, cu
    env._start_rng = np.random.default_rng(5)
    ps = PolicySource(tmp_path, "r", tmp_path, [])
    ps.rng = np.random.default_rng(5)
    pub = SimpleNamespace(draws=dict(kw), starts=starts, curriculum=cu, valid=np.arange(0, 30000))
    seen = set()
    for _ in range(3000):
        k_env = env._draw_start()
        r_env = env._rest_next
        k, rest = ps.pick_start(pub)
        assert k == k_env
        if r_env is False:
            assert rest is None
        elif r_env is True:
            assert rest == {"target": None}
        else:
            assert rest["target"] == list(r_env[:2]) and rest["max_frames"] == r_env[2]
        if 2000 <= k <= 2100:
            seen.add("always")  # below the frontier: only rehearse_always reaches it
        if rest is not None:
            seen.add("drive" if rest["target"] else "brake")
    assert seen == {"always", "drive", "brake"}


def test_link_hello_carries_the_branch_versions(tmp_path):
    """A learner distributing a route-branch bundle accepts the link, with default and with non-default branch rules (its
    hello says branch_assets_version 1 and branch_rules_version 1: the pool's client and replay apply the rules)."""
    import json as _json
    for rules, ok in (({}, True), ({"drive_gain": 0.5}, True)):
        env = SimpleNamespace(branch_assets={"level": 2, "line_npz": b"L", "drive_npz": b"D",
                                             "desc_json": _json.dumps({"rules": rules}).encode()})
        hub, addr = _hub({"env": env, "run": "r", "worker": {}, "pool_ingest": True})
        link = LL.LearnerLink(addr, b"k" * 16, SimpleNamespace(run="r"), None, tmp_path)
        try:
            assert link.connect() is ok, link.last_error
            if not ok:
                assert "branch_rules_version" in link.last_error
        finally:
            link.close()
            hub.close()
