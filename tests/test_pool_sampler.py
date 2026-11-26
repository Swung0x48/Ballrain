"""ballrain.pool.client_sampler on the fake game (ballrain.fakegame; Windows: the fake record is built over the shared-
memory transport): chunks that decode and chain, task-level ends, a replay of the recorded actions reproducing the
chain bit for bit, and reward / flag parity with the learner's own rollout worker (ballrain.train.worker) on the same
forced action sequence."""
import json
import queue
import sys
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ballrain import dataset, fakegame, tas  # noqa: E402
from ballrain.env import BallanceEnv, EnvConfig  # noqa: E402
from ballrain.features import FeatureConfig  # noqa: E402
from ballrain.instances import REPO_ROOT  # noqa: E402
from ballrain.pool import protocol as P  # noqa: E402
from ballrain.pool import task as T  # noqa: E402
from ballrain.pool import weights as W  # noqa: E402
from ballrain.pool.client_sampler import PolicyActions, ReplayActions, TaskSampler  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the fake record is built over named shared memory")
D = np.float32(1000 / 132)
REC = "SR01_0_00_000_132"
LEFT = 3  # ARROWS9 index: off the fake road


class Forced(PolicyActions):
    """Forced actions with the policy's own log-prob / value (what PolicyActions would report for them)."""

    def __init__(self, policy, seq):
        super().__init__(policy, np.random.default_rng(0))
        self.seq, self.i = list(seq), 0

    def __call__(self, raw):
        a = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        x = self.policy.normalize(raw)
        return a, float(self.policy.log_probs(x)[a]), float(self.policy.value(x))


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    """A fake-level record, its starts pack and the repo's L01 line as task assets, a synthetic policy exported like
    the learner's (stall_terminal + split bonus on, so the worker's shaping is exercised)."""
    tmp = tmp_path_factory.mktemp("pool")
    keys = np.zeros(600, np.uint32)
    keys[40:] = tas.KEY_UP
    rec = tmp / f"{REC}.tas"
    tas.save(rec, tas.make_frames(np.full(600, D), keys))
    assert dataset.build_dataset([rec], tmp / "ds", copies=["train-00"], launcher_factory=fakegame.FakeLauncher)[0]["ok"]
    rdir = tmp / "ds" / REC
    from ballrain.curriculum import TasStarts
    TasStarts(rdir)  # builds the start-state cache (starts.npz)
    line = REPO_ROOT / "benchmarks" / "refpaths" / "L01.npz"
    psha, stored = T.export_path_asset(line, tmp / "assets")
    pdir = T.install_path_asset(stored, tmp / "cache", 1, psha)
    ssha, pack = T.export_starts_pack(rdir, tmp / "assets")
    sdir = T.install_starts_pack(pack, tmp / "cache", ssha)
    feats = FeatureConfig(path_sha256=psha)
    probe = BallanceEnv(EnvConfig(level=1, action_repeat=4, start_record=str(sdir), features=feats),
                        launcher=fakegame.FakeLauncher())
    n_obs = probe.observation_space.shape[0]
    probe.close()
    from ballrain.train import ppo
    torch.manual_seed(0)
    model = ppo.ActorCritic(n_obs, 9, (32, 32), "tanh")
    ec = asdict(EnvConfig(level=1, action_repeat=4, start_record=REC, features=feats))
    args = {"activation": "tanh", "chunk": 64, "gamma": 0.997, "stall_s": 3.0, "stall_terminal": True,
            "split_k": 0.5, "split_min_ref_s": 1.0}
    state = {"model": model.state_dict(), "rms": {"mean": np.zeros(n_obs), "var": np.ones(n_obs), "count": 1.0},
             "n_obs": n_obs, "n_act": 9, "args": args, "env_config": ec, "counters": {"update": 1, "steps": 1}}
    m = W.export_state(state, tmp / "weights", "rtest")
    pol = W.load_policy(tmp / "weights" / f"{m['policy_sha']}.npz", tmp / "weights" / f"{m['policy_sha']}.json",
                        m["policy_sha"])
    meta = json.loads((rdir / "meta.json").read_text(encoding="utf-8"))
    k = int(T.valid_start_frames(sdir)[len(T.valid_start_frames(sdir)) // 3])
    return {"tmp": tmp, "manifest": m, "policy": pol, "sdir": sdir, "pdir": pdir, "n_obs": n_obs,
            "task": dict(run="rtest", policy_sha=m["policy_sha"], level=1, record=REC,
                         record_sha256=meta["record_sha256"], start_frame=k, ball_order_seed=7, engine="x87soft",
                         assets={"starts": ssha, "path": psha}, chunk_steps=16, max_steps=400, issued_at=0)}


def _env(pool, task, out):
    lg = T.LocalGame(engine="x87soft", binary="SimTool", root="Ballance")
    return T.build_env(task, pool["manifest"], lg, pool["sdir"], pool["pdir"], out,
                       launcher=fakegame.FakeLauncher(backend="headless"))


def _run(pool, task, actions, out, should_stop=lambda: None):
    env = _env(pool, task, out)
    got = []
    try:
        s = TaskSampler(env, pool["manifest"]["rollout"]).run_task(task, actions, lambda d, h: got.append(d),
                                                                     should_stop, n_act=9)
    finally:
        env.close()
    return s, [P.decode_chunk(d, n_act=9, n_obs=pool["n_obs"]) for d in got]


def _check_chunks(task, chunks):
    assert chunks[0].header["chain_prev"] == P.chain_seed(task.task_id).hex() and chunks[0].header["step0"] == 0
    for a, b in zip(chunks, chunks[1:]):
        assert P.check_chain_link(a, b) is None
    assert [c.header["final"] for c in chunks] == [False] * (len(chunks) - 1) + [True]
    assert all(c.steps == task.chunk_steps for c in chunks[:-1])


def test_task_to_chunks_deaths_respawn_in_one_process(pool):
    task = T.Task(task_id=T.new_task_id(), **pool["task"])
    s, chunks = _run(pool, task, Forced(pool["policy"], [LEFT]), pool["tmp"] / "o1")
    _check_chunks(task, chunks)
    flags = np.concatenate([c.arrays["flags"] for c in chunks])
    # every life dies; the deaths before the last respawn in the same process, GameOver ends the task
    assert s["end"] == "GameOver" and chunks[-1].header["end"] == "GameOver"
    assert int((flags & P.FLAG_TERMINATED).sum()) == s["episodes"] >= 3 and not (flags & P.FLAG_TRUNCATED).any()
    assert s["steps"] == sum(c.steps for c in chunks) == len(flags)
    # rewards: the death penalty lands on the terminal steps
    rew = np.concatenate([c.arrays["rewards"] for c in chunks])
    assert (rew[flags == P.FLAG_TERMINATED] < 0).all()


def test_replay_reproduces_the_chain_and_stop_truncates(pool):
    task = T.Task(task_id=T.new_task_id(), **pool["task"])
    seq = [0, 0, 1, 1] * 10 + [LEFT] * 200
    s, chunks = _run(pool, task, Forced(pool["policy"], seq), pool["tmp"] / "o2")
    acts = np.concatenate([c.arrays["actions"] for c in chunks])
    s2, again = _run(pool, task, ReplayActions(pool["policy"], acts), pool["tmp"] / "o3")
    assert s2["steps"] == s["steps"]
    for a, b in zip(chunks, again):  # the verifier's check: same observations bit for bit, same rewards and log-probs
        assert bytes(a.arrays["obs_hash"]) == bytes(b.arrays["obs_hash"])
        np.testing.assert_array_equal(a.arrays["rewards"], b.arrays["rewards"])
        np.testing.assert_allclose(a.arrays["logp"], b.arrays["logp"], atol=1e-6)
    # the player pauses after 20 steps: the task ends with a truncated last step, end "stopped"
    n = [0]

    def stop():
        n[0] += 1
        return "stopped" if n[0] >= 20 else None
    task3 = T.Task(task_id=T.new_task_id(), **dict(pool["task"], chunk_steps=20))
    s3, ch3 = _run(pool, task3, PolicyActions(pool["policy"], np.random.default_rng(1)), pool["tmp"] / "o4", stop)
    _check_chunks(task3, ch3)
    if s3["end"] == "stopped":  # (a policy that dies within 20 steps ends on its own first)
        assert s3["steps"] == 20 and len(ch3) == 1 and ch3[0].arrays["flags"][-1] & (P.FLAG_TRUNCATED | P.FLAG_TERMINATED)
    # max_steps
    task4 = T.Task(task_id=T.new_task_id(), **dict(pool["task"], max_steps=10))
    s4, ch4 = _run(pool, task4, Forced(pool["policy"], [0]), pool["tmp"] / "o5")
    assert s4["end"] == "max_steps" and s4["steps"] == 10 and ch4[-1].arrays["flags"][-1] == P.FLAG_TRUNCATED


@pytest.mark.parametrize("kind", ["deaths", "stall_terminal", "sectors_split_levelfinish"])
def test_reward_and_flag_parity_with_the_learner_worker(pool, kind):
    """The same forced actions through ballrain.train.worker.Worker and through the pool sampler give the same
    training rewards and terminal / truncation flags (the worker's per-step shaping was ported, not re-derived):
    deaths with respawns, a stall ending as a failure (stall_terminal penalty), sector crossings with the split bonus
    up to LevelFinish."""
    from ballrain.train.worker import Worker, WorkerSpec
    task = T.Task(task_id=T.new_task_id(), **dict(pool["task"], chunk_steps=4096, max_steps=4096))
    seq = {"deaths": [0, 0, 1, 1, 1, 0] * 6 + [LEFT] * 400, "stall_terminal": [0] * 400,
           "sectors_split_levelfinish": [1] * 400}[kind]
    s, chunks = _run(pool, task, Forced(pool["policy"], seq), pool["tmp"] / "p1")
    flags = np.concatenate([c.arrays["flags"] for c in chunks])
    rew = np.concatenate([c.arrays["rewards"] for c in chunks])
    N = s["steps"]
    assert s["end"] == {"deaths": "GameOver", "stall_terminal": "stall", "sectors_split_levelfinish": "LevelFinish"}[kind]

    lg = T.LocalGame(engine="x87soft", binary="SimTool", root="Ballance")
    cfg = T.build_env_config(task, pool["manifest"], lg, pool["sdir"], pool["pdir"], pool["tmp"] / "p2")
    ro = pool["manifest"]["rollout"]
    spec = WorkerSpec(wid=0, copy=cfg.copy, env=cfg, seed=0, launcher_factory=lambda: fakegame.FakeLauncher(backend="headless"),
                      hidden=(32, 32), **dict({k: ro[k] for k in W.ROLLOUT_FIELDS if k in ro}, chunk=N))
    ctrl_q, out = queue.Queue(), []

    class DataQ:
        def put(self, buf):
            out.append(buf)
            ctrl_q.put({"cmd": "stop"})

        def cancel_join_thread(self):
            pass
    w = Worker(spec, ctrl_q, DataQ())
    it = iter(seq)
    w.model.act = lambda x, gen=None: (next(it), 0.0, 0.0)
    w.np_policy = None  # use the stubbed torch-side act
    ctrl_q.put({"rms": {"mean": np.zeros(pool["n_obs"]), "var": np.ones(pool["n_obs"])}})
    w.run()
    buf = out[0]
    np.testing.assert_array_equal(buf["actions"][:N], np.concatenate([c.arrays["actions"] for c in chunks]))
    np.testing.assert_array_equal(buf["rewards"][:N], rew)
    np.testing.assert_array_equal(buf["terminated"][:N], (flags & P.FLAG_TERMINATED) > 0)
    np.testing.assert_array_equal(buf["truncated"][:N], (flags & P.FLAG_TRUNCATED) > 0)
