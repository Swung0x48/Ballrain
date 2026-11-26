"""Skipping early respawn lives (ballrain.train.worker WorkerSpec.p_early_respawn): after a BallOff whose respawn sector
the start record enters before the curriculum frontier, the respawn life is played only with probability p; otherwise
the process ends and a new curriculum start follows. No game is launched (ballrain.fakegame)."""
import queue
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from ballrain import dataset, fakegame, tas
from ballrain.env import EnvConfig
from ballrain.train.worker import Worker, WorkerSpec

D = np.float32(1000 / 132)


def _decider(p, frontier, after_death=True, with_curriculum=True, with_starts=True, seed=0, **spec):
    """A Worker with only what _skip_respawn reads: sectors 1-4 entered at record frames 936 / 7817 / 11756 / 14232."""
    w = object.__new__(Worker)
    w.spec = SimpleNamespace(p_early_respawn=p, **spec)
    valid = np.arange(936, 20587)
    sector = np.select([valid < 7817, valid < 11756, valid < 14232], [1, 2, 3], 4)
    w.env = SimpleNamespace(_after_death=after_death,
                            curriculum=SimpleNamespace(frontier=frontier) if with_curriculum else None,
                            starts=SimpleNamespace(valid=valid, sector=sector) if with_starts else None)
    w._respawn_rng = np.random.default_rng(seed)
    return w


def test_skip_rule_only_for_early_respawn_sectors():
    w = _decider(0.0, 4100)
    assert w._skip_respawn("BallOff", {"sector": 1})  # spawn (936) before the frontier: skipped at p = 0
    assert not w._skip_respawn("BallOff", {"sector": 2})  # 7817 >= 4100: always played
    assert not w._skip_respawn("BallOff", {"sector": 4})
    assert not w._skip_respawn("stall", {"sector": 1})  # only deaths respawn
    assert not w._skip_respawn("LevelFinish", {"sector": 1})
    assert not _decider(0.0, 936)._skip_respawn("BallOff", {"sector": 1})  # cross frame == frontier: played
    assert _decider(0.0, 12000)._skip_respawn("BallOff", {"sector": 3})
    assert not _decider(0.0, 4100, after_death=False)._skip_respawn("BallOff", {"sector": 1})  # process ended (GameOver)
    assert not _decider(0.0, 4100, with_curriculum=False)._skip_respawn("BallOff", {"sector": 1})
    assert not _decider(0.0, 4100, with_starts=False)._skip_respawn("BallOff", {"sector": 1})


def test_skip_probability_and_default_draws_nothing():
    w = _decider(1.0, 4100)
    state = w._respawn_rng.bit_generator.state
    assert not any(w._skip_respawn("BallOff", {"sector": 1}) for _ in range(1000))
    assert w._respawn_rng.bit_generator.state == state  # p = 1 (the default): no draw, old runs unchanged
    w = _decider(0.1, 4100, seed=5)
    skips = np.mean([w._skip_respawn("BallOff", {"sector": 1}) for _ in range(20000)])
    assert 0.88 < skips < 0.92
    assert WorkerSpec(wid=0, copy="x", env=None, seed=0).p_early_respawn == 1.0


def test_skip_respawn_from_sector():
    # bridge override for engines that crash on late respawns: every BallOff in that sector or later, no draw
    w = _decider(1.0, 4100, skip_respawn_from_sector=4)
    state = w._respawn_rng.bit_generator.state
    assert w._skip_respawn("BallOff", {"sector": 4}) and w._skip_respawn("BallOff", {"sector": 5})
    assert not w._skip_respawn("BallOff", {"sector": 3})  # earlier sectors: the p_early_respawn rule (p = 1: played)
    assert w._respawn_rng.bit_generator.state == state
    assert not w._skip_respawn("stall", {"sector": 4}) and not w._skip_respawn("LevelFinish", {"sector": 5})
    assert not _decider(1.0, 4100, after_death=False, skip_respawn_from_sector=4)._skip_respawn("BallOff", {"sector": 4})
    lvl = _decider(0.0, 4100, skip_respawn_from_sector=4, respawn_level_starts=True)
    lvl.env._start = {"process_start_frame": 0}
    assert lvl._skip_respawn("BallOff", {"sector": 4})  # level starts too (the engine crashes either way)
    assert not lvl._skip_respawn("BallOff", {"sector": 2})  # ... their earlier respawns are still played
    assert not _decider(1.0, 4100, skip_respawn_from_sector=0)._skip_respawn("BallOff", {"sector": 5})  # 0 = off
    assert not _decider(1.0, 4100)._skip_respawn("BallOff", {"sector": 5})  # older spec without the field
    assert WorkerSpec(wid=0, copy="x", env=None, seed=0).skip_respawn_from_sector == 0


def test_bridge_local_overrides(tmp_path):
    sys.path.insert(0, str(__import__("ballrain.instances", fromlist=["REPO_ROOT"]).REPO_ROOT / "scripts"))
    import remote_worker_m2 as rw
    logs = []
    a = SimpleNamespace(skip_respawn_from_sector=None)
    assert rw.local_overrides(tmp_path, a, logs.append) == {} and not logs
    (tmp_path / "worker_overrides.json").write_text('{"skip_respawn_from_sector": 5}', encoding="utf-8")
    assert rw.local_overrides(tmp_path, a, logs.append) == {"skip_respawn_from_sector": 5} and logs
    assert rw.local_overrides(tmp_path, SimpleNamespace(skip_respawn_from_sector=4), logs.append) ==         {"skip_respawn_from_sector": 4}  # the flag wins over the file
    (tmp_path / "worker_overrides.json").write_text('{"p_early_respawn": 1.0}', encoding="utf-8")
    with pytest.raises(SystemExit):
        rw.local_overrides(tmp_path, a, logs.append)  # only the listed fields may be overridden locally


def test_train_m2_option_reaches_spec_and_remote_template():
    sys.path.insert(0, str(__import__("ballrain.instances", fromlist=["REPO_ROOT"]).REPO_ROOT / "scripts"))
    import train_m2
    assert train_m2.parse(["--run", "x"]).p_early_respawn == 1.0
    assert train_m2.parse(["--run", "x", "--p-early-respawn", "0.1"]).p_early_respawn == 0.1
    assert "respawn_skipped" in train_m2.EP_COLS
    src = (train_m2.REPO / "scripts" / "train_m2.py").read_text(encoding="utf-8")
    assert '"p_early_respawn": a.p_early_respawn' in src and "p_early_respawn=a.p_early_respawn" in src


@pytest.fixture
def fake_record(tmp_path):
    keys = np.zeros(600, np.uint32)
    keys[40:] = tas.KEY_UP
    rec = tmp_path / "SR01_0_00_000_132.tas"
    tas.save(rec, tas.make_frames(np.full(600, D), keys))
    res = dataset.build_dataset([rec], tmp_path / "ds", copies=["train-00"], launcher_factory=fakegame.FakeLauncher)
    assert res[0]["ok"]
    return tmp_path / "ds" / rec.stem


def _run_worker(tmp_path, fake_record, p, start_back, n_chunks=4):
    cfg = EnvConfig(level=1, copy="train-00", out_root=str(tmp_path / f"ep{p}-{start_back}"), action_repeat=4,
                    start_record=str(fake_record), curriculum={"start_back": start_back, "seed": 1})
    spec = WorkerSpec(wid=0, copy="train-00", env=cfg, seed=3, chunk=64, launcher_factory=fakegame.FakeLauncher,
                      p_early_respawn=p, prune="none")
    ctrl_q, data_q = queue.Queue(), queue.Queue()
    w = Worker(spec, ctrl_q, data_q)
    w.model.act = lambda x, gen=None: (3, np.float32(-2.2), np.float32(0.0))  # left: off the road, a death per life
    w.np_policy = None  # use the stubbed torch-side act
    w.model.value = lambda x: np.float32(0.0)
    n = w.env.observation_space.shape[0]
    ctrl_q.put({"version": 0, "model": {k: v.detach().numpy() for k, v in w.model.state_dict().items()},
                "rms": {"mean": np.zeros(n), "var": np.ones(n), "count": 1.0},
                "curriculum": w.env.get_start_state()})
    th = threading.Thread(target=w.run, daemon=True)
    th.start()
    eps = []
    for _ in range(n_chunks):
        m = data_q.get(timeout=120)
        assert "fatal" not in m and not m["errors"], m.get("errors")
        eps += m["episodes"]
    ctrl_q.put({"cmd": "stop"})
    th.join(timeout=120)
    return eps, int(w.env.curriculum.frontier), w


@pytest.mark.skipif(sys.platform != "win32", reason="named shared memory / events need Windows")
def test_worker_skips_early_respawns_with_a_new_process(tmp_path, fake_record):
    # the fake record enters sector 2 at frame 557 (sector 1 at its first control frame, 30); last valid frame 600
    eps, frontier, w = _run_worker(tmp_path, fake_record, 0.0, start_back=30)  # frontier 570: both sectors early
    assert w._tas_cross_frame(2) == 557 and frontier == 570
    deaths = [e for e in eps if e["end"] == "BallOff"]
    assert deaths and all(e["respawn_skipped"] for e in deaths)
    assert all(e["kind"] == "tas" and e["new_process"] for e in eps)  # never a respawn life
    # frontier 300: sector-1 deaths are skipped, sector-2 respawns (557 >= 300) are played as before
    eps, frontier, _ = _run_worker(tmp_path, fake_record, 0.0, start_back=300, n_chunks=10)
    assert frontier == 300
    d1 = [e for e in eps if e["end"] == "BallOff" and e["end_sector"] == 1]
    d2 = [e for e in eps if e["end"] == "BallOff" and e["end_sector"] == 2]
    assert d1 and d2 and all(e["respawn_skipped"] for e in d1) and not any(e["respawn_skipped"] for e in d2)
    assert {e["start_sector"] for e in eps if e["kind"] == "respawn"} == {2}
    # p = 1 (default): the old behaviour, respawn lives everywhere, nothing skipped
    eps, _, _ = _run_worker(tmp_path, fake_record, 1.0, start_back=30)
    assert any(e["kind"] == "respawn" for e in eps) and not any(e["respawn_skipped"] for e in eps)


@pytest.fixture(autouse=True)
def _clear_geometry_cache():
    """ballrain.features keys its level-geometry cache by id(LevelData): ids of the fake level freed here can be reused
    by a later test's level, which would then get this level's geometry. Leave the cache empty."""
    yield
    from ballrain import features
    features._GEOM_CACHE.clear()
    features._KEY_OF.clear()
