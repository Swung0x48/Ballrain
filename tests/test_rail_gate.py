"""Rail-end gate (ballrain.train.worker WorkerSpec.rail_gate): a 小节-1 life not started at or after TAS frame `below` is
truncated, bootstrapped, once its route arc length reaches s. No game is launched (ballrain.fakegame)."""
import queue
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from ballrain import dataset, fakegame, tas
from ballrain.env import EnvConfig
from ballrain.train.worker import Worker, WorkerSpec, _Episode

D = np.float32(1000 / 132)


def _w(gate, frontier=None):
    w = object.__new__(Worker)
    w.spec = SimpleNamespace(rail_gate=gate) if gate is not None else SimpleNamespace()
    w.env = SimpleNamespace(curriculum=SimpleNamespace(frontier=frontier) if frontier is not None else None)
    return w


def _ep(kind, start_frame, start_sector=1):
    ep = object.__new__(_Episode)
    ep.d = {"kind": kind, "start_frame": start_frame, "start_sector": start_sector}
    return ep


def test_gate_rule():
    g = {"s": 282.5, "below": 4000}
    w = _w(g)
    assert w._gate_hit(_ep("tas", 2100), 282.5, {"sector": 1})  # early 小节-1 start reaches the gate
    assert not w._gate_hit(_ep("tas", 2100), 282.4, {"sector": 1})  # not there yet
    assert w._gate_hit(_ep("tas_rest", 1240), 300.0, {"sector": 1})
    assert w._gate_hit(_ep("tas", 0), 290.0, {"sector": 1})  # level start
    assert w._gate_hit(_ep("respawn", 936), 290.0, {"sector": 1})  # 小节-1 respawn
    assert not w._gate_hit(_ep("tas", 4050), 300.0, {"sector": 1})  # started at / after below: plays through
    assert not w._gate_hit(_ep("tas", 4000), 300.0, {"sector": 1})
    assert not w._gate_hit(_ep("tas", 2100), 300.0, {"sector": 2})  # already past checkpoint 1
    assert not w._gate_hit(_ep("respawn", 7817, start_sector=2), 900.0, {"sector": 2})
    assert not w._gate_hit(_ep("tas", 2100), None, {"sector": 1})
    assert not _w({})._gate_hit(_ep("tas", 2100), 300.0, {"sector": 1})  # off
    assert not _w(None)._gate_hit(_ep("tas", 2100), 300.0, {"sector": 1})  # an older pickled WorkerSpec
    assert _w({"s": 10.0})._gate_hit(_ep("tas", 3999), 10.0, {"sector": 1})  # below defaults to 4000
    assert WorkerSpec(wid=0, copy="x", env=None, seed=0).rail_gate == {}


def test_gate_follows_the_frontier():
    g = {"s": 277.0, "below": 4000, "follow_frontier": True}
    w = _w(g, frontier=3900)  # unpinned: the band starts at 3900
    assert not w._gate_hit(_ep("tas", 3900), 300.0, {"sector": 1})  # band start plays through the platform
    assert not w._gate_hit(_ep("tas", 3950), 300.0, {"sector": 1})
    assert w._gate_hit(_ep("tas", 2000), 300.0, {"sector": 1})  # 小节-1 rehearsal start still gated
    assert w._gate_hit(_ep("tas", 3899), 277.0, {"sector": 1})
    w = _w(g, frontier=4100)  # pinned at 4100: below stays 4000, the 4000-4099 rehearsal plays through
    assert not w._gate_hit(_ep("tas", 4000), 300.0, {"sector": 1})
    assert w._gate_hit(_ep("tas", 3700), 300.0, {"sector": 1})
    assert w._gate_hit(_ep("tas", 3900), 300.0, {"sector": 1})
    assert _w({"s": 277.0, "below": 4000}, frontier=3900)._gate_hit(_ep("tas", 3900), 300.0, {"sector": 1})  # no follow
    assert _w(g)._gate_hit(_ep("tas", 3900), 300.0, {"sector": 1})  # no curriculum: the fixed below


def test_train_m2_option_reaches_spec_and_remote_template():
    sys.path.insert(0, str(__import__("ballrain.instances", fromlist=["REPO_ROOT"]).REPO_ROOT / "scripts"))
    import train_m2
    assert train_m2.parse(["--run", "x"]).rail_gate == ""
    a = train_m2.parse(["--run", "x", "--rail-gate", '{"s": 282.5, "below": 4000}'])
    assert a.rail_gate == '{"s": 282.5, "below": 4000}'
    src = (train_m2.REPO / "scripts" / "train_m2.py").read_text(encoding="utf-8")
    assert src.count("rail_gate=json.loads(a.rail_gate) if a.rail_gate else {}") == 1  # local workers
    assert '"rail_gate": json.loads(a.rail_gate) if a.rail_gate else {}' in src  # remote template


@pytest.fixture
def fake_record(tmp_path):
    keys = np.zeros(600, np.uint32)
    keys[40:] = tas.KEY_UP
    rec = tmp_path / "SR01_0_00_000_132.tas"
    tas.save(rec, tas.make_frames(np.full(600, D), keys))
    res = dataset.build_dataset([rec], tmp_path / "ds", copies=["train-00"], launcher_factory=fakegame.FakeLauncher)
    assert res[0]["ok"]
    return tmp_path / "ds" / rec.stem


@pytest.mark.skipif(sys.platform != "win32", reason="named shared memory / events need Windows")
def test_worker_truncates_at_the_gate_and_bootstraps(tmp_path, fake_record):
    cfg = EnvConfig(level=1, copy="train-00", out_root=str(tmp_path / "ep"), action_repeat=4,
                    start_record=str(fake_record), curriculum={"start_back": 300, "seed": 1})
    spec = WorkerSpec(wid=0, copy="train-00", env=cfg, seed=3, chunk=64, launcher_factory=fakegame.FakeLauncher,
                      prune="none", rail_gate={"s": 6.0, "below": 4000})
    ctrl_q, data_q = queue.Queue(), queue.Queue()
    w = Worker(spec, ctrl_q, data_q)
    w.model.act = lambda x, gen=None: (1, np.float32(-2.2), np.float32(0.0))  # up: along the fake road
    w.model.value = lambda x: np.float32(0.75)
    n = w.env.observation_space.shape[0]
    ctrl_q.put({"version": 0, "model": {k: v.detach().numpy() for k, v in w.model.state_dict().items()},
                "rms": {"mean": np.zeros(n), "var": np.ones(n), "count": 1.0},
                "curriculum": w.env.get_start_state()})
    th = threading.Thread(target=w.run, daemon=True)
    th.start()
    eps, chunks = [], []
    for _ in range(4):
        m = data_q.get(timeout=120)
        assert "fatal" not in m and not m["errors"], m.get("errors")
        eps += m["episodes"]
        chunks.append(m)
    ctrl_q.put({"cmd": "stop"})
    th.join(timeout=120)
    gated = [e for e in eps if e["end"] == "gate_truncate"]
    assert gated, [e["end"] for e in eps]
    assert all(e["end_sector"] == 1 and e["s_end"] >= 6.0 and not e["success_cp"] for e in gated)
    # every gate step is a bootstrapped truncation (not terminal) with the critic's value of the next state
    tr = np.concatenate([c["truncated"] for c in chunks])
    te = np.concatenate([c["terminated"] for c in chunks])
    tv = np.concatenate([c["trunc_values"] for c in chunks])
    assert tr.sum() >= len(gated) and not (tr & te).any()
    assert np.allclose(tv[tr], 0.75)


@pytest.mark.skipif(sys.platform != "win32", reason="named shared memory / events need Windows")
def test_worker_band_starts_play_through_rehearsal_starts_gated(tmp_path, fake_record):
    """Frontier 300 (start_back 300), gate below 4000 following the frontier: curriculum starts (>= 300) are never gated,
    rehearsal starts at 40-100 (below the frontier) are."""
    cfg = EnvConfig(level=1, copy="train-00", out_root=str(tmp_path / "ep2"), action_repeat=4,
                    start_record=str(fake_record), curriculum={"start_back": 300, "seed": 1},
                    rehearse_starts=[[40, 100]], p_rehearse=0.5)
    spec = WorkerSpec(wid=0, copy="train-00", env=cfg, seed=5, chunk=64, launcher_factory=fakegame.FakeLauncher,
                      prune="none", rail_gate={"s": 6.0, "below": 4000, "follow_frontier": True})
    ctrl_q, data_q = queue.Queue(), queue.Queue()
    w = Worker(spec, ctrl_q, data_q)
    w.model.act = lambda x, gen=None: (1, np.float32(-2.2), np.float32(0.0))
    w.model.value = lambda x: np.float32(0.5)
    n = w.env.observation_space.shape[0]
    ctrl_q.put({"version": 0, "model": {k: v.detach().numpy() for k, v in w.model.state_dict().items()},
                "rms": {"mean": np.zeros(n), "var": np.ones(n), "count": 1.0},
                "curriculum": w.env.get_start_state()})
    th = threading.Thread(target=w.run, daemon=True)
    th.start()
    eps = []
    for _ in range(8):
        m = data_q.get(timeout=120)
        assert "fatal" not in m and not m["errors"], m.get("errors")
        eps += m["episodes"]
    ctrl_q.put({"cmd": "stop"})
    th.join(timeout=120)
    assert int(w.env.curriculum.frontier) == 300
    low = [e for e in eps if e["kind"] == "tas" and 0 < e["start_frame"] < 300]
    band = [e for e in eps if e["kind"] == "tas" and e["start_frame"] >= 300]
    assert low and band
    assert any(e["end"] == "gate_truncate" for e in low)
    assert not any(e["end"] == "gate_truncate" for e in band)


@pytest.fixture(autouse=True)
def _clear_geometry_cache():
    yield
    from ballrain import features
    features._GEOM_CACHE.clear()
    features._KEY_OF.clear()


def test_box_gate_pays_once_above_the_top():
    ep = object.__new__(_Episode)
    ep.gates_hit = set()
    g = {"walltop": {"boxes": [[-168.0, -163.0, -165.0, -138.0, 21.5], [-158.6, -153.6, -141.0, -136.0, 23.5]], "bonus": 2.0}}
    v = np.zeros(3)
    assert ep.gate(np.array([-160.0, 19.4, -150.0]), v, g) == 0.0  # middle depression: no
    assert ep.gate(np.array([-165.5, 19.4, -150.0]), v, g) == 0.0  # below the strip top: no
    assert ep.gate(np.array([-156.0, 22.9, -139.0]), v, g) == 0.0  # beside / below the single box top: no
    assert ep.gate(np.array([-165.5, 22.9, -150.0]), v, g) == 2.0  # on the west strip
    assert ep.gate(np.array([-165.5, 22.9, -151.0]), v, g) == 0.0  # once per episode
    ep.gates_hit = set()
    assert ep.gate(np.array([-156.0, 24.6, -139.0]), v, g) == 2.0  # on the single box
    ep.gates_hit = set()  # the centre / radius form still works
    assert ep.gate(np.array([0.0, 0.0, 0.0]), v, {"c": {"center": [0, 0, 1], "radius": 2.0, "bonus": 0.5}}) == 0.5
