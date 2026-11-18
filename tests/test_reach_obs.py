"""The f7 reach observation block (FeatureConfig.reach, ballrain.reach_features): layout (appended after every f6
entry), version, numba requirement, scaling, the directions in the action frame (world actions and camera-relative
keys) on the FakeGame road, the env vector equal to an offline perceive, and the field cache (content-addressed,
one build when several processes need it at once, atomic files)."""
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from ballrain import fakegame, fastreach as FR, reach, reach_features as RF
from ballrain.env import ARROWS9, BallanceEnv, EnvConfig
from ballrain.features import FeatureConfig, FeatureExtractor, make_layout
from ballrain.state import GameTracker

needs_numba = pytest.mark.skipif(not FR.HAVE_NUMBA, reason="the reach block needs numba")
R8 = dict(camera="world", hs_block=False, far_waypoints=(20.0, 30.0, 45.0), ref_speed_at=(0.0, 15.0, 30.0))
NAMES = {n: i for i, n in enumerate(FR.FEATURE_NAMES)}


def test_f6_layout_unchanged_and_f7_appends_the_block_last():
    f6, f7 = FeatureConfig(**R8), FeatureConfig(**R8, reach=True)
    l6, l7 = make_layout(f6), make_layout(f7)
    assert f6.version == "f6" and f7.version == "f7"
    assert l7[:len(l6)] == l6 and l7[-1] == ("reach", FR.N_FEATURES) and len(l7) == len(l6) + 1
    assert sum(k for _, k in l6) == 208  # the r8 layout
    # round trips: an f7 config dict stays f7; an f7 label without the block is the f6 layout
    assert FeatureConfig(**asdict(f7)).version == "f7"
    assert FeatureConfig(**dict(asdict(f6), version="f7")).version == "f6"
    assert FeatureConfig(**{k: v for k, v in asdict(f6).items() if k != "reach"}).reach is False


def test_reach_directions_are_the_action_directions():
    # direction slot q of fastreach is reach.KEY_COMBOS[q] = action q + 1 (ARROWS9[0] is "no key")
    assert tuple(reach.KEY_COMBOS) == tuple(ARROWS9[1:])
    assert FR.DIR_NAMES == ("up", "down", "left", "right", "up_left", "up_right", "down_left", "down_right")


def test_reach_refused_without_numba(monkeypatch):
    monkeypatch.setattr(FR, "HAVE_NUMBA", False)
    with pytest.raises(RuntimeError, match="numba"):
        FeatureExtractor(FeatureConfig(reach=True))
    FeatureExtractor(FeatureConfig())  # f6 does not need it


@needs_numba
def test_non_finite_state_gives_a_zero_block_not_an_error():
    blk, raw, key = RF.ReachBlock().compute(None, [np.nan, 0, 0], [0, 0, 0], 0, 0)  # the tracker is never touched
    assert key is None and not blk.any() and blk.shape == (FR.N_FEATURES,)


def test_scaling_is_finite_and_within_one():
    raw = np.array([1, 1, 1, 1, 3.0, -90, 40, -40, -90, 3.0, 1, 1, 1, 0.3, 0.9, -0.3, -1, 0, 1]
                   + [20, -20, 20, 20, -20] * 8, np.float64)
    s = RF.scale(raw)
    assert np.isfinite(s).all() and s.min() >= -1 and s.max() <= 1
    assert s[NAMES["air_t_first"]] == 1.0 and s[NAMES["air_first_dy"]] == -1.0 and s[NAMES["up_edge"]] == 1.0
    assert s[NAMES["up_brake_margin"]] == -1.0 and s[NAMES["sup_n_y"]] == 0.9
    raw[5], raw[6], raw[7] = np.nan, np.inf, -np.inf
    s = RF.scale(raw)
    assert s[5] == 0 and s[6] == 1 and s[7] == -1
    # the head flags and normals keep their value; a 10-unit edge is 0.5
    raw2 = np.zeros(FR.N_FEATURES)
    raw2[NAMES["right_edge"]] = 10.0
    raw2[NAMES["doomed"]] = 1.0
    s2 = RF.scale(raw2)
    assert s2[NAMES["right_edge"]] == 0.5 and s2[NAMES["doomed"]] == 1.0


# ---------------------------------------------------------------------------------------------- FakeGame road

def make_env(tmp_path, world: bool, **game):
    cfg = EnvConfig(level=1, out_root=str(tmp_path / "ep"), action_repeat=2, world_actions=world,
                    features=FeatureConfig(camera="world" if world else "orient_ref", reach=True))
    return BallanceEnv(cfg, launcher=fakegame.FakeLauncher(**game))


def test_env_refuses_a_reach_frame_other_than_the_action_frame(tmp_path):
    with pytest.raises(ValueError, match="action directions"):
        BallanceEnv(EnvConfig(level=1, out_root=str(tmp_path), world_actions=True,
                              features=FeatureConfig(camera="orient_ref", reach=True)), launcher=fakegame.FakeLauncher())
    with pytest.raises(ValueError, match="action directions"):
        BallanceEnv(EnvConfig(level=1, out_root=str(tmp_path), world_actions=False,
                              features=FeatureConfig(camera="world", reach=True)), launcher=fakegame.FakeLauncher())


def _block(env, obs):
    return obs[env.fx.slices()["reach"]]


@needs_numba
@pytest.mark.parametrize("world", [True, False])
def test_fake_road_directions_in_the_action_frame(tmp_path, monkeypatch, world):
    """The FakeGame road runs from z 0 to z -200 (x within +-5); the ball starts at z -5, the camera looks along -z
    (Cam_OrientRef yaw 180 = quarter 2). World actions: UP = +z, the near end of the road (an edge ~5 ahead), DOWN = -z
    is open for more than L_MAX. Camera-relative keys at quarter 2: UP pushes -z (open), DOWN +z (the near edge)."""
    monkeypatch.setenv("BALLRAIN_FASTREACH_DIR", str(tmp_path / "fr"))
    monkeypatch.setattr(RF, "_FIELDS", type(RF._FIELDS)())  # the process cache would serve the other variant's fields
    env = make_env(tmp_path, world)
    try:
        obs, _ = env.reset()
        t = env.session.tracker
        assert env.fx.action_quarter(t) == (0 if world else 2)
        b = _block(env, obs)
        assert b[NAMES["grounded"]] == 1 and b[NAMES["doomed"]] == 0
        assert abs(b[NAMES["sup_n_y"]] - 1.0) < 1e-6
        open_, near = ("down", "up") if world else ("up", "down")
        assert b[NAMES[f"{open_}_edge"]] == 1.0  # nothing within L_MAX = 20
        assert 0.0 < b[NAMES[f"{near}_edge"]] < 0.5  # the road end ~5 away (< 10)
        assert 0.0 < b[NAMES["left_edge"]] < 0.5 and 0.0 < b[NAMES["right_edge"]] < 0.5  # the sides, 5 away
        # the env vector is the offline perceive of the same state (same fields, zones, bodies, quarter)
        o = t.obs
        pos = np.asarray(o.ball["pos"], np.float64)
        vel = np.asarray(o.ball["speed"], np.float64) + np.asarray(o.ball["speed_change"], np.float64)
        fr, _ = RF.fields_for(t)
        raw, _ = fr.perceive(pos, vel, t.ball_type(), env.fx.action_quarter(t),
                             bodies=RF.moving_bodies(t, pos, 60.0))
        assert np.array_equal(b, RF.scale(raw).astype(np.float32))
        # a few steps: always finite, within [-1, 1]
        for a in (1, 1, 5, 0, 2):
            obs, _, term, trunc, _ = env.step(a)
            b = _block(env, obs)
            assert np.isfinite(b).all() and np.abs(b).max() <= 1.0
            if term or trunc:
                break
        files = sorted(p.name for p in (tmp_path / "fr").iterdir())
        assert len(files) == 1 and files[0].startswith("L01-") and files[0].endswith(".npz"), files
    finally:
        env.close()


def test_unknown_zones_leave_out_the_player_balls():
    """The moving box of the FakeGame level is a zone; the ball's Physicalize record never is (the transformer /
    respawn position is not an obstacle)."""
    tr = fake_tracker()
    zones = RF._zone_boxes(tr.level)
    assert len(zones) == 1
    lo, hi = zones[0][:3], zones[0][3:]
    assert np.allclose(lo, [3 - 1 - 4, 1 - 1 - 4, -60 - 1 - 4]) and np.allclose(hi, [3 + 1 + 4, 1 + 1 + 4, -60 + 1 + 4])
    names = [tr.level.name(ck) for ck in tr.level.phys_by_ent]
    assert "Ball_Wood" in names


# -------------------------------------------------------------------------------------------- field cache

def fake_tracker() -> GameTracker:
    """A tracker holding the FakeGame level message (no game process)."""
    g = fakegame.FakeGame.__new__(fakegame.FakeGame)
    g.frames_done, g.level = 0, 1
    g.box_pos = np.array([3.0, 1.0, -60.0])
    g.sources = [{"pos": fakegame.SCORE_BALL, "suffix": "", "base": fakegame.ENT_SCORE[0]},
                 {"pos": fakegame.SCORE_BALL_2, "suffix": "001", "base": fakegame.ENT_SCORE[1]}]
    g.arrays = {}
    g._init_arrays(3)
    tr = GameTracker(1)
    tr.merge_level(g.level_message())
    return tr


_SUB = r"""
import json, sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
from ballrain import fastreach as F
from test_reach_obs import fake_tracker
fr = F.FastReach.fields_for_tracker(fake_tracker(), cache_dir=sys.argv[3])
print(json.dumps({"built_by": fr.info.get("built_by"), "key": fr.info["key"], "hh": float(abs(fr.Hh[fr.Hh == fr.Hh]).sum())}))
"""


@needs_numba
def test_one_build_when_several_processes_start_at_once(tmp_path):
    root = str(Path(__file__).resolve().parents[1])
    here = str(Path(__file__).resolve().parent)
    cache = tmp_path / "fr"
    procs = [subprocess.Popen([sys.executable, "-I", "-c", _SUB, root, here, str(cache)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for _ in range(4)]
    outs = []
    for p in procs:
        out, err = p.communicate(timeout=600)
        assert p.returncode == 0, err[-3000:]
        outs.append(json.loads(out.strip().splitlines()[-1]))
    assert len({o["built_by"] for o in outs}) == 1  # one builder, the others loaded its file
    assert len({o["key"] for o in outs}) == 1 and len({o["hh"] for o in outs}) == 1
    files = sorted(p.name for p in cache.iterdir())
    assert len(files) == 1 and files[0].endswith(".npz"), files  # no lock, no temporary file left
    fr = FR.FastReach.load(cache / files[0])
    assert fr.info["key"] == FR.FastReach.fields_key(fake_tracker())


@needs_numba
def test_fields_key_ignores_moving_bodies_and_the_file_is_reused(tmp_path):
    tr = fake_tracker()
    k0 = FR.FastReach.fields_key(tr)
    assert k0 != FR.FastReach.tracker_key(tr)[0]  # tracker_key hashes the moving box too
    fr = FR.FastReach.fields_for_tracker(tr, cache_dir=tmp_path)
    assert len(fr.uzones) == 0
    fr2 = FR.FastReach.fields_for_tracker(tr, cache_dir=tmp_path)
    assert fr2.info["built_by"] == fr.info["built_by"] and fr2.info["key"] == k0
    # a stale lock left by a killed process is taken over
    other = fake_tracker()
    other.level.depth = [(0, np.array([-60.0, -40.0, -260.0]), np.array([60.0, -21.0, 60.0]))]
    k1 = FR.FastReach.fields_key(other)
    import hashlib
    name = f"L01-{hashlib.sha256(k1.encode()).hexdigest()[:16]}"
    lock = tmp_path / f"{name}.lock"
    lock.write_text("1\n")
    old = lock.stat().st_mtime - 1000
    os.utime(lock, (old, old))
    fr3 = FR.FastReach.fields_for_tracker(other, cache_dir=tmp_path, lock_timeout_s=60)
    assert fr3.info["key"] == k1 and not lock.exists()


def test_fields_key_does_not_depend_on_ck_ids(monkeypatch):
    """Retail and the headless SimTool give the same L2 static entities and meshes other CK ids (another id order):
    the fields key and the build input are canonical (entity names, triangle coordinates), not in id order."""
    from types import SimpleNamespace

    from ballrain import features, geometry as G
    tri = [np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1.0]]), np.array([[5, 1, 5], [6, 1, 5], [5, 1, 6.0]]),
           np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1.0]]), np.array([[9, 0, 0], [9, 1, 0], [9, 0, 1.0]])]

    def tracker(perm, ids):
        names = {ids[0]: "A01_Floor", ids[1]: "A02_Floor", ids[2]: "Rail_01"}
        owners = [ids[0], ids[1], ids[0], ids[2]]
        T = [tri[i] for i in perm]
        O = [owners[i] for i in perm]
        ts = G.TriangleSet([t[0] for t in T], [t[1] for t in T], [t[2] for t in T], O, grid_cell=None)
        rails = G.TriangleSet([tri[3][0]], [tri[3][1]], [tri[3][2]], [ids[2]], grid_cell=None)
        geo = SimpleNamespace(tris=ts, rails=rails)
        lvl = SimpleNamespace(name=lambda o: names[int(o)], depth=[])
        return SimpleNamespace(level=lvl, level_no=2, _geo=geo)

    monkeypatch.setattr(features, "level_geometry", lambda t: t._geo)
    a, b = tracker([0, 1, 2, 3], (101, 102, 103)), tracker([3, 2, 1, 0], (77, 5, 900))
    ga, gb = FR.FastReach.canonical_geometry(a), FR.FastReach.canonical_geometry(b)
    for x, y in zip(ga, gb):
        assert np.array_equal(x, y)
    assert ga[4].sum() == 1  # the rail triangle
    assert FR.FastReach.fields_key(a) == FR.FastReach.fields_key(b)
    c = tracker([0, 1, 2, 3], (101, 102, 103))
    c._geo.tris.A[1, 1] += 1e-6  # any geometry change changes it
    assert FR.FastReach.fields_key(c) != FR.FastReach.fields_key(a)


# ------------------------------------------------------------------------------------------------ pool gate

def test_pool_refuses_f7_and_f6_manifests_have_no_reach_field(tmp_path):
    pytest.importorskip("torch")
    from ballrain.pool import weights as W
    from ballrain.train import ppo

    def state(fc):
        n_obs = sum(k for _, k in make_layout(fc))
        model = ppo.ActorCritic(n_obs, 9, (16,), "tanh")
        ec = asdict(EnvConfig(level=1, world_actions=True, action_repeat=2, start_record="SR01_test", features=fc))
        return {"model": model.state_dict(), "rms": {"mean": np.zeros(n_obs), "var": np.ones(n_obs), "count": 1.0},
                "n_obs": n_obs, "n_act": 9, "args": {"activation": "tanh", "hidden": "16"}, "env_config": ec,
                "counters": {"update": 1, "steps": 1}}

    with pytest.raises(ValueError, match="f7"):
        W.export_state(state(FeatureConfig(camera="world", reach=True)), tmp_path / "a", "r-test")
    m = W.export_state(state(FeatureConfig(camera="world")), tmp_path / "b", "r-test")
    assert "reach" not in m["features"] and m["features"]["version"] == "f6"
