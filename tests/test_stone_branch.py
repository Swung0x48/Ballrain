"""Regression against bytes measured on the parent commit (FakeLauncher; fixed seed and actions)."""

import hashlib
import struct
import sys


import numpy as np
import pytest

from ballrain.branch import load_branch
from ballrain.path import ReferencePath, REFPATH_DIR


def episode_digest(tmp):
    from ballrain.env import BallanceEnv, EnvConfig
    from ballrain.fakegame import FakeLauncher

    env = BallanceEnv(
        EnvConfig(
            level=2,
            copy="hlw-98",
            out_root=str(tmp),
            action_repeat=4,
            max_episode_frames=200,
        ),
        launcher=FakeLauncher(),
    )
    digest = hashlib.sha256()
    try:
        obs, _ = env.reset(seed=19)
        digest.update(obs.tobytes())
        while True:
            obs, reward, term, trunc, info = env.step(1)
            digest.update(obs.tobytes())
            digest.update(
                struct.pack(
                    "<d??ii", reward, term, trunc, info["frames_done"], info["sector"]
                )
            )
            if term or trunc:
                digest.update(info["episode_end"].encode())
                return digest.hexdigest()
    finally:
        env.close()


@pytest.mark.skipif(
    sys.platform != "win32", reason="FakeLauncher transport requires Windows"
)
def test_off_episode_bytes(tmp_path):
    assert (
        episode_digest(tmp_path)
        == "cbc11a32f30c3f6b050a4c106bb46fecc7ac7a9282251a7e17b0d8ee5a8b8f7e"
    )


def route():
    main = ReferencePath.load(REFPATH_DIR / "L02.npz")
    return load_branch(main, 2, "runtime/refpaths-alt/L02-stone", None)


def test_switch_and_respawn():
    f = route()
    p = [948.74, 2.77, -56.51]
    path, paper = f.update(p, 4, 1)
    assert path is f.main
    path, stone = f.update(p, 4, 2)
    assert path is f.alt and stone["s"] == paper["s"]
    f.reset()
    p = f.alt.point_at(1700.0)
    _, r = f.update(p, 4, 2)
    assert r["branch"] and abs(r["local_s"] - 1700) < 1e-7 and r["ref_elapsed_ms"] == 0
    f.reset()
    assert f.update(p, 5, 2)[0] is f.main
    f.reset()
    assert f.update(f.main.P[0], 1, 1)[1]["s"] == 0


def test_merge_does_not_pay_twice():
    f = route()
    p = f.alt.point_at(f.hi - 0.4)
    _, first = f.update(p, 4, 2)
    for _ in range(5):
        for bt in (1, 2):
            _, r = f.update(p, 4, bt)
            assert r["s"] == first["s"] and r["ref_elapsed_ms"] == 0
    after = f.main.point_at(f.stations[0, -2] + 1)
    f.update(after, 4, 1)  # switch is free in neither direction
    _, limit = f.update(after + [0, 0, 0.2], 4, 1)
    for _ in range(5):
        f.update(p, 4, 2)
        _, r = f.update(after, 4, 1)
        assert r["s"] == limit["s"] and r["ref_elapsed_ms"] == limit["ref_elapsed_ms"]


def test_branch_observation_uses_local_metres(monkeypatch):
    from ballrain.features import FeatureConfig, FeatureExtractor
    from test_features import make_tracker, obs_at

    tr = make_tracker()
    tr.checkpoints = 3
    fx = FeatureExtractor(
        FeatureConfig(camera="world", far_waypoints=(20,), ref_speed_at=(5,))
    )
    fx.branch = f = route()
    fx.path = f.main
    monkeypatch.setattr(fx, "reference_path", lambda tracker: f.main)
    p = f.alt.point_at(1700.0) + [0.25, 0, 0]
    obs_at(tr, p, btype=2)
    obs, info = fx.compute(tr)
    sl, pr = fx.slices(), info["path"]
    assert pr["branch"] and pr["s"] < pr["local_s"]
    assert np.allclose(obs[sl["path_progress"]], pr["s"] / f.main.length)
    assert np.allclose(obs[sl["path_lateral"]], pr["dist"] / fx.cfg.path_lateral_scale)
    assert np.allclose(obs[sl["path_tangent_cam"]], pr["tangent"][[0, 2]])
    assert np.allclose(
        obs[sl["far_waypoints"]],
        (f.alt.point_at(pr["local_s"] + 20) - p) / fx.cfg.pos_scale,
    )
    assert np.allclose(
        obs[sl["waypoints"]][:3],
        (f.alt.point_at(pr["local_s"] + 5) - p) / fx.cfg.pos_scale,
    )
    assert np.allclose(
        obs[sl["ref_speed"]], f.alt.speed_at(pr["local_s"] + 5) / fx.cfg.vel_scale
    )
    obs_at(tr, f.alt.point_at(1705.0), btype=2)
    _, latter = fx.compute(tr)
    assert latter["path"]["ref_elapsed_ms"] == pytest.approx(500, abs=1)
    assert pr["ref_remaining_ms"] - latter["path"]["ref_remaining_ms"] == pytest.approx(
        500, abs=1
    )
