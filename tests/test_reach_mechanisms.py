"""Opt-in mechanism data, worker gating, live pose selection, and legacy byte parity."""
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import numpy as np
import pytest

from ballrain import fastreach as FR, protocol as P, reach, reach_features as RF, reach_mechanisms as M
from ballrain.env import EnvConfig
from ballrain.features import FeatureConfig, make_layout
from ballrain.train.remote import PROTOCOL, RemoteHub
from test_reach_obs import make_env, needs_numba


def test_all_retail_tables_are_pinned_and_no_layout_change():
    base = FeatureConfig(reach=True)
    mech = FeatureConfig(reach=True, reach_version=RF.reach_version(True))
    assert RF.is_mechanism_version(mech.reach_version)
    assert mech.reach_version.endswith(M.asset_pin())
    assert FeatureConfig(**asdict(mech)).reach_version == mech.reach_version
    assert make_layout(base) == make_layout(mech)
    for n in range(1, 14):
        data = M.level_asset(n)
        assert len(data["placements"]) == sum(data["ph_counts"].values())
        assert all(p["type"] in data["rules"] for p in data["placements"])
    with pytest.raises(RF.ReachVersionError):
        RF.ReachBlock(expect_version=mech.reach_version + "-wrong")


@pytest.mark.parametrize("cap", [None, True, "other-version"])
def test_hub_refuses_missing_or_wrong_version_before_config(cap):
    version = RF.reach_version(True)
    cfg = EnvConfig(features=FeatureConfig(reach=True, reach_version=version))
    hub = object.__new__(RemoteHub)
    hub.template = {"env": cfg}
    hub.stats = {"disconnects": 0}
    hub.log, hub._cond = Mock(), threading.Condition()
    conn = Mock()
    conn.recv.return_value = {"type": "hello", "protocol": PROTOCOL, M.CAPABILITY: cap}
    hub._serve(conn)
    conn.send.assert_called_once_with({"type": "error", "error": f"mechanism reach requires {M.CAPABILITY}={version}"})


def test_pose_must_be_present_aligned_and_near_hinge():
    body = np.zeros(2, P.BODY_DT)
    tr = SimpleNamespace(obs=SimpleNamespace(body=body))
    probe = (77, 0, 0, 0, 0, 0, -1, .996)
    body[1]["ckid"] = 77
    body[1]["ck_matrix"][0] = [0, -1, 0]
    assert not M.uses_lowered_plate(tr, [0, 2, 0], probe)
    body[1]["ck_matrix"][0] = [0, 0, -1]
    assert M.uses_lowered_plate(tr, [0, 2, 0], probe)
    assert not M.uses_lowered_plate(tr, [0, 2, -80], probe)
    body[1]["ck_matrix"][3] = [0, 0, 2]
    assert not M.uses_lowered_plate(tr, [0, 2, 0], probe)
    tr.obs = None
    assert not M.uses_lowered_plate(tr, [0, 2, 0], probe)


@needs_numba
def test_plate_layers_match_reference_and_preserve_flag_off_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("BALLRAIN_RUNTIME", str(tmp_path / "rt"))
    # Across the fake road, north of the spawn z=-5. A 10-unit tall wall cannot be mounted.
    corners = np.array([[-5, 0, -1], [5, 0, -1], [5, 10, -1], [-5, 10, -1]], dtype=float)
    wall = corners[[[0, 1, 2], [0, 2, 3]]]
    monkeypatch.setattr(M, "layers_for_tracker", lambda tracker: ((None, wall, None), None))
    env = make_env(tmp_path, True)
    try:
        env.reset()
        t = env.session.tracker
        off = RF.ReachBlock()
        on = RF.ReachBlock(expect_version=RF.reach_version(True))
        ref = reach.estimator_for_tracker(t, mechanisms=True)
        p = np.array(t.obs.ball["pos"], float)
        v = np.zeros(3)
        fr, key = RF.fields_for(t)
        for k in (0, 1, 2):
            baseline, _ = fr.perceive(p, v, k, 0, bodies=RF.moving_bodies(t, p, 60.0))
            a, raw, off_key = off.compute(t, p, v, 0, k)
            b, _, on_key = on.compute(t, p, v, 0, k)
            assert raw.tobytes() == baseline.tobytes()
            assert a.tobytes() == RF.scale(baseline).tobytes()
            assert off_key == key and on_key != key
            idx = FR.FEATURE_NAMES.index("up_obst")
            if k == 1:
                assert b[idx] < a[idx]
                assert ref.estimate(p, v, k)["keys"][0]["blocked_at"] < 4.0
            else:
                assert a.tobytes() == b.tobytes()
        p[2] -= 100
        assert off.compute(t, p, v, 0, 1)[0].tobytes() == on.compute(t, p, v, 0, 1)[0].tobytes()
    finally:
        env.close()


def test_reference_rollout_fan_lifts_paper():
    from ballrain.fastreach import FAN_R_NOM, fan_dv, fan_obbs
    xz = np.array([[-10, 0, -10], [-10, 0, 10], [10, 0, 10], [10, 0, -10]], float)
    A, B, C = xz[[[0, 1, 2], [0, 2, 3]]].transpose(1, 0, 2)
    smap = reach.SupportMap(A, B, C)
    fan = fan_obbs([("fan", np.eye(4)[:, :3])])
    config = reach.ReachConfig(step_psi=1)
    ball = reach.BALLS[1]
    pushes = np.zeros((1, 132, 2))
    grounded = reach.rollout(smap, [0, 2.01, 0], [0, 0, 0], ball, pushes, config)
    lifted = reach.rollout(smap, [0, 2.01, 0], [0, 0, 0], ball, pushes, config,
                           fan=(fan, fan_dv(ball), FAN_R_NOM))
    assert lifted["traj"][0, -1, 1] - grounded["traj"][0, -1, 1] > 4.0
