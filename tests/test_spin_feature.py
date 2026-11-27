"""The spin block (FeatureConfig.spin): layout, the body-to-world rotation of the ball's angular velocity, off = the old
serialized configuration, and growing a pickups policy by the block with zero input weights."""
from dataclasses import asdict

import numpy as np
import torch

from ballrain.features import FeatureConfig, make_layout, spin_along_velocity, spin_vector, to_frame
from ballrain.train.grow import check_prefix, load_grown
from ballrain.train.ppo import ActorCritic, RunningMeanStd, normalize

R10 = dict(camera="world", version="f8", reach=False, pickups=True, far_waypoints=(20.0, 30.0, 45.0),
           ref_speed_at=(0.0, 15.0, 30.0))


def _n(c):
    return sum(k for _, k in make_layout(c))


def test_layout_appends_spin_last_and_off_keeps_the_config():
    a, b = FeatureConfig(**R10), FeatureConfig(**R10, spin=True)
    la, lb = make_layout(a), make_layout(b)
    assert lb[:len(la)] == la and lb[len(la):] == [("spin", 3), ("spin_vel", 3)]
    assert "spin" not in a.to_dict() and b.to_dict()["spin"] is True
    check_prefix(asdict(a), asdict(b), _n(a), _n(b))  # an r10 policy grows into it


def _quat(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    s = np.sin(angle / 2)
    return (axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2))


def test_spin_vector_rotates_the_body_frame_rate_to_the_world_and_the_feature_frame():
    world = (np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))
    w_body = np.array([0.0, 0.0, 6.0])  # spinning about the body z axis
    ball = {"rot_speed": w_body.astype(np.float32), "rot_speed_change": np.zeros(3, np.float32),
            "q_next": (0.0, 0.0, 0.0, 1.0)}
    np.testing.assert_allclose(spin_vector(ball, *world), [0, 0, 6.0], atol=1e-12)
    # the body turned 90 degrees about the world y axis: body z is world x (column-vector rotation R(q) @ v)
    ball["q_next"] = _quat([0, 1, 0], np.pi / 2)
    np.testing.assert_allclose(spin_vector(ball, *world), [6.0, 0, 0], atol=1e-9)
    # rot_speed_change is part of the rate, as for "angvel"
    ball["rot_speed_change"] = np.array([0, 0, -1.0], np.float32)
    np.testing.assert_allclose(spin_vector(ball, *world), [5.0, 0, 0], atol=1e-6)
    # a camera yaw frame: the same world vector expressed like vel_cam
    right, fwd = np.array([0.0, 0, -1.0]), np.array([1.0, 0, 0])
    np.testing.assert_allclose(spin_vector(ball, right, fwd), to_frame([5.0, 0, 0], right, fwd), atol=1e-6)
    # the spin is invariant to how the body has turned about the spin axis itself (a steadily rolling ball)
    ball = {"rot_speed": np.array([0, 0, 6.0], np.float32), "rot_speed_change": np.zeros(3, np.float32)}
    outs = [spin_vector(dict(ball, q_next=_quat([0, 0, 1], a)), *world) for a in (0.0, 0.7, 2.0, 4.0)]
    for o in outs:
        np.testing.assert_allclose(o, outs[0], atol=1e-9)


def test_grown_policy_ignores_the_new_block():
    a, b = FeatureConfig(**R10), FeatureConfig(**R10, spin=True)
    n0, n1 = _n(a), _n(b)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    old = ActorCritic(n0, 9, (64, 64), "tanh", "ball_type")
    rms = RunningMeanStd(n0)
    rms.update(rng.normal(0.2, 1.5, (512, n0)))
    opt = torch.optim.Adam(old.parameters(), lr=1e-4)
    src = {"model": old.state_dict(), "opt": opt.state_dict(), "rms": rms.state_dict(), "n_obs": n0, "n_act": 9,
           "env_config": {"features": asdict(a)}}
    new = ActorCritic(n1, 9, (64, 64), "tanh", "ball_type")
    rms1 = RunningMeanStd(n1)
    assert load_grown(new, torch.optim.Adam(new.parameters(), lr=1e-4), rms1, src, n1, asdict(b), load_opt=False)
    x0 = rng.normal(0.2, 1.5, (64, n0))
    x1 = np.concatenate([x0, rng.uniform(-1, 1, (64, n1 - n0))], 1)
    a0, a1 = normalize(x0, rms.state_dict()), normalize(x1, rms1.state_dict())
    with torch.no_grad():  # float32: summation order only (a longer first-layer dot product)
        la, va = old(torch.as_tensor(a0))
        lb, vb = new(torch.as_tensor(a1))
    assert float((la - lb).abs().max()) <= 1e-6 and float((va - vb).abs().max()) <= 1e-6
    assert torch.equal(la.argmax(-1), lb.argmax(-1))
    o64, n64 = old.double(), new.double()  # float64: the zero columns contribute nothing
    with torch.no_grad():
        d = o64(torch.as_tensor(a0, dtype=torch.float64))[0] - n64(torch.as_tensor(a1, dtype=torch.float64))[0]
    assert float(d.abs().max()) <= 1e-12


def test_spin_along_velocity_roll_side_yaw_and_the_fade():
    r = 2.0
    for v in (np.array([10.0, 0, 0]), np.array([0, -3.0, -7.0]), np.array([4.0, 1.0, 4.0])):
        h = np.hypot(v[0], v[2])
        w_roll = np.cross([0, 1.0, 0], v) / r  # rolling forward without slipping
        roll, side, yaw = spin_along_velocity(w_roll, v, fade=0.0)
        assert roll > 0 and abs(roll - h / r) < 1e-9 and abs(side) < 1e-9 and abs(yaw) < 1e-12
        roll, side, yaw = spin_along_velocity(-w_roll, v, fade=0.0)  # backspin
        assert roll < 0
        f = np.array([v[0], 0, v[2]]) / h
        _, side, _ = spin_along_velocity(3.0 * f, v, fade=0.0)  # spinning about the direction of travel
        assert abs(side - 3.0) < 1e-9
        _, _, yaw = spin_along_velocity([0, 5.0, 0], v, fade=0.0)
        assert yaw == 5.0
    w = np.array([1.0, 2.0, 3.0])
    assert np.all(spin_along_velocity(w, [0, 5.0, 0]) == 0)  # no horizontal motion: no direction, all 0
    small = spin_along_velocity(w, [1e-3, 0, 0], fade=1.0)
    assert np.abs(small).max() < 1e-5  # continuous through rest
    big = spin_along_velocity(w, [100.0, 0, 0], fade=1.0)
    np.testing.assert_allclose(big, spin_along_velocity(w, [100.0, 0, 0], fade=0.0), rtol=2e-4)
