"""The contact-pressure block (FeatureConfig.pressure) and its FRCP wire section: off = the old layout and serialized
configuration, FRCP parsed when present and None when absent (old engines), the block's values (the engine's force per
contact from FRCP, the engine's contact normal from FRIC lt_normal turned towards the ball, slots by force, squashing,
the summed support force), a refusal without the section, and growing a policy by the block with zero input weights."""
import struct
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ballrain import protocol as P
from ballrain.features import (PRESSURE_AGG_FIELDS, PRESSURE_SLOT_FIELDS, PRESSURE_SLOTS, FeatureConfig,
                               FeatureExtractor, make_layout, to_frame)
from ballrain.state import G_RAILS
from ballrain.train.grow import check_prefix, load_grown
from ballrain.train.ppo import ActorCritic, RunningMeanStd, normalize

R10 = dict(camera="world", version="f8", reach=False, pickups=True, far_waypoints=(20.0, 30.0, 45.0),
           ref_speed_at=(0.0, 15.0, 30.0), spin=True, orient=True)


def _n(c):
    return sum(k for _, k in make_layout(c))


def _section(tag, payload, ver=1):
    return struct.pack("<4sHHI", tag.encode(), ver, 0, len(payload)) + payload + b"\0" * (-len(payload) % 4)


def _message(frames_done, sections):
    body = b"".join(sections)
    return struct.pack("<4sHHIIII", b"OBS1", 1, 0, 24 + len(body), frames_done, len(sections), 0) + body


def test_layout_appends_pressure_last_and_off_keeps_the_config():
    a, b = FeatureConfig(**R10), FeatureConfig(**R10, pressure=True)
    la, lb = make_layout(a), make_layout(b)
    assert lb[:len(la)] == la
    assert lb[len(la):] == [("pressure", PRESSURE_SLOTS * len(PRESSURE_SLOT_FIELDS)),
                            ("pressure_agg", len(PRESSURE_AGG_FIELDS))]
    assert _n(b) - _n(a) == 19
    d = a.to_dict()
    assert "pressure" not in d and "pressure_scale" not in d
    assert b.to_dict()["pressure"] is True and "pressure_scale" in b.to_dict()
    assert FeatureConfig().to_dict() == FeatureConfig(pressure=False).to_dict()
    check_prefix(asdict(a), asdict(b), _n(a), _n(b))  # an r10 spin + orient policy grows into it


def _fric_item(partner):
    return struct.pack("<IBBBBff3f3dB", partner, 3, 2, 1, 0, 0.01, 0.56, 0, -1, 0, 1, 2, 3, 1)


def test_frcp_parsed_when_present_and_none_when_absent():
    fric = struct.pack("<H", 2) + _fric_item(501) + _fric_item(777)
    frcp = struct.pack("<H", 2) + struct.pack("<If", 501, 12.5) + struct.pack("<If", 777, 0.25)
    o = P.parse_observation(_message(9, [_section("FRIC", fric), _section("FRCP", frcp)]))
    assert o.unknown == [] and len(o.fric) == 2 and o.frcp is not None and len(o.frcp) == 2
    assert P.FRCP_DT.itemsize == 8 and o.frcp["partner"].tolist() == [501, 777]
    assert o.frcp["pressure"][0] == np.float32(12.5) and o.frcp["pressure"][1] == np.float32(0.25)
    old = P.parse_observation(_message(9, [_section("FRIC", fric)]))  # an engine without the section
    assert old.frcp is None and len(old.fric) == 2
    empty = P.parse_observation(_message(9, [_section("FRIC", struct.pack("<H", 0)),
                                              _section("FRCP", struct.pack("<H", 0))]))
    assert empty.frcp is not None and len(empty.frcp) == 0  # sent, no friction contact (airborne / impact)


def _tracker(items, send_frcp=True):
    """items: (partner, pressure, lt_point, lt_normal[, lt_valid])"""
    fric = np.zeros(len(items), P.FRIC_DT)
    for i, it in enumerate(items):
        fric[i]["partner"], fric[i]["lt_point"], fric[i]["lt_normal"] = it[0], it[2], it[3]
        fric[i]["lt_valid"] = it[4] if len(it) > 4 else 1
    frcp = np.array([(it[0], it[1]) for it in items], P.FRCP_DT) if send_frcp else None
    rails = {501}
    level = SimpleNamespace(in_group=lambda p, g: g == G_RAILS and int(p) in rails)
    return SimpleNamespace(obs=SimpleNamespace(frcp=frcp, fric=fric), level=level)


def _block(cfg, items, pos=(0.0, 10.0, 0.0), send_frcp=True, frame=None):
    fx = FeatureExtractor(cfg)
    sl = fx.slices()
    F = {"pressure": np.zeros(sl["pressure"].stop - sl["pressure"].start, np.float32),
         "pressure_agg": np.zeros(sl["pressure_agg"].stop - sl["pressure_agg"].start, np.float32)}
    right, fwd = frame or (np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))
    info = {"lateral_axis": np.array(right)}
    fx._pressure(_tracker(items, send_frcp), F, info, np.asarray(pos, float), right, fwd)
    return F["pressure"].reshape(PRESSURE_SLOTS, len(PRESSURE_SLOT_FIELDS)), F["pressure_agg"], info


def test_block_values_slots_normals_squash_and_support_force():
    cfg = FeatureConfig(**R10, pressure=True, pressure_scale=10.0)
    # ball centre at y 10: a floor contact below (lt normal pointing away from the ball, as the engine stores it:
    # flipped), a rail contact on the right side with more force, and a weak third contact that gets no slot
    items = [(777, 5.0, (0, 8.0, 0), (0, -1.0, 0)),
             (501, 30.0, (1.0, 10.0, 0), (-2.0, 0, 0)),
             (777, 1.0, (0, 10.0, -1.0), (0, 0, 1.0))]
    slots, agg, info = _block(cfg, items)
    # slot 0: the rail (largest force), normal towards the ball centre (-x), on the route's right axis -1
    np.testing.assert_allclose(slots[0], [1, 30 / 40, -1, 0, 0, -1, 1], atol=1e-6)
    np.testing.assert_allclose(slots[1], [1, 5 / 15, 0, 1, 0, 0, 0], atol=1e-6)  # floor: flipped to +y
    f = np.array([-30.0, 5.0, 1.0])  # sum of force x normal (the third normal points +z, towards the ball)
    m = np.linalg.norm(f)
    np.testing.assert_allclose(agg, [36 / 46, *(f / m * (m / (m + 10.0))), 3 / 4], atol=1e-6)
    assert len(info["pressure_contacts"]) == 3
    assert np.abs(slots).max() <= 1 and np.abs(agg).max() <= 1  # every entry in [-1, 1]
    # no friction contact (airborne / impact): the section is there with n = 0, the block is all zero
    slots, agg, _ = _block(cfg, [])
    assert not slots.any() and not agg.any()
    # a contact without an engine normal (lt_valid 0): no slot, no part in the direction; its force counts in the
    # total and the contact in the count (nothing substituted)
    slots, agg, info = _block(cfg, [(777, 10.0, (0, 0, 0), (0, 0, 0), 0)])
    assert not slots.any()
    np.testing.assert_allclose(agg, [0.5, 0, 0, 0, 0.25], atol=1e-6)
    assert info["pressure_no_normal"] == [(10.0, 777)]
    slots, agg, _ = _block(cfg, [(501, 30.0, (1.0, 10.0, 0), (-2.0, 0, 0)), (777, 10.0, (0, 0, 0), (0, -1, 0), 0)])
    np.testing.assert_allclose(slots[0], [1, 30 / 40, -1, 0, 0, -1, 1], atol=1e-6)
    assert not slots[1].any()
    np.testing.assert_allclose(agg, [40 / 50, -30 / 40, 0, 0, 0.5], atol=1e-6)  # direction from the rail only


def test_block_refuses_an_engine_without_the_section():
    with pytest.raises(RuntimeError, match="FRCP"):
        _block(FeatureConfig(**R10, pressure=True), [], send_frcp=False)


def test_slot_normals_in_the_directional_frame():
    cfg = FeatureConfig(**dict(R10, camera="orient_ref"), pressure=True)
    right, fwd = np.array([0.0, 0, -1.0]), np.array([1.0, 0, 0])
    slots, _, _ = _block(cfg, [(501, 4.0, (1.0, 10.0, 0), (1.0, 0, 0))], frame=(right, fwd))
    np.testing.assert_allclose(slots[0][2:5], to_frame([-1.0, 0, 0], right, fwd), atol=1e-6)  # flipped to the ball


def test_grown_policy_ignores_the_new_block():
    a, b = FeatureConfig(**R10), FeatureConfig(**R10, pressure=True)
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
    o64, n64 = old.double(), new.double()  # float64: the zero columns contribute nothing
    with torch.no_grad():
        d = (o64(torch.as_tensor(normalize(x0, rms.state_dict()), dtype=torch.float64))[0]
             - n64(torch.as_tensor(normalize(x1, rms1.state_dict()), dtype=torch.float64))[0])
    assert float(d.abs().max()) <= 1e-12
