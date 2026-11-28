"""Contact penalty zones (WorkerSpec.zones entries with "rail_touches"): fired from the observation's rail-touch count
(contact_agg n_rail_touch) inside a route-s window, once per life; box zones unchanged."""
from types import SimpleNamespace

import numpy as np

from ballrain.features import CONTACT_AGG_FIELDS, FeatureConfig, layout_slices
from ballrain.train.worker import Worker, WorkerSpec

CFG = FeatureConfig(version="f8")
IDX = layout_slices(CFG)["contact_agg"].start + CONTACT_AGG_FIELDS.index("n_rail_touch")
ZONE = {"double_rail": {"rail_touches": 2, "s": [200.0, 245.0], "penalty": 5.0}}


def _worker(zones):
    w = object.__new__(Worker)
    w.spec = WorkerSpec(wid=0, copy="x", env=None, seed=0, zones=zones)
    ball = {"found": True, "pos": [0.0, 0.0, 0.0]}
    w.env = SimpleNamespace(reward=SimpleNamespace(s=None), fx=SimpleNamespace(cfg=CFG),
                            session=SimpleNamespace(tracker=SimpleNamespace(obs=SimpleNamespace(ball=ball, mind=[]))))
    w._fx_raw = np.zeros(IDX + 10, np.float32)
    w.fx_info = {}
    return w


def _ep():
    return SimpleNamespace(zone_in={}, zone_counts={}, terms={})


def _at(w, s, rails):
    w.env.reward.s = s
    w._fx_raw[IDX] = min(rails, 2) / 2


def test_two_rail_touches_in_window_once_per_life():
    w, ep = _worker(ZONE), _ep()
    _at(w, 210.0, 1)
    assert w._zone_wall(ep, 0, 209.0, 2) == 0.0  # one rail (the single rail / TAS's inner ring)
    _at(w, 199.0, 2)
    assert w._zone_wall(ep, 0, 198.0, 2) == 0.0  # before the window
    _at(w, 212.0, 2)
    assert w._zone_wall(ep, 0, 211.0, 2) == -5.0  # double rail inside the window
    assert ep.zone_counts == {"double_rail": 1} and ep.terms["zone_penalty"] == -5.0
    _at(w, 213.0, 1)
    w._zone_wall(ep, 0, 212.0, 2)
    _at(w, 214.0, 2)
    assert w._zone_wall(ep, 0, 213.0, 2) == 0.0  # contacts flicker: no re-arm within the life
    ep2 = _ep()  # a new life
    assert w._zone_wall(ep2, 0, 213.0, 2) == -5.0
    _at(w, 250.0, 2)
    assert w._zone_wall(_ep(), 0, 249.0, 2) == 0.0  # past the window


def test_box_zones_unchanged_beside_a_contact_zone():
    box = {"pit": {"box": [-1.0, 1.0, -1.0, 1.0], "y_below": 1.0, "penalty": 1.5}}
    w, ep = _worker({**box, **ZONE}), _ep()
    _at(w, 100.0, 2)  # outside the contact window
    assert w._zone_wall(ep, 0, 99.0, 2) == -1.5
    assert ep.zone_counts == {"pit": 1}


def test_terminal_key_marks_the_life():
    """"terminal": true: the zone that fires names itself on the episode (the worker ends the life that step);
    without the key nothing is marked (the life goes on as before)."""
    w, ep = _worker(ZONE), _ep()
    _at(w, 212.0, 2)
    w._zone_wall(ep, 0, 211.0, 2)
    assert getattr(ep, "zone_terminal", None) is None
    w = _worker({"double_rail": {**ZONE["double_rail"], "terminal": True}})
    ep = _ep()
    _at(w, 210.0, 1)
    w._zone_wall(ep, 0, 209.0, 2)
    assert getattr(ep, "zone_terminal", None) is None  # not fired yet
    _at(w, 212.0, 2)
    assert w._zone_wall(ep, 0, 211.0, 2) == -5.0  # the penalty still applies
    assert ep.zone_terminal == "double_rail"
    box = {"pit": {"box": [-1.0, 1.0, -1.0, 1.0], "y_below": 1.0, "penalty": 1.5, "terminal": True}}
    w, ep = _worker(box), _ep()
    _at(w, 100.0, 0)
    assert w._zone_wall(ep, 0, 99.0, 2) == -1.5 and ep.zone_terminal == "pit"


def test_min_steps_needs_consecutive_contact():
    """"min_steps": n: only n consecutive steps with the rail-touch count reached fire the zone (a short brush of the
    other rail while hopping between rails does not); the count resets on a step without it. Without the key the
    first such step fires (test_two_rail_touches_in_window_once_per_life)."""
    w = _worker({"double_rail": {**ZONE["double_rail"], "min_steps": 3, "terminal": True}})
    ep = _ep()
    seq = [2, 2, 1, 2, 2, 1, 2, 2]  # two brushes of 2 steps: never 3 in a row
    for i, n in enumerate(seq):
        _at(w, 210.0 + i, n)
        assert w._zone_wall(ep, 0, 209.0 + i, 2) == 0.0
    assert getattr(ep, "zone_terminal", None) is None and ep.zone_counts == {}
    _at(w, 220.0, 2)
    assert w._zone_wall(ep, 0, 219.0, 2) == -5.0  # the third step in a row
    assert ep.zone_terminal == "double_rail" and ep.zone_counts == {"double_rail": 1}
    ep2 = _ep()  # a new life starts counting from zero
    _at(w, 221.0, 2)
    assert w._zone_wall(ep2, 0, 220.0, 2) == 0.0
    _at(w, 250.0, 2)  # outside the window: the count resets
    w._zone_wall(ep2, 0, 249.0, 2)
    _at(w, 222.0, 2)
    assert w._zone_wall(ep2, 0, 221.0, 2) == 0.0
