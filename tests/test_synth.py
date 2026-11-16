import numpy as np
import pytest

from ballrain import synth, tas


def base(n=500, delta=1000 / 132):
    keys = np.zeros(n, dtype=np.uint32)
    keys[140] = tas.KEY_Q
    keys[300:] = tas.KEY_UP | tas.KEY_SHIFT
    return tas.make_frames(np.full(n, delta), keys)


def test_off_route_deterministic_and_shape():
    f = base()
    a = synth.off_route(f, 320, 1000, seed=7)
    b = synth.off_route(f, 320, 1000, seed=7)
    c = synth.off_route(f, 320, 1000, seed=8)
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != c.tobytes()
    assert len(a) == 1320
    assert a[:320].tobytes() == f[:320].tobytes()
    assert np.all((a["keys"][320:] & ~np.uint32(tas.ARROW_MASK)) == 0)
    assert tas.constant_delta(a) is not None
    assert tas.delta_bits(a)[0] == tas.delta_bits(f)[0]


def test_hold_lengths():
    k = synth.random_hold_keys(5000, seed=1, hold_min=5, hold_max=60)
    change = np.flatnonzero(np.diff(k.astype(np.int64))) + 1
    runs = np.diff(np.concatenate([[0], change, [len(k)]]))
    # consecutive holds may draw the same combo and merge, so only the minimum is strict (except the last run)
    assert runs[:-1].min() >= 5
    assert set(np.unique(k)) <= set(synth.ARROW_COMBOS)


def test_off_route_guards():
    f = base()
    with pytest.raises(ValueError):
        synth.off_route(f, 501, 10, seed=0)
    with pytest.raises(ValueError):
        synth.off_route(f, 0, 10, seed=0)
    g = f.copy()
    g["delta"][3] = 1.0
    with pytest.raises(ValueError):
        synth.off_route(g, 10, 10, seed=0)


def test_idle_then_random():
    a = synth.idle_then_random(800, seed=3)
    assert a.tobytes() == synth.idle_then_random(800, seed=3).tobytes()
    assert len(a) == 940
    assert np.all(a["keys"][:140] == 0)
    assert tas.delta_bits(a)[0] == np.float32(1000 / 132).view("<u4")
    assert tas.constant_delta(a) is not None


def test_write_outputs(tmp_path):
    a = synth.idle_then_random(100, seed=0)
    t, r = synth.write_outputs(a, tmp_path / "sub" / "case")
    assert t.name == "case.tas" and r.name == "case.bin"
    assert tas.load(t).tobytes() == a.tobytes() == r.read_bytes()


def test_off_route_from_real_record(tas_paths):
    f = tas.load(tas_paths[0])
    a = synth.off_route(f, 200, 300, seed=42)
    b = synth.off_route(tas.decode(tas.encode(f)), 200, 300, seed=42)
    assert a.tobytes() == b.tobytes()


def test_idle_then_random_q_frames():
    a = synth.idle_then_random(100, seed=3, idle=300, q_frames=(140,))
    assert a["keys"][140] == tas.KEY_Q
    assert np.all(np.delete(a["keys"][:300], 140) == 0)
    assert a["keys"][300:].tobytes() == synth.idle_then_random(100, seed=3, idle=300)["keys"][300:].tobytes()
    with pytest.raises(ValueError):
        synth.idle_then_random(10, seed=0, idle=100, q_frames=(100,))
