"""ballrain.segsearch: gates / crossings, key-run edits, branch selection, end-state check (no engine), and an engine
round trip (the TAS through the tool gives the TAS) when the headless binary is present."""
import numpy as np
import pytest

from ballrain.segsearch.core import (COMBOS, RING, EndCheck, Gate, Node, crossing, crossings, from_runs, make_gates,
                                     mutate, random_holds, rotate, select, to_runs)


def line(n=101):
    t = np.arange(n, dtype=float)
    p = np.stack([t, np.zeros(n), np.zeros(n)], axis=1)
    return t, p


def test_crossing_interpolates_and_respects_radius():
    t, p = line()
    g = Gate(0, 0, np.array([10.25, 0, 0]), np.array([1.0, 0, 0]), 2.0)
    tc, i = crossing(g, t, p)
    assert tc == pytest.approx(10.25) and i == 11
    far = Gate(0, 0, np.array([10.25, 0, 5.0]), np.array([1.0, 0, 0]), 2.0)
    assert crossing(far, t, p) is None
    back = Gate(0, 0, np.array([10.25, 0, 0]), np.array([-1.0, 0, 0]), 2.0)  # passed against the normal
    assert crossing(back, t, p) is None


def test_crossings_in_order_and_stop_at_first_miss():
    t, p = line()
    gs = [Gate(i, 0, np.array([x, 0, 0.0]), np.array([1.0, 0, 0]), 1.0) for i, x in enumerate((5.5, 20.5, 40.5))]
    cr = crossings(gs, 0, t, p)
    assert sorted(cr) == [0, 1, 2] and cr[1][0] == pytest.approx(20.5)
    gs[1] = Gate(1, 0, np.array([20.5, 0, 9.0]), np.array([1.0, 0, 0]), 1.0)
    assert sorted(crossings(gs, 0, t, p)) == [0]
    # a gate between two consecutive samples right after the previous gate's crossing is still found
    gs2 = [Gate(0, 0, np.array([5.2, 0, 0.0]), np.array([1.0, 0, 0]), 1.0),
           Gate(1, 0, np.array([5.6, 0, 0.0]), np.array([1.0, 0, 0]), 1.0)]
    assert sorted(crossings(gs2, 0, t, p)) == [0, 1]


def test_make_gates_last_is_b_and_normals_follow_velocity():
    n = 1000
    frames = np.arange(100, 100 + n)
    pos = np.stack([np.arange(n) * 0.1, np.zeros(n), np.zeros(n)], axis=1)
    vel = np.tile([26.4, 0, 0], (n, 1))
    gs = make_gates(frames, pos, vel, 100, 900, 264, 5.0)
    assert [g.frame for g in gs] == [364, 628, 900]
    assert np.allclose(gs[0].normal, [1, 0, 0])
    vel[:] = 0  # at rest: direction of travel over the next frames
    assert np.allclose(make_gates(frames, pos, vel, 100, 900, 264, 5.0)[0].normal, [1, 0, 0])


def test_runs_roundtrip_and_rotate():
    keys = np.array([1, 1, 9, 9, 9, 0, 8], np.uint32)
    runs = to_runs(keys)
    assert runs == [[1, 2], [9, 3], [0, 1], [8, 1]]
    assert np.array_equal(from_runs(runs), keys)
    assert rotate(1, 1) == 9 and rotate(1, -1) == 5 and rotate(5, 1) == 1 and rotate(0, 3) == 0
    assert rotate(1 | 16, 2) == 8 | 16  # other bits kept
    assert set(COMBOS) == {0} | set(RING) and len(RING) == 8


def test_mutate_keeps_length_and_tail_outside_window():
    rng = np.random.default_rng(0)
    keys = np.repeat(np.array([1, 9, 8, 10, 2], np.uint32), 200)
    for _ in range(200):
        out = mutate(keys, rng, window=200, n_edits=3)
        assert len(out) == len(keys) and out.dtype == np.uint32
        assert set(np.unique(out)) <= set(COMBOS)
        # runs starting after the window are untouched, apart from the boundary shift of the run right after it
        assert np.array_equal(out[600:], keys[600:])
    assert any(not np.array_equal(mutate(keys, rng, 200, 2), keys) for _ in range(20))


def test_random_holds():
    rng = np.random.default_rng(1)
    h = random_holds(300, rng, bias=9)
    assert len(h) == 300 and set(np.unique(h)) <= set(COMBOS)
    runs = to_runs(h)
    assert all(n % 2 == 0 for _, n in runs[:-1])


def node(t, pos=(0, 0, 0), vel=(1, 0, 0)):
    return Node(0, 0, 0, float(t), np.array(pos, float), np.array(vel, float))


def test_select_earliest_dedup_and_speed():
    ns = [node(10), node(10.5), node(11, pos=(5, 0, 0)), node(12, pos=(9, 0, 0), vel=(9, 0, 0)), node(30, pos=(20, 0, 0))]
    s = select(ns, 2)
    assert [n.time for n in s] == [10, 11]  # 10.5 is a duplicate of 10
    s = select(ns, 2, k_speed=1, slack=5)
    assert [n.time for n in s] == [10, 11, 12]
    assert select([], 3) == []


def test_end_check():
    ref = {"pos": [0, 0, 0], "vel": [10, 0, 0], "type": 0, "counts": {"ExtraPoint": 2}}
    good = {"pos": [1, 0.1, 0], "vel": [10.2, 0, 1], "type": 0, "counts": {"ExtraPoint": 2}}
    assert EndCheck().check(ref, good)["ok"]
    bad = {"pos": [5, -1, 0], "vel": [5, 0, 5], "type": 1, "counts": {"ExtraPoint": 1}, "ball_off": True}
    f = EndCheck().check(ref, bad)["fails"]
    assert set(f) == {"speed", "heading", "position", "height", "ball_type", "pickup:ExtraPoint", "ball_off"}


def _engine_ready():
    import os
    from pathlib import Path
    from ballrain.segsearch.engine import default_engine
    e = default_engine()
    ds = Path(__file__).resolve().parents[1] / "runtime/datasets/tas-v1/HS01_4740_1_30_344/run/framelog.bin"
    return os.environ.get("BALLRAIN_SEG_ENGINE_TEST") == "1" and Path(e.binary).is_file() and ds.is_file(), e, ds


@pytest.mark.skipif(not _engine_ready()[0], reason="engine test: set BALLRAIN_SEG_ENGINE_TEST=1 (needs the headless binary + TAS dataset)")
def test_engine_tas_null_and_framing(tmp_path):
    """The TAS keys from frame 936 through the tool reproduce the record's positions exactly, and a non-TAS sequence
    gives bit-identical states in 1-frame blocks, 4-frame blocks and restored through the prefix."""
    from ballrain import tas
    from ballrain.segsearch.engine import rollout
    _, eng, ds = _engine_ready()
    frames = tas.load_raw(ds)
    st = np.load(ds.parents[1] / "starts.npz")
    a = 936
    d = float(frames["delta"][a])
    r = rollout(eng, tmp_path / "w", "hl-47", frames[:a], [], frames["keys"][a:a + 400], delta=d, block=1)
    i = np.searchsorted(st["frames_done"], a + r["t"])
    assert np.array_equal(r["pos"], st["pos"][i])
    rng = np.random.default_rng(3)
    keys = random_holds(240, rng)
    r1 = rollout(eng, tmp_path / "w", "hl-47", frames[:a], [], keys, delta=d, block=1)
    r4 = rollout(eng, tmp_path / "w", "hl-47", frames[:a], [], keys, delta=d, block=4)
    rp = rollout(eng, tmp_path / "w", "hl-47", frames[:a], keys[:120], keys[120:], delta=d, block=4)
    idx = {int(t): k for k, t in enumerate(r1["t"])}
    for r in (r4, rp):
        assert all(np.array_equal(r["pos"][k], r1["pos"][idx[int(t)]]) for k, t in enumerate(r["t"]))
        assert np.array_equal(r["keys"], keys)


def test_select_survivors_first():
    a = node(10)
    b = node(20, pos=(5, 0, 0))
    b.reach = 3  # its rollout crossed 3 more gates
    c = node(15, pos=(9, 0, 0))
    assert [n.time for n in select([a, b, c], 2, k_surv=1, ahead=2)] == [20, 10]
    assert [n.time for n in select([a, b, c], 2)] == [10, 15]
