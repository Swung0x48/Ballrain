"""The f7 reach block from several threads of one process (the pool verifier's replay threads): each thread queries
its own view of the cached FastReach (reach_features._thread_view), sharing the read-only field arrays, so concurrent
perceive calls give exactly the single-threaded values. Before this, the threads shared one instance's scratch
arrays and the nogil kernels overwrote each other's results (the live H 255 run, 2026-11-18)."""
import threading

import numpy as np
import pytest

from ballrain import fastreach as FR
from ballrain import reach_features as RF
from ballrain.instances import runtime_dir

FIELDS = runtime_dir() / "fastreach" / "L02-86886f7d56fff7a5.npz"
pytestmark = [pytest.mark.skipif(not FR.HAVE_NUMBA, reason="needs numba"),
              pytest.mark.skipif(not FIELDS.is_file(), reason="the L2 fields are not in this runtime (BALLRAIN_RUNTIME)")]


def states(n=400, seed=3):
    """Ball states near the level's rails and floors: rail segment points lifted a little, mixed velocities."""
    fr = FR.FastReach.load(FIELDS)
    rng = np.random.default_rng(seed)
    seg = fr.seg[rng.integers(len(fr.seg), size=n)]
    t = rng.uniform(size=(n, 1))
    pos = seg[:, :3] * (1 - t) + seg[:, 3:6] * t + np.c_[rng.normal(0, 1.5, n), rng.uniform(0.5, 4, n), rng.normal(0, 1.5, n)]
    vel = rng.normal(0, 12, size=(n, 3))
    return pos, vel, rng.integers(0, 3, n), rng.integers(0, 4, n)


def run(fr, pos, vel, ball, quarter):
    return np.stack([fr.perceive(pos[i], vel[i], int(ball[i]), int(quarter[i]))[0] for i in range(len(pos))])


def test_views_share_the_fields_and_not_the_scratch():
    base = FR.FastReach.load(FIELDS)
    v = RF._thread_view(base)
    assert RF._thread_view(base) is v  # reused within a thread
    assert v.tiles is base.tiles and v.Hh is base.Hh and v.bricks is base.bricks  # no copy of the fields
    for k in RF._SCRATCH:
        assert getattr(v, k) is not getattr(base, k)
    other = []
    t = threading.Thread(target=lambda: other.append(RF._thread_view(base)))
    t.start()
    t.join()
    assert other[0] is not v and other[0].tiles is base.tiles


def test_concurrent_perceive_equals_single_threaded():
    pos, vel, ball, quarter = states()
    base = FR.FastReach.load(FIELDS)
    want = run(FR.FastReach.load(FIELDS), pos, vel, ball, quarter)  # one instance, one thread: today's values
    assert np.array_equal(run(RF._thread_view(base), pos, vel, ball, quarter), want)  # a view: bit-identical
    got, errs = {}, []

    def worker(i):
        try:
            order = np.random.default_rng(i).permutation(len(pos))  # each thread its own order: maximal interleaving
            v = RF._thread_view(base)
            out = np.empty_like(want)
            for _ in range(2):
                for j in order:
                    out[j] = v.perceive(pos[j], vel[j], int(ball[j]), int(quarter[j]))[0]
            got[i] = out
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs
    for i in range(4):
        assert np.array_equal(got[i], want), f"thread {i}: {int((got[i] != want).any(1).sum())} states differ"
