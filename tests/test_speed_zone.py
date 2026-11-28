"""Speed zones (a box zone with "speed_below"): counted as entered only while the ball is slower than v inside the box,
for min_steps consecutive steps, and only in a life that was outside the box before (Worker._speed_zone)."""
from types import SimpleNamespace

import numpy as np

from ballrain.train.worker import Worker


def _w(delta_ms=1000 / 264):
    return SimpleNamespace(spec=SimpleNamespace(env=SimpleNamespace(delta_ms=delta_ms)))


def _ep():
    return SimpleNamespace(zone_speed={})


Z = {"box": [0, 100, -10, 10], "y_below": 50, "speed_below": 30.0, "min_steps": 3}


def _run(xs, in_box, frames=2):
    w, ep, out = _w(), _ep(), []
    for x, ib in zip(xs, in_box):
        out.append(Worker._speed_zone(w, ep, "slope", Z, np.array([x, 0.0, 0.0]), ib, frames))
    return out


def test_fast_through_the_box_never_fires():
    dx = 45.0 * 2 / 264  # 45 u/s
    xs = [i * dx for i in range(40)]
    assert not any(_run(xs, [i >= 5 for i in range(40)]))


def test_slow_inside_after_entering_fires_after_min_steps():
    dx_fast, dx_slow = 45.0 * 2 / 264, 10.0 * 2 / 264
    xs, x = [], 0.0
    for i in range(20):
        x += dx_fast if i < 10 else dx_slow
        xs.append(x)
    res = _run(xs, [i >= 5 for i in range(20)])
    first = res.index(True)
    assert first == 12  # slow from step 10: steps 10, 11, 12 -> fires on the third
    assert not any(res[:12])


def test_life_started_inside_is_never_judged():
    dx_slow = 5.0 * 2 / 264
    xs = [i * dx_slow for i in range(30)]
    assert not any(_run(xs, [True] * 30))


def test_one_slow_step_does_not_fire():
    dx_fast, dx_slow = 45.0 * 2 / 264, 10.0 * 2 / 264
    xs, x = [], 0.0
    for i in range(30):
        x += dx_slow if i in (12, 20, 21) else dx_fast
        xs.append(x)
    assert not any(_run(xs, [i >= 5 for i in range(30)]))
