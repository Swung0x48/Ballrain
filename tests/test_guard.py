"""ReachGuard (scripts/train_m2.py --auto-rollback): bins by policy version, trigger after consecutive bad bins,
target = newest earlier good bin, no re-trigger before new bins after a rollback."""
from ballrain.train.guard import ReachGuard


def fill(g, b, n, k, frame=4150):
    for i in range(n):
        g.add("tas", frame, b * g.bin + i % g.bin, i < k)


def test_trigger_and_target():
    g = ReachGuard()
    fill(g, 480, 30, 24)  # 0.8 good
    fill(g, 481, 30, 21)  # 0.7 good
    fill(g, 482, 30, 18)  # 0.6 neither
    fill(g, 483, 30, 12)  # 0.4 bad
    fill(g, 484, 30, 9)   # 0.3 bad
    assert not g.triggered(48450)  # bin 484 not complete (end 48500 + settle 50)
    assert g.triggered(48550)
    assert g.targets(48550) == [48100, 48000]
    g.rolled_back(48550)
    assert not g.triggered(48600) and g.targets(48600) == []


def test_needs_consecutive_bins_and_min_n():
    g = ReachGuard()
    fill(g, 480, 30, 24)
    fill(g, 481, 30, 9)
    fill(g, 482, 10, 0)  # too few episodes: not counted
    fill(g, 483, 30, 25)
    assert not g.triggered(48500)
    g2 = ReachGuard()
    fill(g2, 480, 30, 24)
    fill(g2, 481, 30, 9)
    fill(g2, 483, 30, 9)  # bad, but not adjacent to 481
    assert not g2.triggered(48500)


def test_filters_kind_and_range():
    g = ReachGuard()
    g.add("tas_rest", 4150, 48010, False)
    g.add("tas", 4500, 48010, False)
    g.add("respawn", 4150, 48010, False)
    g.add("tas", 4150, None, False)
    assert dict(g.stats) == {}


def test_no_good_bin_no_target():
    g = ReachGuard()
    fill(g, 480, 30, 15)
    fill(g, 481, 30, 9)
    fill(g, 482, 30, 9)
    assert g.triggered(48400) and g.targets(48400) == []


def test_from_config_rejects_unknown_keys():
    g = ReachGuard.from_config({"lo": 4100, "hi": 4299, "bin": 100})
    assert g.bin == 100
    try:
        ReachGuard.from_config({"lowx": 1})
    except ValueError:
        return
    raise AssertionError("unknown key accepted")
