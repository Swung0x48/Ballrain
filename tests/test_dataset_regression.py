"""Regressions on recorded TAS dataset records (runtime/datasets/tas-v1, built by scripts/m1_gates.py --gates D).
Skipped when the dataset is not there. The tracker is fed the stored observations in order, with the full
level message and all increments merged first (so every physicalize record resolves)."""
from pathlib import Path

import numpy as np
import pytest

from ballrain import dataset, features as FX, instances, protocol as P
from ballrain.state import GameTracker

ROOT = instances.runtime_dir() / "datasets" / f"tas-v{dataset.DATASET_VERSION}"


def _replay(name):
    d = ROOT / name
    if not (d / "raw.npz").is_file():
        pytest.skip(f"dataset record {name} not built")
    t = GameTracker(dataset.level_of(name))
    lm, _ = P.parse_level_message((d / "level.msg").read_bytes())
    t.merge_level(lm)
    if (d / "level.msg.inc").is_file():
        for m in P.parse_level_stream((d / "level.msg.inc").read_bytes())[0]:
            t.merge_level(m)
    for o in dataset.load_observations(d / "raw.npz"):
        t.update(o)
        yield t, o


def _windows(pairs):
    out, start = [], None
    for f, v in pairs:
        if v and start is None:
            start = f
        elif not v and start is not None:
            out.append((start, f))
            start = None
    if start is not None:
        out.append((start, None))
    return out


def test_hs03_transforming_only_between_ball_unphysicalize_and_physicalize():
    fx = FX.FeatureExtractor(FX.FeatureConfig(height_grid=0, fan_dirs=0, mesh_edge=False, bodies=0))
    tr, valid_off, paused = [], [], []
    for t, o in _replay("HS03_4906_1_56_934"):
        tr.append((o.frames_done, t.transforming))
        paused.append((o.frames_done, t.flags.paused))
        if t.flags.paused or t.flags.finished:
            vec, _ = fx.compute(t)
            if vec[fx.slices()["valid"]][0] != 1 or vec[fx.slices()["transforming"]][0] != 0:
                valid_off.append(o.frames_done)
    # the four transformations of HS03 (agent-protocol.md 8.7: the IVP ball is missing in these windows)
    assert _windows(tr) == [(5201, 5800), (9740, 10339), (17738, 18337), (27590, 28189)]
    # f2 bug: "transforming" in every pause and from LevelFinish on (Activetrafo stays set)
    assert _windows(paused) == [(8590, 8719), (14788, 14983), (19972, 20109), (23330, 23461)]
    assert valid_off == []


def test_sr09_deaths_are_not_transformations():
    tr = [(o.frames_done, t.transforming) for t, o in _replay("SR09_1_29_321_264")]
    assert _windows(tr) == [(2817, 3476)]  # one trafo; the four deaths (BallOff, then unphysicalize) are not


def test_followers_at_levelfinish_explain_the_name_score():
    """HS01_4738: 12 small balls still following at LevelFinish; the official readout is the name score."""
    last = None
    for t, o in _replay("HS01_4738_1_31_317"):
        last = t
    assert last.followers_at_finish == 12 and last.hs_source == "tail_end"
    assert last.hs_at_finish == 4738
    assert last.follower_state()["total"] == 12
