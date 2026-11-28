"""EnvConfig.collect_points (BallanceEnv._collect_point_reached) and ballrain.collect_points (points from a record's
pauses): collect once per sector from the sector's point on; without collect_points always (as before)."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from ballrain.env import BallanceEnv

REC = Path(__file__).resolve().parents[1] / "runtime/datasets/tas-v1/HS01_4740_1_30_344"


def _env(points, sector, s, done=()):
    return SimpleNamespace(cfg=SimpleNamespace(collect_points=points),
                           session=SimpleNamespace(tracker=SimpleNamespace(sector=sector)),
                           reward=SimpleNamespace(s=s), _collected_sectors=set(done))


def test_without_points_always():
    assert BallanceEnv._collect_point_reached(_env(None, 1, None))


def test_point_reached_once_per_sector():
    P = [[1, 785.9], [2, 1371.8]]
    assert not BallanceEnv._collect_point_reached(_env(P, 1, 700.0))
    assert BallanceEnv._collect_point_reached(_env(P, 1, 786.0))
    assert not BallanceEnv._collect_point_reached(_env(P, 1, 790.0, done={1}))
    assert not BallanceEnv._collect_point_reached(_env(P, 4, 2300.0))  # no point in the last sector
    assert not BallanceEnv._collect_point_reached(_env(P, 2, None))


@pytest.mark.skipif(not (REC / "events.txt").is_file(), reason="L1 TAS dataset not installed")
def test_l1_record_points():
    from ballrain import collect_points as CP
    from ballrain.curriculum import TasStarts
    from ballrain.path import ReferencePath, asset_path, resolve_path_dir
    st = TasStarts(str(REC), camera_tol=None)
    assert CP.pause_frames(st) == [8921, 14821, 18243]
    pts = CP.points(st, ReferencePath.load(asset_path(1, resolve_path_dir("benchmarks/refpaths-hs"))))
    assert [p[0] for p in pts] == [1, 2, 3]
    assert 780 < pts[0][1] < 792 and 1365 < pts[1][1] < 1373 and 1715 < pts[2][1] < 1743
