"""EnvConfig.own_camera / ballrain.own_camera: the agent's own (spawn) camera in the camera inputs."""
from pathlib import Path

import numpy as np
import pytest

from ballrain import own_camera as OC

SL = {"cam_yaw_ref": slice(14, 16), "cam_yaw": slice(16, 18), "cam_rotating": slice(18, 19), "path_progress": slice(0, 1)}
REC = Path(__file__).resolve().parents[1] / "runtime/datasets/tas-v1/HS01_4740_1_30_344"


def test_quarter_vec_matches_features_convention():
    # features.compute: (cos yaw, sin yaw); the L1 spawn camera shows (-1, 0), the TAS camera after frame 7052 (0, 1)
    assert OC.quarter_vec(2) == (-1, 0)
    assert OC.quarter_vec(1) == (0, 1)
    assert OC.quarter_vec(0) == (1, 0)
    assert OC.quarter_vec(3) == (0, -1)


def test_write_sets_only_camera_inputs():
    x = np.arange(40, dtype=np.float32).reshape(2, 20)
    y = OC.write(x.copy(), SL, 2)
    assert (y[:, 14:18] == np.array([-1, 0, -1, 0], np.float32)).all() and (y[:, 18] == 0).all()
    keep = [i for i in range(20) if not 14 <= i <= 18]
    assert (y[:, keep] == x[:, keep]).all()


def test_view_by_route_s():
    raw = np.zeros((4, 20), np.float32)
    raw[:, 0] = np.array([10, 99, 100, 300]) / 1000.0  # route s / length 1000
    v = OC.view(raw, SL, 0, 1000.0, [(0.0, 2), (100.0, 1)])
    assert (v[:2, 14:18] == np.array([-1, 0, -1, 0])).all()
    assert (v[2:, 14:18] == np.array([0, 1, 0, 1])).all()
    assert (raw[:, 14:18] == 0).all()  # a copy


@pytest.mark.skipif(not (REC / "raw.npz").is_file(), reason="L1 TAS dataset not installed")
def test_l1_record_spawn_and_turn():
    from ballrain.curriculum import TasStarts
    from ballrain.path import ReferencePath, asset_path, resolve_path_dir
    st = TasStarts(str(REC), camera_tol=None)
    assert OC.spawn_quarter(st) == 2
    line = ReferencePath.load(asset_path(1, resolve_path_dir("benchmarks/refpaths-hs")))
    pts = OC.turn_points(st, line)
    assert pts[0] == (0.0, 2) and len(pts) == 2 and pts[1][1] == 1
    j = int(np.argmin(np.abs(line.cum - pts[1][0])))
    assert np.linalg.norm(line.P[j, [0, 2]] - np.array([-176.7, -368.7])) < 10  # the TAS ball at frame 7052
