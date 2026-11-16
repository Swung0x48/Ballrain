"""Reference paths (ballrain.path): airship position, TAS reference lines, continuity projection.
The dataset checks are skipped when runtime/datasets/tas-v1 is not built."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ballrain import dataset, instances, path as RP, protocol as P
from ballrain.path import PathFollower, ReferencePath, build_tas_path
from ballrain.state import LevelData

DS = instances.runtime_dir() / "datasets" / f"tas-v{dataset.DATASET_VERSION}"
PLATFORM = 77


def _m43(pos):
    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], pos], np.float32)


def _m44(pos):
    m = np.eye(4, dtype=np.float32)
    m[3, :3] = pos
    return m


def test_airship_position_from_physicalize_mesh_export_any_merge_order():
    V = np.array([[-2, 0, -1], [2, 0, -1], [2, 1, 1], [-2, 1, 1]], np.float32)
    F = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    # increment at frame 25: Physicalize record + mesh export at the level end
    inc = P.LevelMessage(25)
    inc.phys = [P.PhysRecord(34, 25, PLATFORM, False, False)]
    inc.meshes = [P.Mesh(PLATFORM, 900, 0, _m44((730, -13, -874)), V, F)]
    # the full message (StartLevel, frame 103): the platform parked at the origin, unphysicalized at 34,
    # and its mesh re-exported with the parked matrix
    full = P.LevelMessage(103)
    full.entities = [P.Entity(PLATFORM, "PE_Balloon_Platform", 41, 0, False, _m43((0, 0, 0)), 0, -1, b"")]
    full.phys = [P.PhysRecord(34, 25, PLATFORM, False, False), P.PhysRecord(88, 34, PLATFORM, True, False)]
    full.meshes = [P.Mesh(PLATFORM, 900, 0, _m44((0, 0, 0)), V, F)]
    for order in ((full, inc), (inc, full)):
        L = LevelData()
        for m in order:
            L.merge(m)
        assert L.entity_position("PE_Balloon_Platform").tolist() == [0, 0, 0]  # the StartLevel snapshot
        assert L.physical_position("PE_Balloon_Platform") == pytest.approx([730, -12.5, -874])


def _figure_eight(n=2000, r=40.0):
    t = np.linspace(0, 2 * math.pi, n)
    return np.stack([r * np.sin(t), 0 * t, r * np.sin(t) * np.cos(t)], 1)  # crosses itself at the origin


def test_follower_keeps_continuity_through_a_self_crossing():
    P8 = _figure_eight()
    path = ReferencePath(P8, np.ones(len(P8), int))
    f = PathFollower(path)
    s_prev = None
    for i in range(0, len(P8), 7):
        r = f.update(P8[i] + [0, 0.3, 0], 1)
        assert r["dist"] < 0.5
        if s_prev is not None:
            assert r["s"] >= s_prev - 1e-6 and not r["jump"]  # through the crossing without jumping
        s_prev = r["s"]
    assert s_prev == pytest.approx(path.length, abs=2)
    # a plain nearest-point search is ambiguous at the crossing; far away from the window the follower
    # re-acquires (flagged)
    f.reset()
    f.update(P8[10], 1)
    r = f.update(P8[1500] + [0, 0.1, 0], 1)
    assert r["jump"] and r["mode"] == "reacquire" and r["s"] == pytest.approx(path.cum[1500], abs=1)


def test_build_tas_path_spacing_smoothing_and_sectors():
    rng = np.random.default_rng(0)
    n = 3000
    x = np.linspace(0, 300, n)
    pos = np.stack([x, 0.05 * rng.standard_normal(n), np.zeros(n)], 1)
    pos[100:110] = np.nan  # a transformation without an IVP ball is bridged
    sector = np.where(x < 150, 1, 2)
    p = build_tas_path(pos, sector, np.full(n, 1000 / 264), spacing=0.5, meta={"record": "X"})
    assert p.source == "tas" and p.meta["spacing"] == 0.5 and p.meta["record"] == "X"
    assert np.allclose(np.diff(p.cum)[:-1], 0.5, atol=1e-6) and p.length == pytest.approx(300, abs=1)
    assert p.n_sectors == 2 and np.all(np.diff(p.point_sector) >= 0)
    assert p.sector_range[1][1] == pytest.approx(150, abs=1)
    assert p.points_ahead(10, 3, 5.0)[:, 0] == pytest.approx([15, 20, 25], abs=0.2)


def test_anchor_check_rejects_other_levels():
    meta = {"start": [0, 2, 0], "checkpoints": [[0, 2, 20]]}
    ok = {"start": np.array([0, 2, 0.5]), "checkpoints": [np.array([0, 2, 20])], "airship": None}
    assert RP.anchors_match(meta, ok)
    assert not RP.anchors_match(meta, {**ok, "start": np.array([0, 2, 5])})
    assert not RP.anchors_match(meta, {**ok, "checkpoints": []})


def test_committed_l01_asset():
    p = RP.asset_path(1)
    assert p.is_file()
    path = ReferencePath.load(p)
    m = path.meta
    assert m["record"] == "SR01_1_14_457_264" and m["spacing"] == 0.5 and len(m["record_sha256"]) == 64
    assert path.n_sectors == 4 and 1900 < path.length < 2200 and len(m["checkpoints"]) == 3
    text = json.dumps(m)
    assert ":\\" not in text and "/Users/" not in text and "runtime" not in text  # no machine paths


# ------------------------------------------------------------------------------------- dataset

def _index():
    f = DS / "index.json"
    if not f.is_file():
        pytest.skip("TAS dataset not built")
    return json.loads(f.read_text(encoding="utf-8"))


def _levels(index):
    out = {}
    for m in index:
        out.setdefault(m["level"], []).append(m)
    return out


def test_airship_node_on_every_level_message():
    """Root cause of the old (0,0,0) airship node: the StartLevel ENTS snapshot has the balloon parked at the
    origin; the platform's Physicalize mesh export holds its level-end position (within 3 units of the ball
    at LevelFinish on every completing record)."""
    by = _levels(_index())
    assert set(by) == set(range(1, 14))
    for level, ms in sorted(by.items()):
        for m in ms:
            d = DS / Path(m["record"]).stem
            L = LevelData()
            L.merge(P.parse_level_message((d / "level.msg").read_bytes())[0])
            for lm in P.parse_level_stream((d / "level.msg.inc").read_bytes())[0]:
                L.merge(lm)
            assert np.linalg.norm(L.entity_position("PE_Balloon_Platform")) < 1e-3
            a = L.physical_position("PE_Balloon_Platform")
            assert np.linalg.norm(a) > 50, (m["record"], a)
            lf = m.get("levelfinish_frame")
            if lf:
                z = np.load(d / "raw.npz")
                ball = z["ball_pos"][int(np.searchsorted(z["frames_done"], lf))]
                assert np.linalg.norm(a - ball) < 3.0, (m["record"], a, ball)
                assert np.linalg.norm((a - ball)[[0, 2]]) < 1.6


def test_tas_assets_match_their_levels_and_reference_lateral():
    index = _index()
    by = _levels(index)
    for level in range(1, 14):
        p = RP.asset_path(level)
        completing = [m for m in by.get(level, []) if m.get("levelfinish_frame")]
        if not completing:
            assert not p.exists()  # levels 9 and 13: no completing record, polyline fallback
            continue
        path = ReferencePath.load(p)
        rec = path.meta["record"]
        d = DS / rec
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        assert meta["record_sha256"] == path.meta["record_sha256"]
        if level != 1:
            continue
        tr = dataset.record_trajectory(d, level)
        assert RP.anchors_match(path.meta, tr["anchors"])
        f = PathFollower(path)
        lat, s = [], []
        for i in np.flatnonzero(tr["control"]):
            r = f.update(tr["pos"][i], int(tr["sector"][i]))
            lat.append(r["dist"])
            s.append(r["s"])
        assert max(lat) < 1.0 and np.min(np.diff(s)) > -0.5 and s[-1] > path.length - 2


def test_reference_speed_and_lookahead_points():
    import numpy as np
    from ballrain.features import FeatureConfig, FeatureExtractor
    from ballrain.path import ReferencePath, build_tas_path
    # a straight line at 10 units/s for 2 s, then 20 units/s for 1 s (frames of 10 ms)
    t = np.arange(301) * 10.0
    x = np.where(t <= 2000, t / 100, 20 + (t - 2000) / 50)
    pos = np.stack([x, np.zeros_like(x), np.zeros_like(x)], 1)
    p = build_tas_path(pos, np.ones(len(x), np.int64), np.full(len(x), 10.0), time_sigma_ms=0)
    assert abs(p.speed_at(10.0) - 10.0) < 0.2 and abs(p.speed_at(30.0) - 20.0) < 0.3
    assert np.allclose(p.points_at(5.0, [10, 100])[:, 0], [15.0, p.length])
    assert ReferencePath(p.P).speed_at(1.0) is None  # no reference times
    base = FeatureExtractor(FeatureConfig(hs_block=False)).size
    f6 = FeatureExtractor(FeatureConfig(hs_block=False, far_waypoints=(20, 30, 45), ref_speed_at=(0, 15, 30)))
    assert f6.size == base + 12 and [n for n, _ in f6.layout()][-2:] == ["far_waypoints", "ref_speed"]
