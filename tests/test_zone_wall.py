"""Penalty zones and the unclimbable-wall push (ballrain.train.worker WorkerSpec.zones / wall_push_rate): synthetic
contact cases, and the TAS record SR01_1_14_457_264 (skipped without the dataset): neither the L1 pit zone nor the
wall push fires along the TAS line, while a ball on the pit floor pushing into the pit wall is caught."""
import sys

import numpy as np
import pytest

from ballrain import dataset, geometry as G, instances, tas
from ballrain.train.worker import WorkerSpec, in_zone, push_direction, unclimbable_walls, wall_push_hit

PIT = {"box": [130.0, 169.8, -581.0, -541.0], "y_below": -5.0, "penalty": 1.5}
REC = instances.runtime_dir() / "datasets" / f"tas-v{dataset.DATASET_VERSION}" / "SR01_1_14_457_264"


def _quad(x0, x1, z0, z1, y):
    a, b, c, d = [x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]
    return [a, a], [b, c], [c, d]


def _floor_and_block(top):
    """Floor y = 0 over x -20..20, a block (top y = top) over x 10..20 with its vertical face at x = 10."""
    A, B, C = [], [], []
    for (x0, x1, y) in ((-20, 10, 0.0), (10, 20, top)):
        a, b, c = _quad(x0, x1, -20, 20, y)
        A += a; B += b; C += c
    # the vertical face (not needed for the probe, which looks at the block's top)
    A += [[10, 0, -20], [10, 0, -20]]; B += [[10, top, -20], [10, top, 20]]; C += [[10, top, 20], [10, 0, 20]]
    return G.TriangleSet(A, B, C, grid_cell=None)


def test_zone_box_height_and_rearm_margin():
    assert in_zone([150, -7.07, -560], PIT)  # on the pit floor (ball centre ~-7.07)
    assert not in_zone([150, -2.7, -583], PIT)  # south rim: outside the box
    assert not in_zone([147.2, -2.07, -570], PIT)  # centre strip (y -4.14 + r): above y_below
    assert not in_zone([128, -2.7, -560], PIT)  # west rim
    assert not in_zone([150, -4.5, -560], PIT) and in_zone([150, -4.5, -560], PIT, margin=1.0)
    assert WorkerSpec(wid=0, copy="x", env=None, seed=0).zones == {}
    assert WorkerSpec(wid=0, copy="x", env=None, seed=0).wall_push_rate == 0.0


def test_push_direction_world_and_camera():
    assert np.allclose(push_direction(tas.KEY_UP, True), [0, 0, 1])
    assert np.allclose(push_direction(tas.KEY_RIGHT | tas.KEY_DOWN, True), np.array([1, 0, -1]) / np.sqrt(2))
    assert not np.any(push_direction(0, True))
    cam = np.zeros((4, 3))
    cam[0] = [0, 0, -1]  # camera x row -> world -z
    cam[2] = [1, 0, 0]  # camera z row (forward) -> world +x
    assert np.allclose(push_direction(tas.KEY_UP, False, cam), [1, 0, 0])
    assert np.allclose(push_direction(tas.KEY_RIGHT, False, cam), [0, 0, -1])
    assert not np.any(push_direction("esc", True))


@pytest.mark.parametrize("top,expect", [(4.4, True), (2.5, True), (1.5, False), (0.5, False)])
def test_unclimbable_wall_by_height(top, expect):
    ts = _floor_and_block(top)
    pos = np.array([8.0, 2.0, 0.0])  # radius 2 on the floor, touching the face at x = 10
    n = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # the wall, the floor (floor normals are never walls)
    walls = unclimbable_walls(pos, 2.0, n, [0.0, 0.0], ts.ray_down)
    assert bool(walls) == expect
    if expect:
        assert np.allclose(walls[0], [-1, 0, 0])
        assert wall_push_hit(np.array([1.0, 0, 0]), walls)  # pushing into the wall
        assert wall_push_hit(np.array([1.0, 0, 1.0]) / np.sqrt(2), walls)  # 45 degrees: still into it
        assert not wall_push_hit(np.array([0, 0, 1.0]), walls)  # along the wall
        assert not wall_push_hit(np.array([-1.0, 0, 0]), walls)  # away from it
        assert not wall_push_hit(np.zeros(3), walls)  # no keys


def test_wall_needs_contact_and_a_side_normal():
    ts = _floor_and_block(4.4)
    pos = np.array([8.0, 2.0, 0.0])
    assert not unclimbable_walls(pos, 2.0, [[-1.0, 0, 0]], [0.5], ts.ray_down)  # gap 0.5: not touching
    assert not unclimbable_walls(pos, 2.0, [[-0.6, 0.8, 0]], [0.0], ts.ray_down)  # a slope (|n_y| 0.8)


def _tas_frames():
    if not (REC / "raw.npz").is_file():
        pytest.skip("dataset record SR01_1_14_457_264 not built")
    return dataset.OfflineReplay(REC, 1)


def test_tas_line_never_in_the_pit_zone_nor_pushing_a_wall_slowly():
    from ballrain.features import level_geometry, mind_normals
    rep = _tas_frames()
    fl = np.load(REC / "raw.npz")["framelog"]
    pos_hist, touched, slow_push = {}, 0, []
    for _, o, tr in rep:
        if not rep.loaded or o.ball is None or not bool(o.ball["found"]):
            continue
        pos = np.asarray(o.ball["pos"], np.float64)
        f = int(o.frames_done)
        pos_hist[f] = pos
        assert not in_zone(pos, PIT), f
        if not len(o.mind):
            continue
        walls = unclimbable_walls(pos, float(o.ball["extra_radius"]) or 2.0, mind_normals(tr, pos, o.mind),
                                  o.mind["gap"], level_geometry(tr).tris.ray_down)
        if not walls:
            continue
        touched += 1
        push = push_direction(int(fl[f]["keys"]) if f < len(fl) else 0, False, o.cams["cams"][0]["matrix"])
        if wall_push_hit(push, walls) and f - 8 in pos_hist:
            v = np.linalg.norm(pos - pos_hist[f - 8]) / (8 / 264)  # speed over 8 frames: an upper bound of progress
            if v < 2.0:
                slow_push.append(f)
    assert touched > 0  # the TAS does ride along tall walls (sector 1), fast
    assert slow_push == []


def test_pit_floor_against_the_pit_wall_is_caught():
    from ballrain.features import level_geometry
    rep = _tas_frames()
    for _, o, tr in rep:
        if o.frames_done >= 10300:
            break
    ray = level_geometry(tr).tris.ray_down
    # pit floor y -9.14, ball centre -7.07; the west pit wall at x 129.7 (rim -4.3 above it), the centre strip's side
    # wall at x 144.7 (strip -4.14); a slope-top step (-6.27 -> -4.34, rise 1.9) stays climbable
    assert unclimbable_walls([131.75, -7.07, -570.0], 2.07, [[1.0, 0, 0]], [0.0], ray)
    assert unclimbable_walls([142.63, -7.07, -575.0], 2.07, [[-1.0, 0, 0]], [0.0], ray)
    assert not unclimbable_walls([137.5, -4.2, -543.07], 2.07, [[0, 0, -1.0]], [0.0], ray)
    assert in_zone([150.0, -7.07, -560.0], PIT)


def test_train_m2_options_reach_spec_and_remote_template():
    sys.path.insert(0, str(instances.REPO_ROOT / "scripts"))
    import train_m2
    a = train_m2.parse(["--run", "x", "--zones", '{"pit": {"box": [1, 2, 3, 4], "y_below": -5}}',
                        "--wall-push-rate", "0.5"])
    assert a.wall_push_rate == 0.5 and a.wall_min_rise == 2.0
    assert "zone_penalty" in train_m2.TERMS and "wall_push" in train_m2.TERMS
    assert "zone_entries" in train_m2.EP_COLS


def _worker(zones, rate, pos, keys, s_before, s_after, normals=((-1.0, 0, 0),), gaps=(0.0,), **spec_kw):
    from types import SimpleNamespace
    from ballrain import protocol as P
    from ballrain.train.worker import Worker, _Episode
    w = object.__new__(Worker)
    w.spec = WorkerSpec(wid=0, copy="x", env=SimpleNamespace(delta_ms=1000 / 132, world_actions=True), seed=0,
                        zones=zones, wall_push_rate=rate, **spec_kw)
    mind = np.zeros(len(gaps), P.MIND_DT)
    mind["gap"] = gaps
    ball = {"pos": np.asarray(pos, np.float64), "found": 1, "extra_radius": 2.0}
    obs = SimpleNamespace(ball=ball, mind=mind, cams=None)
    w.env = SimpleNamespace(session=SimpleNamespace(tracker=SimpleNamespace(obs=obs)),
                            reward=SimpleNamespace(s=s_after), actions=(0, keys))
    w.fx_info = {"mind_normals": np.asarray(normals, np.float64)}
    ep = object.__new__(_Episode)
    ep.terms, ep.zone_in, ep.zone_counts = {}, {}, {}
    return w, ep


def test_worker_zone_entry_once_until_rearmed(monkeypatch):
    w, ep = _worker({"pit": PIT}, 0.0, [150, -7.07, -560], tas.KEY_UP, 0.0, 0.0)
    assert w._zone_wall(ep, 1, 0.0, 2) == pytest.approx(-1.5)
    assert w._zone_wall(ep, 1, 0.0, 2) == 0.0  # still inside: no second penalty
    w.env.session.tracker.obs.ball["pos"] = np.array([150, -4.5, -560])  # within the re-arm margin: still armed off
    assert w._zone_wall(ep, 1, 0.0, 2) == 0.0
    w.env.session.tracker.obs.ball["pos"] = np.array([150, -2.7, -560])  # left (above y_below + 1)
    assert w._zone_wall(ep, 1, 0.0, 2) == 0.0
    w.env.session.tracker.obs.ball["pos"] = np.array([150, -7.07, -560])
    assert w._zone_wall(ep, 1, 0.0, 2) == pytest.approx(-1.5)
    assert ep.zone_counts == {"pit": 2} and ep.terms["zone_penalty"] == pytest.approx(-3.0)


def test_worker_wall_push_rate_progress_and_direction(monkeypatch):
    from types import SimpleNamespace
    import ballrain.features as FX
    ts = _floor_and_block(4.4)
    monkeypatch.setattr(FX, "level_geometry", lambda tracker: SimpleNamespace(tris=ts))
    dt = 2 / 132
    w, ep = _worker({}, 0.5, [8.0, 2.0, 0.0], tas.KEY_RIGHT, 10.0, 10.0, wall_push_everywhere=True)  # pushing +x into the face at x = 10
    assert w._zone_wall(ep, 1, 10.0, 2) == pytest.approx(-0.5 * dt)
    assert ep.terms["wall_push"] == pytest.approx(-0.5 * dt)
    w, ep = _worker({}, 0.5, [8.0, 2.0, 0.0], tas.KEY_LEFT, 10.0, 10.0, wall_push_everywhere=True)  # pushing away
    assert w._zone_wall(ep, 1, 10.0, 2) == 0.0
    w, ep = _worker({}, 0.5, [8.0, 2.0, 0.0], tas.KEY_RIGHT, 10.0, 10.0 + 3 * dt,
                    wall_push_everywhere=True)  # progressing 3 u/s: allowed
    assert w._zone_wall(ep, 1, 10.0, 2) == 0.0
    w, ep = _worker({}, 0.5, [8.0, 2.0, 0.0], 0, 10.0, 10.0, wall_push_everywhere=True)  # no keys
    assert w._zone_wall(ep, 1, 10.0, 2) == 0.0


def test_worker_wall_push_only_inside_wall_push_zones(monkeypatch):
    """Default: the wall push fires only inside a zone marked "wall_push" (box and below y_below); a zone without the
    flag, or the same push outside the zone, pays nothing; wall_push_everywhere restores the route-wide push."""
    from types import SimpleNamespace
    import ballrain.features as FX
    ts = _floor_and_block(4.4)
    monkeypatch.setattr(FX, "level_geometry", lambda tracker: SimpleNamespace(tris=ts))
    dt = 2 / 132
    assert WorkerSpec(wid=0, copy="x", env=None, seed=0).wall_push_everywhere is False
    outside = {"pit": {"box": [100.0, 120.0, 100.0, 120.0], "y_below": -5.0, "penalty": 1.5, "wall_push": True}}
    w, ep = _worker(outside, 0.5, [8.0, 2.0, 0.0], tas.KEY_RIGHT, 10.0, 10.0)  # pushing the wall, not in the zone
    assert w._zone_wall(ep, 1, 10.0, 2) == 0.0 and "wall_push" not in ep.terms and not ep.zone_counts
    w, ep = _worker({}, 0.5, [8.0, 2.0, 0.0], tas.KEY_RIGHT, 10.0, 10.0)  # no zones at all: off
    assert w._zone_wall(ep, 1, 10.0, 2) == 0.0
    flagged = {"pit": {"box": [0.0, 9.9, -5.0, 5.0], "y_below": 3.0, "penalty": 1.5, "wall_push": True}}
    w, ep = _worker(flagged, 0.5, [8.0, 2.0, 0.0], tas.KEY_RIGHT, 10.0, 10.0)  # inside the flagged zone
    assert w._zone_wall(ep, 1, 10.0, 2) == pytest.approx(-1.5 - 0.5 * dt)
    assert ep.terms["wall_push"] == pytest.approx(-0.5 * dt) and ep.zone_counts == {"pit": 1}
    assert w._zone_wall(ep, 1, 10.0, 2) == pytest.approx(-0.5 * dt)  # still inside: the push again, no new entry
    unflagged = {"pit": dict(flagged["pit"], wall_push=False)}
    w, ep = _worker(unflagged, 0.5, [8.0, 2.0, 0.0], tas.KEY_RIGHT, 10.0, 10.0)
    assert w._zone_wall(ep, 1, 10.0, 2) == pytest.approx(-1.5) and "wall_push" not in ep.terms
    w, ep = _worker(outside, 0.5, [8.0, 2.0, 0.0], tas.KEY_RIGHT, 10.0, 10.0, wall_push_everywhere=True)
    assert w._zone_wall(ep, 1, 10.0, 2) == pytest.approx(-0.5 * dt)


def test_stall_penalty_zone_exemption():
    """zone_stall_exempt: a life that entered a penalty zone pays no stall penalty (it still ends by stall); other
    lives, or the option off, pay a death's penalty + refund (stall_compensate divides by gamma ** window)."""
    from types import SimpleNamespace
    from ballrain.train.worker import Worker, _Episode
    def mk(exempt, counts, compensate=False):
        w = object.__new__(Worker)
        w.spec = WorkerSpec(wid=0, copy="x", env=SimpleNamespace(action_repeat=2), seed=0, stall_terminal=True,
                            stall_extra=0.5, stall_compensate=compensate, gamma=0.99, zone_stall_exempt=exempt)
        w.env = SimpleNamespace(reward=SimpleNamespace(cfg=SimpleNamespace(death_penalty=5.0, death_refund=0.4),
                                                       sector_paid=10.0))
        ep = object.__new__(_Episode)
        ep.terms, ep.zone_in, ep.zone_counts = {}, {}, dict(counts)
        return w, ep
    full = -(5.0 + 0.4 * 10.0 + 0.5)
    w, ep = mk(True, {"pit": 1})
    assert w._stall_penalty(ep, 528) == 0.0
    w, ep = mk(True, {"pit": 1}, compensate=True)
    assert w._stall_penalty(ep, 528) == 0.0
    w, ep = mk(True, {})
    assert w._stall_penalty(ep, 528) == pytest.approx(full)
    w, ep = mk(False, {"pit": 2})
    assert w._stall_penalty(ep, 528) == pytest.approx(full)
    w, ep = mk(False, {}, compensate=True)
    assert w._stall_penalty(ep, 528) == pytest.approx(full / 0.99 ** 264)
    assert WorkerSpec(wid=0, copy="x", env=None, seed=0).zone_stall_exempt is False


def test_train_m2_zone_limited_wall_push_and_stall_exempt_options():
    sys.path.insert(0, str(instances.REPO_ROOT / "scripts"))
    import train_m2
    a = train_m2.parse(["--run", "x", "--wall-push-rate", "0.5"])
    assert a.wall_push_everywhere is False and a.zone_stall_exempt is False
    a = train_m2.parse(["--run", "x", "--wall-push-rate", "0.5", "--wall-push-everywhere", "--zone-stall-exempt"])
    assert a.wall_push_everywhere and a.zone_stall_exempt
    src = (instances.REPO_ROOT / "scripts" / "train_m2.py").read_text(encoding="utf-8")
    assert src.count("wall_push_everywhere=a.wall_push_everywhere") == 1  # local workers
    assert '"wall_push_everywhere": a.wall_push_everywhere, "zone_stall_exempt": a.zone_stall_exempt' in src  # remote
