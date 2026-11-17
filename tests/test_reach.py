"""Analytic reachability estimate (ballrain.reach): constants, closed-form steps, the sphere support map on synthetic
geometry (floor, edge, rails, wall, gap) and the estimator's flags. The dataset check (exact air dynamics on a TAS
replay) is skipped when runtime/datasets/tas-v1 is not built."""
import math

import numpy as np
import pytest

from ballrain import dataset, env, instances, reach, tas
from ballrain.path import ReferencePath

DS = instances.runtime_dir() / "datasets" / f"tas-v{dataset.DATASET_VERSION}"
WOOD = reach.BALLS[0]


def box(lo, hi):
    """12 triangles of an axis-aligned box."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], np.float64)
    f = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1), (3, 2, 6), (3, 6, 7),
         (0, 3, 7), (0, 7, 4), (1, 5, 6), (1, 6, 2)]
    return [(v[a], v[b], v[c]) for a, b, c in f]


def smap_of(tris, death=()):
    A, B, C = (np.array([t[i] for t in tris]) for i in range(3))
    return reach.SupportMap(A, B, C, death_boxes=death)


# ----------------------------------------------------------------------------------------------- constants

def test_ball_constants_match_the_physicalize_table():
    assert WOOD.air_accel == pytest.approx(0.43 / 1.9)  # 0.22632 per PSI, measured exactly in the air
    assert 1 - WOOD.air_lambda == pytest.approx(0.9 / 66)
    assert WOOD.ground_accel == pytest.approx(0.43 / 1.9 / 1.4)
    assert WOOD.top_speed(1, ground=True) == pytest.approx(15.9, abs=0.05)
    assert WOOD.top_speed(2, ground=True) == pytest.approx(22.5, abs=0.05)
    assert WOOD.top_speed(1, ground=False) == pytest.approx(16.6, abs=0.05)
    assert reach.BALLS[2].air_accel == pytest.approx(0.092)
    assert not reach.BALLS[1].supported


def test_key_push_matches_env_key_push_dirs_at_quarter_turns():
    for q in range(4):
        yaw = q * math.pi / 2
        z = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
        x = np.array([z[2], 0.0, -z[0]])
        cam = np.array([x, [0, 1, 0], z, [0, 0, 0]])
        dirs = env.key_push_dirs(cam)
        for keys in reach.KEY_COMBOS:
            want = sum(dirs[b] for b in env.ARROW_BITS if keys & b)
            got = reach.key_push(keys, q)
            assert np.allclose(got, want[[0, 2]], atol=1e-12)
    assert np.linalg.norm(reach.key_push(tas.KEY_UP | tas.KEY_RIGHT, 1)) == pytest.approx(math.sqrt(2))


def test_closed_form_step_equals_the_per_psi_recurrence():
    v0, a, lam = np.array([3.0, -1.0]), np.array([0.2, -0.3]), WOOD.air_lambda
    v, p = v0.copy(), np.zeros(2)
    for _ in range(7):  # retail: the position moves with the velocity before the PSI's update
        p = p + v / reach.PSI_HZ
        v = lam * v + a
    vn, d = reach._lin_step(v0, a, lam, 7)
    assert np.allclose(vn, v) and np.allclose(d, p)


# ------------------------------------------------------------------------------------------ support map

def test_free_fall_far_from_geometry_follows_the_air_recurrence_exactly():
    sm = smap_of(box((-5, -1, -5), (5, 0, 5)))
    pos, vel = np.array([500.0, 300.0, 0.0]), np.array([4.0, 2.0, -1.0])
    u = reach.key_push(np.full(40, tas.KEY_UP | tas.KEY_LEFT), 0)[None]
    R = reach.rollout(sm, pos, vel, WOOD, u[:, :10], reach.ReachConfig(), grounded=False, step_psi=4)
    p, v = pos.copy(), vel.copy()
    for k in range(40):
        uk = u[0, k]
        p = p + v / 66
        v = WOOD.air_lambda * v + np.array([WOOD.air_accel * uk[0], -reach.GRAVITY / 66, WOOD.air_accel * uk[1]])
    assert np.allclose(R["traj"][0, -1], p, atol=1e-9)


def test_flat_floor_rest_height_normal_and_top_speed():
    sm = smap_of(box((-100, -1, -100), (100, 0, 100)))
    rest, _, wall, n, _ = sm.query(np.array([[0.3, 0.7], [10.1, -3.3]]), np.array([2.0, 2.0]), 2.0, 2.0, normals=True)
    assert np.allclose(rest, 2.0, atol=0.02) and not wall.any()
    assert np.allclose(n, [0, 1, 0], atol=1e-6)
    # hold UP for 3 game seconds from rest: rolls, stays on the floor, approaches the one-key plateau
    u = np.tile(reach.key_push(tas.KEY_UP, 0), (1, 99, 1))
    R = reach.rollout(sm, np.array([0.0, 2.0, -80.0]), np.zeros(3), WOOD, u, reach.ReachConfig())
    assert R["alive"][0] and R["grounded"][0].all()
    v_end = (R["traj"][0, -1, 2] - R["traj"][0, -2, 2]) * 66 / 4
    assert 14.0 < v_end < WOOD.top_speed(1)


def test_rails_support_a_sphere_where_a_vertical_ray_misses():
    rails = box((-1.075, -0.15, -50), (-0.925, 0.0, 50)) + box((0.925, -0.15, -50), (1.075, 0.0, 50))
    sm = smap_of(rails)
    rest, _, _, n, _ = sm.query(np.array([[0.0, 3.0]]), np.array([3.0]), 2.0, 2.0, normals=True)
    assert rest[0] == pytest.approx(math.sqrt(4 - 0.925 ** 2), abs=0.06)  # one sample per 0.5 node
    assert n[0, 1] > 0.99  # two symmetric contacts: vertical net normal
    # rolling along the rails keeps the ball up for the whole horizon
    u = np.tile(reach.key_push(tas.KEY_UP, 0), (1, 60, 1))
    R = reach.rollout(sm, np.array([0.0, rest[0], -40.0]), np.array([0.0, 0.0, 8.0]), WOOD, u, reach.ReachConfig())
    assert R["alive"][0] and R["grounded"][0].all()


def test_tall_wall_blocks_and_low_step_is_climbed():
    floor = box((-50, -1, -50), (50, 0, 50))
    u = np.tile(reach.key_push(tas.KEY_UP, 0), (1, 60, 1))
    sm = smap_of(floor + box((-50, 0, 10), (50, 4.0, 12)))
    R = reach.rollout(sm, np.array([0.0, 2.0, 0.0]), np.array([0.0, 0.0, 10.0]), WOOD, u, reach.ReachConfig())
    tr = R["traj"][0]
    assert R["alive"][0] and 10.0 - 2.5 < tr[:, 2].max() < 10.0 - 1.5  # it reaches the wall and stops there
    sm = smap_of(floor + box((-50, 0, 10), (50, 0.5, 150)))
    R = reach.rollout(sm, np.array([0.0, 2.0, 0.0]), np.array([0.0, 0.0, 10.0]), WOOD, u, reach.ReachConfig())
    assert 20 < R["traj"][0, -1, 2] < 140 and R["traj"][0, -1, 1] == pytest.approx(2.5, abs=0.05)


@pytest.mark.parametrize("height", [6.0, 10.0, 14.0])
def test_walls_taller_than_the_layer_gap_are_solid(height):
    """A block taller than layer_gap (4) and of limited width (its side face's diagonal edges run at varying heights
    across the nodes) must still be one solid column: rolled into at 10 or 25 units / phys-s anywhere along it, the
    ball stops in front of it."""
    sm = smap_of(box((-50, -1, -50), (50, 0, 50)) + box((-15, 0, 10), (15, height, 12)))
    u = np.tile(reach.key_push(tas.KEY_UP, 0), (1, 60, 1))
    for x in np.arange(-11.0, 11.5, 1.1):  # away from the ends (a ball at an end glances off the corner)
        for v0 in (10.0, 25.0):
            R = reach.rollout(sm, np.array([x, 2.0, 0.0]), np.array([0.0, 0.0, v0]), WOOD, u, reach.ReachConfig())
            tr = R["traj"][0]
            assert R["alive"][0] and 10.0 - 2.5 < tr[:, 2].max() < 10.0 - 1.5, (x, v0)


def test_empty_support_map_reports_no_support():
    sm = reach.SupportMap(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)))
    rest, _, wall, _, _ = sm.query(np.array([[0.0, 0.0], [5.0, -3.0]]), np.array([1.0, 1.0]), 2.0, 2.0,
                                    normals=True)
    assert np.isnan(rest).all() and not wall.any()


# --------------------------------------------------------------------------------------------- estimator

def gap_level():
    """Platform A (z 0..40, top 0), a 20-unit gap over a death box, platform B (z 60..140, top -10)."""
    tris = box((-10, -2, 0), (10, 0, 40)) + box((-10, -12, 60), (10, -10, 140))
    death = [(np.array([-10.0, -60.0, 40.0]), np.array([10.0, -30.0, 60.0]))]
    pts = [(0, 2, z) for z in np.arange(0, 40.5, 0.5)] + [(0, 2 - 10 * (z - 40) / 20, z) for z in np.arange(40.5, 60, 0.5)]
    pts += [(0, -8, z) for z in np.arange(60, 140.5, 0.5)]
    return smap_of(tris, death), ReferencePath(np.array(pts, np.float64))


def test_gap_can_be_made_fast_but_not_slow():
    sm, path = gap_level()
    est = reach.ReachEstimator(sm)
    fast = est.estimate(np.array([0.0, 2.0, 30.0]), np.array([0.0, 0.0, 22.0]), 0, 0, path=path)
    assert fast["gap"] is not None and 38 <= fast["gap"][0] <= 44 and 56 <= fast["gap"][1] <= 62
    assert fast["can_make_gap"] and not fast["doomed"]
    slow = est.estimate(np.array([0.0, 2.0, 30.0]), np.array([0.0, 0.0, 2.0]), 0, 0, path=path)
    assert slow["can_make_gap"] is False and not slow["doomed"]  # it can still stop on A
    assert slow["max_safe_s"] < 12


def test_doomed_when_flying_over_the_death_box_and_per_key_landing():
    sm, path = gap_level()
    est = reach.ReachEstimator(sm)
    o = est.estimate(np.array([0.0, -5.0, 45.0]), np.array([0.0, -5.0, 1.0]), 0, 0, path=path)
    assert o["doomed"] and o["n_alive"] == 0
    assert all(k["t_death_s"] is not None and k["t_death_s"] < 1.5 for k in o["keys"])
    # rolling on A towards the gap: UP (+z at quarter 0) lands on B, DOWN stays on A
    o = est.estimate(np.array([0.0, 2.0, 30.0]), np.array([0.0, 0.0, 22.0]), 0, 0, path=path)
    keys = {k["keys"]: k for k in o["keys"]}
    up, down = keys[tas.KEY_UP], keys[tas.KEY_DOWN]
    assert up["alive"] and up["landing"] is not None and up["landing"][2] > 60 and up["landing"][1] == pytest.approx(-8, abs=0.1)
    # braking needs v^2 / (2 * 10.7) units: from 22 it does not stop before the edge 10 units ahead, from 10 it does
    assert not down["alive"]
    o3 = est.estimate(np.array([0.0, 2.0, 30.0]), np.array([0.0, 0.0, 10.0]), 0, 0, path=path)
    end = o3["traj"][:, -1]
    assert not o3["doomed"] and (o3["alive"] & (end[:, 2] > 0) & (end[:, 2] < 40) & (abs(end[:, 1] - 2) < 0.1)).any()
    # the camera quarter only relabels the keys: at quarter 2 DOWN pushes +z
    o2 = est.estimate(np.array([0.0, 2.0, 30.0]), np.array([0.0, 0.0, 22.0]), 0, 2, path=path)
    k2 = {k["keys"]: k for k in o2["keys"]}
    assert np.allclose(k2[tas.KEY_DOWN]["end"], up["end"])


def test_paper_ball_is_reported_unsupported():
    sm, path = gap_level()
    assert reach.ReachEstimator(sm).estimate(np.zeros(3), np.zeros(3), 1, 0)["supported"] is False


# ---------------------------------------------------------------------------------------------- dataset

REC = DS / "SR06_1_36_163_132"


@pytest.mark.skipif(not (REC / "raw.npz").is_file(), reason="tas-v1 dataset not built")
def test_air_dynamics_exact_on_a_retail_replay():
    """In free flight (no mindist, no friction contact) the retail velocity follows the model to float precision."""
    z = np.load(REC / "raw.npz")
    v, keys, fd = z["ball_vel"], z["framelog"]["keys"].astype(np.int64), z["frames_done"]
    air = (z["n_mind"] == 0) & (z["n_fric"] == 0) & (z["ball_type"] == 0) & np.isfinite(v).all(1)
    i = np.flatnonzero(air[1:] & air[:-1]) + 1
    i = i[(i > 600)]  # after the first control frames
    # without the camera: use only frames with no arrow key down (the vertical axis never depends on keys)
    dv = v[i] - v[i - 1]
    moving = np.any(dv != 0, axis=1)  # paused frames keep the velocity
    i, dv = i[moving], dv[moving]
    want_y = -reach.GRAVITY / 66 - (1 - WOOD.air_lambda) * v[i - 1, 1]
    assert len(i) > 500
    assert np.max(np.abs(dv[:, 1] - want_y)) < 1e-4
    # horizontally, what damping leaves is the key push: |F/m u| with |u| = 0, 1 or sqrt 2 (the camera is not stored)
    push = np.linalg.norm(dv[:, [0, 2]] + (1 - WOOD.air_lambda) * v[i - 1][:, [0, 2]], axis=1)
    levels = WOOD.air_accel * np.array([0.0, 1.0, math.sqrt(2)])
    assert np.max(np.min(np.abs(push[:, None] - levels[None]), axis=1)) < 1e-4


# ------------------------------------------------------------------- hard barriers on real L1 geometry

L1 = DS / "SR01_1_14_457_264"


@pytest.fixture(scope="module")
def l1_map():
    if not (L1 / "level.msg").is_file():
        pytest.skip("tas-v1 dataset not built")
    from ballrain import protocol as P
    from ballrain.features import level_geometry
    from ballrain.state import GameTracker
    tr = GameTracker(1)
    tr.merge_level(P.parse_level_message((L1 / "level.msg").read_bytes())[0])
    for m in P.parse_level_stream((L1 / "level.msg.inc").read_bytes())[0]:
        tr.merge_level(m)
    return reach.SupportMap.from_level(level_geometry(tr), tr.level.depth)


def _hold(sm, pos, vel, keys, secs):
    u = np.tile(reach.key_push(keys, 0), (1, int(secs * reach.GAME_PSI_HZ / 4), 1))
    R = reach.rollout(sm, np.array(pos, float), np.array(vel, float), WOOD, u, reach.ReachConfig())
    return R["alive"][0], R["traj"][0]


def test_l1_first_platform_west_guardrail_blocks_and_the_exit_lane_is_open(l1_map):
    """Case (a): the west guardrail of the first big platform (A01_Rail_03, x ~ -177, top ~20.3, 2.9 above the 17.41
    floor) can be neither crossed nor climbed, also where it turns east at the SW corner; the exit lane between the
    rails (x -172.4 .. -168.6) stays open."""
    sm = l1_map
    pos = (-174.0, 19.41, -158.0)
    pr = reach.probe_directions(sm, np.array(pos), np.array([[-1.0, 0.0], [0.0, -1.0]]))
    assert pr["blocked_at"][0] <= 1.0 and pr["wall_top"][0] == pytest.approx(2.9, abs=0.15)
    assert not np.isfinite(pr["blocked_at"][1])
    for speed in (8.0, 15.0):  # driven into it from 4 units away: reaches it, stays on the platform side
        alive, tr = _hold(sm, (-170.0, 19.41, -158.0), (-speed, 0, 0), tas.KEY_LEFT, 3.0)
        assert alive and -176.0 < tr[:, 0].min() < -173.5 and np.allclose(tr[:, 1], 19.41, atol=0.1)
    pr = reach.probe_directions(sm, np.array([-175.0, 19.41, -164.0]), np.array([[0.0, -1.0]]))  # SW corner
    assert pr["blocked_at"][0] <= 3.0 and pr["wall_top"][0] == pytest.approx(2.9, abs=0.15)
    pr = reach.probe_directions(sm, np.array([-170.5, 19.41, -162.0]), np.array([[0.0, -1.0]]))  # exit lane
    assert not np.isfinite(pr["blocked_at"][0]) and not np.isfinite(pr["drop_at"][0])
    alive, tr = _hold(sm, (-170.5, 19.41, -162.0), (0, 0, -6), tas.KEY_DOWN, 1.5)
    assert alive and tr[-1, 2] < -185  # down the lane on the rails
    alive, _ = _hold(sm, (-160.5, 19.41, -162.0), (0, 0, -6), tas.KEY_DOWN, 1.5)
    assert not alive  # the same push east of the raised strip runs off the platform's south edge


def test_l1_sector2_pit_walls_block_from_inside(l1_map):
    """Case (b), walls: the 小节 2 pit (floor -9.14): its walls (rim -4.1..-4.8, central strip -4.14, 5 above the
    floor) cannot be mounted at any speed; driven into them the ball stays on the pit floor."""
    sm = l1_map
    y = -9.14 + 2.0
    for pos, d, keys in (((134.0, y, -565.0), (-1, 0), tas.KEY_LEFT), ((160.0, y, -575.0), (0, -1), tas.KEY_DOWN),
                         ((166.0, y, -552.0), (0, 1), tas.KEY_UP), ((139.0, y, -570.0), (1, 0), tas.KEY_RIGHT)):
        pr = reach.probe_directions(sm, np.array(pos), np.array([d], float), max_dist=10.0, speed=40.0)
        assert np.isfinite(pr["blocked_at"][0]) and pr["wall_top"][0] == pytest.approx(5.0, abs=0.2)
        assert pr["mount_speed"][0] == np.inf
        alive, tr = _hold(sm, pos, (20 * d[0], 0, 20 * d[1]), keys, 3.0)
        assert alive and tr[:, 1].max() < y + 0.2  # never leaves the pit floor
        reach_along = ((tr[:, [0, 2]] - np.array(pos)[[0, 2]]) @ np.array(d, float)).max()
        assert pr["blocked_at"][0] - 1.0 <= reach_along <= pr["blocked_at"][0] + 1.0  # it got to the wall


def test_l1_sector2_pit_ramp_step_centre_never_curb_with_speed(l1_map):
    """Case (b), the ramp-top step onto the north rim (user rule, speed-dependent edge mount, a model estimate): up the
    ramp centre (step ~1.9-2.1 above the ball bottom, at or above MOUNT_CAP) it cannot be mounted at any speed; along
    the west curb (路檐, x ~134.7-135.25, step ~1.5, mount speed ~14) it can: hugging the curb from 15 mounts, coasting
    in at 9 does not."""
    sm = l1_map
    lip_z = -540.75

    def rest_at(x, z):
        return float(sm.query(np.array([[x, z]]), np.array([0.0]), 2.0, 0.5)[0][0])

    for x in (137.5, 138.5):  # centre: blocked even at 40 units / phys-s (flat two-key top speed is 22.5)
        pr = reach.probe_directions(sm, np.array([x, rest_at(x, -546.0), -546.0]), np.array([[0.0, 1.0]]),
                                    max_dist=8.0, speed=40.0)
        assert np.isfinite(pr["blocked_at"][0]) and pr["mount_speed"][0] > WOOD.top_speed(2, ground=True)
        for v0 in (12.0, 15.0, 18.0, 22.0):  # up the ramp a run-up reaches ~12.4
            alive, tr = _hold(sm, (x, rest_at(x, -548.0), -548.0), (0, 0, v0), tas.KEY_UP, 1.5)
            assert tr[:, 2].max() < lip_z - 1.0
    # from rest on the pit floor holding UP: the run-up never gets over the step
    alive, tr = _hold(sm, (137.5, -7.14, -578.0), (0, 0, 0), tas.KEY_UP, 3.0)
    assert tr[:, 2].max() < lip_z - 1.0
    # curb: the probe's barrier is mountable well below the reachable speed
    x = 134.95
    pr = reach.probe_directions(sm, np.array([x, rest_at(x, -546.0), -546.0]), np.array([[0.0, 1.0]]), max_dist=8.0,
                                speed=1.0)
    assert np.isfinite(pr["blocked_at"][0]) and 1.2 < pr["wall_top"][0] < 1.7 and 10.0 < pr["mount_speed"][0] < 16.0
    # hugging the curb (UP + LEFT against the V-ramp's pull to its centre) from 15: mounts
    hi = _hold(sm, (x, rest_at(x, -547.0), -547.0), (0, 0, 15.0), tas.KEY_UP | tas.KEY_LEFT, 1.0)[1]
    assert hi[:, 2].max() > lip_z + 2.0 and hi[:, 1].max() > -4.23 + 1.8  # up on the rim
    lo = _hold(sm, (x, rest_at(x, -547.0), -547.0), (0, 0, 9.0), 0, 1.0)[1]  # coasting in at 9: does not
    assert lo[:, 2].max() < lip_z - 1.0


def test_long_ramp_climbed_but_short_steep_lip_of_lower_height_is_not():
    """Climb rule: a walkable ramp rising 4 over 20 units is climbed; a 2.5 lip (above MOUNT_CAP) is a wall."""
    floor = box((-20, -1, -50), (20, 0, 10))
    ramp = [(np.array([-20.0, 0, 10]), np.array([20.0, 0, 10]), np.array([20.0, 4, 30])),
            (np.array([-20.0, 0, 10]), np.array([20.0, 4, 30]), np.array([-20.0, 4, 30]))]
    top = box((-20, 0, 30), (20, 4, 120))
    sm = smap_of(floor + ramp + top)
    alive, tr = _hold(sm, (0.0, 2.0, 0.0), (0, 0, 8.0), tas.KEY_UP, 3.0)
    assert alive and tr[-1, 2] > 32 and tr[-1, 1] == pytest.approx(6.0, abs=0.1)
    sm = smap_of(floor + box((-20, 0, 10), (20, 2.5, 60)))
    alive, tr = _hold(sm, (0.0, 2.0, 0.0), (0, 0, 8.0), tas.KEY_UP, 3.0)
    assert tr[:, 2].max() < 10 - 1.0 and tr[:, 1].max() < 2.1
    pr = reach.probe_directions(sm, np.array([0.0, 2.0, 0.0]), np.array([[0.0, 1.0]]), max_dist=12.0)
    assert pr["blocked_at"][0] == pytest.approx(8.0, abs=0.6) and pr["wall_top"][0] == pytest.approx(2.5, abs=0.1)


def test_face_normal_decides_between_ramp_and_wall():
    """The face normal decides between ramp and wall: walking at 10 units / phys-s, a 40 degree walkable ramp
    (n_y 0.77) rising 6 is climbed gradually (holding contacts on its face, the crest mounted by the speed rule); a 70
    degree face (n_y 0.34, not walkable) of the same rise is a wall: steep-face samples never support, so its node
    tops do not read as a staircase."""
    floor = box((-20, -1, -50), (20, 0, 10))

    def slope(deg, rise=6.0):
        run = rise / math.tan(math.radians(deg))
        a, b = np.array([-20.0, 0, 10]), np.array([20.0, 0, 10])
        c, d = np.array([20.0, rise, 10 + run]), np.array([-20.0, rise, 10 + run])
        return [(a, b, c), (a, c, d)] + box((-20, 0, 10 + run), (20, rise, 120)), 10 + run

    for deg, passes in ((40.0, True), (70.0, False)):
        tris, z_top = slope(deg)
        sm = smap_of(floor + tris)
        pr = reach.probe_directions(sm, np.array([0.0, 2.0, 4.0]), np.array([[0.0, 1.0]]), max_dist=20.0,
                                    step=0.25, speed=10.0)
        assert np.isfinite(pr["blocked_at"][0]) != passes, deg
        if passes:  # climbed to the top and walked on it, not dropped off anywhere
            assert not np.isfinite(pr["drop_at"][0])
        else:
            assert pr["blocked_at"][0] < z_top - 4.0 and pr["wall_top"][0] > reach.MOUNT_FREE


# ------------------------------------------------------------------------------------------- review fixes

@pytest.mark.skipif(not (REC / "raw.npz").is_file(), reason="tas-v1 dataset not built")
def test_positions_follow_retail_on_a_replayed_flight():
    """Pin the position convention to retail: over a recorded free flight, rollout (step_psi 1, pushes recovered
    from the recorded velocities, no geometry) reproduces BALL.pos: p[k+1] = p[k] + v[k] / 66, then v updates."""
    z = np.load(REC / "raw.npz")
    p, v = z["ball_pos"], z["ball_vel"]
    air = (z["n_mind"] == 0) & (z["n_fric"] == 0) & (z["ball_type"] == 0) & np.isfinite(v).all(1)
    # free flight: consecutive airborne frames whose vertical velocity follows gravity and damping (not paused)
    fall = np.r_[np.abs(v[1:, 1] - (WOOD.air_lambda * v[:-1, 1] - reach.GRAVITY / 66)) < 1e-4, False]
    i = np.flatnonzero(air & fall & np.r_[air[1:], False])
    runs = np.split(i, np.flatnonzero(np.diff(i) != 1) + 1)
    run = max((r for r in runs if r[0] > 600), key=len)
    run = np.r_[run, run[-1] + 1]
    assert len(run) >= 20
    k0, n = run[0], len(run) - 1
    acc = v[run[1:]] - WOOD.air_lambda * v[run[:-1]]  # push per PSI recovered from the data
    u = acc[:, [0, 2]] / WOOD.air_accel
    assert np.allclose(acc[:, 1], -reach.GRAVITY / 66, atol=1e-4)
    sm = reach.SupportMap(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)))
    cfg = reach.ReachConfig(void_drop=1e9)
    R = reach.rollout(sm, p[k0], v[k0], WOOD, u[None], cfg, grounded=False, step_psi=1)
    assert np.abs(R["traj"][0] - p[run]).max() < 1e-3


def test_project_many_projects_onto_segments_not_vertices():
    path = ReferencePath(np.array([[0.0, 0, 0], [0, 0, 10], [0, 0, 20]]))
    s, d = reach.project_many(path, np.array([[1.0, 0, 3.3], [0.0, 0, 17.25]]), 0.0, 20.0)
    assert np.allclose(s, [3.3, 17.25]) and np.allclose(d, [1.0, 0.0])


def test_estimate_batch_accepts_a_scalar_grounded_flag():
    sm, _ = gap_level()
    est = reach.ReachEstimator(sm, reach.ReachConfig(switch_s=(), probe=False))
    out = est.estimate_batch(np.array([[0.0, 2.0, 20.0], [0.0, 2.0, 30.0]]), np.zeros((2, 3)), 0, grounded=True)
    assert out["doomed"].shape == (2,) and not out["doomed"].any()


def test_next_features_gap_not_hidden_by_a_beam_overhead():
    sm0, path = gap_level()
    tris = box((-10, -2, 0), (10, 0, 40)) + box((-10, -12, 60), (10, -10, 140))
    beam = box((-3, 15, 35), (3, 16, 65))  # 13 above the line, over the gap
    sm = smap_of(tris + beam)
    cfg = reach.ReachConfig()
    assert reach.next_features(sm0, path, 30.0, cfg)["gap"] is not None
    assert reach.next_features(sm, path, 30.0, cfg)["gap"] is not None


def test_probe_from_an_airborne_state_starts_on_the_support_or_reports_airborne():
    sm = smap_of(box((-50, -1, -50), (50, 0, 50)))
    dirs = reach.key_push(np.array(reach.KEY_COMBOS), 0)
    for y in (2.6, 30.0):  # in the air above the floor: the walk runs on the floor, no false drops
        pr = reach.probe_directions(sm, np.array([0.0, y, 0.0]), dirs)
        assert pr["airborne"] and not np.isfinite(pr["drop_at"]).any() and not np.isfinite(pr["blocked_at"]).any()
    void = reach.probe_directions(sm, np.array([200.0, 5.0, 0.0]), dirs)  # nothing under it
    assert void["airborne"] and np.isnan(void["drop_at"]).all()


def test_route_flags_see_the_last_trajectory_points():
    """The trajectory subsample used for the path keeps the final index (the horizon end)."""
    sm = smap_of(box((-10, -1, -10), (10, 0, 400)))
    path = ReferencePath(np.array([(0.0, 2.0, z) for z in np.arange(0, 400.5, 0.5)]))
    est = reach.ReachEstimator(sm, reach.ReachConfig(switch_s=(), probe=False))
    o = est.estimate(np.array([0.0, 2.0, 0.0]), np.array([0.0, 0.0, 22.0]), 0, 0, path=path, s0=0.0)
    far = float(o["traj"][:, -1, 2].max())
    assert o["max_safe_s"] == pytest.approx(far, abs=0.6)
