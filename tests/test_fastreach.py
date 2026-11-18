"""Fast field-based reachability (ballrain.fastreach): synthetic geometry (flat platform landing, gap cleared or not by
speed, walls and steps, braking distance, rail tubes), the numba / plain-Python paths giving identical results, and
the TAS frame 4956 landing on L1 (skipped without the tas-v1 dataset)."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from ballrain import dataset, fastreach as F, geometry as G, instances, reach

WOOD = reach.BALLS[0]
DS = instances.runtime_dir() / "datasets" / f"tas-v{dataset.DATASET_VERSION}"
L1 = DS / "SR01_1_14_457_264"
L2 = DS / "SR02_1_28_992_264"


def box(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], np.float64)
    f = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1), (3, 2, 6), (3, 6, 7),
         (0, 3, 7), (0, 7, 4), (1, 5, 6), (1, 6, 2)]
    return [(v[a], v[b], v[c]) for a, b, c in f]


def tube(p0, p1, R=0.25, sides=8):
    """Open-ended prism tube along p0 -> p1 (horizontal), `sides` faces."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    u = np.cross(d, [0, 1, 0])
    u /= np.linalg.norm(u)
    w = np.cross(u, d)
    ang = np.arange(sides) * 2 * np.pi / sides
    ring = [np.cos(a) * u * R + np.sin(a) * w * R for a in ang]
    tris = []
    for i in range(sides):
        a0, a1 = p0 + ring[i], p0 + ring[(i + 1) % sides]
        b0, b1 = p1 + ring[i], p1 + ring[(i + 1) % sides]
        tris += [(a0, b0, b1), (a0, b1, a1)]
    return tris


def make(tris, rails=(), depth=(), fans=()):
    allt = list(tris) + list(rails)
    A, B, C = (np.array([t[i] for t in allt]) for i in range(3))
    owner = np.r_[np.zeros(len(tris), np.int64), np.ones(len(rails), np.int64)]
    ts = G.TriangleSet(A, B, C, owner)
    rs = G.TriangleSet(A[len(tris):], B[len(tris):], C[len(tris):], owner[len(tris):])
    return F.FastReach.build(ts, rs, depth, fans=fans)


DEATH = [(0, np.array([-200.0, -60.0, -200.0]), np.array([200.0, -20.0, 200.0]))]


@pytest.fixture(scope="module")
def platforms():
    """A 20 x 20 platform (top y = 0) with a 2.5-tall wall and a 1.0 step on it, and a 10-unit gap along +x to a
    second one 4 lower (a gap to a platform at the same height cannot be cleared from a flat takeoff: the ball
    always drops), over a death box (top -20)."""
    tris = box((-20, -1, -10), (0, 0, 10)) + box((10, -5, -10), (30, -4, 10))
    tris += box((-20, 0, 8), (0, 2.5, 10))  # wall along the north side (z 8..10)
    tris += box((-20, 0, -10), (-15, 1.0, -6))  # a 1.0 step in the south-west corner
    return make(tris, depth=DEATH)


# --------------------------------------------------------------------------------------------- closed forms

def test_air_tables_match_the_per_psi_recurrence():
    S1, S2 = F.air_tables(WOOD, 50)
    lam, a = WOOD.air_lambda, WOOD.air_accel
    p = np.zeros(3)
    v = np.array([3.0, 2.0, -1.0])
    v0 = v.copy()
    for n in range(1, 51):
        p = p + v / 66
        v = lam * v + np.array([a, -20 / 66, -a])
        want = v0 * S1[n] + np.array([a, -20 / 66, -a]) * S2[n]
        assert np.allclose(p, want, atol=1e-12)


def test_brake_distance_closed_form_equals_the_ground_recurrence():
    lam, a = WOOD.ground_lambda, WOOD.ground_accel
    for v0, g_t in ((14.0, 0.0), (-9.0, 0.0), (12.0, -0.02), (5.0, 0.03)):
        d, _, k = F._brake_axis(v0, g_t, a, lam, 1e9)
        v, x, n = v0, 0.0, 0
        s = 1.0 if v0 > 0 else -1.0
        while v * s > 0:  # p += v / 66 then v' = lam v + a u + g_t, reverse push until the speed changes sign
            x += v / 66
            v = lam * v - s * a + g_t
            n += 1
        assert abs(k - n) < 1.0
        assert d == pytest.approx(x, abs=abs(v0) / 66 + 1e-6)  # within one PSI of travel


def test_gap_takeoff_speed_matches_a_brute_force_flight():
    lam, a, g = WOOD.air_lambda, WOOD.air_accel, 20 / 66

    def clears(v, push, x_need, dy, vy0):
        x, y, vx, vy = 0.0, 0.0, v, vy0
        for _ in range(F.N_PSI):
            x += vx / 66
            y += vy / 66
            vx, vy = lam * vx + a * push, lam * vy - g
            if x >= x_need:
                return y >= dy
        return False
    fr = None

    class _T:  # tables only
        @staticmethod
        def tables(bt):
            return (*F.air_tables(WOOD), WOOD)
    fr = _T()
    for x_need, dy, vy0, push in ((30.2, -34.7, -8.2, 1.0), (30.0, -35.26, 0.0, 0.0), (10.0, -2.0, 0.0, 1.0)):
        v = F.gap_takeoff_speed(fr, x_need, dy, 0.0, push=push, vy0=vy0)
        assert clears(v + 0.02, push, x_need, dy, vy0) and not clears(v - 0.05, push, x_need, dy, vy0)
    # the 小节 4 jump: true vertical speed -8.2, one key, 2 units inside the far edge: ~10.9 (coordinator: ~10.4-11)
    assert F.gap_takeoff_speed(fr, 30.26, 29.9 - 64.59, 0.0, push=1.0, vy0=-8.2) == pytest.approx(10.9, abs=0.2)


# ------------------------------------------------------------------------------------------- synthetic fields

def test_flat_platform_landing_reachable_and_doomed_far_away(platforms):
    fr = platforms
    vec, info = fr.perceive([-10.0, 8.0, 0.0], [0.0, 0.0, 0.0], debug=True)
    named = info["named"]
    assert named["grounded"] == 0 and named["air_any"] == 1 and named["doomed"] == 0
    assert info["dbg"]["near_h"] == pytest.approx(0.0, abs=0.05)
    assert info["dbg"]["first_h"] == pytest.approx(1.0, abs=0.05)  # the 1.0 step is within reach and met earlier
    assert named["air_near_dy"] == pytest.approx(-6.0, abs=0.15)  # lands with the centre at r above the top
    assert named["air_near_effort"] == pytest.approx(0.0, abs=1e-9) and named["air_near_clear"] == 1
    # the contact PSI: the centre falls 6 (from 8 to h + r) under gravity
    S1, S2 = F.air_tables(WOOD)
    n_fall = int(np.argmax(8.0 - 20 / 66 * S2 <= 2.0 + F.CONTACT_TOL))
    assert abs(info["dbg"]["n_near"] - n_fall) <= 1
    # far out over the death box, falling: nothing reachable
    vec, info = fr.perceive([100.0, 5.0, 100.0], [0.0, -5.0, 0.0], debug=True)
    assert info["named"]["air_any"] == 0 and info["named"]["doomed"] == 1
    # off the platform edge sideways but within air control: reachable
    vec, _ = fr.perceive([3.0, 4.0, 0.0], [2.0, 0.0, 0.0])
    assert vec[F.FEATURE_NAMES.index("air_any")] == 1


def test_gap_cleared_with_speed_not_without(platforms):
    fr = platforms
    i = F.FEATURE_NAMES.index("right_gap_deficit")
    e = F.FEATURE_NAMES.index("right_edge")
    pos = np.array([-6.0, 2.0, 0.0])
    slow, info = fr.perceive(pos, [1.0, 0.0, 0.0], debug=True)
    fast, _ = fr.perceive(pos, [20.0, 0.0, 0.0])
    assert slow[F.FEATURE_NAMES.index("grounded")] == 1
    assert slow[e] == pytest.approx(6.0 + 1.5, abs=0.6)  # support lost ~1.7 past the edge (sphere over the corner)
    d = info["dirs"]["right"]
    assert d["s_far"] - d["s_near"] == pytest.approx(10.0, abs=1.1)
    assert slow[i] > 0 > fast[i]  # from 1 the full push cannot reach the takeoff speed in 6 units; from 20 it can
    # the airborne check from a fast takeoff agrees: the far platform is reachable
    vec, _ = fr.perceive([3.0, 2.0, 0.0], [20.0, 0.0, 0.0])
    assert vec[F.FEATURE_NAMES.index("air_any")] == 1
    # going left (no far side within reach) the deficit is the "impossible" clip
    assert slow[F.FEATURE_NAMES.index("left_gap_deficit")] == F.CLIP_V


def test_wall_blocks_and_step_needs_speed(platforms):
    fr = platforms
    m = fr.march([-10.0, 2.0, 0.0], (0, 1))
    assert m["obst_s"] == pytest.approx(6.0, abs=0.6) and m["obst_h"] == pytest.approx(2.5, abs=0.1)
    assert math.isinf(m["edge_s"])
    vec, _ = fr.perceive([-10.0, 2.0, 0.0], [0.0, 0.0, 5.0])
    assert vec[F.FEATURE_NAMES.index("up_obst_deficit")] == F.CLIP_V  # a 2.5 wall is never mounted
    m = fr.march([-17.5, 2.0, 2.0], (0, -1))
    assert m["obst_h"] == pytest.approx(1.0, abs=0.1) and m["obst_s"] == pytest.approx(6.0, abs=0.6)
    vm = float(reach.mount_speed(1.0))
    # 2 units before the step: a full push from 2 does not reach the mount speed (8.3), rolling in at 14 does
    slow, info = fr.perceive([-17.5, 2.0, -2.0], [0.0, 0.0, -2.0], debug=True)
    fast, _ = fr.perceive([-17.5, 2.0, -2.0], [0.0, 0.0, -14.0])
    k = F.FEATURE_NAMES.index("down_obst_deficit")
    assert info["dirs"]["down"]["mount_v"] == pytest.approx(vm, rel=1e-6)
    assert slow[k] > 0 > fast[k]


def test_braking_margin_sign(platforms):
    fr = platforms
    k = F.FEATURE_NAMES.index("right_brake_margin")
    slow, _ = fr.perceive([-6.0, 2.0, 0.0], [3.0, 0.0, 0.0])
    fast, _ = fr.perceive([-6.0, 2.0, 0.0], [22.0, 0.0, 0.0])
    assert slow[k] > 0 > fast[k]
    # a ball at rest is never doomed; one rolling off at speed toward the void with nothing beyond is
    vec, _ = fr.perceive([-6.0, 2.0, 0.0], [0.0, 0.0, 0.0])
    assert vec[F.FEATURE_NAMES.index("doomed")] == 0


def test_rail_tube_extraction_and_support():
    rails = tube((-20, 1.0, -1.9), (20, 1.0, -1.9)) + tube((-20, 1.0, 1.9), (20, 1.0, 1.9))
    seg, fb, st = F.extract_rails(*(np.array([t[i] for t in rails]) for i in range(3)), np.zeros(len(rails)))
    assert st["fallback_tris"] == 0 and st["resid_p95"] < 0.02
    assert seg[:, 6] == pytest.approx(0.25, abs=0.01)
    assert np.allclose(np.sort(np.unique(np.round(seg[:, 2], 2))), [-1.9, 1.9], atol=0.02)
    fr = make(box((-40, -1, -10), (-20, 0, 10)), rails, DEATH)
    # a ball between the two rails rests on both: centre at axis + sqrt((r + R)^2 - 1.9^2)
    want = 1.0 + math.sqrt((2.0 + seg[0, 6]) ** 2 - 1.9 ** 2)
    rest, n, fl = fr.support([0.0, want, 0.0])
    assert fl == 2 and rest == pytest.approx(want, abs=0.06)
    d, nx, ny, nz = fr.rail_dist([0.0, 5.0, -1.9])
    assert d == pytest.approx(4.0 - seg[0, 6], abs=0.01) and ny == pytest.approx(1.0)
    # landing onto the rails from above is reachable
    vec, _ = fr.perceive([0.0, 8.0, 0.0], [0.0, 0.0, 0.0])
    assert vec[F.FEATURE_NAMES.index("air_any")] == 1


def test_feature_vector_layout(platforms):
    vec, info = platforms.perceive([-10.0, 2.0, 0.0], [1.0, 0.0, 0.0], debug=True)
    assert vec.shape == (F.N_FEATURES,) == (len(F.FEATURE_NAMES),)
    assert list(info["named"]) == list(F.FEATURE_NAMES)
    assert np.isfinite(vec).all()
    # the slots, on known geometry: the +x edge (8 from x -6) is "right" at camera quarter 0 and "up" at quarter 1
    # (UP pushes along +x there); the 2.5 wall to +z is "up" at quarter 0 and "left" at quarter 1
    pos, vel = [-6.0, 2.0, 0.0], [0.0, 0.0, 0.0]
    q0 = dict(zip(F.FEATURE_NAMES, platforms.perceive(pos, vel, 0, 0)[0]))
    q1 = dict(zip(F.FEATURE_NAMES, platforms.perceive(pos, vel, 0, 1)[0]))
    assert reach.key_push(reach.tas.KEY_UP, 1) == pytest.approx([1.0, 0.0])
    assert q0["right_edge"] == q1["up_edge"] == pytest.approx(8.0, abs=0.6)
    assert q0["up_obst"] == q1["left_obst"] == pytest.approx(6.0, abs=0.6)
    assert q0["up_obst_deficit"] == q1["left_obst_deficit"] == F.CLIP_V
    assert q0["grounded"] == 1 and q0["sup_n_y"] == pytest.approx(1.0, abs=1e-6) and q0["sup_n_r"] == 0
    # a 30 degree slope rising to +x: support normal (-sin, cos, 0), camera right at quarter 0 is +x
    th = math.radians(30)
    ramp = [(np.array([-10.0, 0, -10]), np.array([10.0, 20 * math.tan(th), 10]), np.array([10.0, 20 * math.tan(th), -10])),
            (np.array([-10.0, 0, -10]), np.array([-10.0, 0, 10]), np.array([10.0, 20 * math.tan(th), 10]))]
    fr = make(ramp, depth=DEATH)
    y = 10 * math.tan(th) + 2.0 / math.cos(th)
    v = dict(zip(F.FEATURE_NAMES, fr.perceive([0.0, y, 0.0], [0.0, 0.0, 0.0], 0, 0)[0]))
    assert v["grounded"] == 1
    assert v["sup_n_r"] == pytest.approx(-math.sin(th), abs=0.02) and v["sup_n_y"] == pytest.approx(math.cos(th), abs=0.02)
    assert v["sup_n_f"] == pytest.approx(0.0, abs=0.02)
    with pytest.raises(ValueError):
        fr.perceive([np.nan, 0, 0], [0, 0, 0])


def test_cache_roundtrip(platforms, tmp_path):
    platforms.save(tmp_path / "f.npz")
    fr2 = F.FastReach.load(tmp_path / "f.npz")
    for p, v in (([-10.0, 8.0, 0.0], [1.0, 0.0, 2.0]), ([-6.0, 2.0, 0.0], [9.0, 0.0, 0.0])):
        assert np.array_equal(platforms.perceive(p, v)[0], fr2.perceive(p, v)[0])


# --------------------------------------------------------------------------- numba vs plain Python: identical

_SUB = r"""
import json, sys, numpy as np
sys.path.insert(0, sys.argv[1])
from ballrain import fastreach as F
assert not F.HAVE_NUMBA
fr = F.FastReach.load(sys.argv[2])
states = json.loads(sys.argv[3])
out = [fr.perceive(p, v, 0, q)[0].tolist() for p, v, q in states]
# a tiny build in plain Python
sys.path.insert(0, sys.argv[4])
from test_fastreach import box, make
small = make(box((-3, -1, -3), (3, 0, 3)) + box((-3, 0, 2), (3, 1.0, 3)))
np.savez(sys.argv[5], **{k: getattr(small, k) for k in F.FastReach.ARRAYS})
print(json.dumps(out))
"""

STATES = [([-10.0, 8.0, 0.0], [0.0, 0.0, 0.0], 0), ([-6.0, 2.0, 0.0], [1.0, 0.0, 0.0], 1),
          ([-6.0, 2.0, 0.0], [20.0, 0.0, 3.0], 2), ([-17.5, 2.0, 2.0], [0.0, 0.0, -14.0], 3),
          ([3.0, 4.0, 0.0], [2.0, 1.0, 0.0], 0), ([100.0, 5.0, 100.0], [0.0, -5.0, 0.0], 0)]


@pytest.mark.skipif(not F.HAVE_NUMBA, reason="numba not installed: only the plain-Python path exists")
def test_numba_and_plain_python_give_identical_results(platforms, tmp_path):
    platforms.save(tmp_path / "f.npz")
    want = [platforms.perceive(p, v, 0, q)[0].tolist() for p, v, q in STATES]
    env = dict(os.environ, BALLRAIN_NO_NUMBA="1")
    root = str(instances.REPO_ROOT)
    r = subprocess.run([sys.executable, "-I", "-c", _SUB, root, str(tmp_path / "f.npz"), json.dumps(STATES),
                        os.path.dirname(os.path.abspath(__file__)), str(tmp_path / "small.npz")],
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert np.array_equal(np.array(got), np.array(want))
    small = make(box((-3, -1, -3), (3, 0, 3)) + box((-3, 0, 2), (3, 1.0, 3)))
    z = np.load(tmp_path / "small.npz")
    for k in F.FastReach.ARRAYS:
        assert np.array_equal(z[k], getattr(small, k), equal_nan=z[k].dtype.kind == "f"), k


# ------------------------------------------------------------------------------------------ L1: TAS frame 4956

@pytest.fixture(scope="module")
def l1_fields(tmp_path_factory):
    if not (L1 / "level.msg").is_file():
        pytest.skip("tas-v1 dataset not built")
    from ballrain import protocol as P
    from ballrain.state import GameTracker
    tr = GameTracker(1)
    tr.merge_level(P.parse_level_message((L1 / "level.msg").read_bytes())[0])
    for m in P.parse_level_stream((L1 / "level.msg.inc").read_bytes())[0]:
        tr.merge_level(m)
    if not F.HAVE_NUMBA:
        pytest.skip("the L1 fields are built and queried with numba (plain Python is far too slow for a whole level)")
    # the shared cache only when it is current; never rewritten by the tests
    path = instances.runtime_dir() / "fastreach" / "L01.npz"
    if path.is_file():
        fr = F.FastReach.load(path)
        if fr.info.get("key") == F.FastReach.tracker_key(tr)[0]:
            return fr
    return F.FastReach.for_tracker(tr, cache_dir=tmp_path_factory.mktemp("fr"), name="L01")


def test_tas_frame_4956_landing_is_reachable(l1_fields):
    z = np.load(L1 / "raw.npz")
    p, v = z["ball_pos"][4956], z["ball_vel"][4956]
    vec, info = l1_fields.perceive(p, v, 0, 0, debug=True)
    assert info["named"]["grounded"] == 0 and info["named"]["air_any"] == 1 and info["named"]["doomed"] == 0
    # the TAS lands on the lower platform (centre ~15.1): the first support-query contact after 4956
    pos = z["ball_pos"]
    j = 4957
    while not np.isfinite(l1_fields.support(pos[j])[0]):
        j += 1
    land = pos[j]
    assert land[1] == pytest.approx(15.1, abs=0.4)
    rest, nrm, fl = l1_fields.support(land)
    ok, n_first = l1_fields.landing_reachable(p, v, (land - 2.0 * nrm)[[0, 2]], half=0.5)
    n_true = (j - 4956) / 2  # 264 fps record: one PSI every 2 frames
    assert ok and n_first <= n_true + 2


# ---------------------------------------------------------------------------------- review findings (regressions)

def test_least_effort_landing_is_minimised_over_the_contact_window(platforms):
    """A key-free fall that touches the platform edge (ball at x 1.0 / 1.5 beyond the edge at x 0) needs no keys:
    the effort is minimised over the whole contact window, not taken at its first PSI."""
    for x in (1.0, 1.5):
        vec, info = platforms.perceive([x, 8.0, 0.0], [0.0, 0.0, 0.0], debug=True)
        named = info["named"]
        assert named["air_any"] == 1 and named["air_near_effort"] == pytest.approx(0.0, abs=1e-9)
        assert named["air_near_r"] == pytest.approx(0.0, abs=1e-9)  # the landing centre is the key-free one
    # straddling two rails, falling dead centre: lands on both without keys, centre not pulled toward one rail
    rails = tube((-20, 1.0, -1.9), (20, 1.0, -1.9)) + tube((-20, 1.0, 1.9), (20, 1.0, 1.9))
    fr = make(box((-40, -1, -10), (-20, 0, 10)), rails, DEATH)
    vec, info = fr.perceive([0.0, 8.0, 0.0], [0.0, 0.0, 0.0], debug=True)
    assert info["named"]["air_near_effort"] == pytest.approx(0.0, abs=1e-9)
    assert abs(info["dbg"]["land_cz"]) < 1e-9
    # the rail landing centre is a sphere-tube contact (r + R from the axis), never inside the tube
    vec, info = fr.perceive([0.0, 5.0, -5.0], [0.0, 0.0, 2.0], debug=True)
    d = info["dbg"]
    if info["named"]["air_any"]:
        R = fr.seg[0, 6]
        near_axis = min(math.hypot(d["land_cy"] - 1.0, d["land_cz"] - zr) for zr in (-1.9, 1.9))
        assert near_axis >= 2.0 + R - 1e-6


def test_gap_to_a_same_height_platform_can_be_cleared_fast():
    """The gap criterion matches the air scan's contact model: the ball lands on the far side when its centre is still
    at or above the far top as it reaches the far edge; a fast enough flat takeoff touches the far corner from above."""
    fr = make(box((-20, -1, -10), (0, 0, 10)) + box((6, -1, -10), (26, 0, 10)), depth=DEATH)
    k = F.FEATURE_NAMES.index("right_gap_deficit")
    slow, info = fr.perceive([-6.0, 2.0, 0.0], [2.0, 0.0, 0.0], debug=True)
    fast, _ = fr.perceive([-6.0, 2.0, 0.0], [30.0, 0.0, 0.0])
    assert np.isfinite(info["dirs"]["right"]["v_need"])
    assert slow[k] > 0 > fast[k]


def test_cache_key_covers_moving_bodies_death_boxes_and_constants():
    k0 = F.FastReach.cache_key("abc", DEATH, [])
    assert k0 != F.FastReach.cache_key("abc", DEATH, [(np.zeros(3), np.ones(3))])
    assert k0 != F.FastReach.cache_key("abc", (), [])
    small = [(0, np.zeros(3), np.full(3, 4.0))]  # a small DEPT prop: not a death box, not in the key
    assert k0 == F.FastReach.cache_key("abc", DEATH + small, [])
    assert F.CONSTS == (reach.GRAVITY, reach.PSI_HZ, reach.ROLL_INERTIA, reach.UP_FACE, reach.MOUNT_CAP,
                        reach.MOUNT_FREE)


def test_sdf_is_unsigned_and_has_no_false_zero_crossings(platforms):
    assert platforms.bricks.min() >= 0
    # just above the platform top, near its edge, the distance is the height above it (no sign flip across corners)
    for x in np.linspace(-3.0, -0.6, 7):
        d = platforms.sdf([x, 1.7, 0.3])[0]
        assert d == pytest.approx(1.7, abs=0.05)


def test_no_landing_within_the_horizon_but_above_the_death_height_is_unknown_not_doomed(platforms):
    vec, info = platforms.perceive([-10.0, 300.0, 0.0], [0.0, 0.0, 0.0], debug=True)
    assert info["named"]["air_any"] == 0 and info["named"]["doomed"] == 0 and info["named"]["unknown_zone"] == 1
    assert info["dbg"]["doom_reason"] == 4


# ------------------------------------------------------------------------------------------------ fans (P_Modul_18)

def fan_at(x, y, z, yscale=1.0, name=F.FAN_ENTITY):
    """A fan push box (local x, z -1..1, y 0..16) with its base at (x, y, z); yscale stretches it vertically."""
    return (name, np.array([[1.0, 0, 0], [0, yscale, 0], [0, 0, 1.0], [x, y, z]]))


@pytest.mark.parametrize("bt, v_inf", [(1, 8.667), (0, -18.36)])
def test_fan_push_steady_speed_and_time_constant(bt, v_inf):
    """Inside a push box v' = lam v + F/m - g per PSI: paper rises to (0.5 - 0.303) / (1.5/66) = 8.67 with a time
    constant of 44 PSIs, wood falls at 18.4 instead of 22.2; the kernel's profile follows that recurrence."""
    ball = reach.BALLS[bt]
    v_ss, tau = F.fan_steady(ball)
    assert v_ss == pytest.approx(v_inf, abs=0.01)
    assert tau == pytest.approx(66.0 / ball.lin_damp)
    if bt == 1:
        assert tau == pytest.approx(44.0)
    lam, dv = ball.air_lambda, F.FAN_FORCE / ball.mass
    v = 0.0
    for n in range(3000):
        v = lam * v + dv - 20 / 66
    assert v == pytest.approx(v_ss, rel=1e-9)
    # the field-level profile in a 1600-tall push box far above the ground
    fr = make(box((-20, -1, -10), (0, 0, 10)), depth=DEATH, fans=[fan_at(100.0, -500.0, 0.0, yscale=100.0)])
    prof = fr.fan_profile([100.0, 300.0, 0.0], [0.0, 0.0, 0.0], bt)
    assert prof["inside"].all()
    vy = np.diff(prof["y"]) * 66.0  # v[n] = (y[n+1] - y[n]) * 66
    want = v_ss * (1.0 - lam ** np.arange(len(vy)))  # v[n] = lam^n v0 + a (1 - lam^n) / (1 - lam), v0 = 0
    assert np.allclose(vy, want, atol=1e-9)
    k = int(round(tau))
    assert 0.62 < vy[k] / v_ss < 0.65  # ~1 - 1/e after one time constant
    # the fan-free reference is the flight formula
    S1, S2 = F.air_tables(ball)
    assert np.allclose(prof["y_nofan"], 300.0 - 20 / 66 * S2)


def test_stone_ignores_fans_and_fan_boxes_from_the_level():
    assert F.fan_dv(reach.BALLS[2]) == 0.0 and F.fan_dv(reach.BALLS[1]) == pytest.approx(0.5)
    assert F.fan_steady(reach.BALLS[2])[0] == pytest.approx(-64.47, abs=0.01)  # (with the push it would give)

    class _E:
        def __init__(self, m):
            self.matrix = m

    class _L:
        by_name = {F.FAN_ENTITY: 1, F.FAN_ENTITY + "003": 2, F.FAN_ENTITY + "_Mesh": 3, "P_Modul_18_Rotor": 4}
        entities = {1: _E(np.eye(4, 3)), 2: _E(np.r_[np.eye(3), [[5.0, 6.0, 7.0]]]), 3: _E(np.eye(4, 3)),
                    4: _E(np.eye(4, 3))}
    fans = F.fan_boxes(_L)
    assert [n for n, _ in fans] == [F.FAN_ENTITY, F.FAN_ENTITY + "003"]
    o = F.fan_obbs(fans)
    assert o.shape == (2, F.FAN_COLS)
    assert np.allclose(o[1, :3], [5.0, 14.0, 7.0]) and np.allclose(o[1, 3:6], [1.0, 8.0, 1.0])
    # a box turned 45 degrees about y: its world AABB half-extent is sqrt 2
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    o = F.fan_obbs([("f", np.array([[c, 0, -s], [0, 1.0, 0], [s, 0, c], [0, 0, 0]]))])
    assert o[0, 15] == pytest.approx(math.sqrt(2)) and o[0, 17] == pytest.approx(math.sqrt(2))
    assert F.FastReach.cache_key("abc", DEATH, []) != F.FastReach.cache_key("abc", DEATH, [], fans=fans)


FAN_TRIS = box((-20, -1, -10), (0, 0, 10)) + box((35, -16, -5), (45, -15, 5)) + box((42, -1, -5), (50, 2, 5))


@pytest.fixture(scope="module")
def fan_level():
    """The test platform plus a fan at x 40 (push box x 39..41, z -1..1, y -15..1) with a floor under it (top -15) and
    a ledge beside its top (x 42..50, top 2), and a fan over the death box at x 80 (nothing under it)."""
    return make(FAN_TRIS, depth=DEATH, fans=[fan_at(40.0, -15.0, 0.0), fan_at(80.0, -15.0, 0.0, name=F.FAN_ENTITY + "001")])


def test_fan_lifts_paper_never_doomed_inside(fan_level):
    fr = fan_level
    nofan = make(FAN_TRIS, depth=DEATH)
    # paper inside the push box, at rest: the flight formula drops it onto the floor under the fan, the push lifts it to
    # the box top, where the ledge beside it is the key-free landing
    vec, info = fr.perceive([40.0, -5.0, 0.0], [0.0, 0.0, 0.0], 1, debug=True)
    n = info["named"]
    assert n["doomed"] == 0 and n["air_any"] == 1 and info["dbg"]["fan"] == 1
    assert info["dbg"]["near_h"] == pytest.approx(2.0, abs=0.05) and n["air_near_dy"] > 0
    prof = fr.fan_profile([40.0, -5.0, 0.0], [0.0, 0.0, 0.0], 1)
    assert prof["y"].max() > 1.0 + F.FAN_R_NOM - 0.5 and prof["y_nofan"][-1] < -60
    vec, info = nofan.perceive([40.0, -5.0, 0.0], [0.0, 0.0, 0.0], 1, debug=True)
    assert info["dbg"]["near_h"] == pytest.approx(-15.0, abs=0.05) and info["named"]["air_near_dy"] < 0
    # falling fast through the fan over the void (x 80, nothing under it): paper can be lifted before it dies, so
    # unknown, not doomed; wood (pushed 0.053 per PSI against gravity 0.303) stays below the band's upper bound (pushed
    # at every PSI), which dies with no surface in reach: doomed (fan 7, doom_reason 6); stone ignores fans. Without
    # the fans all three are doomed (the negative control of the paper case)
    for bt, doomed in ((0, 1), (1, 0), (2, 1)):
        vec, info = fr.perceive([80.0, -5.0, 0.0], [0.0, -20.0, 0.0], bt, debug=True)
        assert info["named"]["doomed"] == doomed, (bt, info["dbg"])
        if bt == 0:
            assert info["dbg"]["fan"] == 7 and info["dbg"]["doom_reason"] == 6 and info["named"]["unknown_zone"] == 0
        assert nofan.perceive([80.0, -5.0, 0.0], [0.0, -20.0, 0.0], bt)[0][1] == 1
    # far from the fan over the death box: still doomed (no reachable centre comes near a push box before it dies)
    vec, info = fr.perceive([150.0, 5.0, 150.0], [0.0, -5.0, 0.0], 1, debug=True)
    assert info["named"]["doomed"] == 1 and info["dbg"]["fan_t"] == -1


def test_fan_band_keeps_wood_alive_only_with_a_surface_in_reach():
    """Wood beside a tall fan (outside the rotation-free push radius, inside FAN_R_OUT: the key-free flight is never
    pushed, keys can bring it in): a ledge in reach of the band between no push and a push at every PSI is a possible
    landing (unknown, fan 3; the fan-free flight alone cannot reach it: doomed without fans); without the ledge the
    upper bound dies: doomed (fan 7)."""
    death = [(0, np.array([-200.0, -120.0, -200.0]), np.array([200.0, -60.0, 200.0]))]
    fan = [fan_at(0.0, -60.0, 0.0, yscale=4.0)]  # push box x, z -1..1, y -60..4
    base = box((-30, 9, -10), (-20, 10, 10))
    ledge = box((14.0, -11.0, -1.0), (15.0, -10.0, 1.0))
    p, v = [4.3, 0.0, 0.0], [0.0, 0.0, 0.0]
    assert F.FAN_R_NOM + 1.0 < 4.3 < F.FAN_R_OUT + 1.0
    vec, info = make(base + ledge, depth=death, fans=fan).perceive(p, v, 0, debug=True)
    assert vec[1] == 0 and vec[2] == 1 and vec[3] == 0 and info["dbg"]["fan"] == 3 and info["dbg"]["fan_t"] == 0
    assert make(base + ledge, depth=death).perceive(p, v, 0)[0][1] == 1  # negative control: no fans
    vec, info = make(base, depth=death, fans=fan).perceive(p, v, 0, debug=True)
    assert vec[1] == 1 and vec[2] == 0 and info["dbg"]["fan"] == 7


@pytest.mark.parametrize("bt, fan_xy, x0, code, doomed", [
    (1, (8.0, -15.0), -1.0, 5, 0),  # paper: the brake departure off the edge can be lifted by the fan
    (1, (20.0, -5.0), -4.0, 6, 0),  # paper: only the full-push jump reaches the farther fan
    (0, (8.0, -15.0), -1.0, 9, 1),  # wood: the departures reach the fan, but no key plan lands: doomed
])
def test_fan_grounded_takeoff_band(bt, fan_xy, x0, code, doomed):
    """Grounded at 20 toward the platform edge (x 0) with a fan over the void beyond it: the takeoff scans (brake
    departure, fan 5 / 8; full-push jump, fan 6 / 9) that find no landing but can reach a push box run the band test.
    Without the fan every case is doomed (the negative control)."""
    tris = box((-20, -1, -10), (0, 0, 10))
    p, v = [x0, reach.BALLS[bt].radius, 0.0], [20.0, 0.0, 0.0]
    vec, info = make(tris, depth=DEATH, fans=[fan_at(fan_xy[0], fan_xy[1], 0.0)]).perceive(p, v, bt, debug=True)
    assert vec[0] == 1 and info["dbg"]["fan"] == code and vec[1] == doomed, info["dbg"]
    vec0, info0 = make(tris, depth=DEATH).perceive(p, v, bt, debug=True)
    assert vec0[0] == 1 and vec0[1] == 1 and info0["dbg"]["doom_reason"] == 2


def test_fan_hover_lands_only_in_reach_of_the_column():
    """Paper hovering at the top of the void fan (x 80; key-free: exit, fall back, pushed again until the horizon end)
    with a ledge at the hover height: beside the column it is a landing early in the hover (not at the horizon end);
    10 away it is not (a centre steered out of the column is not pushed and falls), so no landing: unknown. The far
    ledge is inside the key square at the hover height, i.e. a profile applied to every offset would land on it."""
    p, v = [80.0, 3.0, 0.0], [0.0, 0.0, 0.0]
    fans = [fan_at(80.0, -15.0, 0.0)]
    base = box((-20, -1, -10), (0, 0, 10))
    near = make(base + box((83.0, 1.0, -5), (88.0, 2.0, 5)), depth=DEATH, fans=fans)
    vec, info = near.perceive(p, v, 1, debug=True)
    assert vec[3] == 1 and info["dbg"]["fan"] == 1 and info["dbg"]["near_h"] == pytest.approx(2.0)
    assert info["dbg"]["n_near"] < 200 and info["dbg"]["n_first"] < 100
    far = make(base + box((90.0, 1.0, -5), (95.0, 2.0, 5)), depth=DEATH, fans=fans)
    vec, info = far.perceive(p, v, 1, debug=True)
    assert vec[3] == 0 and vec[1] == 0 and vec[2] == 1 and info["dbg"]["fan"] == 3
    prof = far.fan_profile(p, v, 1)
    b = reach.BALLS[1]
    assert prof["inside"][-5:].any() and prof["y"][100:].min() > 2.0 + 1.0  # still hovering above the ledge at the end
    assert b.air_accel * F.air_tables(b)[1][-1] > 10.0  # the far ledge is within the key square


def test_fan_collect_holds_the_reported_scans_landings():
    """collect: fan code 2 (the fan-free landing kept) returns the fan-free scan's landings (as a fan-free build),
    fan code 1 those of the pushed scan only (none before its first landing PSI)."""
    fans = [fan_at(80.0, -15.0, 0.0)]
    tris = box((-20, -1, -10), (0, 0, 10)) + box((86.0, -4.0, -5), (91.0, -3.0, 5))
    fr, nofan = make(tris, depth=DEATH, fans=fans), make(tris, depth=DEATH)
    p, v = [80.0, 3.0, 0.0], [0.0, 0.0, 0.0]
    _, info = fr.perceive(p, v, 1, debug=True, collect=4000)
    _, info0 = nofan.perceive(p, v, 1, debug=True, collect=4000)
    assert info["dbg"]["fan"] == 2 and len(info["landings"]) > 0
    assert np.array_equal(info["landings"], info0["landings"])
    _, info = fr.perceive([80.0, -5.0, 0.0], [0.0, 0.0, 0.0], 1, debug=True, collect=4000)  # lifted into the hover
    tris1 = box((-20, -1, -10), (0, 0, 10)) + box((83.0, 1.0, -5), (88.0, 2.0, 5))
    _, info = make(tris1, depth=DEATH, fans=fans).perceive([80.0, -5.0, 0.0], [0.0, 0.0, 0.0], 1, debug=True,
                                                            collect=4000)
    assert info["dbg"]["fan"] == 1 and len(info["landings"]) > 0
    assert info["landings"][:, 3].min() >= info["dbg"]["n_first"]


def test_fan_grounded_inside_is_never_doomed(fan_level):
    fr = fan_level
    nofan = make(FAN_TRIS, depth=DEATH)
    # paper on the floor under the fan (inside its push box): lifted off, so unknown; never doomed. Fast toward the
    # floor's edge the fan-free build calls it doomed (the negative control)
    for vx in (12.0, 30.0):
        vec, info = fr.perceive([40.0, -15.0 + 2.004, 0.0], [vx, 0.0, 0.0], 1, debug=True)
        n = info["named"]
        assert n["grounded"] == 1 and n["doomed"] == 0 and n["unknown_zone"] == 1 and info["dbg"]["fan"] == 4
    assert nofan.perceive([40.0, -15.0 + 2.004, 0.0], [30.0, 0.0, 0.0], 1)[0][1] == 1
    # wood there is not lifted (net push down): grounded, not doomed, not unknown
    vec, info = fr.perceive([40.0, -13.0, 0.0], [12.0, 0.0, 0.0], 0, debug=True)
    assert info["named"]["grounded"] == 1
    assert info["named"]["doomed"] == 0 and info["named"]["unknown_zone"] == 0


def test_unknown_zone_clears_doomed(platforms):
    """unknown_zone wins: a ball doomed by the fields is not doomed when a moving body is within UNKNOWN_R (the
    perceive bodies check) or inside an unknown zone; dbg doom_reason keeps the reason."""
    p, v = [100.0, 5.0, 100.0], [0.0, -5.0, 0.0]
    vec, info = platforms.perceive(p, v, 1, debug=True)
    assert vec[1] == 1 and vec[2] == 0
    vec, info = platforms.perceive(p, v, 1, debug=True, bodies=[[100.0, 0.0, 103.0]])
    assert vec[1] == 0 and vec[2] == 1 and info["dbg"]["doom_reason"] == 1
    old = platforms.uzones
    try:
        platforms.set_unknown_boxes([((95.0, 0.0, 95.0), (105.0, 10.0, 105.0))])
        vec, info = platforms.perceive(p, v, 1, debug=True)
        assert vec[1] == 0 and vec[2] == 1 and info["dbg"]["doom_reason"] == 1
    finally:
        platforms.uzones = old


def test_fan_free_levels_are_unchanged(platforms):
    """No fans: the fan code is never reached: dbg fan stays nan."""
    assert platforms.fans.shape == (0, F.FAN_COLS)
    vec, info = platforms.perceive([100.0, 5.0, 100.0], [0.0, -5.0, 0.0], 1, debug=True)
    assert info["named"]["doomed"] == 1 and info["dbg"]["fan"] != info["dbg"]["fan"]


_SUB_FAN = r"""
import json, sys, numpy as np
sys.path.insert(0, sys.argv[1])
from ballrain import fastreach as F
assert not F.HAVE_NUMBA
fr = F.FastReach.load(sys.argv[2])
print(json.dumps([fr.perceive(p, v, bt, 0)[0].tolist() for p, v, bt in json.loads(sys.argv[3])]))
"""
FAN_STATES = [([40.0, -5.0, 0.0], [0.0, 0.0, 0.0], 1), ([40.0, -5.0, 0.0], [0.0, -20.0, 0.0], 0),
              ([40.0, -5.0, 0.0], [0.0, -20.0, 0.0], 2), ([40.0, -12.996, 0.0], [12.0, 0.0, 0.0], 1),
              ([36.0, -12.996, 0.0], [-9.0, 0.0, 0.0], 1), ([44.0, 6.0, 0.0], [3.0, 1.0, 0.0], 1),
              ([80.0, -5.0, 0.0], [0.0, -20.0, 0.0], 0), ([80.0, 3.0, 0.0], [0.0, 0.0, 0.0], 1),
              ([-1.0, 2.0, 0.0], [20.0, 0.0, 0.0], 0), ([-1.0, 2.0, 0.0], [30.0, 0.0, 0.0], 1)]


@pytest.mark.skipif(not F.HAVE_NUMBA, reason="numba not installed: only the plain-Python path exists")
def test_fan_numba_and_plain_python_identical(fan_level, tmp_path):
    fan_level.save(tmp_path / "fan.npz")
    want = [fan_level.perceive(p, v, bt, 0)[0].tolist() for p, v, bt in FAN_STATES]
    env = dict(os.environ, BALLRAIN_NO_NUMBA="1")
    r = subprocess.run([sys.executable, "-I", "-c", _SUB_FAN, str(instances.REPO_ROOT), str(tmp_path / "fan.npz"),
                        json.dumps(FAN_STATES)], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    assert np.array_equal(np.array(json.loads(r.stdout.strip().splitlines()[-1])), np.array(want))


def test_load_of_a_file_without_fans(platforms, tmp_path):
    """A cache file written before BUILD_VERSION 5 has no fans array: it loads (no fans); its key never matches."""
    np.savez(tmp_path / "old.npz", **{k: getattr(platforms, k) for k in F.FastReach.ARRAYS if k != "fans"})
    old = F.FastReach.load(tmp_path / "old.npz")
    assert old.fans.shape == (0, F.FAN_COLS)
    assert np.array_equal(old.perceive([-10.0, 8.0, 0.0], [1.0, 0.0, 2.0])[0],
                          platforms.perceive([-10.0, 8.0, 0.0], [1.0, 0.0, 2.0])[0])


# ------------------------------------------------------------------------------------------ L2: the TAS fan stretch

@pytest.fixture(scope="module")
def l2_fields(tmp_path_factory):
    if not (L2 / "level.msg").is_file():
        pytest.skip("tas-v1 dataset not built")
    if not F.HAVE_NUMBA:
        pytest.skip("the L2 fields are built and queried with numba (plain Python is far too slow for a whole level)")
    from ballrain import protocol as P
    from ballrain.state import GameTracker
    tr = GameTracker(2)
    tr.merge_level(P.parse_level_message((L2 / "level.msg").read_bytes())[0])
    for m in P.parse_level_stream((L2 / "level.msg.inc").read_bytes())[0]:
        tr.merge_level(m)
    tr.level_no = 2
    fr = F.FastReach.for_tracker(tr, cache_dir=tmp_path_factory.mktemp("fr2"), name="L02")
    states = {}
    for i, o in enumerate(dataset.load_observations(L2 / "raw.npz")):
        tr.update(o)
        if not (18900 <= i <= 22400) or o.ball is None:
            continue
        p = np.asarray(o.ball["pos"], float)
        if np.abs(p).sum() < 1e-6:
            continue
        bt = tr.ball_type()
        states[i] = (p, np.asarray(o.ball["speed"], float) + np.asarray(o.ball["speed_change"], float),
                     0 if bt is None else int(bt))
    return fr, states


def test_l2_fans_found(l2_fields):
    fr, _ = l2_fields
    assert fr.fans.shape == (9, F.FAN_COLS)
    assert np.allclose(fr.fans[:, 3:6], [1.0, 8.0, 1.0], atol=1e-3)
    names = [f["name"] for f in fr.info["fans"]]
    assert names[0] == F.FAN_ENTITY and names[-1] == F.FAN_ENTITY + "008"


def test_l2_tas_fan_stretch_never_doomed_and_vertical_motion(l2_fields):
    """TAS 18900-22400 rides fans 003-006 and 008 and passes 001 (paper): no doomed flag (a regression guard only: the
    fan-free fields flag none there either); the fans lift the least-effort landing (air_near_dy more than 1 above the
    fan-free one) for hundreds of states, none of them at the horizon end (the hover envelope pushed 374 there);
    inside fan 005 (frames 20720-20960) the key-free vertical motion with the push matches the TAS over 0.5 s, the
    flight formula misses by ~3."""
    fr, st = l2_fields
    arrays = {k: getattr(fr, k) for k in F.FastReach.ARRAYS}
    arrays["fans"] = np.zeros((0, F.FAN_COLS))
    nofan = F.FastReach(arrays, dict(fr.info))
    k = {n: F.FEATURE_NAMES.index(n) for n in ("doomed", "air_any", "air_near_dy", "air_t_near")}
    doomed, lifted, late = [], 0, 0
    for i, (p, v, bt) in st.items():
        a, b = fr.perceive(p, v, bt, 0)[0], nofan.perceive(p, v, bt, 0)[0]
        if a[k["doomed"]] > .5:
            doomed.append(i)
        if a[k["air_any"]] > .5:
            late += a[k["air_t_near"]] * 132 >= 390
            lifted += b[k["air_any"]] > .5 and a[k["air_near_dy"]] - b[k["air_near_dy"]] > 1.0
    assert doomed == []
    assert lifted > 150 and late == 0, (lifted, late)
    psi = sorted(st)
    psi = [psi[0]] + [k for a, k in zip(psi, psi[1:]) if np.any(st[k][1] != st[a][1])]  # a new PSI: speed changes
    errs, errs0 = [], []
    for j, i0 in enumerate(psi):
        if not (20760 <= i0 <= 20880) or psi[j + 33] - i0 != 66:
            continue
        p, v, bt = st[i0]
        prof = fr.fan_profile(p, v, bt)
        y = np.array([st[psi[j + n]][0][1] for n in range(34)])
        errs.append(np.abs(prof["y"][:34] - y).max())
        errs0.append(np.abs(prof["y_nofan"][:34] - y).max())
    assert len(errs) > 20
    assert np.median(errs) < 0.01 and max(errs) < 0.5
    assert min(errs0) > 2.0
