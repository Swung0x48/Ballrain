"""Geometry and feature tests on a hand-built level (no game)."""
import math

import numpy as np
import pytest

from ballrain import features as FX, geometry as G, protocol as P
from ballrain.fakegame import box_mesh
from ballrain.state import GameTracker

FLOOR, RAIL, STOPPER, BOX, STONE = 10, 11, 12, 13, 14


def quad(x0, x1, z0, z1, y=0.0):
    V = np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]], np.float32)
    return V, np.array([[0, 2, 1], [0, 3, 2]], np.uint32)


def make_tracker(cam_forward=(0, 0, 1)):
    lm = P.LevelMessage(0)
    I43 = np.zeros((4, 3), np.float32)
    lm.entities = [P.Entity(FLOOR, "Floor", 41, 0, True, I43, 2, 2, b""),
                   P.Entity(RAIL, "Rail", 41, 0, True, I43, 3, 3, b""),
                   P.Entity(STOPPER, "Stopper", 41, 0, True, I43, 0, 0, b""),
                   P.Entity(BOX, "P_Box_MF", 41, 0, True, I43, 0, 1, b""),
                   P.Entity(STONE, "Stone", 41, 0, True, I43, 1, 1, b"")]
    lm.groups = [P.Group(1, "Phys_Floors", np.array([FLOOR, STONE], np.uint32)),
                 P.Group(2, "Phys_FloorRails", np.array([RAIL], np.uint32)),
                 P.Group(3, "Phys_FloorStopper", np.array([STOPPER], np.uint32))]
    lm.phys = [P.PhysRecord(0, 0, FLOOR, False, True), P.PhysRecord(1, 0, RAIL, False, True),
               P.PhysRecord(2, 0, STOPPER, False, True), P.PhysRecord(3, 0, BOX, False, False),
               P.PhysRecord(4, 0, STONE, False, True)]
    eye = np.eye(4, dtype=np.float32)
    fv, ff = quad(-5, 5, 0, 20)  # road x in [-5, 5], z in [0, 20], y = 0
    sv, sf = quad(-5, 5, 20, 30, y=-0.5)  # a lower stone step
    rv, rf = box_mesh((0.1, 0.1, 10.0))
    stv, stf = box_mesh((0.5, 3.0, 0.5))
    bv, bf = box_mesh((1.0, 1.0, 1.0))
    rail_m = eye.copy()
    rail_m[3, :3] = (6.0, 0.5, 10.0)
    stop_m = eye.copy()
    stop_m[3, :3] = (0.0, 0.0, 5.0)  # a stopper right on the road: must be ignored
    lm.meshes = [P.Mesh(FLOOR, 100, 1, eye, fv, ff), P.Mesh(STONE, 104, 1, eye, sv, sf),
                 P.Mesh(RAIL, 101, 1, rail_m, rv, rf), P.Mesh(STOPPER, 102, 1, stop_m, stv, stf),
                 P.Mesh(BOX, 103, 0, eye, bv, bf)]
    lm.depth = [(99, np.array([-100, -60, -100], np.float32), np.array([100, -50, 100], np.float32))]
    lm.arrays = [P.ArrayDef(500, "ResetPoints", [P.ArrayColumn("Resetpoint", 5, "Matrix")],
                            [[_m44((0, 2, 0)).tobytes()]]),
                 P.ArrayDef(501, "Checkpoints", [P.ArrayColumn("Matrix", 5, "Matrix"), P.ArrayColumn("Object", 5, None)],
                            [[_m44((0, 2, 20)).tobytes(), b"\0\0\0\0"]]),
                 P.ArrayDef(502, "Energy", [P.ArrayColumn(n, 1, None) for n in ("Points", "Lifes", "S", "L", "T", "LifeBonus")],
                            [[np.int32(v).tobytes() for v in (950, 3, 1000, 3, 500, 200)]])]
    lm.entities.append(P.Entity(20, "PE_Balloon_Platform", 41, 0, True, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 2, 40]], np.float32), 0, 0, b""))
    t = GameTracker(1)
    t.merge_level(lm)
    t.flags.ball_nav = True
    f = np.asarray(cam_forward, float)
    f /= np.linalg.norm(f)
    r = np.array([f[2], 0, -f[0]])
    t._cam = np.array([r, [0, 1, 0], f, [0, 0, 0]], np.float32)
    return t


def _m44(pos):
    m = np.eye(4, dtype=np.float32)
    m[3, :3] = pos
    return m


def obs_at(t, pos, vel=(0, 0, 0), gap=None, partner=FLOOR, normal=(0, 1, 0), fric=False, bodies=(), radius=2.0, btype=0,
           mind=None):
    """mind: extra MIND entries [(partner, gap, normal, self_status)]; gap / partner / normal give one entry of a
    sphere (self_status 3)."""
    o = P.Observation(frames_done=100)
    b = np.zeros(1, P.BALL_DT)[0]
    b["pos"], b["speed"], b["found"], b["type"], b["extra_radius"] = pos, vel, 1, btype, radius
    o.ball = b
    rows = ([(partner, gap, normal, FX.IVP_ST_BALL)] if gap is not None else []) + list(mind or [])
    if rows:
        m = np.zeros(len(rows), P.MIND_DT)
        for k, (pk, gk, nk, st) in enumerate(rows):
            m["partner"][k], m["gap"][k], m["plane"][k], m["self_status"][k] = pk, gk, (*nk, 0), st
        o.mind = m
    if fric:
        f = np.zeros(1, P.FRIC_DT)
        f["partner"], f["gap"] = partner, 0.013
        o.fric = f
    if bodies:
        o.body = np.array(bodies, P.BODY_DT)
    c = np.zeros(1, P.CAMS_DT)[0]
    for i in range(5):
        c["cams"][i]["matrix"] = t._cam
    o.cams = c
    t.obs = o
    return o


def test_quaternion_and_transforms():
    q = (0, math.sin(math.pi / 4), 0, math.cos(math.pi / 4))  # +90 deg about y
    R = G.quat_matrix(q)
    assert np.allclose(R @ [1, 0, 0], [0, 0, -1])
    v = np.array([[1.0, 0, 0]])
    assert np.allclose(G.body_to_world(v, (10, 0, 0), q, (0, 0, 0, 1), (0, 1, 0)), [[10, 1, -1]])
    m = np.eye(4)
    m[3, :3] = (1, 2, 3)
    m[:3, :3] = R.T  # CK row-vector matrix of the same rotation
    assert np.allclose(G.ck_to_world(v, m), [[1, 2, 2]])


def test_closest_point_regions():
    A, B, C = np.array([[0.0, 0, 0]]), np.array([[1.0, 0, 0]]), np.array([[0.0, 0, 1]])
    for p, q, reg in (([0.2, 1, 0.2], [0.2, 0, 0.2], 2), ([0.5, 1, -1], [0.5, 0, 0], 1), ([-1, 1, -1], [0, 0, 0], 0)):
        Q, r = G.closest_point_triangles(np.array(p, float), A, B, C)
        assert np.allclose(Q[0], q) and r[0] == reg


def test_ray_down_and_grid():
    t = make_tracker()
    geo = FX.LevelGeometry(t)
    assert STOPPER not in set(geo.tris.owner.tolist())  # stoppers never collide with the player ball
    y, tri = geo.tris.ray_down(np.array([[0, 5, 10], [0, 5, 25], [0, 5, 50], [0, -1, 10]], float))
    assert y[0] == 0 and y[1] == -0.5 and np.isnan(y[2]) and np.isnan(y[3])
    assert geo.tris.owner[tri[0]] == FLOOR and tri[2] == -1
    c = geo.tris.closest(np.array([1, 2.01, 10.0]))
    assert c["owner"] == FLOOR and c["d"] == pytest.approx(2.01) and c["region"] == 2


def test_boundary_edges_mesh_method():
    t = make_tracker()
    geo = FX.LevelGeometry(t)
    assert geo.edges.distance(np.array([3.0, 0, 10])) == pytest.approx(2.0)  # road side at x = 5
    assert geo.edges.distance(np.array([0.0, 0, 1])) == pytest.approx(1.0)  # road start at z = 0


def test_contact_features_and_surface():
    t = make_tracker()
    fx = FX.FeatureExtractor(FX.FeatureConfig(fan_dirs=4, height_grid=3, bodies=0))
    obs_at(t, (3, 2.01, 10), vel=(0, 0, 5), gap=0.01, fric=True)
    v, info = fx.compute(t)
    s = fx.slices()
    assert v[s["valid"]][0] == 1 and v[s["geom_contact"]][0] == 1 and v[s["fric_contact"]][0] == 1
    assert v[s["surface"]].tolist() == [0, 0, 1, 0, 0]  # wood (Continuous Contact ID 2)
    assert np.allclose(v[s["normal_cam"]], [0, 1, 0])
    assert info["contact_point"] == pytest.approx([3, 0, 10], abs=1e-6)
    # fan: 0 forward (+z), 1 right (+x), 2 back, 3 left (step 0.25: the first sample past the edge)
    assert info["edge_fan"].tolist() == [8.0, 2.25, 8.0, 8.0]
    assert info["edge_mesh"] == pytest.approx(2.0)
    hs = v[s["heights"]].reshape(3, 3) * fx.cfg.height_clip
    assert np.allclose(hs, -0.01, atol=1e-5)  # all nine samples on the road, ball bottom at y = 0.01
    assert v[s["vel_cam"]][2] == pytest.approx(5 / fx.cfg.vel_scale)
    assert v[s["path_progress"]][0] == pytest.approx(10 / 40, abs=0.01)
    # rail on the right: 6 - 0.1 = 5.9 in x, ball at x=3, radius 2
    assert v[s["rail_dist"]][0] == pytest.approx((np.hypot(2.9, 1.4) - 2) / fx.cfg.rail_radius, abs=1e-3)
    assert v[s["rail_dir"]][0] > 0.8  # to the right
    # stone step under a ball beyond z = 20
    obs_at(t, (0, 1.51, 25), gap=0.01, partner=STONE, fric=True)
    v, info = fx.compute(t)
    assert info["surface"] == "stone"
    assert info["edge_fan"][0] == pytest.approx(5.25)  # stone ends at z = 30
    assert info["edge_fan"][2] == 8.0  # behind: the higher road is not a drop


def test_camera_frame_rotation():
    t = make_tracker(cam_forward=(1, 0, 0))  # camera looks along +x
    fx = FX.FeatureExtractor(FX.FeatureConfig(fan_dirs=4, height_grid=0, bodies=0))
    obs_at(t, (0, 2.01, 10), vel=(4, 0, 0), gap=0.01)
    v, info = fx.compute(t)
    s = fx.slices()
    assert v[s["vel_cam"]][2] == pytest.approx(4 / fx.cfg.vel_scale)  # forward in camera frame
    assert v[s["vel_cam"]][0] == pytest.approx(0, abs=1e-6)
    assert info["edge_fan"][0] == pytest.approx(5.25)  # forward = +x: road edge at x = 5
    assert v[s["cam_yaw_ref"]].tolist() == pytest.approx([math.cos(math.pi / 2), 1.0], abs=1e-6)


def test_no_contact_paper_ball_and_bodies():
    t = make_tracker()
    fx = FX.FeatureExtractor(FX.FeatureConfig(fan_dirs=0, height_grid=3, bodies=2))
    body = (BOX, 1, (0.0, 1.0, 14.0), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0), (1, 0, 0), (0, 0, 0), np.zeros((4, 3)))
    obs_at(t, (0, 5, 10), radius=0.0, btype=1, bodies=[body])  # paper ball: extra_radius 0, no mindists
    v, info = fx.compute(t)
    s = fx.slices()
    assert v[s["ball_type"]].tolist() == [0, 1, 0]
    assert v[s["geom_contact"]][0] == 0 and v[s["min_gap"]][0] == FX.NO_CONTACT_GAP
    assert info["surface"] == "none"
    hs = v[s["heights"]].reshape(3, 3) * fx.cfg.height_clip
    assert hs[1, 1] == pytest.approx(-3.0)  # floor at 0, ball bottom at 5 - 2 (radius fallback 2)
    b = v[s["bodies"]].reshape(2, 7)
    assert b[0, 0] == 1 and b[1, 0] == 0
    assert b[0, 1:4] == pytest.approx(np.array([0, -4, 4]) / fx.cfg.pos_scale)
    # the moving box (top at y = 2, z 13..15) is in the height grid
    fx2 = FX.FeatureExtractor(FX.FeatureConfig(fan_dirs=0, height_grid=3, height_spacing=4.0, bodies=0))
    obs_at(t, (0, 5, 10), bodies=[body])
    v2, _ = fx2.compute(t)
    hs2 = v2[fx2.slices()["heights"]].reshape(3, 3) * fx2.cfg.height_clip
    assert hs2[1, 2] == pytest.approx(2 - 3) and hs2[1, 1] == pytest.approx(-3)


def test_invalid_frames_and_depth_margin():
    t = make_tracker()
    fx = FX.FeatureExtractor()
    o = obs_at(t, (0, 2, 10))
    o.ball["found"] = 0
    v, _ = fx.compute(t)
    assert v[fx.slices()["valid"]][0] == 0
    obs_at(t, (0, 2, 10))
    v, _ = fx.compute(t)
    assert v[fx.slices()["depth_margin"]][0] == pytest.approx((2 + 50) / fx.cfg.pos_scale)
    assert v.shape == (fx.size,) and v.dtype == np.float32
    assert len(FX.feature_names(fx)) == fx.size


def _events_obs(frame, *events, ball_found=1):
    o = P.Observation(frames_done=frame)
    o.events = np.array([(frame, c, p) for c, p in events], P.EVT_DT)
    b = np.zeros(1, P.BALL_DT)[0]
    b["found"], b["type"], b["extra_radius"] = ball_found, 0, 2.0
    o.ball = b
    return o


def test_transforming_follows_the_ball_unphysicalize_physicalize_window():
    # f3: a transformation is the window from the Unphysicalize of the active ball (a Ball_* entity) to the
    # Physicalize of the new ball (HS03 5201 -> 5800). IngameParameter[0,0] (Activetrafo) keeps the last
    # trafo's ckid for the rest of the level and BallNav is also inactive in pauses and after LevelFinish:
    # neither makes a transformation (f2 got this wrong).
    t = make_tracker()
    lm = P.LevelMessage(0)
    lm.entities = [P.Entity(30, "Ball_Wood", 41, 0, True, np.zeros((4, 3), np.float32), -1, -1, b""),
                   P.Entity(31, "Ball_Paper", 41, 0, True, np.zeros((4, 3), np.float32), -1, -1, b"")]
    lm.phys = [P.PhysRecord(10, 0, 30, False), P.PhysRecord(11, 0, 30, True), P.PhysRecord(12, 0, 31, False),
               P.PhysRecord(13, 0, BOX, True), P.PhysRecord(14, 0, 31, True), P.PhysRecord(15, 0, 31, False)]
    t.merge_level(lm)
    t.arrays.define(P.ArrayDef(503, "IngameParameter", [P.ArrayColumn("Activetrafo", 4, None)],
                               [[np.uint32(45354).tobytes()]]))
    E = P.Ev
    t.update(_events_obs(10, (E.StartLevel, 0), (E.BallNavActive, 0)))
    assert not t.transforming and t.flags.ball_nav
    t.update(_events_obs(20, (E.PauseLevel, 0), (E.BallNavInactive, 0)))  # Activetrafo set, no BallNav
    assert t.flags.paused and not t.transforming
    t.update(_events_obs(30, (E.UnpauseLevel, 0), (E.BallNavActive, 0)))
    t.update(_events_obs(40, (E.BallNavInactive, 0), (E.Unphysicalize, 13)))  # some other object
    assert not t.transforming
    t.update(_events_obs(41, (E.Unphysicalize, 11), ball_found=0))  # the old ball leaves the physics
    assert t.transforming and not t.control_valid()
    fx = FX.FeatureExtractor()
    assert fx.compute(t)[0][fx.slices()["valid"]][0] == 0
    t.update(_events_obs(50, (E.ActiveBallChanged, 0), (E.Physicalize, 12)))
    assert not t.transforming
    t.update(_events_obs(51, (E.BallNavActive, 0)))
    t.update(_events_obs(60, (E.BallOff, 0), (E.BallNavInactive, 0)))
    t.update(_events_obs(70, (E.Unphysicalize, 14), ball_found=0))  # a death is not a transformation
    assert not t.transforming
    t.update(_events_obs(80, (E.Physicalize, 15), (E.BallNavActive, 0)))
    t.update(_events_obs(90, (E.LevelFinish, 0), (E.BallNavInactive, 0)))
    assert not t.transforming


def test_edge_fan_reference_airborne_side_contact_and_void():
    """f5: the fan is referenced to the floor support, else the floor under the ball centre, else nothing."""
    t = make_tracker()
    fx = FX.FeatureExtractor(FX.FeatureConfig(fan_dirs=4, height_grid=0, bodies=0))
    ground = [8.0, 2.25, 8.0, 8.0]  # rolling at x = 3: the road edge (x = 5) to the right
    obs_at(t, (3, 2.01, 10), gap=0.01)
    _, info = fx.compute(t)
    assert info["fan_ref"] == "support" and info["edge_fan"].tolist() == ground and info["supported"]
    # airborne 4 above the road (no mindist): f4 compared the samples with the ball bottom (all 0.25)
    obs_at(t, (3, 6.0, 10))
    _, info = fx.compute(t)
    assert info["fan_ref"] == "below" and info["edge_fan"].tolist() == ground and not info["supported"]
    # nearest contact is a wall on the left (normal +x), the floor is touched too: the floor is the support
    obs_at(t, (3, 2.01, 10), gap=0.005, partner=BOX, normal=(1, 0, 0), mind=[(FLOOR, 0.01, (0, 1, 0), 3)])
    _, info = fx.compute(t)
    assert info["normal"].tolist() == [1, 0, 0] and info["fan_ref"] == "support"
    assert info["support_point"] == pytest.approx([3, 0, 10], abs=1e-6) and info["edge_fan"].tolist() == ground
    # only a side contact (pushed against a box): the floor under the centre
    obs_at(t, (3, 2.01, 10), gap=0.01, partner=BOX, normal=(1, 0, 0))
    _, info = fx.compute(t)
    assert info["fan_ref"] == "below" and info["edge_fan"].tolist() == ground and not info["supported"]
    # over the void (z < 0, the road starts at z = 0): no reference, only misses count
    obs_at(t, (0, 2.0, -3))
    _, info = fx.compute(t)
    assert info["fan_ref"] == "none" and info["edge_fan"].tolist() == [0.25] * 4


def test_fan_edges_without_reference_counts_only_misses():
    cfg = FX.FeatureConfig(fan_dirs=2)
    steps = np.array([0.25, 0.5, 0.75])
    y = np.array([[-3.0, -3.0, np.nan], [-3.0, -5.0, -5.0]])
    assert FX.fan_edges(y, float("nan"), steps, cfg).tolist() == [0.75, 0.5]  # the -2 drop still counts
    assert FX.fan_edges(y, 0.0, steps, cfg).tolist() == [0.25, 0.25]


def test_paper_ball_mindist_normals_are_oriented_towards_the_ball():
    """The paper ball (polyhedron synapses) gets reversed mindist planes; a sphere's plane is kept as it is."""
    t = make_tracker()
    fx = FX.FeatureExtractor(FX.FeatureConfig(fan_dirs=4, height_grid=0, bodies=0))
    obs_at(t, (3, 2.01, 10), radius=0.0, btype=1, mind=[(FLOOR, 0.01, (0, -1, 0), 1)])
    v, info = fx.compute(t)
    assert info["normal"].tolist() == [0, 1, 0] and info["fan_ref"] == "support"
    assert v[fx.slices()["normal_cam"]].tolist() == pytest.approx([0, 1, 0])
    assert info["edge_fan"].tolist() == [8.0, 2.25, 8.0, 8.0]
    # a sphere under a ceiling keeps its downward normal (no flip for self_status 3)
    o = obs_at(t, (3, 2.01, 10), gap=0.01, normal=(0, -1, 0))
    assert FX.mind_normals(t, o.ball["pos"], o.mind)[0].tolist() == [0, -1, 0]


def test_dynamic_triangles_footprint_cull_is_exact():
    t = make_tracker()
    near = (BOX, 1, (4.5, 0.5, 12.0), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0), (1, 0, 0), (0, 0, 0), np.zeros((4, 3)))
    far = (BOX, 1, (30.0, 0.5, 12.0), (0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0), (1, 0, 0), (0, 0, 0), np.zeros((4, 3)))
    obs_at(t, (0, 2.01, 10), bodies=[near, far])
    pos = np.array([0, 2.01, 10.0])
    full = FX.dynamic_triangles(t, pos, 60.0)
    box = (np.array([-4.0, 6.0]), np.array([4.0, 14.0]))
    cut = FX.dynamic_triangles(t, pos, 60.0, box)
    assert len(full) == 24 and 0 < len(cut) < 12  # the far box is skipped, the near one (x 3.5..5.5) loses its +x face
    g = np.stack(np.meshgrid(np.linspace(-4, 4, 33), np.linspace(6, 14, 33)), -1).reshape(-1, 2)
    P = np.c_[g[:, 0], np.full(len(g), 5.0), g[:, 1]]
    assert np.array_equal(full.ray_down(P)[0], cut.ray_down(P)[0], equal_nan=True)
