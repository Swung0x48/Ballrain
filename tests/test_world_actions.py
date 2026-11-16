"""World-direction actions (EnvConfig.world_actions), the stale-key re-press (repress_stale_keys) and the camera guard."""
import json
import math
from pathlib import Path

import numpy as np

from ballrain import fakegame, tas
from ballrain.env import ARROWS9, BallanceEnv, EnvConfig, world_to_camera_keys
from ballrain.features import FeatureConfig, yaw_basis

U, D, L, R = tas.KEY_UP, tas.KEY_DOWN, tas.KEY_LEFT, tas.KEY_RIGHT


def cam(yaw_deg):
    """A CK matrix (rows x, y, z axis, pos) whose z axis has yaw atan2(z.x, z.z) = yaw."""
    y = math.radians(yaw_deg)
    z = np.array([math.sin(y), 0.0, math.cos(y)])
    x = np.array([z[2], 0.0, -z[0]])
    return np.array([x, [0, 1, 0], z, [0, 0, 0]])


def world_dir(keys):
    return np.array([bool(keys & R) - bool(keys & L), bool(keys & U) - bool(keys & D)])


def test_world_keys_push_in_the_world_direction_under_every_camera():
    for yaw in (0, 90, 180, -90):
        right, fwd = yaw_basis(cam(yaw))
        for a in ARROWS9:
            k = world_to_camera_keys(a, cam(yaw))
            # the force the keys give: up/down along the camera forward, right/left along the camera right
            push = (bool(k & U) - bool(k & D)) * fwd + (bool(k & R) - bool(k & L)) * right
            assert np.allclose(push[[0, 2]], world_dir(a)), (yaw, a, k)
    assert world_to_camera_keys(U, cam(0)) == U  # camera looking along +z: identity
    assert world_to_camera_keys(U | tas.KEY_SHIFT, cam(180)) == D | tas.KEY_SHIFT


def make_env(tmp_path, **kw):
    cfg = EnvConfig(level=1, out_root=str(tmp_path), action_repeat=4, world_actions=True,
                    features=FeatureConfig(camera="world"), **kw)
    return BallanceEnv(cfg, launcher=fakegame.FakeLauncher())


def test_stale_key_is_released_for_one_frame(tmp_path):
    env = make_env(tmp_path)
    try:
        env.reset()
        q = env._cam_quarter()
        assert q is not None
        a = ARROWS9.index(U)
        keys = world_to_camera_keys(U, env.session.tracker.obs.cams["cams"][0]["matrix"])
        _, _, _, _, info = env.step(a)
        assert "repressed" not in info and env._held == keys and env._press_q[keys] == q
        env._press_q[keys] = (q + 1) % 4  # as if the camera turned while the key was held
        n0 = env.session.frames_done
        _, _, _, _, info = env.step(a)
        assert info["repressed"] == keys
        log = env.session.frame_log()
        assert int(log["keys"][n0]) == 0 and all(int(x) == keys for x in log["keys"][n0 + 1:n0 + 4])
        assert env._press_q[keys] == q
        _, _, _, _, info = env.step(a)
        assert "repressed" not in info
        meta_opts = env.recorder.meta.env_options
        assert meta_opts["world_actions"] and meta_opts["repress_stale_keys"]
    finally:
        env.close()


def push_of(k, m):
    from ballrain.env import key_push_dirs
    d = key_push_dirs(m)
    return sum((d[b] for b in (U, D, L, R) if k & b), np.zeros(3))


def test_off_quarter_cameras_pick_the_nearest_push():
    rng = np.random.default_rng(0)
    for yaw in (0, 90, 180, -90):
        m = cam(yaw)
        m[[0, 2]] += rng.normal(0, 1.3e-5, (2, 3))  # retail at-rest bit patterns are up to ~1e-5 off
        for a in ARROWS9[1:]:
            assert world_to_camera_keys(a, m) == world_to_camera_keys(a, cam(yaw))
    # mid-turn / interrupted turn: the chosen keys push within 22.5 degrees of the wanted direction (8 directions)
    for yaw in np.linspace(-180, 180, 73):
        for a in ARROWS9[1:]:
            k = world_to_camera_keys(a, cam(yaw))
            p = push_of(k, cam(yaw))[[0, 2]]
            w = world_dir(a) / np.linalg.norm(world_dir(a))
            assert p @ w / np.linalg.norm(p) >= math.cos(math.radians(22.5)) - 1e-9, (yaw, a, k)
    # 10 degrees off: north is pushed by UP alone (10 deg off); 30 degrees off: by UP + LEFT (15 deg off, UP is 30)
    assert world_to_camera_keys(U, cam(10)) == U
    assert world_to_camera_keys(U, cam(30)) == U | L


def test_camera_guard_ends_the_episode_without_credit(tmp_path):
    env = make_env(tmp_path)
    try:
        env.reset()
        env.step(ARROWS9.index(U))
        env._camera_dev = lambda: 0.2  # Cam_OrientRef 11 degrees off a quarter turn, no turn in progress
        _, r, term, trunc, info = env.step(ARROWS9.index(U))
        assert term and not trunc and info["episode_end"] == "camera_off_quarter" and info["camera_off_quarter"]
        assert env.reward.total == -env.reward.cfg.death_penalty
        meta = json.loads(Path(info["run_meta"]).read_text(encoding="utf-8"))
        assert meta["invalid"] == "camera_off_quarter" and env.session is None
    finally:
        env.close()


def test_camera_guard_allows_a_turn_in_progress(tmp_path):
    env = make_env(tmp_path, camera_actions=True)
    try:
        env.reset()
        env._camera_dev = lambda: 0.5
        _, _, term, _, info = env.step(len(ARROWS9))  # shift + left: a turn starts
        assert not term
        for _ in range(12):  # 12 x 4 frames x 7.6 ms = 364 ms < 400 ms grace
            _, _, term, _, info = env.step(0)
            if term:
                break
        assert not term
        for _ in range(3):
            _, _, term, _, info = env.step(0)
            if term:
                break
        assert term and info["episode_end"] == "camera_off_quarter"
    finally:
        env.close()


def test_held_keys_are_stale_after_death_or_pause(tmp_path):
    env = make_env(tmp_path)
    try:
        env.reset()
        env.step(ARROWS9.index(U))
        held = env._held
        env._invalidate_held()
        _, _, _, _, info = env.step(ARROWS9.index(U))
        assert info["repressed"] == held
    finally:
        env.close()


def test_camera_ok_filters_off_quarter_start_frames():
    from ballrain.curriculum import TasStarts
    st = TasStarts.__new__(TasStarts)
    n = 400
    st.frames = np.zeros(n, tas.FRAME_DTYPE)
    st.frames["delta"] = 1000 / 264
    st.frames["keys"][100] = tas.KEY_SHIFT | tas.KEY_LEFT
    dev = np.zeros(n, np.float32)
    dev[110:130] = 0.3  # mid-turn, 10-30 frames after the press: allowed
    dev[300:310] = 0.3  # off a quarter at rest: dropped
    st._cam = {"quarter": np.zeros(n, np.int8), "dev": dev}
    st.valid = np.arange(n)
    ok = st.camera_ok(1e-4, 106)
    assert ok[110:130].all() and not ok[300:310].any() and ok[:300].all() and ok[310:].all()
