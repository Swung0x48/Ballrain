"""Real-game checks of the world-action key mapping (launches the game on two training copies).

    uv run python tools/analysis/m2_key_checks.py [--copies train-12,train-13] [--k 6350] [--k-death 15000]

1. Stale keys at a TAS hand-over (repress_stale_keys): from frame k of SR01_1_14_457_264 (default 6350: the prefix
   holds DOWN + RIGHT, pressed under Cam_OrientRef 180 deg, the camera is 90 deg at the hand-over), three processes:
   idle; the world action whose keys are exactly the held ones, with the re-press; the same without it. The push of
   each = its ball displacement minus the idle run's, per step; expected: with the re-press along the world action,
   without it along the direction latched at the old camera.
2. Keys after a respawn: push the ball off the road until BallOff (world direction perpendicular to the reference
   line), respawn, then idle or hold the world action closest to the line's tangent for 30 steps; compare the
   displacement (two processes with identical inputs up to the respawn).
Writes runtime/analysis/m2_key_checks.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from ballrain import instances, tas  # noqa: E402
from ballrain.env import ARROWS9, BallanceEnv, EnvConfig, key_push_dirs, world_to_camera_keys  # noqa: E402
from ballrain.features import FeatureConfig  # noqa: E402

RECORD = "SR01_1_14_457_264"
OUT = instances.runtime_dir() / "analysis"


def env_for(copy, k, repress, tag):
    cfg = EnvConfig(level=1, mode="sr", copy=copy, action_repeat=4, start_record=RECORD, start_frame=k,
                    world_actions=True, repress_stale_keys=repress, features=FeatureConfig(camera="world", hs_block=False),
                    out_root=str(OUT / "m2_key_checks" / tag))
    return BallanceEnv(cfg)


def ball_pos(env):
    return np.asarray(env.session.tracker.obs.ball["pos"], np.float64)


def cam_matrix(env):
    return np.asarray(env.session.tracker.obs.cams["cams"][0]["matrix"], np.float64)


def world_vec(a):
    return np.array([bool(a & tas.KEY_RIGHT) - bool(a & tas.KEY_LEFT), 0.0, bool(a & tas.KEY_UP) - bool(a & tas.KEY_DOWN)])


def run_actions(env, actions):
    obs, info = env.reset()
    start = {"held": env._held, "press_q": {int(b): q for b, q in env._press_q.items()}, "cam_q": env._cam_quarter(),
             "cam": cam_matrix(env).tolist()}
    pos = [ball_pos(env)]
    infos = []
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        pos.append(ball_pos(env))
        infos.append({k: info.get(k) for k in ("repressed", "episode_end")})
        if term or trunc:
            break
    return start, np.array(pos), infos


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def check_stale(copies, k, steps=6):
    probe = env_for(copies[0], k, True, "probe")
    try:
        probe.reset()
        held, m = probe._held, cam_matrix(probe)
        press_q = dict(probe._press_q)
    finally:
        probe.close()
    cands = [i for i, a in enumerate(ARROWS9) if a and world_to_camera_keys(a, m) == held]
    if not cands:
        return {"k": k, "held": held, "error": "no world action maps to the held keys"}
    ai = cands[0]
    res = {"k": k, "held_keys": held, "press_quarters": {int(b): q for b, q in press_q.items()}, "world_action": ARROWS9[ai],
           "world_dir": world_vec(ARROWS9[ai]).tolist()}
    runs = {}
    for tag, acts, rep in (("idle", [0] * steps, True), ("repress", [ai] * steps, True), ("no_repress", [ai] * steps, False)):
        env = env_for(copies[0], k, rep, tag)
        try:
            runs[tag] = run_actions(env, acts)
        finally:
            env.close()
    p_idle = runs["idle"][1]
    # the push directions: current camera rows for the held keys, and the rows of the camera they were pressed under
    cur = sum(key_push_dirs(m)[b] for b in (tas.KEY_UP, tas.KEY_DOWN, tas.KEY_LEFT, tas.KEY_RIGHT) if held & b)
    q_old = list(press_q.values())[0]
    yaw = q_old * np.pi / 2
    zr, xr = np.array([np.sin(yaw), 0, np.cos(yaw)]), np.array([np.cos(yaw), 0, -np.sin(yaw)])
    m_old = np.array([xr, [0, 1, 0], zr, [0, 0, 0]])
    old = sum(key_push_dirs(m_old)[b] for b in (tas.KEY_UP, tas.KEY_DOWN, tas.KEY_LEFT, tas.KEY_RIGHT) if held & b)
    res["expected_push_now"] = unit(cur).tolist()
    res["latched_push_old_camera"] = unit(old).tolist()
    for tag in ("repress", "no_repress"):
        p = runs[tag][1]
        n = min(len(p), len(p_idle))
        d = p[:n] - p_idle[:n]
        rows = []
        for i in range(1, n):
            v = d[i].copy()
            v[1] = 0
            rows.append({"step": i, "dpos_minus_idle": d[i].round(6).tolist(),
                         "cos_world_action": float(unit(v) @ unit(world_vec(ARROWS9[ai]))),
                         "cos_latched_old": float(unit(v) @ unit(old))})
        res[tag] = {"steps": rows, "infos": runs[tag][2]}
    return res


def check_respawn(copies, k, steps=30):
    # find a world action that kills: perpendicular to the line, either side
    def tangent(env, obs):
        sl = env.fx.slices()["path_tangent_cam"]
        t = obs[sl]
        return np.array([t[0], 0, t[1]])

    out = {"k": k}
    env = env_for(copies[1], k, True, "respawn_probe")
    kill = None
    try:
        obs, _ = env.reset()
        t = unit(tangent(env, obs))
        perp = np.array([t[2], 0, -t[0]])
        for sgn in (1, -1):
            a = max(range(1, 9), key=lambda i: unit(world_vec(ARROWS9[i])) @ (sgn * perp))
            env.close()
            env = env_for(copies[1], k, True, "respawn_probe")
            obs, _ = env.reset()
            for n in range(400):
                obs, r, term, trunc, info = env.step(a)
                if term or trunc:
                    break
            if info.get("episode_end") == "BallOff":
                kill = (a, n + 1)
                break
    finally:
        env.close()
    if kill is None:
        out["error"] = "no death found"
        return out
    a_kill, n_kill = kill
    out["kill_action"], out["kill_steps"] = ARROWS9[a_kill], n_kill
    for tag in ("idle", "push"):
        env = env_for(copies[1], k, True, f"respawn_{tag}")
        try:
            env.reset()
            for _ in range(n_kill):
                obs, r, term, trunc, info = env.step(a_kill)
            assert info.get("episode_end") == "BallOff", info.get("episode_end")
            obs, info = env.reset()
            assert info["start"]["kind"] == "respawn"
            t = unit(tangent(env, obs))
            a = 0 if tag == "idle" else max(range(1, 9), key=lambda i: unit(world_vec(ARROWS9[i])) @ t)
            p0 = ball_pos(env)
            ps = []
            for _ in range(steps):
                obs, r, term, trunc, info = env.step(a)
                ps.append(ball_pos(env) - p0)
                if term or trunc:
                    break
            out[tag] = {"action": ARROWS9[a], "tangent": t.round(4).tolist(), "held_at_respawn": 0,
                        "disp_after": {str(i + 1): ps[i].round(4).tolist() for i in (0, 4, 9, 19, len(ps) - 1) if i < len(ps)},
                        "horizontal_dist": float(np.linalg.norm(ps[-1][[0, 2]]))}
        finally:
            env.close()
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--copies", default="train-12,train-13")
    p.add_argument("--k", type=int, default=6350)
    p.add_argument("--k-death", type=int, default=15000)
    a = p.parse_args(argv)
    instances.LAUNCH_RECIPE = instances.launch_direct
    copies = a.copies.split(",")
    res = {"stale": check_stale(copies, a.k), "respawn": check_respawn(copies, a.k_death)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m2_key_checks.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
