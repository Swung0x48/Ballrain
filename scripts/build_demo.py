"""Demonstration set for the imitation term (train_m2.py --demo): the states a TAS record passes through in given
frame ranges, each with the action the record takes there, as the policy sees them.

    uv run python scripts/build_demo.py --ckpt CKPT --ranges 4600-5900,12100-14300,17000-18300 --out demo.npz
        [--backend headless --headless JSON --copy hl-91]

For each range the env (the checkpoint's own configuration: features, reference line, pickups) hands over at the
range's first frame (the TAS prefix replayed by the mod, as a curriculum start), then the record's own frames are fed
in blocks of one agent decision (action_repeat agent frames, in the record's frames: their deltas summed). Before each
block the observation is computed exactly as the env does (FeatureExtractor, pickup state updated from the counters);
the label is the world-frame action (ARROWS9 index) whose push direction is closest to the arrows the record holds for
most of the block (camera keys mapped through the camera reference at that moment; no arrow = index 0). Blocks while
paused, transforming, without control or with a camera turn held are skipped. No game is started for anything else.
Output: obs (raw, float32), act, frame, sector, ball type, and the feature config (the learner checks the layout).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402


def world_action(keys: int, cam_matrix) -> int:
    """ARROWS9 index (world directions under world_actions) closest to the push of camera arrow keys `keys`."""
    from ballrain import tas
    from ballrain.env import ARROW_BITS, ARROWS9, key_push_dirs
    if not keys & (tas.KEY_UP | tas.KEY_DOWN | tas.KEY_LEFT | tas.KEY_RIGHT):
        return 0
    dirs = key_push_dirs(cam_matrix)
    v = sum(dirs[b] for b in ARROW_BITS if keys & b)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return 0
    v = v / n
    best, bc = 0, -np.inf
    for i, a in enumerate(ARROWS9):
        if not a:
            continue
        w = np.array([bool(a & tas.KEY_RIGHT) - bool(a & tas.KEY_LEFT), 0.0, bool(a & tas.KEY_UP) - bool(a & tas.KEY_DOWN)])
        c = float(v @ (w / np.linalg.norm(w)))
        if c > bc + 1e-9:
            best, bc = i, c
    return best


def collect_range(env, k0: int, k1: int, log=print) -> dict:
    from ballrain import tas
    from ballrain.env import ARROW_MASK, EPISODE_STOP
    env.cfg.start_frame = k0
    env.reset(options={"new_process": True})
    s = env.session
    fr = env.starts.frames
    k0 = int(s.frames_done)  # the curriculum snaps the hand-over to a valid start frame
    # one agent decision = action_repeat agent frames; in the record's frames, as many as cover the same time
    agent_ms = env.cfg.action_repeat * float(env.delta)
    out = {"obs": [], "act": [], "frame": [], "sector": [], "ball": []}
    k = k0
    skipped = Counter()
    while k < k1 and not s.ended:
        n, t = 0, 0.0
        while k + n < len(fr) and t < agent_ms - 1e-6:
            t += float(fr["delta"][k + n])
            n += 1
        block = fr[k:k + n]
        t_ = s.tracker
        keys = Counter(int(x) & ARROW_MASK for x in block["keys"]).most_common(1)[0][0]
        cam_turn = any((int(x) & tas.KEY_SHIFT) and (int(x) & (tas.KEY_LEFT | tas.KEY_RIGHT)) for x in block["keys"])
        esc = any(int(x) & tas.KEY_ESC for x in block["keys"])
        if t_.flags.paused or esc:
            skipped["paused"] += 1
        elif t_.transforming or not t_.control_valid():
            skipped["no_control"] += 1
        elif cam_turn:
            skipped["camera_turn"] += 1
        else:
            obs, info = env.fx.compute(t_)
            cams = t_.obs.cams if t_.obs is not None else None
            if cams is None:
                skipped["no_camera"] += 1
            else:
                out["obs"].append(np.asarray(obs, np.float32))
                out["act"].append(world_action(keys, cams["cams"][0]["matrix"]))
                out["frame"].append(k)
                out["sector"].append(int(t_.sector))
                out["ball"].append(int(t_.ball_type()))
        s.act(block, repeat=1, stop=EPISODE_STOP)
        if env.pickups:
            env.pickups.update(s.tracker, env._ball_pos())
        k += n
        if s.frames_done != k:
            raise RuntimeError(f"frames_done {s.frames_done} != {k} (the record ended or diverged)")
    log(f"range {k0}-{k1}: {len(out['act'])} samples, skipped {dict(skipped)}, actions {Counter(out['act'])}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ranges", required=True, help="TAS frame ranges lo-hi,lo-hi")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="headless")
    ap.add_argument("--headless", default=None, help="HeadlessConfig fields as JSON")
    ap.add_argument("--copy", default="hl-91")
    a = ap.parse_args(argv)
    import torch
    import eval_m2
    from ballrain import instances
    from ballrain.env import BallanceEnv
    instances.LAUNCH_RECIPE = instances.launch_direct
    ck = torch.load(a.ckpt, weights_only=False, map_location="cpu")
    ranges = [tuple(int(v) for v in r.split("-")) for r in a.ranges.split(",") if r.strip()]
    out_root = Path(a.out).with_suffix("").as_posix() + "-runs"
    cfg = eval_m2.build_env_config(ck, out_root, ranges[0][0], 200000, backend=a.backend,
                                   headless=json.loads(a.headless) if a.headless else None)
    cfg.copy = a.copy
    cfg.trace = False
    env = BallanceEnv(cfg)
    parts = []
    try:
        for k0, k1 in ranges:
            parts.append(collect_range(env, k0, k1))
    finally:
        env.close()
    res = {k: np.concatenate([np.asarray(p[k]) for p in parts]) for k in parts[0]}
    if len(res["obs"]) and res["obs"].shape[1] != int(ck["n_obs"]):
        raise SystemExit(f"observation size {res['obs'].shape[1]} != checkpoint n_obs {ck['n_obs']}")
    feats = json.dumps(ck["env_config"]["features"], sort_keys=True, default=str)
    np.savez_compressed(a.out, obs=res["obs"].astype(np.float32), act=res["act"].astype(np.int64),
                        frame=res["frame"].astype(np.int64), sector=res["sector"].astype(np.int64),
                        ball=res["ball"].astype(np.int64), features=np.array(feats), record=np.array(cfg.start_record),
                        ranges=np.array(a.ranges))
    print(f"{a.out}: {len(res['act'])} samples, n_obs {res['obs'].shape[1] if len(res['obs']) else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
