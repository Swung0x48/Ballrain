"""Golden recordings for the native volunteer client (native/, docs/native-client.md).

Runs the Python side (BallanceEnv / AgentSession on the headless engine) and records, in the order they happen:
- every message of the engine socket (the PONG, every OBS payload, every ACT1 the Python side sends),
- the level-message bytes at the moment AgentSession reads them (level.msg when it parses it, the unread tail of
  level.msg.inc at every increment read, or a "skipped" record when the file was absent / not grown),
- every FeatureExtractor.reset_episode() and every feature vector compute() returned,
- in the policy / random modes the policy's outputs per step (log-probs, value, the draw u, the action).

The native client's offline replay (native/tests/golden_replay) feeds the recorded engine bytes through its own
tracker / features / policy and compares; its live runner writes the same stream format, so two runs can be
diffed record by record (tools/native/compare_streams.py).

    python tools/native/golden.py policy --manifest runtime/nc/policy/<sha>.json --out runtime/nc/golden/policy_s1 --seed 1
    python tools/native/golden.py random --manifest ... --out runtime/nc/golden/random_s2 --seed 2
    python tools/native/golden.py tas --manifest ... --record <dataset record dir> --out runtime/nc/golden/tas_sr01

Engine: --engine (BallanceMMOSimTool), --root (a Ballance folder); the x87-soft source physics is selected with
BALLRAIN_HL_PHYS2002=0 (the Windows 2002 host build ignores the folder's physics_RT.dll then) and checked against the
hello string. Outputs stay under runtime/ (they contain level geometry and local paths are stripped from meta.json).

Stream format (little-endian): b"BRGS", u32 version; then records u32 kind, u32 length, payload (kinds below).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ballrain import client as C, protocol as P, tas, transport as TR  # noqa: E402
from ballrain.features import FeatureConfig, FeatureExtractor  # noqa: E402

MAGIC = b"BRGS"
VERSION = 1
K_HELLO, K_OBS, K_ACT, K_LVL_FULL, K_LVL_INC, K_RESET, K_FEAT, K_POLICY, K_NOTE = 1, 2, 3, 4, 5, 6, 7, 8, 9


class Stream:
    def __init__(self, path: Path):
        self.f = open(path, "wb")
        self.f.write(MAGIC + struct.pack("<I", VERSION))
        self.counts: dict = {}

    def put(self, kind: int, payload: bytes) -> None:
        self.f.write(struct.pack("<II", kind, len(payload)) + payload)
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def note(self, obj) -> None:
        self.put(K_NOTE, json.dumps(obj, sort_keys=True).encode())

    def close(self):
        self.f.close()


_STREAM: Stream | None = None
_LAUNCH: dict = {}


def install_taps(stream: Stream) -> None:
    """Record the socket traffic and the level-message reads (monkeypatches; this process only)."""
    global _STREAM
    _STREAM = stream
    SC = TR.SocketChannel
    orig_hello, orig_recv, orig_send = SC.wait_hello, SC._recv, SC._send

    def wait_hello(self, timeout_s, process_handle=None):
        h = orig_hello(self, timeout_s, process_handle)
        _STREAM.put(K_HELLO, h["mod_version"].encode("latin-1"))
        return h

    def _recv(self, deadline):
        m = orig_recv(self, deadline)
        if m is not None and m[0] == TR.BR_OBS:
            _STREAM.put(K_OBS, m[1])
        return m

    def _send(self, typ, payload):
        if typ == TR.BR_ACT:
            _STREAM.put(K_ACT, payload)
        return orig_send(self, typ, payload)

    SC.wait_hello, SC._recv, SC._send = wait_hello, _recv, _send

    AS = C.AgentSession

    def _load_level(self, full):
        p = self.paths["levelmsg"]
        if not p.exists() or (self._level_loaded and not full):
            _STREAM.put(K_LVL_FULL, b"\0")
            return
        data = p.read_bytes()
        _STREAM.put(K_LVL_FULL, b"\1" + data)
        lm, _ = P.parse_level_message(data)
        self.tracker.merge_level(lm)
        self._level_loaded = True

    def _load_increments(self):
        p = Path(str(self.paths["levelmsg"]) + ".inc")
        if not p.exists() or p.stat().st_size <= self._inc_pos:
            _STREAM.put(K_LVL_INC, b"\0")
            return
        data = p.read_bytes()
        tail = data[self._inc_pos:]
        _STREAM.put(K_LVL_INC, b"\1" + tail)
        msgs, used = P.parse_level_stream(tail)
        for lm in msgs:
            self.tracker.merge_level(lm)
        self._inc_pos += used

    AS._load_level, AS._load_increments = _load_level, _load_increments

    from ballrain import headless as H
    orig_start = H.HeadlessLauncher.start

    def start(self, env):
        _LAUNCH["argv"] = self.arguments()
        _LAUNCH["env"] = {k: v for k, v in self.environment(env).items() if k.startswith("BALLRAIN_")}
        return orig_start(self, env)

    H.HeadlessLauncher.start = start

    orig_compute, orig_reset = FeatureExtractor.compute, FeatureExtractor.reset_episode

    def compute(self, tracker):
        vec, info = orig_compute(self, tracker)
        _STREAM.put(K_FEAT, np.asarray(vec, "<f4").tobytes())
        return vec, info

    def reset_episode(self):
        _STREAM.put(K_RESET, b"")
        return orig_reset(self)

    FeatureExtractor.compute, FeatureExtractor.reset_episode = compute, reset_episode


def sanitize_env(env: dict) -> dict:
    out = {}
    for k, v in sorted(env.items()):
        s = str(v)
        if ("/" in s or "\\" in s) and not k.endswith("BALL_ORDER"):
            s = "<path>/" + Path(s.replace("\\", "/")).name
        if k == "BALLRAIN_PARENT_PID":
            s = "<pid>"
        if k == "BALLRAIN_SOCK":
            s = "tcp:<port>"
        out[k] = s
    return out


def headless_cfg(a, manifest) -> dict:
    hl = {"binary": str(Path(a.engine).resolve()), "root": str(Path(a.root).resolve()), "socket": "tcp",
          "ball_order": f"retail:{a.ball_seed}", "extra_env": {"BALLRAIN_HL_PHYS2002": "0"}}
    for k, v in (manifest.get("engine_options") or {}).items():
        hl[k] = v
    return hl


def env_config(a, manifest):
    from ballrain.env import EnvConfig
    from ballrain.rewards import RewardConfig
    env = dict(manifest["env"])
    env.pop("start_record", None)
    feats = dict(manifest["features"])
    feats["path_dir"] = a.path_dir
    for k in ("far_waypoints", "ref_speed_at"):
        feats[k] = tuple(feats.get(k) or ())
    bundle = None
    if feats.get("branch_sha256"):  # the env installs the learner's bundle itself (EnvConfig.branch_assets)
        if not a.branch_dir:
            raise SystemExit("the policy follows a route branch: give --branch-dir")
        from ballrain.branch_assets import read_bundle
        bundle = read_bundle(int(env["level"]), a.branch_dir)
    return EnvConfig(**env, branch_assets=bundle, start_record=None, start_frame=None, curriculum=None, rest_starts=None, p_rest=0.0,
                     rehearse_starts=None, p_rehearse=0.0, warm_pool=False, warm_copy=None, trace=False, prefix=None,
                     seed=int(a.ball_seed), copy="nc-py", out_root=str(Path(a.out).resolve() / "runs"),
                     backend="headless", headless=headless_cfg(a, manifest), features=FeatureConfig(**feats),
                     reward=RewardConfig(**manifest["reward"]))


def run_env(a, manifest, stream: Stream) -> dict:
    """policy / random: the pool's per-task loop (deaths respawn in the same process, the process end ends it)."""
    from ballrain.env import BallanceEnv
    from ballrain.pool.sampling import sample_action, sample_u
    from ballrain.pool.weights import load_policy
    pol = load_policy(Path(a.manifest).with_suffix(".npz"), manifest)
    launcher = None
    if a.fake:  # ballrain.fakegame's synthetic level over the headless socket transport (CI: no game, no engine)
        from ballrain.fakegame import FakeLauncher
        launcher = FakeLauncher(transport="socket")
    env = BallanceEnv(env_config(a, manifest), launcher=launcher)
    rng = np.random.default_rng(a.seed)
    t0 = time.perf_counter()
    steps, deaths, ends = 0, 0, []
    sticky, left = 0, 0
    try:
        obs, info = env.reset()
        t_start = time.perf_counter()
        while steps < a.max_steps:
            x = pol.normalize(obs)
            lp = pol.log_probs(x)
            v = float(pol.value(x))
            u = sample_u(a.seed, steps)
            if a.mode == "policy":
                act = sample_action(lp, u)
            else:  # sticky random actions (2-39 steps), as scripts/m2_start_states.py "random"
                if left <= 0:
                    sticky, left = int(rng.integers(len(env.actions))), int(rng.integers(2, 40))
                act = sticky
                left -= 1
            stream.put(K_POLICY, struct.pack("<IdIf", steps, u, act, v) + np.asarray(lp, "<f4").tobytes())
            obs, r, term, trunc, info = env.step(act)
            steps += 1
            if term or trunc:
                ends.append(info.get("episode_end"))
                stream.note({"step": steps, "episode_end": info.get("episode_end"), "frames": info["frames_done"]})
                if info.get("episode_end") == "BallOff":
                    deaths += 1
                if env._after_death:
                    obs, i2 = env.reset(options={"relaunch": False})
                    if obs is None:
                        ends.append("process_ended")
                        break
                else:
                    break
        t_loop = time.perf_counter() - t_start
    finally:
        env.close()
    return {"steps": steps, "deaths": deaths, "ends": ends, "wall_s": time.perf_counter() - t0, "loop_s": t_loop,
            "steps_per_s": steps / max(t_loop, 1e-9), "policy_sha": manifest["policy_sha"]}


def run_tas(a, manifest, stream: Stream) -> dict:
    """tas: replay a dataset record's frames in blocks of --block frames through an AgentSession, features after every
    block (geometry coverage of the whole level). The follower is reset at every BallNavActive."""
    from ballrain.headless import HeadlessConfig, HeadlessLauncher
    z = np.load(Path(a.record) / "raw.npz", allow_pickle=False)
    frames = z["record_frames"]
    if a.mode == "branch":  # the record up to the drive's prefix frame, then its verified key stream through the trafo
        # wire v2 (branches.json): the member's own measured drive (--branch-member)
        ddir = Path(a.branch_dir) / a.branch_member if a.branch_member else Path(a.branch_dir)
        with np.load(ddir / "drive.npz", allow_pickle=False) as dz:
            k = int(dz["prefix_frame"])
            meta = json.loads((Path(a.record) / "meta.json").read_text(encoding="utf-8"))
            if str(dz["record_sha256"]) != meta.get("record_sha256"):
                raise SystemExit("drive.npz is not for this record")
            drive = dz["frames"]
        frames = np.concatenate([frames[:k].astype(drive.dtype), drive])
    feats = dict(manifest["features"])
    feats["path_dir"] = a.path_dir
    root = None
    if a.branch_dir:  # as the learner's workers: the pinned bundle (ballrain.branch_assets), never the directory itself
        from ballrain.branch_assets import install, read_bundle, validate
        from ballrain.path import asset_path, file_sha256
        lv = int(manifest["env"]["level"])
        bundle = read_bundle(lv, a.branch_dir)
        main_sha = file_sha256(asset_path(lv, Path(a.path_dir)))
        feats["branch_sha256"] = validate(bundle, lv, main_sha)
        feats["path_sha256"] = main_sha
        feats.pop("branch_dir", None)
        tmp = install(bundle, lv, main_sha, feats["branch_sha256"])
        root = tmp.name
    for k in ("far_waypoints", "ref_speed_at"):
        feats[k] = tuple(feats.get(k) or ())
    fx = FeatureExtractor(FeatureConfig(**feats), branch_root=root)
    out = Path(a.out).resolve() / "runs" / "tas"
    sc = C.SessionConfig(level=int(manifest["env"]["level"]), out_dir=out, copy="nc-py", max_frames=len(frames) + 10,
                         body_radius=60.0, sr_tail=32)
    s = C.AgentSession(sc, HeadlessLauncher(HeadlessConfig(**headless_cfg(a, manifest))))
    t0 = time.perf_counter()
    s.start()
    fx.reset_episode()
    fx.compute(s.tracker)
    i, n = 0, 0
    try:
        while i < len(frames) and not s.ended and n < a.max_steps:
            blk = frames[i:i + a.block]
            ev0 = len(s.tracker.events)
            s.act(blk, repeat=1)
            i += len(blk)
            n += 1
            if any(c == P.Ev.BallNavActive for _, c, _ in s.tracker.events[ev0:]):
                fx.reset_episode()
            fx.compute(s.tracker)
    finally:
        s.stop()
        s.close()
    return {"steps": n, "frames": int(s.frames_done), "wall_s": time.perf_counter() - t0, "trace_hash": s.trace_hash}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("policy", "random", "tas", "branch"))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default=str(REPO / "runtime/nc/engine/BallanceMMOSimTool.exe"))
    ap.add_argument("--root", default=str(REPO / "runtime/nc/game"))
    ap.add_argument("--path-dir", default="runtime/refpaths-alt")
    ap.add_argument("--seed", type=int, default=1, help="sample seed (the u_t stream; random mode: the action rng)")
    ap.add_argument("--ball-seed", type=int, default=12, help="retail:<seed> ball order")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--record", help="tas mode: dataset record directory (raw.npz)")
    ap.add_argument("--block", type=int, default=2, help="tas mode: frames per block")
    ap.add_argument("--branch-member", help="branch mode with a multi-branch --branch-dir: the member whose drive.npz is "
                    "replayed")
    ap.add_argument("--branch-dir", help="the ball-type diversion assets (features.branch_dir); branch mode replays "
                    "<record>[:drive.prefix_frame] + <branch-dir>/drive.npz")
    ap.add_argument("--fake", action="store_true", help="policy / random on ballrain.fakegame (the synthetic level)")
    a = ap.parse_args(argv)
    from ballrain.pool.weights import read_manifest
    manifest = read_manifest(a.manifest)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stream = Stream(out / "stream.bin")
    install_taps(stream)
    try:
        res = run_tas(a, manifest, stream) if a.mode in ("tas", "branch") else run_env(a, manifest, stream)
    finally:
        stream.close()
    meta = {"mode": a.mode, "seed": a.seed, "ball_seed": a.ball_seed, "ball_order": f"retail:{a.ball_seed}",
            "max_steps": a.max_steps, "block": a.block if a.mode == "tas" else None,
            "record": Path(a.record).name if a.record else None, "policy_sha": manifest["policy_sha"],
            "argv": ["<engine>"] + [x if not ("/" in x or "\\" in x) else "<path>" for x in _LAUNCH.get("argv", [])[1:]],
            "env": sanitize_env(_LAUNCH.get("env", {})), "counts": {str(k): v for k, v in stream.counts.items()},
            "result": res}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta["result"]), json.dumps(meta["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
