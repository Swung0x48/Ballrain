"""精彩回放, second stage: the replay driven through the training environment itself, and the side-panel video.

`env_replay` plays the selected run on the video copy through `ballrain.env.BallanceEnv` built from the run's own
checkpoint config (features f6, world actions, action repeat 2, SR reward with its death refund) plus the worker's
split bonus (ballrain.train.worker), with rendering and capture on. The env is given, at every step, the action
decoded from the run's .tas (the world direction whose camera keys are the step's keys); the env turns it back into
keys (world_to_camera_keys, stale-key re-press) and sends them. After every step the frames the env sent must equal
the .tas frames: a decoded action that does not re-encode to the recorded keys aborts the replay (the plain replay
is used instead). So the observation, the reward terms and the policy outputs on the panel are recomputed from the
verified retail replay with the trainer's own code, not taken from the original run's logs. The policy outputs come
from the run's checkpoint nearest to (at or before) the run's update; the behaviour policy sampled at temperature 1
from weights up to ~25 updates apart, so its argmax need not be the action taken (reported, not required).

During the TAS prefix the mod reports an observation after every frame (BALLRAIN_PREFIX_EVENTS; read-only, the
capture gate covers it) and the panel shows the observation features of those frames, without reward or policy.

`render_overlay` composes 1366x768 frames: the captured 1024x768 game picture on the left, the panel on the right
(drawn with PIL in the dashboard's Ballance look), and encodes them. The panel shows no paths, hosts or copy names.
The reference-route waypoints of the observation are not drawn (the route is derived from a human recording, which
the dashboard does not publish).
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ballrain import capture, tas

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

FAIL_AFTER_STEPS = 0  # test hook (highlight.py --test-env-fail N): make env_replay fail to exercise the fallback
PANEL_W, H, GAME_W = 342, 768, 1024
W = GAME_W + PANEL_W
FONT_TTF = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")
GAME_FONT = ROOT / "runtime" / "dashboard" / "assets"

# Chinese labels (Ballance terms in the community's wording; one place to rename)
LABELS = {
    "checkpoint": "盘点", "sector": "小节", "airship": "飞船",
    "ball": {0: "木球", 1: "纸球", 2: "石球"},
    "mode_tas": "TAS 控制", "mode_agent": "智能体控制", "mode_env": "开场 / 刹车", "mode_end": "已到达飞船",
    "src_tas": "TAS", "src_agent": "智能体", "src_env": "开场", "src_end": "终点后",
}
TERM_LABELS = [("time", "时间"), ("progress", "进度"), ("split_bonus", "分段奖励"), ("gate_bonus", "关口奖励"),
               ("finish", "通关"),
               ("death", "死亡"), ("death_refund", "返还"), ("stall_penalty", "卡住")]
KEY_BITS = [(tas.KEY_UP, "↑"), (tas.KEY_DOWN, "↓"), (tas.KEY_LEFT, "←"), (tas.KEY_RIGHT, "→"),
            (tas.KEY_SHIFT, "Shift"), (tas.KEY_SPACE, "Space"), (tas.KEY_Q, "Q"), (tas.KEY_ESC, "Esc")]

AMBER, AMBER2, BRASS, BRASS_HI, INK, MUTED = (255, 207, 90), (255, 159, 28), (184, 137, 58), (233, 196, 106), \
    (235, 227, 207), (169, 160, 137)
GOOD, BAD, BLUE = (126, 214, 154), (255, 122, 102), (120, 160, 255)
PANEL_BG = (9, 14, 32)


# ======================================================================================= env replay

def nearest_ckpt(run, update: int) -> Path | None:
    cands = []
    for f in run.files("ckpt_*.pt"):
        m = re.search(r"ckpt_(\d+)\.pt$", f)
        if m:
            cands.append((int(m.group(1)), Path(f)))
    if not cands:
        return None
    before = [c for c in cands if c[0] <= update]
    return (max(before) if before else min(cands))[1]


def _local_path(p: str | None) -> str | None:
    """A config path of the training machine -> the same place under this checkout's runtime/."""
    if not p:
        return p
    if Path(p).exists():
        return p
    m = re.search(r"[\\/]runtime[\\/](.+)$", str(p))
    if m:
        q = ROOT / "runtime" / Path(m.group(1).replace("\\", "/"))
        if q.exists():
            return str(q)
    return p


def make_env(ck: dict, launcher, out_root: Path, k: int, kind: str):
    from ballrain.env import BallanceEnv, EnvConfig
    from ballrain.features import FeatureConfig
    from ballrain.rewards import RewardConfig
    ec = dict(ck["env_config"])
    ec["branch_assets"] = ck.get("branch_assets")  # the run's route bundle lives beside env_config in the checkpoint
    fc = dict(ec.pop("features"))
    rc = dict(ec.pop("reward"))
    fc["path_dir"] = _local_path(fc.get("path_dir"))
    ec.update(copy="video-00", out_root=str(out_root), backend="retail", headless=None, curriculum=None,
              start_frame=int(k), p_rest=0.0, p_rehearse=0.0, warm_pool=False, warm_copy=None, skip_render=False,
              trace=True, seed=0, launch="nofocus")
    cfg = EnvConfig(**ec, features=FeatureConfig(**fc), reward=RewardConfig(**rc))
    env = BallanceEnv(cfg, launcher=launcher)

    def draw():  # the run's own start: frame k, at-rest start when the run was one
        env._rest_next = kind == "tas_rest"
        return int(k)
    env._draw_start = draw
    return env


def _split_helper(ck: dict, env):
    """The worker's split-bonus methods on a light object (they only use .spec and .env)."""
    from ballrain.train.worker import Worker, WorkerSpec
    a = ck["args"]
    w = object.__new__(Worker)
    w.spec = SimpleNamespace(split_k=float(a.get("split_k") or 0.0), split_min_ref_s=float(a.get("split_min_ref_s", 5.0)),
                             split_rest_allow_s=WorkerSpec.__dataclass_fields__["split_rest_allow_s"].default,
                             split_extra=json.loads(a["split_extra"]) if a.get("split_extra") else [])
    w.env = env
    return w


def _display(fx, vec, tracker, finfo=None) -> dict:
    """What the panel shows of one observation (floats, all from the observation vector / tracker)."""
    c, sl = fx.cfg, fx.slices()
    o = tracker.obs
    cams = o.cams if o is not None else None
    from ballrain.features import yaw_of
    d = {"sr": tracker.sr_ms, "sector": int(tracker.sector), "cps": int(tracker.checkpoints),
         "ball": int(tracker.ball_type()) if tracker.ball_type() in (0, 1, 2) else None,
         "valid": bool(vec[sl["valid"]][0] > 0.5),
         "speed": float(vec[sl["speed"]][0] * c.vel_scale),
         "fan": (vec[sl["edge_fan"]] * c.fan_max).round(3).tolist() if "edge_fan" in sl else None,
         "heights": (vec[sl["heights"]] * c.height_clip).round(3).tolist() if "heights" in sl else None,
         "hit": vec[sl["height_hit"]].astype(int).tolist() if "height_hit" in sl else None,
         "yaw": float(yaw_of(cams["cams"][0]["matrix"])) if cams is not None else 0.0,
         "cam": np.asarray(cams["cams"][0]["matrix"], np.float64).tolist() if cams is not None else None}
    return d


def env_replay(run, row: dict, meta: dict, frames: np.ndarray, k: int, out: Path, *, post_s: float, fps: float,
               launcher_cls, log=print) -> tuple[dict, dict]:
    """The capture replay through the training env. Returns (res for highlight.verify, panel data)."""
    import shutil

    import torch

    torch.set_num_threads(1)  # tiny MLP; many threads only fight the training processes for the cores
    import ballrain.env as envmod
    from ballrain.env import ARROWS9, world_to_camera_keys
    from ballrain.features import FeatureExtractor
    from ballrain.train.ppo import ActorCritic, normalize
    from ballrain.train.worker import _Episode

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ckp = nearest_ckpt(run, int(row["update"]))
    if ckp is None:
        raise RuntimeError("no checkpoint of the run")
    ck = torch.load(ckp, weights_only=False)
    a = ck["args"]
    # the env comes from the checkpoint: it must be the run's level and start record (config --level / --record)
    import data as D
    want = D.run_level(run)
    ec0 = ck.get("env_config") or {}
    got_level, got_rec = int(ec0.get("level") or 1), ec0.get("start_record") or D.LEVEL_RECORDS[1]
    if got_level != want["level"] or (want["record"] and got_rec != want["record"]):
        raise RuntimeError(f"checkpoint env is level {got_level} / {got_rec}, the run is level {want['level']} / "
                           f"{want['record']}")
    hidden = tuple(int(x) for x in str(a.get("hidden", "256,256")).split(",") if x.strip())
    model = ActorCritic(int(ck["n_obs"]), int(ck["n_act"]), hidden, a.get("activation", "tanh"),
                        a.get("actor_heads", "single"))
    model.load_state_dict(ck["model"])
    model.eval()
    rms = ck["rms"]
    kind = (meta.get("start_state") or {}).get("kind") or "tas"
    last_delta = float(frames["delta"][-1])
    post = int(math.ceil(post_s * 1000.0 / max(last_delta, 1.0)))
    reward_scale = float(a.get("reward_scale") or 1.0)  # the critic learns returns of reward * reward_scale
    gates = json.loads(a["gates"]) if a.get("gates") else {}
    probes = json.loads(a["route_probes"]) if a.get("route_probes") else {}
    sink = capture.FfmpegSink(out / "capture.mp4", fps)
    srv = capture.CaptureServer(sink)
    env = None
    try:  # setup errors (config, route files) must not leak the pipe server and its thread
        launcher = launcher_cls("video-00", srv.path, fps, extra={"BALLRAIN_CAPTURE_TAIL": str(post),
                                                                  "BALLRAIN_PREFIX_EVENTS": "1"})
        env = make_env(ck, launcher, out / "env", k, kind)
        fx_tas = FeatureExtractor(env.cfg.features, branch_root=env._branch_temp.name if env._branch_temp else None)  # TAS frames: features for the panel only (env's own untouched)
    except BaseException:
        srv.close()
        sink.close()
        raise
    tas_rec: list = []
    prefix_n = [0]

    class _Sess(envmod.AgentSession):
        def __init__(self, *aa, **kw):
            super().__init__(*aa, **kw)
            self.on_prefix_obs = self._prefix_obs

        def _prefix_obs(self, obs):
            prefix_n[0] += 1
            if self.frames_done % 4 == 0 or self.frames_done >= k - 1:
                v, fi = fx_tas.compute(self.tracker)
                r = _display(fx_tas, v, self.tracker, fi)
                r["f"] = int(self.frames_done)
                tas_rec.append(r)

    orig = envmod.AgentSession
    envmod.AgentSession = _Sess
    steps: list = []
    agree = {"steps": 0, "argmax": 0, "p_taken": []}
    t0 = time.time()
    trk = None
    try:
        obs, info = env.reset()
        sess = env.session
        trk = sess.tracker
        agent_start = int(sess.frames_done)
        got = sess.frame_log()
        if got.tobytes() != frames[:len(got)].tobytes() or len(got) != agent_start:
            raise RuntimeError(f"frames up to the hand-over differ from the run ({len(got)} / {agent_start})")
        w = _split_helper(ck, env)
        ep = _Episode(info, env, int(row["update"]), None)
        w._split_start(ep, info)
        ret = 0.0
        while True:
            f = int(env.session.frames_done)
            if f >= len(frames):
                raise RuntimeError(f"the run's frames end at {len(frames)} but the episode goes on")
            keys2 = int(frames["keys"][f + 1]) if f + 1 < len(frames) else int(frames["keys"][f])
            cam = env.session.tracker.obs.cams["cams"][0]["matrix"]
            cands = [i for i, act in enumerate(env.actions) if act in ARROWS9
                     and world_to_camera_keys(act, cam) == (keys2 & tas.ARROW_MASK) and not keys2 & ~tas.ARROW_MASK]
            if not cands:
                raise RuntimeError(f"step at frame {f}: keys {keys2:#x} are no action of the env")
            act = cands[0]
            x = normalize(obs, rms)
            with torch.no_grad():
                logits, v = model(torch.as_tensor(x)[None])
            probs = torch.softmax(logits[0], -1).numpy()
            disp = _display(env.fx, obs, env.session.tracker)
            n_log = env.session.n_logged
            obs2, r, term, trunc, info2 = env.step(act)
            s = env.session
            if s is not None:
                new = s.frame_log()[n_log:]
                if new.tobytes() != frames[n_log:n_log + len(new)].tobytes():
                    raise RuntimeError(f"step at frame {f}: the env sent other frames than the run (action {act})")
            terms = dict(info2.get("reward_terms") or {})
            if w.spec.split_k and (int(info2.get("sector") or 0) > ep.max_sector_seen
                                   or info2.get("episode_end") == "LevelFinish"):
                sb = w._split_bonus(ep, info2, info2.get("episode_end") == "LevelFinish")
                if sb:
                    r += sb
                    terms["split_bonus"] = terms.get("split_bonus", 0.0) + sb
            ep.max_sector_seen = max(ep.max_sector_seen, int(info2.get("sector") or 0))
            if (probes or gates) and env.session is not None:  # the worker's gate bonus (ballrain.train.worker)
                o = env.session.tracker.obs
                if o is not None and o.ball is not None and bool(o.ball["found"]) and gates and not term:
                    gb = ep.gate(np.asarray(o.ball["pos"], np.float64), np.asarray(o.ball["speed"], np.float64)
                                 + np.asarray(o.ball["speed_change"], np.float64), gates)
                    if gb:
                        r += gb
                        terms["gate_bonus"] = terms.get("gate_bonus", 0.0) + gb
            ep.step(act, r, info2, None)
            ret += r
            agree["steps"] += 1
            agree["argmax"] += int(int(np.argmax(probs)) == act)
            agree["p_taken"].append(float(probs[act]))
            disp.update(f=f, a=int(act), probs=[round(float(p), 4) for p in probs],
                        v=round(float(v[0]) / reward_scale, 4),  # in reward units, like r and ret beside it
                        r=round(float(r), 5), terms={kk: round(float(vv), 5) for kk, vv in terms.items()},
                        ret=round(ret, 4), step=agree["steps"])
            steps.append(disp)
            if term or trunc:
                end = info2.get("episode_end")
                break
            if FAIL_AFTER_STEPS and agree["steps"] >= FAIL_AFTER_STEPS:  # test hook: the fallback path
                raise RuntimeError(f"test: env replay stopped after {agree['steps']} steps")
            obs = obs2
    finally:  # always release the game, the pipe and the encoder (a later replay reuses the work directory)
        envmod.AgentSession = orig
        try:
            if env is not None:
                env.close()
        finally:
            srv.join(60)
            srv.close()
            cap = sink.close()
    cap["server_error"] = srv.error
    cap["launches"] = launcher.starts  # a hand-over relaunch runs without capture: the video would be incomplete
    (out / "capture.json").write_text(json.dumps(cap), encoding="utf-8")
    wall = time.time() - t0
    mp = Path(env.finished_runs[-1]) if env.finished_runs else None
    if mp is None:
        raise RuntimeError("the env wrote no run")
    m2 = json.loads(mp.read_text(encoding="utf-8"))
    rd = mp.parent
    for name in ("trace.txt", "events.txt", "framelog.bin"):
        if (rd / name).is_file():
            shutil.copyfile(rd / name, out / name)
    got = tas.load_any(rd / m2["tas_path"])
    evs = [(int(n), str(nm)) for n, nm, _ in m2.get("events") or []]
    res = {"end": m2.get("end_reason"), "levelfinish_frame": m2.get("levelfinish_frame"),
           "balloffs": sum(1 for _, nm in evs if nm == "BallOff"), "sr_double": m2.get("sr_double"),
           "sr_ms": m2.get("sr_ms"), "wall_s": round(wall, 1), "post_frames": post,
           # the mod's own frame log must exist and equal the run: the recorder's .tas falls back to what Python
           # sent when the mod wrote none (framelog_match None)
           "framelog_equal": bool(m2.get("framelog_match") is True and (out / "framelog.bin").is_file()
                                  and tas.load_raw(out / "framelog.bin").tobytes() == frames.tobytes()
                                  and len(got) == len(frames) and got.tobytes() == frames.tobytes()),
           "framelog_frames": int(len(got)), "env_end": end, "env_return": round(ret, 4),
           "capture": {kk: vv for kk, vv in cap.items() if kk != "rows"}}
    # the SR timer per frame, rebuilt from the frames and the run's events (BML+ 0.3.12 rule) and checked against
    # the mod's own sum: the panel's clock
    sr = sr_per_frame(frames, last_delta, [(int(n), int(c)) for n, c, _ in trk.events], len(frames) + 32 + post + 8)
    lf = m2.get("levelfinish_frame")
    sr_ok = m2.get("sr_double") is not None and abs(sr[-1] - float(m2["sr_double"])) < 1e-6
    p_taken = agree.pop("p_taken")
    agree.update(argmax_rate=round(agree["argmax"] / max(agree["steps"], 1), 4),
                 p_taken_mean=round(float(np.mean(p_taken)), 4) if p_taken else None,
                 reencoded=agree["steps"], ckpt_update=int(ck["counters"]["update"]),
                 run_update=int(row["update"]))
    panel = {"k": int(k), "agent_start": int(agent_start), "n_frames": int(len(frames)), "levelfinish_frame": lf,
             "tas": tas_rec, "steps": steps, "sr": sr.tolist(), "sr_check": bool(sr_ok), "agree": agree,
             "prefix_obs": prefix_n[0], "run": run.name, "update": int(row["update"])}
    (out / "panel.json").write_text(json.dumps(panel), encoding="utf-8")
    log(f"  env replay: {agree['steps']} agent steps re-encoded to the run's keys, argmax = taken in "
        f"{agree['argmax_rate']:.1%}, mean p(taken) {agree['p_taken_mean']}, return {ret:.3f}, "
        f"SR clock check {sr_ok}, checkpoint u{agree['ckpt_update']}")
    return res, panel


def sr_per_frame(frames: np.ndarray, tail_delta: float, events: list, n_total: int) -> np.ndarray:
    """SR double sum after each frame (index i = after frame i), frames beyond the .tas at tail_delta (the SR tail /
    post-finish frames the mod runs at the last delta). Event number n belongs to frame n - 1."""
    from ballrain.protocol import Ev
    by: dict = {}
    for n, c in events:
        by.setdefault(n - 1, []).append(c)
    d = np.full(n_total, np.float32(tail_delta), np.float32)
    d[:len(frames)] = frames["delta"]
    out = np.zeros(n_total, np.float64)
    acc, run = 0.0, False
    for i in range(n_total):
        for c in by.get(i, ()):
            if c == Ev.StartLevel:
                acc, run = 0.0, False
            elif c in (Ev.CounterActive, Ev.UnpauseLevel):
                run = True
            elif c in (Ev.CounterInactive, Ev.PauseLevel):
                run = False
        if run:
            acc += float(d[i])
        out[i] = acc
    # the value stops at the last event that matters; trim to the frames that ran
    last = max([n for n, _ in events], default=len(frames))
    return out[:max(last, len(frames))]


# ======================================================================================= drawing

class GameFont:
    """The game's bitmap menu font (dashboard assets) with the amber gradient of the page."""

    def __init__(self, folder: Path = GAME_FONT):
        from PIL import Image
        self.ok = (folder / "font.png").is_file() and (folder / "font.json").is_file()
        if self.ok:
            self.img = Image.open(folder / "font.png").convert("RGBA")
            self.meta = json.loads((folder / "font.json").read_text(encoding="utf-8"))
        self.cache: dict = {}

    def render(self, text: str, h: int):
        from PIL import Image
        key = (text, h)
        if key in self.cache:
            return self.cache[key]
        g = self.meta["glyphs"]
        top, bot = 2, 30
        sc = h / (bot - top)
        chars = [g.get(str(ord(ch))) or g.get("63") for ch in text]
        wpx = sum((c[2] + 2) if c else 8 for c in chars)
        mask = Image.new("L", (max(1, int((wpx + 4) * sc)), h + 4), 0)
        x = 2.0
        for c in chars:
            if c:
                gl = self.img.crop((c[0], c[1] + top, c[0] + c[2], c[1] + bot)).getchannel("A")
                gl = gl.resize((max(1, int(round(c[2] * sc))), h))
                mask.paste(gl, (int(round(x * sc)), 2), gl)
            x += (c[2] if c else 6) + 2
        hh, ww = mask.size[1], mask.size[0]
        px = np.zeros((hh, ww, 4), np.uint8)
        t = np.linspace(0, 1, hh)
        stops = np.array([[255, 240, 168], list(AMBER), [232, 138, 18]], np.float64)
        for ch in range(3):
            px[..., ch] = np.interp(t, [0, 0.5, 1], stops[:, ch]).astype(np.uint8)[:, None]
        px[..., 3] = np.asarray(mask)
        grad = Image.fromarray(px, "RGBA")
        self.cache[key] = grad
        return grad


class Panel:
    def __init__(self, panel: dict, frames: np.ndarray, meta_pub: dict):
        from PIL import ImageFont
        self.p = panel
        self.frames = frames
        self.k = panel["k"]
        self.agent_start = panel["agent_start"]
        self.n = len(frames)
        self.pub = meta_pub
        self.f = {s: ImageFont.truetype(str(FONT_TTF), s) for s in (11, 12, 13, 14, 16, 18)}
        self.fb = {s: ImageFont.truetype(str(FONT_BOLD), s) for s in (13, 14, 16, 18, 20)}
        sym = Path(r"C:\Windows\Fonts\seguisym.ttf")
        self.sym = ImageFont.truetype(str(sym if sym.is_file() else FONT_BOLD), 18)
        self.gf = GameFont()
        self.tas_f = np.array([r["f"] for r in panel["tas"]], np.int64)
        self.step_f = np.array([r["f"] for r in panel["steps"]], np.int64)
        self.sr = np.asarray(panel["sr"], np.float64)
        keys = np.asarray(frames["keys"], np.int64)
        ch = np.flatnonzero(np.r_[keys[0] != 0, keys[1:] != keys[:-1]])
        self.changes = [(int(i) + 1, int(keys[i - 1]) if i > 0 else 0, int(keys[i])) for i in ch]  # (tick, before, after)
        if keys[-1] != 0:
            self.changes.append((self.n + 1, int(keys[-1]), 0))  # the tail / post-finish frames run with no keys
        self.change_ticks = np.array([c[0] for c in self.changes], np.int64)

    # ------------------------------------------------------------------------------ lookups
    def source(self, tick: int) -> str:
        i = tick - 1
        if i >= self.n:
            return LABELS["src_end"]
        if i < self.k:
            return LABELS["src_tas"]
        if i < self.agent_start:
            return LABELS["src_env"]
        return LABELS["src_agent"]

    def state_at(self, g: int):
        """(kind, record) for the game frame count g: the agent step whose frames are on screen (the step that
        contains the last injected frame g - 1: its observation, policy output and the keys it sent), or the latest
        TAS observation."""
        if g - 1 >= self.agent_start and len(self.step_f):
            j = int(np.searchsorted(self.step_f, g - 1, side="right")) - 1
            if j >= 0:
                return "agent", self.p["steps"][j]
        if len(self.tas_f):
            j = int(np.searchsorted(self.tas_f, g, side="right")) - 1
            if j >= 0:
                return "tas", self.p["tas"][j]
        return "none", None

    # ------------------------------------------------------------------------------ drawing
    def draw(self, g: int):
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (PANEL_W, H), PANEL_BG)
        d = ImageDraw.Draw(im)
        # brass frame
        d.rectangle([0, 0, PANEL_W - 1, H - 1], outline=(107, 74, 23), width=2)
        d.line([(2, 2), (2, H - 3)], fill=(60, 44, 18), width=1)
        kind, rec = self.state_at(g)
        tick = g
        x0 = 14
        # mode badge
        if g > self.n:
            mode, col = LABELS["mode_end"], GOOD
        elif kind == "agent" and g > self.agent_start:
            mode, col = LABELS["mode_agent"], AMBER
        elif g <= self.k:
            mode, col = LABELS["mode_tas"], BLUE
        else:
            mode, col = LABELS["mode_env"], MUTED
        d.rounded_rectangle([x0 - 4, 10, x0 + 128, 38], radius=5, fill=(30, 22, 8) if col == AMBER else (14, 22, 44),
                            outline=col, width=2)
        d.text((x0 + 62, 24), mode, font=self.fb[16], fill=col, anchor="mm")
        lv = f"第 {self.pub['level']} 关 · " if self.pub.get("level") else ""
        # the title stays right of the mode badge: when the one-line form does not fit, the run name gets a line
        # of its own (three smaller lines above the tick row)
        room = PANEL_W - 14 - (x0 + 128 + 10)
        title = f"{self.pub['run']} · {lv}第 {self.pub['update']} 次更新"
        start = f"从 TAS 第 {self.k} 帧接手"
        if d.textlength(title, font=self.f[12]) <= room:
            lines, font, ys = [title, start], self.f[12], (16, 31)
        else:
            lines, font, ys = [str(self.pub["run"]), f"{lv}第 {self.pub['update']} 次更新", start], self.f[11], (7, 20, 33)
        for text, yy in zip(lines, ys):
            d.text((PANEL_W - 14, yy), text, font=font, fill=MUTED, anchor="ra")
        # tick / step / SR
        y = 48
        d.text((x0, y + 4), "第", font=self.fb[16], fill=INK)
        tw = self._game_text(im, str(tick), (x0 + 22, y), 26)
        d.text((x0 + 26 + tw, y + 4), "tick", font=self.fb[16], fill=INK)
        if g > self.agent_start and g <= self.n:
            step = (g - self.agent_start - 1) // 2 + 1
            d.text((PANEL_W - 14, y + 6), f"智能体第 {step} 步", font=self.f[13], fill=AMBER, anchor="ra")
            d.text((PANEL_W - 14, y + 23), "每步 2 tick", font=self.f[11], fill=MUTED, anchor="ra")
        y = 86
        sr = self.sr[min(max(g - 1, 0), len(self.sr) - 1)] if len(self.sr) else 0.0
        d.text((x0, y + 5), "SR 计时", font=self.f[14], fill=INK)
        tw = self._game_text(im, f"{sr / 1000:.3f}", (x0 + 62, y), 24)
        d.text((x0 + 66 + tw, y + 5), "秒", font=self.f[14], fill=INK)
        y = 118
        if rec is not None:
            ball = LABELS["ball"].get(rec.get("ball"), "—")
            d.text((x0, y), f"{ball} · 速度 {rec['speed']:.1f}", font=self.f[14], fill=INK)
            d.text((PANEL_W - 14, y), f"第 {rec['sector']} {LABELS['sector']} · {LABELS['checkpoint']} {rec['cps']}",
                   font=self.f[14], fill=INK, anchor="ra")
        self._hr(d, 142)
        # live keys
        held = int(self.frames["keys"][g - 1]) if 1 <= g <= self.n else 0
        self._keys(d, x0, 150, held)
        # key log
        self._hr(d, 222)
        d.text((x0, 228), "按键记录（游戏收到的输入）", font=self.fb[13], fill=BRASS_HI)
        self._key_log(d, x0, 248, tick)
        self._hr(d, 436)
        # policy + terrain
        self._policy(d, x0, 442, kind, rec, held)
        self._terrain(im, d, 200, 442, rec)
        self._hr(d, 612)
        self._reward(d, x0, 618, kind, rec, g)
        d.text((PANEL_W // 2, H - 14), "画面：原版游戏录制 · 面板：由这次重放重算", font=self.f[11], fill=MUTED, anchor="mm")
        return im

    def _hr(self, d, y):
        d.line([(10, y), (PANEL_W - 10, y)], fill=(107, 74, 23), width=1)

    def _game_text(self, im, text, xy, h) -> int:
        """Draws text in the game font; returns its width."""
        if self.gf.ok:
            t = self.gf.render(text, h)
            im.paste(t, xy, t)
            return t.size[0]
        from PIL import ImageDraw
        dr = ImageDraw.Draw(im)
        dr.text(xy, text, font=self.fb[20], fill=AMBER)
        return int(dr.textlength(text, font=self.fb[20]))

    def _cap(self, d, box, label, on, small=False):
        fill = (255, 207, 90) if on else (28, 30, 44)
        d.rounded_rectangle(box, radius=4, fill=fill, outline=BRASS if not on else (255, 240, 168), width=1)
        d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), label, font=self.fb[13 if small else 16],
               fill=(30, 18, 4) if on else MUTED, anchor="mm")

    def _keys(self, d, x0, y0, held):
        s = 30
        cx = x0 + s + 4
        self._cap(d, [cx, y0, cx + s, y0 + s - 2], "↑", held & tas.KEY_UP)
        self._cap(d, [cx - s - 2, y0 + s, cx - 2, y0 + 2 * s - 2], "←", held & tas.KEY_LEFT)
        self._cap(d, [cx, y0 + s, cx + s, y0 + 2 * s - 2], "↓", held & tas.KEY_DOWN)
        self._cap(d, [cx + s + 2, y0 + s, cx + 2 * s + 2, y0 + 2 * s - 2], "→", held & tas.KEY_RIGHT)
        bx = x0 + 120
        for i, (bit, lab) in enumerate([(tas.KEY_SHIFT, "Shift"), (tas.KEY_SPACE, "Space"), (tas.KEY_Q, "Q"),
                                        (tas.KEY_ESC, "Esc")]):
            xx = bx + (i % 2) * 96
            yy = y0 + (i // 2) * s
            self._cap(d, [xx, yy, xx + 90, yy + s - 2], lab, held & bit, small=True)

    @staticmethod
    def _key_names(mask):
        return " ".join(lab for bit, lab in KEY_BITS if mask & bit)

    def _key_log(self, d, x0, y0, tick):
        j = int(np.searchsorted(self.change_ticks, tick, side="right"))
        rows = self.changes[max(0, j - 12):j]
        d.text((x0, y0), "tick", font=self.f[11], fill=MUTED)
        d.text((x0 + 62, y0), "变化", font=self.f[11], fill=MUTED)
        d.text((PANEL_W - 14, y0), "来源", font=self.f[11], fill=MUTED, anchor="ra")
        y = y0 + 15
        for n, (t, before, after) in enumerate(reversed(rows)):
            down, up = after & ~before, before & ~after
            txt = " ".join(([f"+{self._key_names(down)}"] if down else []) + ([f"−{self._key_names(up)}"] if up else []))
            src = self.source(t)
            alpha = 1.0 - n * 0.055
            col = tuple(int(c * alpha + 9 * (1 - alpha)) for c in (INK if n else AMBER))
            d.text((x0, y), str(t), font=self.f[12], fill=col)
            d.text((x0 + 62, y), txt[:22], font=self.f[12], fill=col)
            sc = {LABELS["src_tas"]: BLUE, LABELS["src_agent"]: AMBER}.get(src, MUTED)
            d.text((PANEL_W - 14, y), src, font=self.f[12], fill=tuple(int(c * alpha + 9 * (1 - alpha)) for c in sc),
                   anchor="ra")
            y += 13.5

    def _policy(self, d, x0, y0, kind, rec, held):
        """3x3 grid in screen directions (the arrow keys each action presses now): fill and bar = probability,
        outline = the action taken."""
        from ballrain.env import ARROWS9, world_to_camera_keys
        d.text((x0, y0), "策略（按键方向）" if kind == "agent" else "TAS 输入（按键方向）", font=self.fb[13], fill=BRASS_HI)
        cs, gap = 45, 3
        gy = y0 + 20
        pos = {0: (1, 1), tas.KEY_UP: (1, 0), tas.KEY_DOWN: (1, 2), tas.KEY_LEFT: (0, 1), tas.KEY_RIGHT: (2, 1),
               tas.KEY_UP | tas.KEY_LEFT: (0, 0), tas.KEY_UP | tas.KEY_RIGHT: (2, 0),
               tas.KEY_DOWN | tas.KEY_LEFT: (0, 2), tas.KEY_DOWN | tas.KEY_RIGHT: (2, 2)}
        arrow = {(1, 1): "·", (1, 0): "↑", (1, 2): "↓", (0, 1): "←", (2, 1): "→", (0, 0): "↖", (2, 0): "↗", (0, 2): "↙",
                 (2, 2): "↘"}
        cell = {}
        if kind == "agent" and rec is not None and rec.get("cam") is not None:
            cam = np.asarray(rec["cam"])
            for i, a in enumerate(ARROWS9):
                cell[pos.get(world_to_camera_keys(a, cam) & tas.ARROW_MASK, (1, 1))] = (rec["probs"][i], i == rec["a"])
        taken_pos = pos.get(held & tas.ARROW_MASK)
        for (cx, cy), lab in arrow.items():
            bx, by = x0 + cx * (cs + gap), gy + cy * (cs + gap)
            p, taken = cell.get((cx, cy), (None, False))
            if kind != "agent":
                taken = (cx, cy) == taken_pos
            shade = int(28 + 120 * (p or 0) ** 0.5) if p is not None else 28
            fill = (shade, int(shade * 0.75), int(shade * 0.3)) if p is not None else (22, 26, 40)
            d.rectangle([bx, by, bx + cs, by + cs], fill=fill, outline=(255, 240, 168) if taken else (70, 56, 30),
                        width=3 if taken else 1)
            d.text((bx + cs / 2, by + 14), lab, font=self.sym, fill=AMBER if taken else INK, anchor="mm")
            if p is not None:
                d.text((bx + cs / 2, by + 30), f"{100 * p:.0f}%", font=self.f[11], fill=INK, anchor="mm")
                d.rectangle([bx + 4, by + cs - 7, bx + 4 + (cs - 8) * p, by + cs - 4], fill=AMBER2)

    def _terrain(self, im, d, x0, y0, rec):
        """What the agent senses around the ball, turned so that up = the camera's forward (as on the screen):
        the 7 x 7 floor heights under and around the ball and the 8 distance-to-edge rays."""
        d.text((x0, y0), "周围地形（观测）", font=self.fb[13], fill=BRASS_HI)
        size = 128
        cx, cy = x0 + size / 2, y0 + 20 + size / 2
        d.rectangle([x0, y0 + 20, x0 + size, y0 + 20 + size], fill=(5, 8, 18), outline=(70, 56, 30))
        if rec is None or not rec.get("valid") or rec.get("fan") is None:
            d.text((cx, cy), "—", font=self.f[14], fill=MUTED, anchor="mm")
            return
        yaw = rec["yaw"]
        cy_, sy_ = math.cos(yaw), math.sin(yaw)

        def scr(wx, wz, scale):  # world (x right, z forward) -> panel, camera forward up
            # world direction angle from +z toward +x is atan2(x, z); on screen subtract the camera yaw
            sx = wx * cy_ - wz * sy_
            sz = wx * sy_ + wz * cy_
            return cx + sx * scale, cy - sz * scale

        hs, hit = rec.get("heights"), rec.get("hit")
        if hs is not None:
            g = int(round(len(hs) ** 0.5))
            sp = 1.0
            sc = size / (2 * 8.5)
            for ix in range(g):
                for iz in range(g):
                    j = ix * g + iz
                    ox, oz = (ix - (g - 1) / 2) * sp, (iz - (g - 1) / 2) * sp
                    px, py = scr(ox, oz, sc)
                    if not hit[j]:
                        col = (40, 16, 16)
                    else:
                        h = max(-5.0, min(5.0, hs[j]))
                        t = (h + 5) / 10
                        col = (int(60 + 140 * t), int(50 + 110 * t), int(30 + 60 * t))
                    r = sc * 0.45
                    d.rectangle([px - r, py - r, px + r, py + r], fill=col)
        fan = rec["fan"]
        sc = size / (2 * 8.5)
        for i, dist in enumerate(fan):
            ang = i * 2 * math.pi / len(fan)  # 0 = +z, clockwise toward +x
            wx, wz = math.sin(ang), math.cos(ang)
            ex, ey = scr(wx * dist, wz * dist, sc)
            near = dist < 2.0
            d.line([(cx, cy), (ex, ey)], fill=BAD if near else AMBER, width=2)
            d.ellipse([ex - 2.5, ey - 2.5, ex + 2.5, ey + 2.5], fill=BAD if near else AMBER)
        d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(214, 170, 110), outline=(60, 40, 10))
        d.text((x0 + size / 2, y0 + 20 + size + 8), "射线 = 到边缘的距离（最远 8）", font=self.f[11], fill=MUTED, anchor="mm")

    def _reward(self, d, x0, y0, kind, rec, g):
        if kind != "agent" or rec is None or g <= self.agent_start:
            d.text((x0, y0), "奖励与价值", font=self.fb[13], fill=BRASS_HI)
            d.text((x0, y0 + 26), "TAS 控制中：没有奖励、价值或策略输出", font=self.f[13], fill=MUTED)
            d.text((x0, y0 + 46), "（智能体从接手帧开始控制）", font=self.f[12], fill=MUTED)
            return
        d.text((x0, y0), "奖励与价值", font=self.fb[13], fill=BRASS_HI)
        d.text((PANEL_W - 14, y0), f"价值估计 {rec['v']:+.2f}", font=self.fb[14], fill=AMBER, anchor="ra")
        r = rec["r"]
        d.text((x0, y0 + 22), "本步奖励", font=self.f[13], fill=INK)
        d.text((x0 + 70, y0 + 22), f"{r:+.4f}", font=self.fb[14], fill=GOOD if r >= 0 else BAD)
        d.text((PANEL_W - 14, y0 + 22), f"累计回报 {rec['ret']:+.2f}", font=self.fb[14], fill=INK, anchor="ra")
        terms = rec.get("terms") or {}
        y = y0 + 46
        for n, (key, lab) in enumerate(TERM_LABELS):
            v = terms.get(key, 0.0)
            xx = x0 + (n % 2) * 160
            yy = y + (n // 2) * 19
            col = MUTED if abs(v) < 1e-9 else (GOOD if v > 0 else BAD)
            d.text((xx, yy), lab, font=self.f[12], fill=MUTED)
            d.text((xx + 150, yy), f"{v:+.4f}", font=self.f[12], fill=col, anchor="ra")


def render_overlay(work: Path, frames: np.ndarray, panel: dict, meta_pub: dict, dst: Path, *, fps: float,
                   crf: int = 26, maxrate: str = "1400k", log=print) -> Path:
    """Decode the captured video, put the panel next to every frame and encode the published file."""
    from PIL import Image
    cap = json.loads((work / "capture.json").read_text(encoding="utf-8"))
    if tuple(cap.get("size") or ()) != (GAME_W, H):  # the panel layout and the published size assume 1024x768
        raise RuntimeError(f"capture size {cap.get('size')} is not {GAME_W}x{H}")
    rows = cap["rows"]
    nvid = int(cap["video_frames"])
    game_frame = np.zeros(nvid, np.int64)
    for first, rep, gf, _ms in rows:
        game_frame[first:first + rep] = gf
    pn = Panel(panel, frames, meta_pub)
    ff = capture.ffmpeg_exe()
    dec = subprocess.Popen([ff, "-hide_banner", "-loglevel", "error", "-i", str(work / "capture.mp4"), "-f", "rawvideo",
                            "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
                           creationflags=capture.low_priority_flags(idle=True))
    enc = subprocess.Popen([ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow",
                            "-crf", str(crf), "-maxrate", maxrate, "-bufsize", "2800k", "-profile:v", "high",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2", str(dst)],
                           stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=capture.low_priority_flags(idle=True))
    fsize = GAME_W * H * 3
    canvas = Image.new("RGB", (W, H))
    last_g, last_panel = None, None
    t0 = time.time()
    try:
        for j in range(nvid):
            buf = dec.stdout.read(fsize)
            if len(buf) < fsize:
                raise RuntimeError(f"capture ended at video frame {j} of {nvid}")
            canvas.paste(Image.frombuffer("RGB", (GAME_W, H), buf, "raw", "RGB", 0, 1), (0, 0))
            g = int(game_frame[j])
            if g != last_g:
                last_panel = pn.draw(g)
                last_g = g
            canvas.paste(last_panel, (GAME_W, 0))
            enc.stdin.write(canvas.tobytes())
        if dec.stdout.read(1):
            raise RuntimeError(f"the capture has more than the {nvid} frames its sidecar lists")
    finally:
        enc.stdin.close()
        err = enc.stderr.read().decode("utf-8", "replace")
        enc.wait()
        dec.stdout.close()
        dec.wait()
    if enc.returncode != 0:
        raise RuntimeError(f"overlay encode failed: {err[-500:]}")
    if dec.returncode != 0:
        raise RuntimeError(f"decoding the capture failed (exit {dec.returncode})")
    log(f"  overlay: {nvid} frames in {time.time() - t0:.0f}s, {dst.stat().st_size} bytes")
    return dst
