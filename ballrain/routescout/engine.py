"""Cold-start agent search. Geometry proposes controls; only observations score them."""

from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import tempfile
import re

import numpy as np
from PIL import Image, ImageDraw
from ballrain import tas
from ballrain.client import AgentSession, DEFAULT_DELTA, SessionConfig
from ballrain.env import world_to_camera_keys
from ballrain.features import level_geometry
from ballrain.headless import HeadlessLauncher
from ballrain.path import level_anchors
from ballrain.protocol import Ev
from ballrain.reach import KEY_COMBOS
from ballrain.state import G_RAILS
from .contract import MATERIALS, UNVERIFIED, difficulty, digest
from .graph import SurfaceGraph
from .guidance import ribbon, project, point, local_line

STOPS = (
    Ev.BallOff,
    Ev.GameOver,
    Ev.PostCheckpoint,
    Ev.LevelFinish,
    Ev.ExtraPoint,
    Ev.PostLifeUp,
    Ev.PauseLevel,
    Ev.UnpauseLevel,
)


def build_id(path):
    if path.name.lower() not in ("ballancemmosimtool.exe", "ballancemmosimtool"):
        raise ValueError("only a headless BallanceMMOSimTool executable is supported")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def launch(level, root, build, instance, out):
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        launcher = HeadlessLauncher(
            binary=str(build.resolve()),
            root=str(root.resolve()),
            socket="tcp",
            log=False,
        )
        s = AgentSession(
            SessionConfig(
                level, Path(tmp), copy=instance, hello_timeout_s=20, obs_timeout_s=20
            ),
            launcher,
        )
        try:
            s.start()
            tr = s.tracker
            if not tr.flags.started:
                s.run_until(0, [Ev.StartLevel], limit=3000)
            if not tr.flags.cam_nav and not tr.flags.ball_nav:
                s.run_until(0, [Ev.CamNavActive, Ev.BallNavActive], limit=400)
            if not tr.flags.cam_nav and not tr.flags.ball_nav:
                for _ in range(200):
                    s.act(
                        tas.make_frames([DEFAULT_DELTA] * 2, [tas.KEY_Q, 0]),
                        stop=[Ev.BallNavActive, Ev.CamNavActive],
                    )
                    if tr.flags.cam_nav or tr.flags.ball_nav:
                        break
            if not tr.flags.ball_nav:
                s.run_until(0, [Ev.BallNavActive], limit=3000)
            if not tr.flags.ball_nav:
                raise ValueError("no controllable map spawn")
            yield s
        finally:
            s.close(kill=True)


def position(s):
    return np.asarray(s.obs.ball["pos"])


def guide(s, target):
    a = level_anchors(s.tracker)
    pts = [a["start"], *a["checkpoints"]]
    tris = level_geometry(s.tracker).tris
    graph = SurfaceGraph(tris)
    if target >= len(pts):
        raise ValueError("target checkpoint outside map")
    paths = []
    for x, y in zip(pts[:target], pts[1:]):
        line = local_line(tris, np.asarray(x), np.asarray(y))
        paths.append(line or ribbon([x, *graph.corridor(x, y), y]))
    triangles = np.stack([tris.A, tris.B, tris.C], axis=1)[abs(tris.N[:, 1]) > 0.45][
        :, :, [0, 2]
    ]
    return paths, triangles


def draw(out, triangles, rows):
    # Zoom to the union of the explored sectors, rather than shrinking the split
    # to a few pixels to show distant floors that no candidate visited.
    points = (
        np.asarray([p for r in rows for _, p in r["centre_path"]])[:, [0, 2]]
        if rows
        else triangles.reshape(-1, 2)
    )
    lo = points.min(axis=0) - 6
    extent = points.max(axis=0) + 6 - lo
    scale = min(940 / extent[0], 620 / extent[1])
    width = 980
    height = 52 + int(scale * extent[1])

    def xy(v):
        return tuple(map(int, (v - lo) * scale + [20, 32]))

    im = Image.new("RGB", (width, height + 20 * ((len(rows) + 2) // 3) + 16), "#14202b")
    pen = ImageDraw.Draw(im)
    for t in triangles:
        if np.all(t.max(axis=0) >= lo) and np.all(t.min(axis=0) <= lo + extent):
            pen.polygon([xy(v) for v in t], fill="#526371")
    pen.rectangle((0, 0, width, 24), fill="#14202b")
    pen.text(
        (10, 6),
        "Route scout  |  top view (+X right, +Z down)  |  sampled ball centres",
        fill="white",
    )
    pen.rectangle((0, height, width, im.height), fill="#14202b")
    colors = ("#ffbb55", "#70e0ff", "#ff8ac1", "#a2f38a", "#dca5ff", "#fb826b")
    for i, r in enumerate(rows):
        color = colors[i % len(colors)]
        line = [xy(np.asarray(p)[[0, 2]]) for _, p in r["centre_path"]]
        if len(line) > 1:
            pen.line(line, fill=color, width=2)
        x, y = line[-1]
        pen.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        label = "checkpoint" if r["complete"] else "partial"
        pen.text(
            (12 + 318 * (i % 3), height + 20 * (i // 3)),
            f"{r['id']}   {label}   {r['sr_ms'] / 1000:.2f}s",
            fill=color,
        )
    path = out / "top-view.tmp"
    im.save(path, format="PNG")
    path.replace(out / "top-view.png")


def steer(s, path, look, speed, rng):
    if not s.tracker.flags.ball_nav or not s.obs.ball["found"]:
        return 0
    p = position(s)
    station, _ = project(path, p)
    target = point(path, station + look)
    d = (target - p)[[0, 2]]
    v = (s.obs.ball["speed"] + s.obs.ball["speed_change"])[[0, 2]]
    want = d * min(2.7, speed / max(0.01, np.linalg.norm(d))) - v
    key = (8 if want[0] > 0.55 else 4 if want[0] < -0.55 else 0) | (
        1 if want[1] > 0.55 else 2 if want[1] < -0.55 else 0
    )
    if rng.random() < 0.008:
        key = rng.choice(KEY_COMBOS)
    return int(world_to_camera_keys(key, s.obs.cams["cams"][0]["matrix"]))


def act(s, k, n):
    before = s.frames_done
    o = s.act(tas.make_frames([DEFAULT_DELTA], [k]), repeat=n, stop=STOPS)
    used = s.frames_done - before
    return (
        used
        if used
        and o is not None
        and not (o.has_event(Ev.BallOff) or o.has_event(Ev.GameOver))
        else 0
    )


def ready(s, cfg):
    if s.ended or not s.obs.ball["found"] or s.tracker.ball_type() not in range(3):
        return False
    if cfg["material"] and MATERIALS[s.tracker.ball_type()] != cfg["material"]:
        return False
    if s.tracker.checkpoints != (cfg["from_checkpoint"] or 0):
        return False
    pose = cfg["spawn"]
    if not pose:
        return True
    if np.linalg.norm(position(s) - pose[:3]) > 0.5:
        return False
    if len(pose) == 4:
        f = s.obs.cams["cams"][0]["matrix"][2]
        yaw = math.degrees(math.atan2(f[0], f[2]))
        if abs((yaw - pose[3] + 180) % 360 - 180) > 2:
            return False
    return True


class Trace:
    def __init__(self, s, rules):
        self.s = s
        self.rules = rules
        tr = s.tracker
        self.frame = s.frames_done
        self.sr = tr.sr_ms
        self.hs = tr.hs_partial()
        self.banks = []
        self.pause = None
        self.keys = []
        self.points = [[0, position(s).tolist()]]
        self.balls = [MATERIALS[tr.ball_type()]]
        self.transform = []
        self.cp = []
        self.hit = []
        self.pick = []
        self.rail = 0
        self.samples = 0
        self.rails = (
            tr.level.groups[G_RAILS].members if G_RAILS in tr.level.groups else set()
        )
        self.proposal = "engine exploration"
        self.witness = hashlib.sha256()
        self.observe_hash()

    def observe_hash(self):
        s = self.s
        o = s.obs
        # Same contact/body witness as the segment certifier, at most four frames apart.
        contacts = set(map(int, o.fric["partner"]))
        bodies = [
            [
                int(body["ckid"]),
                *[body[k].tolist() for k in ("pos", "q", "speed", "rot_speed")],
            ]
            for body in o.body
            if int(body["ckid"]) in contacts
        ]
        self.witness.update(
            digest(
                [
                    bodies,
                    s.frames_done,
                    position(s).tolist(),
                    o.ball["speed"].tolist(),
                    o.ball["speed_change"].tolist(),
                    s.tracker.ball_type(),
                    o.fric["partner"].tolist(),
                    s.tracker.sr_ms,
                    s.tracker.hs_partial(),
                ]
            ).encode()
        )

    def step(self, key, n):
        used = 0
        while used < n:
            got = self._sample(key, min(4, n - used))
            if not got:
                return False
            used += got
            if any(self.s.obs.has_event(code) for code in STOPS):
                break
        self.keys.append([key, used])
        return True

    def _sample(self, key, n):
        s = self.s
        tr = s.tracker
        prev_points = tr.points
        followers = tr.followers()
        used = act(s, key, n)
        if not used:
            return False
        frame = s.frames_done - self.frame
        o = s.obs
        for e in o.events:
            t = int(e["frame"]) - self.frame
            code = int(e["code"])
            if code == Ev.PostCheckpoint:
                self.cp.append([tr.checkpoints, t])
                self.rules.refresh()
            if code == Ev.PostLifeUp:
                self.pick.append([t, "life"])
            if code == Ev.PauseLevel:
                self.pause = [t, int(followers), int(tr.points)]
            if code == Ev.UnpauseLevel and self.pause:
                f, b, p = self.pause
                gained = tr.points - p
                if gained > 0 and tr.followers() < b:
                    self.banks.append(
                        dict(
                            pause=f,
                            unpause=t,
                            points=gained,
                            followers=b - tr.followers(),
                        )
                    )
                self.pause = None
        if tr.points - prev_points >= 98 and not tr.flags.paused:
            hidden = [tr.level.name(int(w["ckid"])) for w in o.wtch if not w["visible"]]
            if any(
                re.fullmatch(r"P_Extra_Point_Ball0\d*", name or "") for name in hidden
            ):
                self.pick.append([frame, "point"])
        if o.ball["found"] and tr.ball_type() in range(3):
            p = position(s)
            b = tr.ball_type()
            self.samples += 1
            if frame - self.points[-1][0] >= 16 or o.has_event(Ev.PostCheckpoint):
                self.points.append([frame, p.tolist()])
            if self.balls[-1] != MATERIALS[b]:
                self.transform.append([frame, self.balls[-1], MATERIALS[b]])
                self.balls.append(MATERIALS[b])
            contacts = set(map(int, o.fric["partner"]))
            self.rail += bool(contacts.intersection(self.rails))
            for ck in sorted(contacts):
                name = tr.level.name(ck)
                typ = self.rules.bindings.get(
                    (tr.sector, name), self.rules.parts.get(name)
                )
                if typ:
                    hit = [name, typ, self.rules.rule(typ, b)]
                    if hit not in self.hit:
                        self.hit.append(hit)
        self.observe_hash()
        return used

    def candidate(self, cfg, prefix, goal):
        s = self.s
        frac = self.rail / max(1, self.samples)
        path = list(self.points)
        frame = s.frames_done - self.frame
        if path[-1][0] != frame:
            path.append([frame, position(s).tolist()])
        points = np.array([p for _, p in path])
        # Signed excursion relative to the start-to-goal chord separates geometric lanes.
        axis = (goal - points[0])[[0, 2]]
        side = np.array([-axis[1], axis[0]]) / max(1, np.linalg.norm(axis))
        offsets = (points - points[0])[:, [0, 2]] @ side
        excursion = offsets[np.argmax(abs(offsets))]
        family = [
            self.balls,
            sorted({x[1] for x in self.hit}),
            "rail" if self.rail else "no-rail-observed",
            int(np.trunc(excursion / 8)),
            sorted({kind for _, kind in self.pick}),
            bool(self.banks),
        ]
        return dict(
            id=digest([prefix, self.keys])[:12],
            family=json_family(family),
            segments=list(
                range(
                    (cfg["from_checkpoint"] or 0) + 1,
                    min(s.tracker.checkpoints + 1, cfg["target"]) + 1,
                )
            ),
            checkpoints=list(self.cp),
            complete=s.tracker.checkpoints >= cfg["target"],
            distance_to_goal=round(float(np.linalg.norm(points[-1] - goal)), 3),
            start_keys=list(prefix),
            keys=list(self.keys),
            centre_path=path,
            proposal=self.proposal,
            ball_sequence=list(self.balls),
            transformer_events=list(self.transform),
            mechanisms=list(self.hit),
            sr_ms=s.tracker.sr_ms - self.sr,
            game_ms=(s.frames_done - self.frame) * float(DEFAULT_DELTA),
            hs=dict(
                pickups=list(self.pick),
                esc_banks=list(self.banks),
                score_delta=s.tracker.hs_partial() - self.hs,
            ),
            difficulty=difficulty(self.keys, self.balls, frac),
            witness=self.witness.hexdigest(),
            search_build=cfg["build"],
            verification=dict(status=UNVERIFIED),
        )


def json_family(parts):
    import json

    return json.dumps(parts, separators=(",", ":"))


class Detours:
    """Alternate between direct, pickup/bank and material exploration episodes."""

    def __init__(self, rules, rng):
        self.rules = rules
        self.mode = rng.choice(("direct", "direct", "pickup", "transform"))
        self.visited = set()
        self.target = None
        self.until = self.release = self.pause_start = 0

    def action(self, s, path, look, speed, rng, block):
        tr = s.tracker
        if self.mode == "pickup":
            if self.release:
                self.release = 0
                return 0, 2
            if tr.flags.paused:
                if tr.followers() and s.frames_done - self.pause_start < 1500:
                    return 0, 12
                self.release = 1
                return tas.KEY_ESC, 1
            if tr.flags.ball_nav and tr.followers():
                self.pause_start = s.frames_done
                self.release = 1
                return tas.KEY_ESC, 1
        if (
            self.mode != "direct"
            and tr.flags.ball_nav
            and s.obs.ball["found"]
            and tr.ball_type() in range(3)
        ):
            p = position(s)
            if self.target is not None and (
                s.frames_done >= self.until or np.linalg.norm(p - self.target) < 2
            ):
                self.target = None
            if self.target is None:
                for i in np.argsort(np.linalg.norm(self.rules.centers - p, axis=1)):
                    q = self.rules.centers[i]
                    if np.linalg.norm(q - p) > 18:
                        break
                    row = self.rules.rows[i]
                    if i in self.visited or row["sector"] not in (0, tr.sector):
                        continue
                    rule = self.rules.rule(row["type"], tr.ball_type())
                    if (rule == "pickup" and self.mode == "pickup") or (
                        rule.startswith("transform_")
                        and self.mode == "transform"
                        and rule != "transform_" + MATERIALS[tr.ball_type()]
                    ):
                        self.visited.add(i)
                        self.target = q
                        self.until = s.frames_done + 360
                        break
            if self.target is not None:
                path = ribbon([p, self.target])
        return steer(s, path, look, speed, rng), block
