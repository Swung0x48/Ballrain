"""Single-life controller; only reserved copies, isolated output, no respawn."""

from pathlib import Path
import sys, tempfile, json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ballrain import tas
from ballrain.client import AgentSession, SessionConfig
from ballrain.headless import HeadlessConfig, HeadlessLauncher
from ballrain.env import world_to_camera_keys, EPISODE_STOP
from ballrain.protocol import Ev

LIVE = Path(__import__("os").environ.get("BALLRAIN_RUNTIME") or Path(__file__).resolve().parents[1] / "runtime")
OUT = LIVE / "viz/l2branch2"
REC = LIVE / "datasets/tas-v1/SR02_1_28_992_264"
# (x,z) gates, optionally requiring material or minimum altitude
ROUTES = {
    2: [
        (709.6, -282.2, "stone"),
        (704, -273),
        (698, -265),
        (688.47, -265.57, "wood"),
        (697.8, -269),
        *[(697.7, z) for z in range(-276, -337, -6)],
        (747, -358),
        (766, -348),
        (790, -342),
        (843, -342),
    ],
    3: [
        (906, -346.8),
        (916.35, -346.74, "paper"),
        (988, -346.74),
        (991.4, -366.74),
        (1016.67, -366.74, 61),
        (1006.67, -366.74, 72),
        (1026.68, -366.74, "stone"),
        (1020, -370),
        (992, -370),
        (991.5, -386.7),
        (931.7, -386.7),
        (931.7, -375),
        (931.7, -367),
        (931.4, -358),
        (927, -320),
        (926.5, -300),
        (927.07, -261.52, "paper"),
        (941, -206),
        (935, -145),
        (929, -96),
    ],
}


def run(section, build):
    copy = {"x86soft": "hlw-96", "x86p2002f-respawnfix": "hlw-97"}[build]
    out = Path(tempfile.mkdtemp(prefix=f"{copy}-s{section}-", dir=OUT))
    print(out, flush=True)
    if section == 3 and "--unlock" in sys.argv:
        route_override = (
            ROUTES[3][:7]
            + [
                (1015, -366.7),
                (992, -366.7),
                (941, -366.7),
                (936, -366.7),
                (991, -366.7),
            ]
            + ROUTES[3][9:]
        )
    elif section == 2 and "--pit" in sys.argv:
        route_override = [
            ROUTES[2][0],
            (715.98, -267.86, "rest"),
            (716, -261.9),
            (714, -275),
            (703, -272),
            *ROUTES[2][3:],
        ]
    else:
        route_override = ROUTES[section]
    k = {2: 8200, 3: 10660}[section]
    prefix = out / "prefix.bin"
    tas.save_raw(prefix, tas.load_raw(REC / "run/framelog.bin")[:k])
    hl = LIVE / "hl-win"
    s = AgentSession(
        SessionConfig(
            level=2,
            copy=copy,
            out_dir=out,
            prefix=prefix,
            hello_timeout_s=30,
            obs_timeout_s=30,
        ),
        HeadlessLauncher(
            HeadlessConfig(
                binary=str(hl / f"bin-{build}" / "BallanceMMOSimTool.exe"),
                root=str(hl / "game"),
                socket="tcp",
            )
        ),
    )
    stage = 0
    route = route_override
    records = []
    contacts = []
    mechanisms = []
    try:
        o = s.start()
        for step in range(2300):
            if s.ended or o is None or o.has_event(Ev.BallOff):
                break
            b = o.ball
            keys = 0
            if b is not None and b["found"]:
                p = np.asarray(b["pos"])
                v = np.asarray(b["speed"]) + b["speed_change"]
                bt = int(b["type"])
                goal = route[stage]
                d = np.array(goal[:2]) - p[[0, 2]]
                norm = np.linalg.norm(d)
                records.append(
                    (
                        s.frames_done - k,
                        *p,
                        bt,
                        stage,
                        s.tracker.sector,
                        s.tracker.sr_ms,
                    )
                )
                contacts.extend(
                    (s.frames_done - k, s.tracker.level.name(int(c["partner"])), bt)
                    for c in o.fric
                )
                if section == 3 and step % 8 == 0:
                    for body in o.body:
                        name = s.tracker.level.name(int(body["ckid"]))
                        if "P_Modul_25" in name or "P_Modul_01_Pusher" in name:
                            mechanisms.append(
                                (
                                    s.frames_done - k,
                                    name,
                                    *body["pos"].tolist(),
                                    *body["q"].tolist(),
                                )
                            )
                if s.tracker.sector > section or p[1] < -5:
                    break
                if (
                    len(goal) == 2
                    and (
                        (
                            section == 3
                            and "--unlock" not in sys.argv
                            and stage in (11, 12, 13)
                            and p[2] >= goal[1]
                        )
                        or section == 2
                        and "--pit" in sys.argv
                        and stage == 4
                        and p[0] < 705
                        or norm < (2.3 if section == 3 and stage == 10 else 1.5)
                        or section == 2
                        and stage >= (18 if "--pit" in sys.argv else 16)
                        and p[0] >= goal[0] - 1
                        or section == 3
                        and "--unlock" in sys.argv
                        and stage >= 17
                        and p[2] >= goal[1] - 1
                    )
                    or len(goal) == 3
                    and (
                        bt == {"wood": 0, "paper": 1, "stone": 2}.get(goal[2], -1)
                        or isinstance(goal[2], int)
                        and p[1] >= goal[2]
                        or goal[2] == "rest"
                        and norm < 0.5
                        and np.linalg.norm(v[[0, 2]]) < 0.5
                    )
                ) and stage + 1 < len(route):
                    stage += 1
                fast = section == 2 or (stage >= 11 and "--unlock" not in sys.argv)
                desired = (
                    np.clip(d * 2, -26, 26)
                    if fast
                    else d * min(2, 14 / max(0.01, norm))
                )
                desired[0] = (
                    26
                    if section == 2 and stage >= (18 if "--pit" in sys.argv else 16)
                    else desired[0]
                )
                desired[1] = (
                    26
                    if section == 3
                    and "--unlock" not in sys.argv
                    and stage in (11, 12, 13)
                    else desired[1]
                )
                if_fast_exit = section == 3 and "--unlock" in sys.argv and stage >= 17
                desired[1] = 27 if if_fast_exit and stage != 19 else desired[1]
                desired[0] = (
                    np.clip(
                        ((931.4 - p[0]) if stage == 17 and p[2] < -328 else d[0]) * 2,
                        -18,
                        18,
                    )
                    if if_fast_exit
                    else desired[0]
                )
                w = desired - v[[0, 2]]
                threshold = 0.5 if fast or if_fast_exit else 0.28 * np.linalg.norm(w)
                mask = (
                    (tas.KEY_RIGHT if w[0] > 0 else tas.KEY_LEFT)
                    if abs(w[0]) > threshold
                    else 0
                )
                if abs(w[1]) > threshold:
                    mask |= tas.KEY_UP if w[1] > 0 else tas.KEY_DOWN
                if o.cams is not None:
                    keys = world_to_camera_keys(mask, o.cams["cams"][0]["matrix"])
            o = s.act(
                tas.make_frames([1000 / 132], [keys]), repeat=6, stop=EPISODE_STOP
            )
    finally:
        s.close(kill=True)
        np.savez_compressed(
            out / "trace.npz", samples=records, frames=s.frame_log()[k:], prefix_frame=k
        )
        (out / "mechanisms.json").write_text(
            json.dumps(dict(contacts=contacts, poses=mechanisms)), encoding="utf8"
        )
        print(records[-1:], flush=True)


if __name__ == "__main__":
    for arg in sys.argv[2:]:
        if arg.isdigit():
            run(int(arg), sys.argv[1])
