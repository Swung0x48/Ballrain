"""Single-life L2 TAS handover. Never respawn; keep logs in a new private run directory."""

from pathlib import Path
import sys, tempfile, json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ballrain import tas
from ballrain.client import AgentSession, SessionConfig
from ballrain.headless import HeadlessConfig, HeadlessLauncher
from ballrain.env import world_to_camera_keys, EPISODE_STOP
from ballrain.path import ReferencePath, asset_path, file_sha256
from ballrain.protocol import Ev


def run(live, build, k):
    copy = {"x86soft": "hlw-98", "x86p2002d": "hlw-99"}[build]
    out = Path(tempfile.mkdtemp(prefix=copy + "-", dir=live / "viz/l2s4/stone"))
    prefix = out / "prefix.bin"
    rec = live / "datasets/tas-v1/SR02_1_28_992_264"
    frames = tas.load_raw(rec / "run/framelog.bin")
    tas.save_raw(prefix, frames[:k])
    hl = live / "hl-win"
    cfg = HeadlessConfig(
        binary=str(hl / ("bin-" + build) / "BallanceMMOSimTool.exe"),
        root=str(hl / "game"),
        socket="tcp",
    )
    s = AgentSession(
        SessionConfig(
            level=2,
            copy=copy,
            out_dir=out,
            prefix=prefix,
            hello_timeout_s=12,
            obs_timeout_s=30,
        ),
        HeadlessLauncher(cfg),
    )
    print(out, flush=True)
    with (out / "controller.jsonl").open("w") as log:
        try:
            o = s.start()
            target = np.array([948.74, -56.51])
            branch = None
            prev = None
            stone = False
            board = False
            samples = []
            for i in range(6500):
                if s.ended or o is None:
                    break
                b = o.ball
                keys = 0
                if b is not None and b["found"]:
                    p = np.asarray(b["pos"], float)
                    v = np.asarray(b["speed"]) + b["speed_change"]
                    bt = int(b["type"])
                    if i == 0 and (
                        s.tracker.sector != 4 or np.linalg.norm(p[[0, 2]] - target) > 40
                    ):
                        raise ValueError(
                            f"prefix mismatch: {p}, sector {s.tracker.sector}"
                        )
                    if bt == 2 or stone:
                        stone = True
                        if branch is None:
                            branch = ReferencePath.load(
                                Path(__file__).resolve().parents[1]
                                / "benchmarks/refpaths-alt/L02-stone/L02.npz"
                            )
                        pr = branch.project(p, 4, prev)
                        prev = pr["s"]
                        target = (
                            branch.point_at(
                                prev
                                + (
                                    16
                                    if p[0] < 1052
                                    and p[2] < -70
                                    and p[2] > -209
                                    and p[0] > 1029
                                    else 4
                                )
                            )[[0, 2]]
                            if s.tracker.sector == 4
                            else target
                        )
                        board |= p[2] < -201 and p[0] > 1031 and p[1] > 11
                        if board and p[2] < -20:
                            target = np.array([1040.55, p[2] + 24])
                        if p[2] > 20:
                            target = np.array([1040.6, 48.0])
                    aim = target.copy()
                    if not stone and p[2] < -60:
                        aim = np.array([933.8, -55.5])
                    d = aim - p[[0, 2]]
                    w = (
                        d
                        * min(
                            3,
                            (
                                26
                                if stone and 1029 < p[0] < 1052 and -209 < p[2] < -24
                                else 8
                            )
                            / max(0.01, np.linalg.norm(d)),
                        )
                        - v[[0, 2]]
                    )
                    mask = (
                        (tas.KEY_RIGHT if w[0] > 0 else tas.KEY_LEFT)
                        if abs(w[0]) > 0.4 * np.linalg.norm(w)
                        else 0
                    )
                    if abs(w[1]) > 0.4 * np.linalg.norm(w):
                        mask |= tas.KEY_UP if w[1] > 0 else tas.KEY_DOWN
                    if o.cams is not None:
                        keys = world_to_camera_keys(mask, o.cams["cams"][0]["matrix"])
                    print(
                        json.dumps(
                            dict(
                                frame=s.frames_done,
                                pos=p.tolist(),
                                speed=float(np.linalg.norm(v)),
                                bt=bt,
                                sector=s.tracker.sector,
                                sr=s.tracker.sr_ms,
                            )
                        ),
                        file=log,
                    )
                    if bt == 2 and prev is not None:
                        samples.append((s.frames_done - k, prev))
                    if s.tracker.sector >= 5:
                        np.savez_compressed(
                            out / "drive.npz",
                            branch_sha256=file_sha256(
                                asset_path(
                                    2,
                                    Path(__file__).resolve().parents[1]
                                    / "benchmarks/refpaths-alt/L02-stone",
                                )
                            ),
                            record_sha256=json.loads((rec / "meta.json").read_text())[
                                "record_sha256"
                            ],
                            prefix_frame=k,
                            frames=s.frame_log()[k:],
                            samples=np.asarray(samples),
                        )
                        break
                    if p[1] < -5:
                        break
                if o.has_event(Ev.BallOff):
                    break
                o = s.act(
                    tas.make_frames([1000 / 132], [keys]), repeat=4, stop=EPISODE_STOP
                )
        except Exception as exc:
            print(repr(exc), file=log)
            print(repr(exc))
        finally:
            s.close(kill=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
