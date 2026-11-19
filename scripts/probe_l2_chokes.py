"""Real south-loop resets. Only the two reserved copies; times are simulation frames."""

import argparse, csv, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ballrain.protocol import Ev
from ballrain import tas
from ballrain.branch_assets import read_bundle, validate
from ballrain.env import BallanceEnv, EnvConfig, EPISODE_STOP, world_to_camera_keys
from ballrain.features import FeatureConfig
from ballrain.path import ReferencePath, file_sha256


def probe(live, build, station, mode, v1=False, observer=None, corner_cap=None):
    repo = Path(__file__).resolve().parents[1]
    assets = repo / "benchmarks/refpaths-alt/L02-multi"
    path = ReferencePath.load(assets / "south-loop/L02.npz")
    main = repo / "benchmarks/refpaths-v2/L02.npz"
    bundle = read_bundle(
        2, repo / "benchmarks/refpaths-alt/L02-stone" if v1 else assets
    )
    sha = file_sha256(main)
    pin = validate(bundle, 2, sha)
    copy = {"x86soft": "hlw-96", "x86p2002f-respawnfix": "hlw-97"}[build]
    out = live / "viz/l2chokes"
    hl = live / "hl-win"
    env = BallanceEnv(
        EnvConfig(
            level=2,
            backend="headless",
            copy=copy,
            out_root=str(out),
            headless=dict(
                binary=str(hl / ("bin-" + build) / "BallanceMMOSimTool.exe"),
                root=str(hl / "game"),
                socket="tcp",
            ),
            features=FeatureConfig(
                path_dir=str(main.parent), path_sha256=sha, branch_sha256=pin
            ),
            branch_assets=bundle,
            start_record=str(live / "datasets/tas-v1/SR02_1_28_992_264"),
            p_rest=1,
            rest_starts=[[14730, 14730, dict(branch_s=float(station))]],
        )
    )
    try:
        _, info = env.reset(seed=1)
        se = env.session
        tr = se.tracker
        o = tr.obs
        s = info["local_s"]
        if not (
            info.get("branch")
            and info["supported"]
            and abs(s - station) < 2
            and tr.ball_type() == 2
        ):
            print(
                copy,
                station,
                "REJECT",
                s,
                info["start"].get("branch_rest_error"),
                flush=True,
            )
            return
        start = se.frames_done
        sign = 1
        entered = False
        recovering = mode == "detour"
        seen = set()
        if mode == "reach":
            from ballrain.reach_features import ReachBlock, REACH_NAMES

            rb = ReachBlock(cache_dir=out / "fields")
        suffix = "-v1" if v1 else ""
        with (out / f"{copy}-{station}-{mode}{suffix}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow("frame s dist x y z vx vy vz ball sector mask".split())
            while o and not se.ended and se.frames_done - start <= 132 * 45:
                b = o.ball
                p = np.array(b["pos"])
                v = np.array(b["speed"]) + b["speed_change"]
                pr = path.project(p, 4, s)
                s = pr["s"]
                frame = se.frames_done - start
                if mode == "reach" and s <= 1630 and int(s) not in seen:
                    seen.add(int(s))
                    raw = rb.compute(tr, p, v, 0, 2)[1]
                    print("reach", s, dict(zip(REACH_NAMES, raw.tolist())), flush=True)
                tgt = path.point_at(s + 4)[[0, 2]]
                d = tgt - p[[0, 2]]
                aim = d * min(3, 8 / max(0.01, np.linalg.norm(d))) - v[[0, 2]]
                if corner_cap is not None and s < 1780:
                    target = path.point_at(s + 4)[[0, 2]] - p[[0, 2]]
                    aim = (
                        target * (corner_cap / max(0.01, np.linalg.norm(target)))
                        - v[[0, 2]]
                    )
                if mode.startswith("late") and s < float(mode[4:]):
                    aim = np.array([0.0, -40.0])
                if observer is not None:
                    observer(tr, s, p, v, frame)
                entered |= p[2] < -201 and p[0] > 1031 and p[1] > 11
                if recovering and (p[0] < 1053 or abs(v[0]) > 1.5 or p[1] > 7):
                    aim = np.array(
                        [np.clip(2 * (1055.43 - p[0]), -7, 7) - v[0], -4 - v[2]]
                    )
                elif recovering:
                    recovering = False
                elif (
                    (
                        mode in ("runup", "detour")
                        or corner_cap is not None
                        or mode in ("idle4", "north22", "align")
                    )
                    and entered
                    and p[2] < -22
                ):
                    d = np.array([1040.55 - p[0], 24.0])
                    aim = (
                        d * ((22 if mode == "north22" else 26) / np.linalg.norm(d))
                        - v[[0, 2]]
                    )
                    if mode == "align":
                        aim[0] = np.clip(3 * (1040.55 - p[0]), -12, 12) - v[0]
                elif (s > 1944 or (station >= 1960 and mode != "detour")) and p[
                    2
                ] < -22:
                    if mode == "rock" and abs(v[2]) > 0.5:
                        if sign * v[2] < -0.5:
                            sign = -sign
                    aim = np.array(
                        [
                            3 * (1040.55 - p[0]) - v[0],
                            (sign if mode == "rock" else 1) * 8,
                        ]
                    )
                    if mode == "diag":
                        aim[0] = 5 if p[0] < 1040.55 else -5
                elif s > 2120 or station >= 2110:
                    aim = np.array(
                        [3 * (1040.55 - p[0]) - v[0], min(12, 3 * (48 - p[2])) - v[2]]
                    )
                n = np.linalg.norm(aim)
                mask = (
                    (tas.KEY_RIGHT if aim[0] > 0 else tas.KEY_LEFT)
                    if abs(aim[0]) > 0.4 * n
                    else 0
                )
                if abs(aim[1]) > 0.4 * n:
                    mask |= tas.KEY_UP if aim[1] > 0 else tas.KEY_DOWN
                if mode == "idle4" and frame < 528:
                    mask = 0
                w.writerow([frame, s, pr["dist"], *p, *v, b["type"], tr.sector, mask])
                if tr.sector >= 5 or p[1] < -3 or o.has_event(Ev.BallOff):
                    break
                keys = world_to_camera_keys(mask, o.cams["cams"][0]["matrix"])
                o = se.act(
                    tas.make_frames([1000 / 132], [keys]), repeat=4, stop=EPISODE_STOP
                )
        print(copy, station, mode, frame, s, tr.sector, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("runtime", type=Path)
    a.add_argument("build")
    a.add_argument("mode")
    a.add_argument("stations", type=int, nargs="+")
    a.add_argument("--v1", action="store_true")
    v = a.parse_args()
    for station in v.stations:
        probe(v.runtime, v.build, station, v.mode, v.v1)
