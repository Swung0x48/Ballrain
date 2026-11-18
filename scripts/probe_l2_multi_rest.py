"""Exercise branch drive + settle + fresh-extractor handover. No fallback counts as success."""

import sys, json, numpy as np
from probe_l2_multi import OUT, REC, LIVE
from build_l2_multi import DEST, MAIN
from ballrain.env import BallanceEnv, EnvConfig
from ballrain.features import FeatureConfig
from ballrain.branch_assets import validate, read_bundle
from ballrain.path import file_sha256, ReferencePath


def run(build):
    copy = {"x86soft": "hlw-96", "x86p2002f-respawnfix": "hlw-97"}[build]
    bundle = read_bundle(2, DEST)
    sha = file_sha256(MAIN)
    pin = validate(bundle, 2, sha)
    for key, pts in [
        (
            "back-to-wood",
            [
                (715.9, 32.24, -267.85),
                "phase:1",
                (715.8, 32.2, -263.1),
                (691.7, 32.24, -265.5),
                (692.5, 32.2, -267),
                (697.3, 32.4, -269),
            ],
        ),
        (
            "east-platform",
            [
                (917.3, 48.7, -346.7),
                (988, 48, -346.7),
                (991.7, 48, -366),
                (1013, 48, -367),
                (971, 48, -387),
                (932, 48, -385),
            ],
        ),
    ]:
        path = ReferencePath.load(DEST / key / "L02.npz")
        drive = np.load(DEST / key / "drive.npz")
        samples = drive["samples"]
        positions = np.array([path.point_at(s) for s in samples[:, 1]])
        k = int(drive["prefix_frame"])
        for point in pts:
            if isinstance(point, str):
                entry = next(
                    e
                    for e in json.loads(bundle["manifest"])["branches"]
                    if e["id"] == key
                )
                target = entry["phases"][int(point.split(":")[1])]["local_s"][0]
            else:
                target = float(
                    samples[np.argmin(np.linalg.norm(positions - point, axis=1)), 1]
                )
            rest = [k, k, dict(id=key, branch_s=target)]
            cfg = EnvConfig(
                level=2,
                backend="headless",
                copy=copy,
                out_root=str(OUT / "rest"),
                headless=dict(
                    binary=str(
                        LIVE / "hl-win" / ("bin-" + build) / "BallanceMMOSimTool.exe"
                    ),
                    root=str(LIVE / "hl-win/game"),
                    socket="tcp",
                ),
                features=FeatureConfig(
                    path_dir=str(MAIN.parent), path_sha256=sha, branch_sha256=pin
                ),
                branch_assets=bundle,
                start_record=str(REC),
                p_rest=1,
                rest_starts=[rest],
            )
            env = BallanceEnv(cfg)
            try:
                _, info = env.reset(seed=1)
                tr = env.session.tracker
                b = tr.obs.ball
                print(
                    json.dumps(
                        dict(
                            build=build,
                            pin=pin,
                            rest=rest,
                            id=info.get("branch_id"),
                            phase=info.get("phase"),
                            bt=tr.ball_type(),
                            supported=bool(info["supported"]),
                            speed=float(
                                np.linalg.norm(
                                    (np.asarray(b["speed"]) + b["speed_change"])[[0, 2]]
                                )
                            ),
                            error=info["start"].get("branch_rest_error"),
                        )
                    ),
                    flush=True,
                )
            finally:
                env.close()


if __name__ == "__main__":
    run(sys.argv[1])
