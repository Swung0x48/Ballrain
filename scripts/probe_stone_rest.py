"""Exercise a real branch rest hand-over, not a mocked type assignment."""

import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ballrain.env import BallanceEnv, EnvConfig
from ballrain.features import FeatureConfig

live = Path(sys.argv[1])
hl = live / "hl-win"
build, copy = sys.argv[2:4]
assert (build, copy) in {("bin-x86soft", "hlw-98"), ("bin-x86p2002f-respawnfix", "hlw-99")}
for target in (1525, 1700, 1850, 1975, 2100):
    cfg = EnvConfig(level=2, backend="headless", copy=copy, out_root=str(live / "viz/l2s4/stone"),
                    headless=dict(binary=str(hl / build / "BallanceMMOSimTool.exe"),
                                  root=str(hl / "game"), socket="tcp"),
                    features=FeatureConfig(branch_dir="benchmarks/refpaths-alt/L02-stone"),
                    start_record=str(live / "datasets/tas-v1/SR02_1_28_992_264"), p_rest=1,
                    rest_starts=[[14730, 14730, dict(branch_s=float(target))]])
    env = BallanceEnv(cfg)
    try:
        _, info = env.reset(seed=1)
        tr = env.session.tracker
        ball = tr.obs.ball
        speed = np.asarray(ball["speed"]) + ball["speed_change"]
        print(json.dumps(dict(target=target, build=build, kind=info["start"]["kind"],
                              error=info["start"].get("branch_rest_error"), local_s=info.get("local_s"),
                              horizontal_speed=float(np.linalg.norm(speed[[0, 2]])), rest_speed=cfg.rest_speed,
                              supported=bool(info["supported"]), ball=tr.ball_type())), flush=True)
    finally:
        env.close()
