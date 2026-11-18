"""Exercise a real branch rest hand-over, not a mocked type assignment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ballrain.env import BallanceEnv, EnvConfig
from ballrain.features import FeatureConfig

live = Path(sys.argv[1])
hl = live / "hl-win"
cfg = EnvConfig(
    level=2,
    backend="headless",
    copy="hlw-99",
    out_root=str(live / "viz/l2s4/stone"),
    headless=dict(
        binary=str(hl / "bin-x86p2002d/BallanceMMOSimTool.exe"),
        root=str(hl / "game"),
        socket="tcp",
    ),
    features=FeatureConfig(branch_dir="runtime/refpaths-alt/L02-stone"),
    start_record=str(live / "datasets/tas-v1/SR02_1_28_992_264"),
    p_rest=1,
    rest_starts=[[14730, 14730, dict(branch_s=1700.0)]],
)
env = BallanceEnv(cfg)
try:
    o, i = env.reset(seed=1)
    print(
        i["start"],
        i["branch"],
        i["local_s"],
        i["supported"],
        env.session.tracker.ball_type(),
    )
finally:
    env.close()
