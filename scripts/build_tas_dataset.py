"""Build the TAS raw-state dataset: every TAS record replayed through BallrainIO agent mode with a
replay policy (one frame per action), raw observation bytes stored per frame (ballrain.dataset).

LAUNCHES THE GAME (training copies only). Owned by the gates phase.

  uv run python scripts/build_tas_dataset.py --copies 8
  uv run python scripts/build_tas_dataset.py --only SR01_1_14_457_264 SR06_1_36_163_132 --copies 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ballrain import dataset, instances  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--records", default=str(instances.runtime_dir() / instances.REFERENCE_COPY / "ModLoader" / "TASRecords"))
    p.add_argument("--only", nargs="*", help="record stems")
    p.add_argument("--out", default=str(instances.runtime_dir() / "datasets" / f"tas-v{dataset.DATASET_VERSION}"))
    p.add_argument("--copies", type=int, default=1, help="number of training copies used in parallel")
    p.add_argument("--frames-per-action", type=int, default=1)
    a = p.parse_args(argv)
    recs = sorted(Path(a.records).glob("*.tas"))
    if a.only:
        recs = [r for r in recs if r.stem in set(a.only)]
    if not recs:
        print("no records", file=sys.stderr)
        return 2
    res = dataset.build_dataset(recs, a.out, instances.training_copy_names(a.copies),
                                frames_per_action=a.frames_per_action)
    bad = [m["record"] for m in res if not m["ok"]]
    print(json.dumps({"records": len(res), "failed": bad}))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
