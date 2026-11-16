"""Create / verify the training copies runtime/train-01..train-(n-1) from runtime/train-00, and
optionally install a freshly built BallrainIO.bmodp into all of them.

  uv run python scripts/make_copies.py                 # create missing copies, verify all
  uv run python scripts/make_copies.py --sync-mod      # also install BallrainIO/build/BallrainIO.bmodp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ballrain import instances  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=8)
    p.add_argument("--source", default=instances.TRAINING_COPY)
    p.add_argument("--hash-all", action="store_true", help="sha256 every file, not only binaries")
    p.add_argument("--sync-mod", nargs="?", const=str(instances.REPO_ROOT / "BallrainIO" / "build" / "BallrainIO.bmodp"))
    a = p.parse_args(argv)
    res = instances.create_training_copies(a.n, a.source, hash_all=a.hash_all)
    print(json.dumps(res))
    names = instances.training_copy_names(a.n)
    if a.sync_mod:
        instances.sync_mod(a.sync_mod, names)
    mods = instances.mod_hashes(names)
    print(json.dumps(mods))
    if len(set(mods.values())) != 1:
        print("warning: the copies hold different BallrainIO.bmodp builds; run with --sync-mod", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
