"""Create / verify the training copies runtime/train-01..train-(n-1) from runtime/train-00, and
optionally install a freshly built BallrainIO.bmodp into all of them.

  uv run python scripts/make_copies.py                 # create missing copies, verify all
  uv run python scripts/make_copies.py --sync-mod      # also install BallrainIO/build/BallrainIO.bmodp
  uv run python scripts/make_copies.py -n 16 --splash off --also ref   # no start-up splash in every train-NN and ref

--splash on|off renames Bin/splash.bmp <-> Bin/splash.bmp.off in every runtime/train-NN and the --also copies (README,
"游戏窗口"); nothing is changed unless no Player.exe runs in any of them.
--volume V sets the game's own volume option (Database.tdb DB_Options.Volume, stock 1.0; only those 4 bytes change) in
the same copies. Measured 2026-11-16: 0 does not silence BallrainIO / TASSupport replays (README, "游戏声音"); the launcher's
per-process mute is what silences the games.
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
    p.add_argument("--splash", choices=("on", "off"), help="enable / disable Player.exe's start-up splash")
    p.add_argument("--volume", type=float, help="set Database.tdb DB_Options.Volume (0..1, stock 1.0)")
    p.add_argument("--also", default="", help="comma-separated extra copies for --splash / --volume (e.g. ref)")
    a = p.parse_args(argv)
    if a.splash or a.volume is not None:
        # every train-NN present (not only the -n range), so no copy is left with the other state
        names = sorted(p.name for p in instances.runtime_dir().glob("train-[0-9][0-9]") if p.is_dir())
        targets = [instances.get_copy(n) for n in names + [x for x in a.also.split(",") if x]]
        busy = [c.name for c in targets if instances.running_in_copy(c)]
        if busy:
            print(f"Player.exe running in {busy}; nothing changed", file=sys.stderr)
            return 1
        if a.splash:
            print(json.dumps({c.name: instances.set_splash(c, a.splash == "on") for c in targets}))
        if a.volume is not None:
            print(json.dumps({c.name: instances.set_tdb_volume(c, a.volume) for c in targets}))
    res = instances.create_training_copies(a.n, a.source, hash_all=a.hash_all)
    print(json.dumps(res))
    names = instances.training_copy_names(a.n)
    if a.sync_mod:
        instances.sync_mod(a.sync_mod, names)
    mods = instances.mod_hashes(names)
    print(json.dumps(mods))
    if len(set(mods.values())) > 1:
        print("warning: the copies hold different BallrainIO.bmodp builds; run with --sync-mod", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
