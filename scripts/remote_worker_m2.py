"""Remote rollout workers for scripts/train_m2.py (ballrain.train.remote): local game copies feeding a learner on
another machine over the LAN (launches the game).

    BALLRAIN_REMOTE_KEY=<secret> uv run python scripts/remote_worker_m2.py --learner HOST:PORT \
        --copies train-00,train-01,... [--tag pc] [--fake]

The learner (train_m2.py --remote-listen HOST:PORT) sends the worker template (env config, chunk length, limits) and
the newest weights / observation statistics / curriculum state after every update; this process runs one worker
process per copy (ballrain.train.worker), forwards their chunks and keeps reconnecting when the link drops.
Outputs: <runtime>/m2-train/remote-<tag>/ (episodes/<copy>/..., remote.log). Stop: create remote-<tag>/STOP; the
workers close their game processes, then any Player.exe still running in the copies used is stopped by path.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--learner", required=True, help="HOST:PORT of train_m2.py --remote-listen")
    p.add_argument("--key", default=None, help="shared secret (default $BALLRAIN_REMOTE_KEY)")
    p.add_argument("--copies", required=True)
    p.add_argument("--tag", default="pc")
    p.add_argument("--backend", default="retail")
    p.add_argument("--fake", action="store_true", help="FakeLauncher (tests)")
    p.add_argument("--launch", default=None,
                   help="launch recipe for this machine's copies (a key of ballrain.instances.LAUNCH_RECIPES); default: "
                        "the learner template's env.launch. Lets the bridge switch launchers without a learner restart")
    a = p.parse_args(argv)
    from ballrain import instances
    from ballrain.train.remote import RemoteBridge, parse_addr
    from train_m2 import stop_copies
    if a.launch is not None and a.launch not in instances.LAUNCH_RECIPES:
        raise SystemExit(f"--launch {a.launch!r}: not one of {sorted(instances.LAUNCH_RECIPES)}")
    instances.LAUNCH_RECIPE = instances.LAUNCH_RECIPES[a.launch] if a.launch else instances.launch_direct
    key = (a.key or os.environ.get("BALLRAIN_REMOTE_KEY") or "").encode()
    if not key:
        raise SystemExit("needs --key or $BALLRAIN_REMOTE_KEY")
    out = instances.runtime_dir() / "m2-train" / f"remote-{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "STOP").unlink(missing_ok=True)
    logf = open(out / "remote.log", "a", encoding="utf-8")

    def log(s):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    copies = a.copies.split(",")
    if not a.fake:
        for c in copies:
            if instances.running_in_copy(instances.get_copy(c)):
                raise SystemExit(f"Player.exe already running in {c}")
    overrides = {}
    if a.fake:
        from ballrain.fakegame import FakeLauncher
        overrides["launcher_factory"] = FakeLauncher
    br = RemoteBridge(parse_addr(a.learner), key, copies, a.backend, lambda c: out / "episodes" / c, log=log,
                      worker_overrides=overrides)
    if a.launch:
        _spec_l = br._spec

        def spec_launch(i, name, restart):
            s = _spec_l(i, name, restart)
            s.env.launch = a.launch
            return s
        br._spec = spec_launch
        log(f"launch recipe: {a.launch}")
    if a.fake:
        _spec = br._spec

        def spec_fake(i, name, restart):  # the fake game ignores the copy; keep runs apart by out_root
            s = _spec(i, name, restart)
            s.env.copy = instances.TRAINING_COPY
            return s
        br._spec = spec_fake
    br.heartbeat = out / "heartbeat.json"
    ctx = mp.get_context("spawn")
    data_q = ctx.Queue()
    try:
        br.run(ctx, data_q, should_stop=lambda: (out / "STOP").is_file(),
               on_fatal=None if a.fake else (lambda c: stop_copies([c], log)))
    finally:
        log(f"stopping (forwarded {br.counts['forwarded']}, dropped {br.counts['dropped']}, "
            f"reconnects {br.counts['reconnects']})")
        br.shutdown()
        if not a.fake:
            st = stop_copies(copies, log)
            if st:
                log(f"stopped leftover game processes: {st}")
        data_q.cancel_join_thread()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
