"""Remote rollout workers for scripts/train_m2.py (ballrain.train.remote): local game copies feeding a learner on
another machine over the LAN (launches the game).

    BALLRAIN_REMOTE_KEY=<secret> uv run python scripts/remote_worker_m2.py --learner HOST:PORT \
        --copies train-00,train-01,... [--tag pc] [--fake]
    ... --backend headless --copies hlw-00,...,hlw-07 --label headless2002 --hl-bin <SimTool> --hl-root <game dir>
        (headless instances on this machine, e.g. the Windows host with the original 2002 physics_RT.dll; the learner's
        template carries no headless paths, so the HeadlessConfig comes from here; ball order default "retail")

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
    p.add_argument("--backend", default="retail", help="env backend of the local workers: retail | headless")
    p.add_argument("--label", default=None,
                   help="data source name the learner logs for these episodes (episodes.csv backend, TB); default = "
                        "--backend, e.g. headless2002 for the Windows 2002-DLL host")
    p.add_argument("--hl-bin", default=None, help="headless: the SimTool binary (default $BALLRAIN_HL_BIN)")
    p.add_argument("--hl-root", default=None, help="headless: the game directory (default $BALLRAIN_HL_ROOT)")
    p.add_argument("--host-label", default=None,
                   help="host name sent to the learner (episodes.csv host column, the hub's connection key: one per "
                        "bridge); default: this machine's name. Use a generic name, e.g. runner2")
    p.add_argument("--quiet-file", default=None,
                   help="while this file exists all workers are stopped (their game / engine processes exit) and the "
                        "bridge stays connected; deleting it starts them again (shared machines, e.g. C:\ballrain-quiet)")
    p.add_argument("--priority", default="normal", choices=["normal", "below_normal"],
                   help="below_normal: this bridge, its workers and their engines run below normal priority")
    p.add_argument("--hl-ball-order", default="retail",
                   help="headless: BALLRAIN_HL_BALL_ORDER (default retail: a sampled order per process, recorded per "
                        "episode and run as ball_order)")
    p.add_argument("--fake", action="store_true", help="FakeLauncher (tests)")
    p.add_argument("--launch", default=None,
                   help="launch recipe for this machine's copies (a key of ballrain.instances.LAUNCH_RECIPES); default: "
                        "the learner template's env.launch. Lets the bridge switch launchers without a learner restart")
    a = p.parse_args(argv)
    from ballrain import instances
    from ballrain.train.remote import RemoteBridge, parse_addr
    from ballrain.train.remote import lower_priority
    from train_m2 import stop_copies
    if a.priority == "below_normal":
        lower_priority()  # inherited by the worker processes and the engines they start
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
    retail = a.backend == "retail" and not a.fake  # Player.exe copies (headless engines are the workers' children)
    headless = None
    if a.backend == "headless":
        from ballrain.headless import HeadlessConfig, parse_ball_order
        parse_ball_order(a.hl_ball_order, allow_seedless=True)  # fail early on a bad mode ("retail": a seed per process)
        hc = HeadlessConfig(ball_order=a.hl_ball_order)
        if a.hl_bin:
            hc.binary = a.hl_bin
        if a.hl_root:
            hc.root = a.hl_root
        if not a.fake:
            if not hc.binary or not os.path.isfile(hc.binary):
                raise SystemExit(f"--hl-bin {hc.binary!r}: not a file")
            if not hc.root or not os.path.isdir(hc.root):
                raise SystemExit(f"--hl-root {hc.root!r}: not a directory")
        headless = hc.to_dict()
    elif a.backend != "retail":
        raise SystemExit(f"--backend {a.backend!r}: retail | headless")
    if retail:
        for c in copies:
            if instances.running_in_copy(instances.get_copy(c)):
                raise SystemExit(f"Player.exe already running in {c}")
    overrides = {}
    if a.fake:
        from ballrain.fakegame import FakeLauncher
        overrides["launcher_factory"] = FakeLauncher
    br = RemoteBridge(parse_addr(a.learner), key, copies, a.backend, lambda c: out / "episodes" / c, log=log,
                      worker_overrides=overrides, label=a.label, headless=headless, quiet_file=a.quiet_file, host=a.host_label)
    log(f"backend {a.backend} (label {br.label}), {len(copies)} workers, priority {a.priority}"
        + (f", quiet file {a.quiet_file}" if a.quiet_file else "") + (f", headless {headless}" if headless else ""))
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
               on_fatal=(lambda c: stop_copies([c], log)) if retail else None)
    finally:
        log(f"stopping (forwarded {br.counts['forwarded']}, dropped {br.counts['dropped']}, "
            f"reconnects {br.counts['reconnects']})")
        br.shutdown()
        if retail:
            st = stop_copies(copies, log)
            if st:
                log(f"stopped leftover game processes: {st}")
        data_q.cancel_join_thread()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
