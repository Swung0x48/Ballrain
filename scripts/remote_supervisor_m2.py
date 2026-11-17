"""Supervisor for scripts/remote_worker_m2.py: restarts the remote bridge (with the current code) when it dies, hangs or
crash-loops (launches the game through the bridge).

    BALLRAIN_REMOTE_KEY=<secret> uv run python scripts/remote_supervisor_m2.py --tag r4 -- --learner HOST:PORT --copies ...

Everything after "--" goes to remote_worker_m2.py (with --tag added). Health comes from the bridge's heartbeat file
(<runtime>/m2-train/remote-<tag>/heartbeat.json, written every 30 s): the bridge is restarted when
- its process exited,
- the heartbeat is older than --stale-s (default 180 s: the bridge's main loop is stuck),
- it has been connected but forwarded nothing for --idle-s (default 600 s),
- more than --max-fatal-per-min worker fatals per minute (crash loop, default 5).
A restart: STOP file, wait up to 90 s, then kill the bridge's process tree (only its own PID) and stop any Player.exe left
in its copies by path; then a new bridge process (new code). Log: remote-<tag>/supervisor.log. Stop the supervisor
(and the bridge): create remote-<tag>/SUPERVISOR_STOP.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--stale-s", type=float, default=180)
    p.add_argument("--idle-s", type=float, default=600)
    p.add_argument("--max-fatal-per-min", type=float, default=5)
    p.add_argument("--check-s", type=float, default=30)
    a = p.parse_args(own)
    from ballrain import instances
    from train_m2 import stop_copies
    out = instances.runtime_dir() / "m2-train" / f"remote-{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "SUPERVISOR_STOP").unlink(missing_ok=True)
    logf = open(out / "supervisor.log", "a", encoding="utf-8")
    copies = rest[rest.index("--copies") + 1].split(",") if "--copies" in rest else []

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    def start():
        (out / "STOP").unlink(missing_ok=True)
        (out / "heartbeat.json").unlink(missing_ok=True)
        cmd = [sys.executable, str(REPO / "scripts" / "remote_worker_m2.py"), "--tag", a.tag] + rest
        so = open(out / "bridge_stdout.txt", "ab")
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=so, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        log(f"bridge started (pid {proc.pid})")
        return proc

    def stop(proc, why):
        log(f"restarting the bridge: {why}")
        (out / "STOP").write_text("supervisor", encoding="utf-8")
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            log(f"bridge pid {proc.pid} did not stop: killing its process tree")
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        time.sleep(10)  # the workers exit on their own once their parent is gone
        try:
            st = stop_copies(copies, log)
            if st:
                log(f"stopped leftover game processes: {st}")
        except Exception as e:  # noqa: BLE001  (e.g. fake copies in tests)
            log(f"stop_copies: {type(e).__name__}: {e}")
        (out / "STOP").unlink(missing_ok=True)

    proc = start()
    t_start = time.time()
    last_fwd, t_fwd = -1, time.time()
    fatal_hist: list = []
    while True:
        time.sleep(a.check_s)
        if (out / "SUPERVISOR_STOP").is_file():
            stop(proc, "supervisor stop requested")
            log("supervisor exiting")
            return 0
        why = None
        hb = None
        try:
            hb = json.loads((out / "heartbeat.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        now = time.time()
        if proc.poll() is not None:
            why = f"process exited (code {proc.returncode})"
        elif hb is None:
            if now - t_start > a.stale_s:
                why = "no heartbeat"
        else:
            if now - hb["t"] > a.stale_s:
                why = f"heartbeat {now - hb['t']:.0f} s old"
            if hb["forwarded"] != last_fwd:
                last_fwd, t_fwd = hb["forwarded"], now
            elif hb.get("connected") and now - t_fwd > a.idle_s:
                why = f"connected but nothing forwarded for {(now - t_fwd) / 60:.0f} min"
            fatal_hist.append((now, hb.get("fatal", 0)))
            fatal_hist = [x for x in fatal_hist if now - x[0] <= 60]
            if len(fatal_hist) >= 2 and fatal_hist[-1][1] - fatal_hist[0][1] > a.max_fatal_per_min:
                why = f"crash loop ({fatal_hist[-1][1] - fatal_hist[0][1]} worker fatals in the last minute)"
        if why:
            stop(proc, why)
            proc = start()
            t_start = time.time()
            last_fwd, t_fwd = -1, time.time()
            fatal_hist = []


if __name__ == "__main__":
    raise SystemExit(main())
