"""The pool verifier as its own process (the gateway can also run it as a thread: gateway.py serve).

    python tools/pool/verifier.py [--config runtime/pool/config.toml] [--once]

Replays closed tasks with the configured engines ([engines.2002] / [engines.x87soft] binary + root), one engine
process at a time, and records the outcome in <pool>/state.sqlite3 (ballrain.pool.server_verify).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gateway import build, log  # noqa: E402
from ballrain.pool.server_config import load_config  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--pool-dir", default=None)
    ap.add_argument("--once", action="store_true", help="verify what is ready now, then exit")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.pool_dir)
    _, store, _, verifier = build(cfg)
    if a.once:
        n = verifier.run_once(limit=1000)
        log(f"verified {n} task(s)")
        return
    log("verifier running")
    try:
        verifier.run_forever()
    except KeyboardInterrupt:
        verifier.stop_event.set()


if __name__ == "__main__":
    main()
