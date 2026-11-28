"""Engine gate for a new machine: replay a TAS record's own frames (with their deltas) in a headless engine and save the
ball position after every block, so two machines' runs of the same binary can be compared.

    python tools/hl_gate.py run  --record HS01_4740_1_30_344 --frames 8000 --out gate_pc.npz [--bin .. --root ..]
    python tools/hl_gate.py cmp  gate_pc.npz gate_remote.npz

Ball order is pinned (ckid) so both runs see the same ball-pair order. Also compares with the dataset's raw.npz
positions where the record has them."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def run(a) -> int:
    from ballrain import tas
    from ballrain.client import AgentSession, SessionConfig
    from ballrain.headless import HeadlessLauncher
    rec = REPO / "runtime/datasets/tas-v1" / a.record
    frames = tas.load_raw(rec / "run/framelog.bin")[: a.frames]
    binary = a.bin or str(REPO / "runtime/hl-win/bin-x86p2002f-pressure/BallanceMMOSimTool.exe")
    root = a.root or str(REPO / "runtime/hl-win/game")
    work = Path(tempfile.mkdtemp(prefix="hlgate-"))
    launcher = HeadlessLauncher(binary=binary, root=root, socket="tcp", log=False, ball_order="ckid")
    s = AgentSession(SessionConfig(1, work / "run", copy=a.copy, hello_timeout_s=90, obs_timeout_s=90, events_txt=False),
                     launcher)
    T, P = [], []
    t0 = time.perf_counter()
    try:
        o = s.start()
        k = s.frames_done
        while True:
            if o is not None and o.ball is not None:
                T.append(k)
                P.append(np.asarray(o.ball["pos"], float))
            if k >= len(frames) or s.ended:
                break
            o = s.act(frames[k:k + a.block])
            if o is None:
                break
            k = s.frames_done
    finally:
        s.close(kill=True)
    np.savez(a.out, t=np.asarray(T), pos=np.asarray(P))
    print(json.dumps({"frames": int(T[-1]), "obs": len(T), "wall_s": round(time.perf_counter() - t0, 2),
                      "last_pos": [round(float(x), 4) for x in P[-1]]}))
    return 0


def cmp(a) -> int:
    x, y = np.load(a.a), np.load(a.b)
    common, ia, ib = np.intersect1d(x["t"], y["t"], return_indices=True)
    d = np.linalg.norm(x["pos"][ia] - y["pos"][ib], axis=1)
    exact = bool(np.array_equal(x["pos"][ia], y["pos"][ib]))
    first = int(common[np.argmax(d > 0)]) if (d > 0).any() else None
    out = {"compared_obs": len(common), "last_frame": int(common[-1]), "bit_identical": exact,
           "max_dist": float(d.max()), "first_diff_frame": first}
    print(json.dumps(out))
    return 0 if exact else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--record", default="HS01_4740_1_30_344")
    r.add_argument("--frames", type=int, default=8000)
    r.add_argument("--block", type=int, default=4)
    r.add_argument("--copy", default="hl-gate")
    r.add_argument("--bin")
    r.add_argument("--root")
    r.add_argument("--out", required=True)
    c = sp.add_parser("cmp")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else cmp(a)


if __name__ == "__main__":
    raise SystemExit(main())
