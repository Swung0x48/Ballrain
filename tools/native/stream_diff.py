"""Compare two golden / native streams (tools/native/golden.py format) record by record.

    python tools/native/stream_diff.py A/stream.bin B/stream.bin [--feat-tol 1e-4]

Reports, per record kind, how many records differ; for OBS records the sections that differ (raw payload bytes,
OBS1 header fields and the socket header separately), for LVL records the level-message sections, for FEAT records
the largest absolute difference per feature, for ACT records the first differing one. Exit code 1 when the ACT
streams differ (the trajectories are then not the same run).
"""
from __future__ import annotations

import argparse
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ballrain import protocol as P  # noqa: E402

NAMES = {1: "HELLO", 2: "OBS", 3: "ACT", 4: "LVL_FULL", 5: "LVL_INC", 6: "RESET", 7: "FEAT", 8: "POLICY", 9: "NOTE"}


def read_stream(path):
    b = Path(path).read_bytes()
    if b[:4] != b"BRGS":
        raise SystemExit(f"{path}: not a stream")
    p, out = 8, []
    while p + 8 <= len(b):
        k, n = struct.unpack_from("<II", b, p)
        out.append((k, b[p + 8:p + 8 + n]))
        p += 8 + n
    return out


def sections(msg: bytes, magic: int) -> dict:
    m = P.split_message(msg, 0, magic)
    out = defaultdict(list)
    for s in m.sections:
        out[s.tag].append(s.payload)
    return {"_hdr": [(m.version, m.flags, m.frames_done)], **out}


def lvl_sections(rec: bytes) -> dict:
    if not rec or rec[0] == 0:
        return {"_status": [rec[:1]]}
    data = rec[1:]
    out = defaultdict(list)
    pos = 0
    while len(data) - pos >= P.MSG_HDR.size:
        (bl,) = struct.unpack_from("<I", data, pos + 8)
        if pos + bl > len(data):
            out["_partial"].append(data[pos:])
            break
        m = P.split_message(data, pos, P.LVL_MAGIC)
        for s in m.sections:
            out[s.tag].append(s.payload)
        pos += bl
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--feat-tol", type=float, default=1e-4)
    ap.add_argument("--names", help="manifest json: name the feature columns")
    a = ap.parse_args(argv)
    A, B = read_stream(a.a), read_stream(a.b)
    by = defaultdict(lambda: ([], []))
    for i, (k, p) in enumerate(A):
        by[k][0].append(p)
    for k, p in B:
        by[k][1].append(p)
    rc = 0
    for k in sorted(by):
        xa, xb = by[k]
        name = NAMES.get(k, str(k))
        n = min(len(xa), len(xb))
        diff = [i for i in range(n) if xa[i] != xb[i]]
        print(f"{name:9s} {len(xa):6d} / {len(xb):6d} records, {len(diff)} differ" + (f" (first {diff[0]})" if diff else ""))
        if k == 3 and (diff or len(xa) != len(xb)):
            rc = 1
        if k == 2 and diff:
            sec = Counter()
            for i in diff:
                ha, hb = xa[i][:24], xb[i][:24]
                if ha != hb:
                    sec["<socket hdr>"] += 1
                sa, sb = sections(xa[i][24:], P.OBS_MAGIC), sections(xb[i][24:], P.OBS_MAGIC)
                for t in set(sa) | set(sb):
                    if sa.get(t) != sb.get(t):
                        sec[t] += 1
            print("          sections:", dict(sec.most_common()))
        if k in (4, 5) and diff:
            sec = Counter()
            for i in diff:
                sa, sb = lvl_sections(xa[i]), lvl_sections(xb[i])
                for t in set(sa) | set(sb):
                    if sa.get(t) != sb.get(t):
                        sec[t] += 1
            print("          sections:", dict(sec.most_common()))
        if k == 7 and n:
            fa = np.stack([np.frombuffer(x, "<f4") for x in xa[:n]])
            fb = np.stack([np.frombuffer(x, "<f4") for x in xb[:n]])
            d = np.abs(fa.astype(np.float64) - fb)
            bad = d > a.feat_tol
            print(f"          max |diff| {d.max():.3g}; rows with any > {a.feat_tol}: {int(bad.any(1).sum())}")
            cols = np.flatnonzero(bad.any(0))
            for c in cols[:20]:
                r = int(np.argmax(d[:, c]))
                print(f"          col {c}: {int(bad[:, c].sum())} rows, max {d[r, c]:.3g} at row {r} "
                      f"({fa[r, c]!r} vs {fb[r, c]!r})")
        if k == 8 and n:
            ua = [struct.unpack_from("<IdIf", x) for x in xa[:n]]
            ub = [struct.unpack_from("<IdIf", x) for x in xb[:n]]
            act = [i for i in range(n) if ua[i][2] != ub[i][2]]
            dv = max(abs(ua[i][3] - ub[i][3]) for i in range(n))
            la = np.stack([np.frombuffer(x[20:], "<f4") for x in xa[:n]]).astype(np.float64)
            lb = np.stack([np.frombuffer(x[20:], "<f4") for x in xb[:n]]).astype(np.float64)
            print(f"          actions differ at {len(act)} steps" + (f" (first {act[0]})" if act else "")
                  + f"; max |value diff| {dv:.3g}; max |logp diff| {np.abs(la - lb).max():.3g}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
