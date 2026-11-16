"""Compare dumped agent observations (obs.bin) field by field with the probe's frames.txt (same record)."""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ana"))
import pf  # noqa: E402  (probe frames.txt parser)


def sections(buf, off, end):
    out = {}
    while off + 12 <= end:
        tag = buf[off:off + 4].decode("latin-1")
        ver, flags, ln = struct.unpack_from("<HHI", buf, off + 4)
        out[tag] = buf[off + 12:off + 12 + ln]
        off = off + 12 + ((ln + 3) & ~3)
    return out


def load_obs(path):
    data = Path(path).read_bytes()
    pos = 0
    out = {}
    while pos < len(data):
        (n,) = struct.unpack_from("<I", data, pos)
        msg = data[pos + 4:pos + 4 + n]
        pos += 4 + n
        frames = struct.unpack_from("<I", msg, 12)[0]
        flags = struct.unpack_from("<H", msg, 6)[0]
        if flags & 1:
            continue  # final (tail) observation
        out[frames] = sections(msg, 24, len(msg))
    return out


def main():
    obs = load_obs(sys.argv[1])
    frames, _ev = pf.parse(sys.argv[2], want={"K", "BW", "C", "D", "F"})
    stats = dict(frames=0, ball_cmp=0, ball_bad=0, mind_cmp=0, mind_count_bad=0, mind_bad=0, fric_cmp=0,
                 fric_count_bad=0, fric_bad=0, psi_ck_big=0)
    bad_examples = []
    big = []
    fd = []
    for fr in frames:
        o = obs.get(fr["idx"])
        if o is None or "BALL" not in o:
            continue
        stats["frames"] += 1
        b = o["BALL"]
        found = b[5]
        K = fr.get("K")
        if found and K and "pos" in K:
            stats["ball_cmp"] += 1
            pos = struct.unpack_from("<3d", b, 54)
            ql = struct.unpack_from("<4d", b, 78)
            qn = struct.unpack_from("<4d", b, 110)
            s = struct.unpack_from("<3f", b, 150)
            ok = all(abs(a - c) <= 1e-9 * max(1, abs(c)) for a, c in zip(pos, K["pos"])) and \
                all(abs(a - c) <= 1e-12 for a, c in zip(ql, K["ql"])) and \
                all(abs(a - c) <= 1e-12 for a, c in zip(qn, K["qn"])) and \
                all(abs(a - c) <= 1e-5 * max(1, abs(c)) for a, c in zip(s, K["s"]))
            if not ok:
                stats["ball_bad"] += 1
                if len(bad_examples) < 5:
                    bad_examples.append(("ball", fr["idx"], pos, K["pos"]))
            ck = struct.unpack_from("<3f", b, 6 + 36)
            d = max(abs(x - y) for x, y in zip(ck, pos))
            if d > 2.0:
                stats["psi_ck_big"] += 1
                big.append((fr["idx"], fr["ball"], round(d, 3), [round(v, 3) for v in K["s"]]))
        elif found != (1 if (K and "pos" in K) else 0):
            stats["ball_found_diff"] = stats.get("ball_found_diff", 0) + 1
            fd.append((fr["idx"], found))
        # MIND vs probe D lines (order: list order of the exact synapses)
        m = o.get("MIND")
        if m is not None:
            n = struct.unpack_from("<H", m, 0)[0]
            items = [m[2 + 112 * i:2 + 112 * (i + 1)] for i in range(n)]
            D = [d for d in fr["D"] if len(d["raw"]) >= 0x78 and int(d["pid"]) != 0]
            stats["mind_cmp"] += 1
            if len(items) != len(D):
                stats["mind_count_bad"] += 1
                if len(bad_examples) < 10:
                    bad_examples.append(("mind_n", fr["idx"], len(items), len(D), [int(d["pid"]) for d in fr["D"]]))
            else:
                for it, d in zip(items, D):
                    pid = struct.unpack_from("<I", it, 0)[0]
                    ln = struct.unpack_from("<f", it, 8)[0]
                    plane = struct.unpack_from("<3f", it, 12)
                    if pid != int(d["pid"]) or ln != pf.f32(d["raw"], 0x54) or plane != struct.unpack_from("<3f", d["raw"], 0x68):
                        stats["mind_bad"] += 1
                        if len(bad_examples) < 10:
                            bad_examples.append(("mind", fr["idx"], pid, d["pid"], ln, pf.f32(d["raw"], 0x54)))
        c = o.get("FRIC")
        if c is not None:
            n = struct.unpack_from("<H", c, 0)[0]
            items = [c[2 + 53 * i:2 + 53 * (i + 1)] for i in range(n)]
            C = fr["C"]
            stats["fric_cmp"] += 1
            if len(items) != len(C):
                stats["fric_count_bad"] += 1
            else:
                for it, cc in zip(items, C):
                    pid = struct.unpack_from("<I", it, 0)[0]
                    gap, fric = struct.unpack_from("<ff", it, 8)
                    if pid != int(cc["pid"]) or gap != pf.f32(cc["raw"], 0x58) or fric != pf.f32(cc["raw"], 0x44):
                        stats["fric_bad"] += 1
    print(stats)
    runs = []
    for f, v in fd:
        if runs and f == runs[-1][1] + 1 and v == runs[-1][2]:
            runs[-1][1] = f
        else:
            runs.append([f, f, v])
    print("found differs (mine) runs:", runs[:30])
    for e in bad_examples:
        print(" ", e)
    if big:
        runs = []
        for f, ball, d, s in big:
            if runs and f == runs[-1][1] + 1:
                runs[-1][1] = f
                runs[-1][2] = max(runs[-1][2], d)
            else:
                runs.append([f, f, d, ball, s])
        print("frames with |PSI - CK| > 2:", len(big), "runs:", [(a, b, d, ball) for a, b, d, ball, _ in runs][:20])


if __name__ == "__main__":
    main()
