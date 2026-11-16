"""Parser for the probe's frames.txt (per-frame records)."""
import re
import struct


def kv(tokens):
    out = {}
    for t in tokens:
        if "=" in t:
            k, v = t.split("=", 1)
            out[k] = v
    return out


def vec(s):
    return [float(x) for x in s.split(",")]


def parse(path, want=None):
    frames = []
    cur = None
    events = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            tag, _, rest = line.rstrip("\n").partition(" ")
            if want and tag not in want and tag not in ("F", "EV"):
                continue
            if tag == "F":
                p = rest.split(" ")
                cur = {"idx": int(p[0]), "ball": p[1], "C": [], "D": [], "B": [], "M": {}, "A": [], "W": [], "CC": [], "CD": [], "P": []}
                frames.append(cur)
            elif tag == "EV":
                p = rest.split(" ")
                events.append((int(p[0]), p[1]))
            elif cur is None:
                continue
            elif tag == "T":
                p = rest.split(" ")
                d = kv(p)
                cur["T"] = {"cur": float(p[0]), "next": float(p[1]), "last": float(p[2]), "c8": float(p[3]), "cc": float(p[4]),
                            "d0": float(p[5]), "d": float(d["d"]), "sr": float(d["sr"]), "hs": int(d["hs"])}
            elif tag == "BW":
                cur["BW"] = [float(x) for x in rest.split(" ")]
            elif tag == "K":
                if rest.startswith("none"):
                    continue
                d = kv(rest.split(" "))
                k = {"xrad": float(d["xrad"]), "type": int(d["type"]), "tpsi": float(d.get("tpsi", "nan"))}
                for n in ("s", "sc", "rs", "rsc", "pos", "dp", "ql", "qn", "axis", "cur"):
                    if n in d:
                        k[n] = vec(d[n])
                k["cflags"] = int(d.get("cflags", "0"), 16)
                k["psif"] = int(d.get("psif", "0"), 16)
                cur["K"] = k
            elif tag == "C":
                p = rest.split(" ")
                d = kv(p[1:])
                d["n"] = int(p[0])
                d["raw"] = bytes.fromhex(d["raw"]) if d["raw"] != "??" else b""
                cur["C"].append(d)
            elif tag == "CT":
                p = rest.split(" ")
                cur["C"][-1]["tri"] = parse_tri(p[1:])
            elif tag == "D":
                p = rest.split(" ")
                d = kv(p[1:])
                d["n"] = int(p[0])
                d["raw"] = bytes.fromhex(d["raw"]) if d["raw"] != "??" else b""
                cur["D"].append(d)
            elif tag == "DT":
                p = rest.split(" ")
                s = " ".join(p[1:])
                a, _, b = s.partition(" | opp ")
                cur["D"][-1]["tri"] = parse_tri(a.split(" "))
                if b:
                    cur["D"][-1]["opp"] = parse_tri(b.split(" "))
            elif tag == "P":
                cur["P"].append(rest)
            elif tag == "B":
                cur["B"].append(rest)
            elif tag == "M":
                p = rest.split(" ")
                cur["M"][p[0]] = [float(x) for x in p[2:]]
            elif tag == "A":
                cur["A"].append(rest)
            elif tag == "W":
                cur["W"].append(rest)
            elif tag == "CC":
                cur["CC"].append(rest)
            elif tag == "CD":
                cur["CD"].append(rest)
    return frames, events


def parse_tri(p):
    d = kv(p[:3])
    nums = [float(x) for x in p[3:12]]
    return {"ledge": d["ledge"], "tri": d["tri"], "e": int(d["e"]), "v": [nums[0:3], nums[3:6], nums[6:9]]}


def f32(raw, off):
    return struct.unpack_from("<f", raw, off)[0]


def f64(raw, off):
    return struct.unpack_from("<d", raw, off)[0]


def i32(raw, off):
    return struct.unpack_from("<i", raw, off)[0]


def parse_partners(frames):
    """P lines -> {obj: {name, id, movable, core 16 doubles}} (last seen)."""
    out = {}
    for fr in frames:
        for s in fr["P"]:
            p = s.split(" ")
            obj = p[0].split("=")[1]
            name = p[1]
            d = kv(p[2:6])
            i = p.index("core=")
            core = [float(x) for x in p[i + 1:i + 17]]
            out.setdefault(obj, {"name": name, "id": int(d["id"]), "movable": d["movable"] == "1", "cores": {}})
            out[obj]["cores"][fr["idx"]] = core
    return out
