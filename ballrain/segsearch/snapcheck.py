"""Acceptance check of the process snapshot (docs/segsearch.md "Rollback"): save at TAS frame F, run N frames (blocks of 4)
of keys, load, run them again (raw OBS bytes must be identical); optionally LOADS more loads with other keys and the
reference run again (identical, handles and private memory flat); two cold prefix replays (masked: the three fields that
differ between processes, MIND ledge / plane w and SCOR energy[4], must be the only difference to the loaded run).
KIND: tas (the record's keys), rand (random holds), trafo (TAS to 5240, then into the L1 stone trafo: transform
5428-6088). Prints one JSON line.
    python -m ballrain.segsearch.snapcheck F N KIND [LOADS]"""
import sys, time, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
import numpy as np
from ballrain import tas
from ballrain.client import SessionConfig
from ballrain.headless import HeadlessLauncher
from ballrain.segsearch.engine import default_engine, keys_to_frames
from ballrain.segsearch.core import random_holds
from ballrain.segsearch.snapsession import SnapSession

F, N, KIND = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
LOADS = int(sys.argv[4]) if len(sys.argv) > 4 else 0
frames = tas.load_raw(REPO / "runtime/datasets/tas-v1/HS01_4740_1_30_344/run/framelog.bin")
delta = float(frames["delta"][F])
if KIND == "trafo":  # TAS to 5240, then down+right 120 frames into the stone trafo (transform 5428-6088), then no keys
    full = np.r_[frames["keys"][:5240], np.full(120, 10, np.uint32), np.zeros(4000, np.uint32)].astype(np.uint32)
    frames = tas.make_frames(np.full(len(full), frames["delta"][0], np.float32), full)
keys = frames["keys"][F:F + N].copy() if KIND in ("tas", "trafo") else random_holds(N, np.random.default_rng(F))
eng = default_engine()
W = REPO / "runtime/segsearch/snapcheck"; W.mkdir(parents=True, exist_ok=True)
tas.save_raw(W / "prefix.bin", frames[:F])


def session(tag, raw):
    L = HeadlessLauncher(binary=eng.binary, root=eng.root, socket="tcp", log=False, ball_order="ckid")
    s = SnapSession(SessionConfig(1, W / f"run-{tag}", copy="hl-47", prefix=W / "prefix.bin", hello_timeout_s=90,
                                  obs_timeout_s=90, events_txt=False), L)
    s.raw_hook = raw.append
    s.start()
    return s


def run(s, ks, raw):
    raw.clear()
    fr = keys_to_frames(ks, delta)
    for k in range(0, len(fr), 4):
        s.act(fr[k:k + 4])
        if s.ended:
            break
    return list(raw)


def cmp(a, b):
    if len(a) != len(b):
        return f"count {len(a)} vs {len(b)}"
    bad = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if not bad:
        return "identical"
    i = bad[0]
    x, y = a[i], b[i]
    off = [j for j in range(min(len(x), len(y))) if x[j] != y[j]]
    return f"{len(bad)}/{len(a)} obs differ, first obs {i}, len {len(x)}/{len(y)}, {len(off)} bytes differ, first byte offsets {off[:12]}"


res = {"F": F, "N": N, "kind": KIND}
raw = []
s = session("a", raw)
img = s.save()
res["save"] = {"MB": round(img.nbytes / 2**20, 1), "s": round(img.seconds, 3), "regions": len(img.regions), "threads": len(img.contexts)}
r1 = run(s, keys, raw)
st = s.load(img)
res["load1"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()}
r2 = run(s, keys, raw)
res["save_run_vs_load_run"] = cmp(r1, r2)
if LOADS:
    rng = np.random.default_rng(1)
    h0 = s.snap.handle_count(); m0 = None
    import psutil
    p = psutil.Process(s.handle.pid); m0 = p.memory_info().private
    t0 = time.perf_counter()
    for i in range(LOADS):
        s.load(img)
        run(s, random_holds(int(rng.integers(40, 400)), rng), raw)
    res["loads"] = {"n": LOADS, "s_per_load_and_run": round((time.perf_counter() - t0) / LOADS, 4),
                    "handles": [h0, s.snap.handle_count()], "privMB": [round(m0 / 2**20, 1), round(p.memory_info().private / 2**20, 1)]}
    s.load(img)
    r3 = run(s, keys, raw)
    res["after_loads_vs_first"] = cmp(r1, r3)
s.close(kill=True)
raw_b, raw_c = [], []
b = session("b", raw_b); rb = run(b, keys, raw_b); b.close(kill=True)
c = session("c", raw_c); rc = run(c, keys, raw_c); c.close(kill=True)
res["cold_vs_cold"] = cmp(rb, rc)
res["save_run_vs_cold"] = cmp(r1, rb)
from ballrain.protocol import split_message


def secdiff(x, y):
    """{section tag: set of differing byte offsets in the payload} (plus '#count' if the section lists differ)."""
    mx, my = split_message(x), split_message(y)
    out = {}
    tx = [s.tag for s in mx.sections]; ty = [s.tag for s in my.sections]
    if tx != ty:
        out["#sections"] = {(tuple(tx), tuple(ty))}
    for a, b in zip(mx.sections, my.sections):
        if a.payload != b.payload:
            out[a.tag] = {j for j in range(min(len(a.payload), len(b.payload))) if a.payload[j] != b.payload[j]} | ({-1} if len(a.payload) != len(b.payload) else set())
    return out


cc_tags, lc_tags, extra = {}, {}, []
for i, (x, y, z) in enumerate(zip(r1, rb, rc)):
    dcc = secdiff(y, z); dlc = secdiff(x, y)
    for t, o in dcc.items(): cc_tags[t] = cc_tags.get(t, 0) + 1
    for t, o in dlc.items():
        lc_tags[t] = lc_tags.get(t, 0) + 1
        if t not in dcc or not o <= dcc[t]:
            extra.append((i, t, sorted(o - dcc.get(t, set()))[:8]))
import struct
from ballrain.protocol import MSG_HDR, SEC_HDR, align4


def masked(buf):
    """The message with the two process-address fields zeroed: MIND ledge (entry +32, u32) and SCOR energy[3]
    (+20, u32: a parameter cell's pointer); docs/headless.md 13.4."""
    b = bytearray(buf)
    n = MSG_HDR.unpack_from(b, 0)[5]
    pos = MSG_HDR.size
    for _ in range(n):
        tag, sv, sf, plen = SEC_HDR.unpack_from(b, pos)
        pos += SEC_HDR.size
        t = tag.to_bytes(4, "little").decode("latin-1")
        if t == "MIND":
            cnt = struct.unpack_from("<H", b, pos)[0]
            for j in range(cnt):
                o = pos + 2 + 112 * j
                b[o + 32:o + 36] = bytes(4)  # ledge (address)
                b[o + 24:o + 28] = bytes(4)  # plane w (not maintained by IVP)
        elif t == "SCOR":
            b[pos + 24:pos + 28] = bytes(4)  # energy[4] (a parameter cell's pointer)
        pos += align4(plen)
    return bytes(b)


def cmpm(a, b):
    return cmp([masked(x) for x in a], [masked(y) for y in b])


res["masked_cold_vs_cold"] = cmpm(rb, rc)
res["masked_load_run_vs_cold"] = cmpm(r2, rb)
res["masked_save_run_vs_cold"] = cmpm(r1, rb)
res["sections_cold_vs_cold"] = cc_tags
res["sections_load_vs_cold"] = lc_tags
res["load_vs_cold_outside_cold_vs_cold"] = {"n": len(extra), "first": extra[:5]}
print(json.dumps(res))
