"""Throwaway stub client for BallrainIO agent mode (docs/agent-protocol.md). Not the real client.

Drives one game process through agent mode, sending the frames of a recorded .tas / raw frames file:
  --mode k1      one frame per action, TRAC on, every section, every frame
  --mode blocks  run-length acts (one listed frame + repeat) mixed with multi-frame blocks, a stop mask
  --mode prefix  BALLRAIN_PREFIX with the first N frames, then blocks
Checks per observation: section lengths, (float)SR == GetSRScore, frames_done bookkeeping.
"""
import os as _os, pathlib as _pl
_REPO = _os.environ.get("BALLRAIN_REPO") or str(_pl.Path(__file__).resolve().parents[3])  # repo root
import argparse
import ctypes
import json
import mmap
import os
import struct
import sys
import time
from ctypes import wintypes
from pathlib import Path

REPO = Path(_REPO)
sys.path.insert(0, str(REPO))
from ballrain import instances, tas  # noqa: E402
from ballrain import protocol as BP  # noqa: E402  (the real client's parser, read-only use)

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateEventW.restype = wintypes.HANDLE
k32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
k32.SetEvent.argtypes = [wintypes.HANDLE]
k32.WaitForMultipleObjects.argtypes = [wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE), wintypes.BOOL, wintypes.DWORD]
k32.WaitForMultipleObjects.restype = wintypes.DWORD
k32.CloseHandle.argtypes = [wintypes.HANDLE]

SIZE = 4 << 20
OBS_OFF, OBS_CAP = 0x1000, 3 << 20
ACT_OFF, ACT_CAP = OBS_OFF + OBS_CAP, 512 << 10

EV = {1: "PreLoadLevel", 2: "PostLoadLevel", 3: "StartLevel", 4: "CamNavActive", 5: "CamNavInactive",
      6: "BallNavActive", 7: "BallNavInactive", 8: "CounterActive", 9: "CounterInactive", 10: "PauseLevel",
      11: "UnpauseLevel", 12: "BallOff", 13: "Dead", 14: "PreSubLife", 15: "PostSubLife", 16: "PreCheckpoint",
      17: "PostCheckpoint", 18: "ExtraPoint", 19: "PreLifeUp", 20: "PostLifeUp", 21: "LevelFinish", 22: "GameOver",
      23: "PreEndLevel", 24: "PostEndLevel", 25: "Physicalize", 26: "Unphysicalize", 27: "ActiveBall",
      28: "PreResetLevel", 29: "PreExitLevel", 30: "RunEnd"}
CODE = {v: k for k, v in EV.items()}

ITEM = {"FRIC": 53, "MIND": 112, "BODY": 180}


def sections(buf, off, end):
    out = {}
    while off + 12 <= end:
        tag = bytes(buf[off:off + 4]).decode("latin-1")
        ver, flags, ln = struct.unpack_from("<HHI", buf, off + 4)
        out[tag] = (ver, flags, bytes(buf[off + 12:off + 12 + ln]))
        off = off + 12 + ((ln + 3) & ~3)
    return out


def parse_obs(mm):
    magic, ver, flags, blen, frames, nsec, _ = struct.unpack_from("<4sHHIIII", mm, OBS_OFF)
    assert magic == b"OBS1", magic
    secs = sections(mm, OBS_OFF + 24, OBS_OFF + blen)
    assert len(secs) == nsec, (len(secs), nsec, list(secs))
    return flags, frames, secs


def check_obs(secs, stats):
    t = secs["TIME"][2]
    cur, nxt, last, c8, cc, d0, delta, sr, bml, running, covered = struct.unpack_from("<dddffffdfBI", t, 0)
    sr32, rawframe, tailf = struct.unpack_from("<fII", t, 57)
    if float(struct.unpack("<f", struct.pack("<f", sr))[0]) != bml:
        stats["sr_mismatch"] += 1
        if stats["sr_mismatch"] <= 5:
            print(f"  SR mismatch: double {sr!r} -> f32 {struct.unpack('<f', struct.pack('<f', sr))[0]!r} vs bml {bml!r}")
    stats["sr_checked"] += 1
    for tag, size in ITEM.items():
        if tag in secs:
            p = secs[tag][2]
            n = struct.unpack_from("<H", p, 0)[0]
            assert len(p) == 2 + n * size, (tag, n, len(p))
            stats[tag + "_max"] = max(stats.get(tag + "_max", 0), n)
            stats[tag + "_frames"] = stats.get(tag + "_frames", 0) + (n > 0)
    if "BALL" in secs:
        p = secs["BALL"][2]
        assert len(p) == 4 + 1 + 1 + 48 + 24 + 32 + 32 + 8 + 60 + 8 + 8 + 4, len(p)
        stats["ball_found"] = stats.get("ball_found", 0) + p[5]
        if p[5]:
            ck = struct.unpack_from("<3f", p, 6 + 36)
            psi = struct.unpack_from("<3d", p, 54)
            d = max(abs(a - b) for a, b in zip(ck, psi))
            if d > stats.get("ball_psi_ck_max", 0):
                stats["ball_psi_ck_max"] = d
            if d > 2.0:
                stats["ball_psi_ck_bad"] = stats.get("ball_psi_ck_bad", 0) + 1
    stats["sections_seen"] = sorted(set(stats.get("sections_seen", [])) | set(secs))
    if "GLCT" in secs:
        p = secs["GLCT"][2]
        assert len(p) == 30, len(p)
        g = struct.unpack_from("<6i", p, 6)
        if any(g[1::2]):
            stats["glct_contact"] = stats.get("glct_contact", 0) + 1
    if "CAMS" in secs:
        assert len(secs["CAMS"][2]) == 5 * 52 + 4
    if "SCOR" in secs:
        p = secs["SCOR"][2]
        assert len(p) == 8 + 24 + 4 + 4 + 4 + 12, len(p)
        stats["hs_last"] = struct.unpack_from("<i", p, 40)[0]
        stats["points_last"] = struct.unpack_from("<I", p, 8)[0]
    if "ARRS" in secs:
        p = secs["ARRS"][2]
        n = struct.unpack_from("<H", p, 0)[0]
        o = 2
        for _ in range(n):
            ln = p[o + 9]
            o += 10 + ln
        assert o == len(p), (o, len(p))
        stats["arr_cells"] = stats.get("arr_cells", 0) + n
    if "WTCH" in secs:
        p = secs["WTCH"][2]
        n = struct.unpack_from("<H", p, 0)[0]
        assert len(p) == 2 + 17 * n
        stats["watch_changes"] = stats.get("watch_changes", 0) + n
    return sr, bml, running


def parse_events(secs):
    if "EVTS" not in secs:
        return []
    p = secs["EVTS"][2]
    n = struct.unpack_from("<H", p, 0)[0]
    return [struct.unpack_from("<IHH", p, 2 + 8 * i) for i in range(n)]


def parse_level(path):
    data = Path(path).read_bytes()
    msgs = []
    off = 0
    while off + 24 <= len(data):
        magic, ver, flags, blen, frames, nsec, _ = struct.unpack_from("<4sHHIIII", data, off)
        assert magic == b"LVL1", (magic, off)
        secs = sections(data, off + 24, off + blen)
        assert len(secs) == nsec
        msgs.append((flags, frames, {k: (v[0], len(v[2]), struct.unpack_from("<I", v[2], 0)[0] if len(v[2]) >= 4 else None)
                                     for k, v in secs.items()}))
        off += blen
    assert off == len(data), (off, len(data))
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="k1", choices=["k1", "blocks", "prefix", "prefixev"])
    ap.add_argument("--prefix-len", type=int, default=3000)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stop-after", type=int, default=0, help="send STOP once frames_done >= this")
    ap.add_argument("--dump", action="store_true", help="append every observation to <out>/obs.bin")
    ap.add_argument("--obs-mask", type=int, default=0)
    ap.add_argument("--die-after", type=int, default=0, help="exit without answering once frames_done >= this")
    args = ap.parse_args()

    frames = tas.load_any(args.frames)
    deltas = [float(x) for x in frames["delta"]]
    keys = [int(x) for x in frames["keys"]]
    raw = [struct.pack("<fI", d, k) for d, k in zip(deltas, keys)]
    N = len(raw)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    name = f"Local\\ballrain-train-00-{os.getpid()}-{int(time.time() * 1000) & 0xFFFFFF:06x}"
    mm = mmap.mmap(-1, SIZE, tagname=name)
    struct.pack_into("<4sIIIIIII", mm, 0, b"BRSM", 1, SIZE, OBS_OFF, OBS_CAP, ACT_OFF, ACT_CAP, 0)
    struct.pack_into("<I", mm, 0x3C, os.getpid())
    ev_obs = k32.CreateEventW(None, False, False, name + ".obs")
    ev_act = k32.CreateEventW(None, False, False, name + ".act")
    assert ev_obs and ev_act

    env = {k: None for k in instances.MANAGED_VARS}
    env.update({
        "BALLRAIN_MODE": "agent", "BALLRAIN_SHM": name, "BALLRAIN_LEVEL": str(args.level),
        "BALLRAIN_PARENT_PID": str(os.getpid()), "BALLRAIN_TIMEOUT_MS": "60000",
        "BALLRAIN_LEVELMSG": str(out / "level.bin"), "BALLRAIN_FRAMELOG": str(out / "framelog.bin"),
        "BALLRAIN_TRACE": str(out / "trace.txt"), "BALLRAIN_EVENTS": str(out / "events.txt"),
        "BALLRAIN_SKIP_RENDER": "1", "BALLRAIN_EXIT": "1",
        "BALLRAIN_PREFIX": None, "BALLRAIN_PREFIX_EVENTS": None,
        "BALLRAIN_MAX_FRAMES": str(args.max_frames) if args.max_frames else None,
    })
    if args.mode in ("prefix", "prefixev"):
        pf = out / "prefix.bin"
        pf.write_bytes(b"".join(raw[:args.prefix_len]))
        env["BALLRAIN_PREFIX"] = str(pf)
        if args.mode == "prefixev":
            env["BALLRAIN_PREFIX_EVENTS"] = "1"
    for f in ("framelog.bin", "trace.txt", "events.txt", "level.bin", "level.bin.inc"):
        (out / f).unlink(missing_ok=True)

    copy = instances.get_copy("train-00")
    instances.preflight(copy)
    t0 = time.monotonic()
    pid = instances.launch_explorer_cmd(copy, env)
    h = instances.ProcessHandle(pid, copy.exe)
    print(f"launched pid {pid}, shm {name}")
    handles = (wintypes.HANDLE * 2)(ev_obs, h._h)

    dumpf = open(out / "obs.bin", "wb") if args.dump else None
    py_seq = 0
    mod_seq = 0
    stats = {"sr_mismatch": 0, "sr_checked": 0, "obs": 0, "acts": 0}
    all_events = []
    t_wait = 0.0
    final = None
    block_i = 0
    sent_stop = False
    while True:
        tw = time.perf_counter()
        r = k32.WaitForMultipleObjects(2, handles, False, int(args.timeout * 1000))
        t_wait += time.perf_counter() - tw
        if r != 0:
            print("wait result", r, "state", struct.unpack_from("<I", mm, 0x30)[0])
            break
        seq = struct.unpack_from("<Q", mm, 0x20)[0]
        state, endr = struct.unpack_from("<II", mm, 0x30)
        if seq != mod_seq + 1:
            print(f"event without new obs: seq {seq} (had {mod_seq}) state {state} end {endr} err {struct.unpack_from('<I', mm, 0x44)[0]}")
            if state == 7:
                break
            continue
        mod_seq = seq
        flags, done, secs = parse_obs(mm)
        blen = struct.unpack_from("<I", mm, OBS_OFF + 8)[0]
        if dumpf:
            dumpf.write(struct.pack("<I", blen) + bytes(mm[OBS_OFF:OBS_OFF + blen]))
        try:
            BP.parse_observation(bytes(mm[OBS_OFF:OBS_OFF + blen]))
            stats["client_parse_ok"] = stats.get("client_parse_ok", 0) + 1
        except Exception as e:  # noqa: BLE001
            stats["client_parse_err"] = stats.get("client_parse_err", 0) + 1
            stats["client_parse_last_err"] = repr(e)
        stats["obs"] += 1
        sr, bml, running = check_obs(secs, stats)
        evs = parse_events(secs)
        all_events += evs
        if "TRAC" in secs:
            p = secs["TRAC"][2]
            n = struct.unpack_from("<I", p, 0)[0]
            assert len(p) == 4 + 52 * n
            stats["trac"] = stats.get("trac", 0) + n
        if "PHYS" in secs:
            stats["phys_new"] = stats.get("phys_new", 0) + struct.unpack_from("<H", secs["PHYS"][2], 0)[0]
        if flags & 1:
            final = dict(frames_done=done, state=state, end_reason=endr, flags=flags, sr=sr, bml=bml, running=running)
            if "SREN" in secs:
                p = secs["SREN"][2]
                srd, sr32, bmlx, tail, fin, inact, run_, _ = struct.unpack_from("<dffIBBBB", p, 0)
                srfin, sr32fin = struct.unpack_from("<df", p, 24)
                hsv = struct.unpack_from("<B6sB6IIii", p, 36)
                final.update(sren=dict(sr=srd, sr_ms=int(srd), sr32=sr32, bml=bmlx, tail=tail, finished=fin,
                                       inactive_seen=inact, running=run_, sr_at_finish=srfin, sr32_at_finish=sr32fin,
                                       hs_valid=hsv[0], energy=list(hsv[3:9]), level_bonus=hsv[9], bml_hs=hsv[10],
                                       hs=hsv[11]))
            break
        # choose the next action, starting at frame `done`
        if args.die_after and done >= args.die_after:
            print(f"dying at frames_done {done}, game pid {pid}", flush=True)
            os._exit(0)
        if args.stop_after and done >= args.stop_after and not sent_stop:
            act_flags, lst, rep, mask = 1, [], 1, 0
            sent_stop = True
        elif done >= N:
            act_flags, lst, rep, mask = 1, [], 1, 0
        elif args.mode == "k1" or (args.mode == "prefixev" and flags & 2 == 0 and False):
            act_flags, lst, rep, mask = 2, [raw[done]], 1, 0
        elif args.mode == "prefixev" and flags & 2:
            act_flags, lst, rep, mask = 0, [], 1, 0   # acknowledge a prefix observation
        else:
            block_i += 1
            mask = (1 << CODE["StartLevel"]) | (1 << CODE["BallNavActive"]) | (1 << CODE["PreCheckpoint"])
            if block_i % 3 == 0:
                # run-length: one listed frame repeated
                j = done
                while j < N and raw[j] == raw[done] and j - done < 400:
                    j += 1
                lst, rep = [raw[done]], j - done
            else:
                lst, rep = raw[done:done + 37 + (block_i % 50)], 1
            act_flags = 2 if block_i % 5 == 0 else 0
        py_seq += 1
        body = b"".join(lst)
        struct.pack_into("<4sHHIIIIIII", mm, ACT_OFF, b"ACT1", 1, act_flags, py_seq, len(lst), rep,
                         mask & 0xFFFFFFFF, mask >> 32, args.obs_mask, 0)
        mm[ACT_OFF + 0x24:ACT_OFF + 0x24 + len(body)] = body
        struct.pack_into("<Q", mm, 0x28, py_seq)
        k32.SetEvent(ev_act)
        stats["acts"] += 1

    if dumpf:
        dumpf.close()
    exited = h.wait(60)
    code = h.exit_code()
    if not exited:
        print("game did not exit; terminating our own pid", pid)
        h.terminate()
        h.wait(10)
    h.close()
    wall = time.monotonic() - t0
    hdr_ver = bytes(mm[0x4C:0x4C + 180]).split(b"\0")[0].decode()
    res = dict(stats=stats, final=final, exit_code=code, wall_s=wall, py_wait_s=t_wait, version=hdr_ver,
               events=[(f, EV.get(c, c), p) for f, c, p in all_events if c not in (25, 26)],
               n_phys_events=sum(1 for _, c, _ in all_events if c in (25, 26)))
    try:
        lm, _n = BP.parse_level_message((out / "level.bin").read_bytes())
        msgs, used = BP.parse_level_stream((out / "level.bin.inc").read_bytes())
        res["client_level"] = dict(ents=len(lm.entities), grps=len(lm.groups), arrs=len(lm.arrays), phys=len(lm.phys),
                                   meshes=len(lm.meshes), ivp=len(lm.ivp), depth=len(lm.depth), watch=len(lm.watch),
                                   unknown=lm.unknown, inc_msgs=len(msgs), inc_bytes_used=used,
                                   inc_bytes=(out / "level.bin.inc").stat().st_size,
                                   inc_phys=sum(len(m.phys) for m in msgs), inc_mesh=sum(len(m.meshes) for m in msgs),
                                   inc_ivp=sum(len(m.ivp) for m in msgs),
                                   contact_ids=sorted({e.contact_id for e in lm.entities}),
                                   watch_groups=sorted({g for _, g in lm.watch if g})[:8],
                                   param_types=sorted({c.param_type for a in lm.arrays for c in a.columns if c.param_type}))
    except Exception as e:  # noqa: BLE001
        res["client_level_error"] = repr(e)
    try:
        res["level_full"] = parse_level(out / "level.bin")
        inc = parse_level(out / "level.bin.inc")
        res["level_inc_msgs"] = len(inc)
        res["level_inc_phys"] = sum(m[2]["PHYS"][2] for m in inc)
        res["level_inc_mesh"] = sum(m[2]["MESH"][2] for m in inc)
    except Exception as e:  # noqa: BLE001
        res["level_error"] = repr(e)
    (out / "stub_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "events"}, indent=1)[:4000])
    k32.CloseHandle(ev_obs)
    k32.CloseHandle(ev_act)
    mm.close()


if __name__ == "__main__":
    main()
