"""Stdlib-only replay driver for the headless agent mode (no numpy, no ballrain package: runs on any Python 3).

Starts one engine process, plays a raw frame file (headerless <f32 ms><u32 keys> frames) through agent mode, one
frame per action (as ballrain.dataset.replay_record does), and writes every observation:

    python3 replay_stub.py --bin SIMTOOL --root GAME --level N --frames FILE.raw --out DIR [--k 1] [--prefix N]

DIR gets obs.bin (OBS1 messages back to back), obs_index.txt (frames_done offset length per message), level.msg,
level.msg.inc, framelog.bin, trace.txt, events.txt, headless.log, summary.json. ballrain.headless_gate parses them.
"""
import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time

FRAME_HDR = struct.Struct("<II")
OBS_HDR = struct.Struct("<IIIIQ")
ACT_HDR = struct.Struct("<IHHIIIIIII")
ACT_MAGIC = 0x31544341
BR_PING, BR_PONG, BR_STATE, BR_OBS, BR_ACT = 0x100, 0x101, 0x102, 0x103, 0x104


def recv_exact(s, n):
    b = bytearray()
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c:
            return None
        b += c
    return bytes(b)


def recv(s):
    h = recv_exact(s, 8)
    if h is None:
        return None
    t, n = FRAME_HDR.unpack(h)
    p = recv_exact(s, n) if n else b""
    return None if p is None else (t, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=1, help="frames per action")
    ap.add_argument("--prefix", type=int, default=0, help="BALLRAIN_PREFIX frames (replayed by the engine)")
    ap.add_argument("--obs-mask", type=int, default=0)
    ap.add_argument("--flags", type=int, default=0)
    ap.add_argument("--max", type=int, default=0, help="stop after this many frames (0 = the whole file)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--wrap", default="", help="a launcher to run the engine under (diagnostics)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = os.path.abspath(a.out)
    data = open(a.frames, "rb").read()
    n = len(data) // 8
    if a.max:
        n = min(n, a.max)
    path = os.path.join(tempfile.gettempdir(), f"brs-{os.getpid()}.sock")
    if os.path.exists(path):
        os.unlink(path)
    ls = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    ls.bind(path)
    ls.listen(1)
    env = {k: v for k, v in os.environ.items() if not k.startswith("BALLRAIN_") or k.startswith("BALLRAIN_HL_")}
    env.update(BALLRAIN_SOCK="unix:" + path, BALLRAIN_LEVEL=str(a.level), BALLRAIN_LEVELMSG=out + "/level.msg",
               BALLRAIN_FRAMELOG=out + "/framelog.bin", BALLRAIN_TRACE=out + "/trace.txt",
               BALLRAIN_EVENTS=out + "/events.txt", BALLRAIN_TIMEOUT_MS="60000")
    if a.prefix:
        open(out + "/prefix.bin", "wb").write(data[:8 * a.prefix])
        env["BALLRAIN_PREFIX"] = out + "/prefix.bin"
    t0 = time.perf_counter()
    log = open(out + "/headless.log", "wb")
    proc = subprocess.Popen(([a.wrap] if a.wrap else []) + [a.bin, "--ballrain-agent", "--root", a.root], env=env, stdout=subprocess.DEVNULL,
                            stderr=log)
    ls.settimeout(60)
    s, _ = ls.accept()
    s.sendall(FRAME_HDR.pack(BR_PING, 0))
    m = recv(s)
    assert m and m[0] == BR_PONG, m
    t_hello = time.perf_counter()
    obs = open(out + "/obs.bin", "wb") if not a.no_save else None
    idx = open(out + "/obs_index.txt", "w") if not a.no_save else None
    pos = 0
    seq = 0
    n_obs = 0
    last = None
    first_obs_t = None
    state = None
    while True:
        m = recv(s)
        if m is None:
            break
        t, p = m
        if t == BR_STATE:
            state = struct.unpack_from("<IIII", p)
            continue
        if t != BR_OBS:
            continue
        st, end, fd, err, mseq = OBS_HDR.unpack_from(p)
        body = p[OBS_HDR.size:]
        if first_obs_t is None:
            first_obs_t = time.perf_counter()
        n_obs += 1
        flags = struct.unpack_from("<H", body, 6)[0]
        last = (st, end, fd, err, flags)
        if obs is not None:
            obs.write(body)
            idx.write(f"{fd} {pos} {len(body)}\n")
            pos += len(body)
        if flags & 1:  # final
            break
        if flags & 2:  # prefix observation: acknowledge
            seq += 1
            s.sendall(FRAME_HDR.pack(BR_ACT, ACT_HDR.size) + ACT_HDR.pack(ACT_MAGIC, 1, 0, seq, 0, 1, 0, 0, a.obs_mask, 0))
            continue
        i = fd
        if i >= n:
            seq += 1
            s.sendall(FRAME_HDR.pack(BR_ACT, ACT_HDR.size) + ACT_HDR.pack(ACT_MAGIC, 1, 1, seq, 0, 1, 0, 0, 0, 0))
            continue
        blk = data[8 * i:8 * min(n, i + a.k)]
        seq += 1
        s.sendall(FRAME_HDR.pack(BR_ACT, ACT_HDR.size + len(blk))
                  + ACT_HDR.pack(ACT_MAGIC, 1, a.flags, seq, len(blk) // 8, 1, 0, 0, a.obs_mask, 0) + blk)
    t_end = time.perf_counter()
    s.close()
    proc.wait(30)
    log.close()
    if obs is not None:
        obs.close()
        idx.close()
    summ = {"frames_file": n, "observations": n_obs, "last": last, "state": state, "rc": proc.returncode,
            "hello_s": t_hello - t0, "first_obs_s": (first_obs_t or t_end) - t0, "run_s": t_end - (first_obs_t or t_end),
            "frames_done": last[2] if last else None}
    summ["frames_per_s"] = (summ["frames_done"] or 0) / max(summ["run_s"], 1e-9)
    json.dump(summ, open(out + "/summary.json", "w"), indent=1)
    print(json.dumps(summ))
    os.unlink(path)


if __name__ == "__main__":
    main()
