"""Receiving the game's own picture from BallrainIO >= 0.3.3 (BALLRAIN_CAPTURE, BallrainIO/src/Capture.cpp).

The mod reads back its back buffer after every game frame that passes a 1/fps game-time step and writes it to a
named pipe this module serves (Windows, ctypes). Each record: a 48-byte header (see Capture.cpp) and the pixels.
`CaptureServer` accepts one connection in a thread and hands every record to a sink; `FfmpegSink` turns the
records into a video with the ffmpeg of imageio-ffmpeg (each record written `repeat` times, so the video runs at
game speed) and keeps a sidecar of (video frame, game frame, game time).

Nothing here reads the screen: the only input is what the mod sends.
"""
from __future__ import annotations

import ctypes
import json
import os
import struct
import subprocess
import sys
import threading
import time
import uuid
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<4s9Id")
assert HEADER.size == 48
MAGIC = b"BRVF"
CAPTURE_VARS = ("BALLRAIN_CAPTURE", "BALLRAIN_CAPTURE_FPS", "BALLRAIN_CAPTURE_TAIL")

PIPE_ACCESS_INBOUND = 0x00000001
PIPE_TYPE_BYTE = 0x0
PIPE_WAIT = 0x0
ERROR_PIPE_CONNECTED = 535
ERROR_BROKEN_PIPE = 109
INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value

if sys.platform == "win32":
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateNamedPipeW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID]
    _k32.CreateNamedPipeW.restype = wintypes.HANDLE
    _k32.ConnectNamedPipe.argtypes = [wintypes.HANDLE, wintypes.LPVOID]
    _k32.DisconnectNamedPipe.argtypes = [wintypes.HANDLE]
    _k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                              wintypes.LPVOID]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                 wintypes.DWORD, wintypes.HANDLE]
    _k32.CreateFileW.restype = wintypes.HANDLE


@dataclass
class Frame:
    frame: int  # game frames played when this picture was drawn
    repeat: int  # video frames it stands for
    width: int
    height: int
    pitch: int
    bpp: int
    masks: tuple
    game_ms: float
    pixels: bytes

    def bgr0(self) -> np.ndarray:
        """(height, width, 4) uint8 in B, G, R, X byte order (32-bit X8R8G8B8 back buffer)."""
        if self.bpp != 32 or self.masks != (0xFF0000, 0xFF00, 0xFF):
            raise ValueError(f"unsupported back buffer format: {self.bpp} bpp, masks {self.masks}")
        a = np.frombuffer(self.pixels, np.uint8).reshape(self.height, self.pitch)
        return a[:, : self.width * 4].reshape(self.height, self.width, 4)


class CaptureServer:
    """One inbound named pipe; `sink(frame)` is called from the reader thread for every record."""

    def __init__(self, sink, name: str | None = None, buf: int = 8 << 20):
        self.name = name or f"ballrain-capture-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.path = rf"\\.\pipe\{self.name}"
        self.sink = sink
        self.error: str | None = None
        self.records = 0
        self.connected = threading.Event()
        self._h = _k32.CreateNamedPipeW(self.path, PIPE_ACCESS_INBOUND, PIPE_TYPE_BYTE | PIPE_WAIT, 1, 0, buf, 0, None)
        if self._h == INVALID_HANDLE_VALUE:
            raise OSError(ctypes.get_last_error(), f"CreateNamedPipe {self.path}")
        self._closed = False
        self._t = threading.Thread(target=self._run, name="capture-reader", daemon=True)
        self._t.start()

    def _read(self, n: int) -> bytes | None:
        out = bytearray(n)
        view = (ctypes.c_char * n).from_buffer(out)
        got = 0
        while got < n:
            done = wintypes.DWORD()
            ok = _k32.ReadFile(self._h, ctypes.byref(view, got), min(n - got, 4 << 20), ctypes.byref(done), None)
            if not ok or done.value == 0:
                err = ctypes.get_last_error()
                if got == 0 and err in (ERROR_BROKEN_PIPE, 0):
                    return None
                raise OSError(err, f"pipe read ({got}/{n} bytes)")
            got += done.value
        return bytes(out)

    def _run(self):
        try:
            if not _k32.ConnectNamedPipe(self._h, None) and ctypes.get_last_error() != ERROR_PIPE_CONNECTED:
                if self._closed:
                    return
                raise OSError(ctypes.get_last_error(), "ConnectNamedPipe")
            self.connected.set()
            while True:
                hb = self._read(HEADER.size)
                if hb is None:
                    return
                magic, frame, repeat, w, h, pitch, bpp, rm, gm, bm, ms = HEADER.unpack(hb)
                if magic != MAGIC or not (0 < w <= 8192 and 0 < h <= 8192 and pitch >= w * bpp // 8):
                    raise ValueError(f"bad capture record header {hb[:16]!r}")
                px = self._read(h * pitch)
                if px is None:
                    raise ValueError("capture stream ended inside a record")
                self.records += 1
                self.sink(Frame(frame, repeat, w, h, pitch, bpp, (rm, gm, bm), ms, px))
        except Exception as e:  # noqa: BLE001  reported by join()
            self.error = f"{type(e).__name__}: {e}"
            # nobody reads the pipe any more: disconnect it, so the game's next write fails at once and the mod turns
            # its capture off, instead of the game blocking on a full pipe until the observation timeout
            if self.connected.is_set():
                _k32.DisconnectNamedPipe(self._h)

    def join(self, timeout: float | None = None) -> bool:
        self._t.join(timeout)
        return not self._t.is_alive()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if not self.connected.is_set():  # unblock ConnectNamedPipe: connect and drop a client ourselves
            h = _k32.CreateFileW(self.path, 0x40000000, 0, None, 3, 0, None)  # GENERIC_WRITE, OPEN_EXISTING
            if h != INVALID_HANDLE_VALUE:
                _k32.CloseHandle(h)
        self._t.join(10)
        _k32.CloseHandle(self._h)


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


IDLE_PRIORITY_CLASS = 0x40
BELOW_NORMAL_PRIORITY_CLASS = 0x4000


def low_priority_flags(idle: bool = True) -> int:
    if sys.platform != "win32":
        return 0
    return (IDLE_PRIORITY_CLASS if idle else BELOW_NORMAL_PRIORITY_CLASS) | subprocess.CREATE_NO_WINDOW


class FfmpegSink:
    """Raw BGR0 records -> an intermediate video (fast, near-lossless H.264) + sidecar json.

    The intermediate keeps the capture cheap for the game (the pipe never waits on a slow encoder); the published
    files are encoded from it afterwards (`encode_final`)."""

    def __init__(self, out: Path, fps: float = 60.0, crf: int = 12, threads: int = 4):
        self.out = Path(out)
        self.fps = fps
        self.crf = crf
        self.threads = threads
        self.proc: subprocess.Popen | None = None
        self.size = None
        self.video_frames = 0
        self.rows: list = []  # [first video frame, repeat, game frame, game ms]
        self.first_png: Path | None = None

    def __call__(self, f: Frame):
        if self.proc is None:
            self.size = (f.width, f.height)
            self.proc = subprocess.Popen(
                [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr0",
                 "-s", f"{f.width}x{f.height}", "-r", f"{self.fps:g}", "-i", "-", "-an", "-c:v", "libx264",
                 "-preset", "ultrafast", "-crf", str(self.crf), "-pix_fmt", "yuv420p", "-threads", str(self.threads),
                 str(self.out)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                creationflags=low_priority_flags(idle=False))
        elif (f.width, f.height) != self.size:
            raise ValueError(f"capture size changed from {self.size} to {(f.width, f.height)}")
        img = np.ascontiguousarray(f.bgr0()).tobytes()
        self.rows.append([self.video_frames, f.repeat, f.frame, round(f.game_ms, 4)])
        for _ in range(f.repeat):
            self.proc.stdin.write(img)
        self.video_frames += f.repeat

    def close(self) -> dict:
        err = ""
        if self.proc is not None:
            self.proc.stdin.close()
            err = self.proc.stderr.read().decode("utf-8", "replace")
            self.proc.wait()
        info = {"fps": self.fps, "size": self.size, "video_frames": self.video_frames, "records": len(self.rows),
                "rows": self.rows, "ffmpeg_exit": None if self.proc is None else self.proc.returncode,
                "ffmpeg_err": err[-2000:]}
        return info


def encode_final(src: Path, dst: Path, *, fps: float = 60.0, crf: int = 26, maxrate: str = "1400k",
                 threads: int = 2, vf: str | None = None, timeout: float = 3 * 3600) -> None:
    """The published file: H.264 high, yuv420p, faststart, no audio, capped bitrate; idle priority."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-an", "-c:v", "libx264",
           "-preset", "slow", "-crf", str(crf), "-maxrate", maxrate, "-bufsize", "2800k", "-profile:v", "high",
           "-pix_fmt", "yuv420p", "-r", f"{fps:g}", "-movflags", "+faststart", "-threads", str(threads)]
    if vf:
        cmd += ["-vf", vf]
    r = subprocess.run(cmd + [str(dst)], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout,
                       creationflags=low_priority_flags(idle=True))
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg exit {r.returncode}: {r.stderr[-800:]}")


def probe_duration(path: Path) -> float | None:
    """Duration in seconds from ffmpeg's own banner (no ffprobe in imageio-ffmpeg)."""
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                       creationflags=low_priority_flags(idle=True))
    import re
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", r.stderr)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None


def write_json(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    for i in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # a reader (the dashboard) holds the file open for a moment
            time.sleep(0.25 * (i + 1))
    os.replace(tmp, path)
