"""Legacy .tas codec (the format read by TASSupport with LegacyMode 1).

File layout: u32 LE uncompressed byte count, then a zlib-wrapped deflate stream.
Uncompressed payload: 8-byte frames, f32 LE deltaTime (ms) + u32 LE key bits.
Frame 0 is the first frame after the PreLoadLevel message.

The real loader (reference/TASSupport/TASRecord.cpp, legacy branch of TASRecord::Load) checks:
file size >= 8, zlib uncompress() succeeds, header == decompressed length. It then reads
len // 8 frames. This module is stricter on load (len % 8 == 0, no trailing bytes after the
zlib stream) so that malformed files are noticed rather than silently truncated.

Raw frames (BALLRAIN_FRAMES / BALLRAIN_FRAMELOG) are the same 8-byte records without header
or compression.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

FRAME_DTYPE = np.dtype([("delta", "<f4"), ("keys", "<u4")])
FRAME_SIZE = FRAME_DTYPE.itemsize  # 8

KEY_UP = 1 << 0
KEY_DOWN = 1 << 1
KEY_LEFT = 1 << 2
KEY_RIGHT = 1 << 3
KEY_SHIFT = 1 << 4
KEY_SPACE = 1 << 5
KEY_Q = 1 << 6
KEY_ESC = 1 << 7
KEY_ENTER = 1 << 8

KEY_NAMES = {
    KEY_UP: "up",
    KEY_DOWN: "down",
    KEY_LEFT: "left",
    KEY_RIGHT: "right",
    KEY_SHIFT: "shift",
    KEY_SPACE: "space",
    KEY_Q: "q",
    KEY_ESC: "esc",
    KEY_ENTER: "enter",
}
ARROW_MASK = KEY_UP | KEY_DOWN | KEY_LEFT | KEY_RIGHT
ALL_KEYS_MASK = (1 << 9) - 1

# zlib level 9 reproduces the 0x78 0xDA header found in all known legacy records.
ZLIB_LEVEL = 9


class TASFormatError(ValueError):
    pass


def make_frames(deltas, keys) -> np.ndarray:
    """Build a frame array; deltas are converted to float32 (round to nearest), keys to uint32."""
    deltas = np.atleast_1d(np.asarray(deltas, dtype=np.float32))
    keys = np.atleast_1d(np.asarray(keys, dtype=np.uint32))
    deltas, keys = np.broadcast_arrays(deltas, keys)
    out = np.empty(deltas.shape[0], dtype=FRAME_DTYPE)
    out["delta"] = deltas
    out["keys"] = keys
    return out


def as_frames(frames) -> np.ndarray:
    """Return `frames` as a contiguous 1-D FRAME_DTYPE array.

    Only the exact dtype (<f4 delta, <u4 keys, 8 bytes, no padding) is accepted. Anything else,
    including a structured array with the same field names but other widths or byte order, is
    rejected rather than cast: a cast would silently round f8 deltas or truncate u8 keys.
    """
    frames = np.asarray(frames)
    if frames.dtype != FRAME_DTYPE:
        raise TypeError(f"expected structured array with dtype {FRAME_DTYPE}, got {frames.dtype}")
    return np.ascontiguousarray(frames.reshape(-1))


_as_frames = as_frames


def frames_to_bytes(frames) -> bytes:
    return _as_frames(frames).tobytes()


def frames_from_bytes(data: bytes) -> np.ndarray:
    if len(data) % FRAME_SIZE:
        raise TASFormatError(f"payload length {len(data)} is not a multiple of {FRAME_SIZE}")
    return np.frombuffer(data, dtype=FRAME_DTYPE).copy()


def decode(data: bytes) -> np.ndarray:
    """Decode the bytes of a legacy .tas file."""
    if len(data) < 8:
        raise TASFormatError(f"file is {len(data)} bytes, legacy loader requires >= 8")
    (header,) = struct.unpack_from("<I", data, 0)
    stream = data[4:]
    d = zlib.decompressobj()  # default wbits: zlib wrapper required, raw deflate / gzip rejected
    try:
        payload = d.decompress(stream)
        payload += d.flush()
    except zlib.error as e:
        raise TASFormatError(f"not a zlib stream: {e}") from e
    if not d.eof:
        raise TASFormatError("zlib stream is truncated")
    if d.unused_data:
        raise TASFormatError(f"{len(d.unused_data)} trailing bytes after the zlib stream")
    if header != len(payload):
        raise TASFormatError(f"header says {header} bytes, stream decompresses to {len(payload)}")
    return frames_from_bytes(payload)


def encode(frames) -> bytes:
    """Encode frames as a legacy .tas file (u32 decompressed size + zlib stream)."""
    payload = frames_to_bytes(frames)
    if len(payload) > 0xFFFFFFFF:
        raise TASFormatError("payload too large for u32 header")
    return struct.pack("<I", len(payload)) + zlib.compress(payload, ZLIB_LEVEL)


def load(path) -> np.ndarray:
    return decode(Path(path).read_bytes())


def save(path, frames) -> Path:
    path = Path(path)
    path.write_bytes(encode(frames))
    return path


def load_raw(path) -> np.ndarray:
    return frames_from_bytes(Path(path).read_bytes())


def save_raw(path, frames) -> Path:
    path = Path(path)
    path.write_bytes(frames_to_bytes(frames))
    return path


def load_any(path) -> np.ndarray:
    """Load a legacy .tas (by extension .tas) or a raw frame file (anything else)."""
    path = Path(path)
    return load(path) if path.suffix.lower() == ".tas" else load_raw(path)


def delta_bits(frames) -> np.ndarray:
    """deltaTime bit patterns (u32), for exact comparison without float semantics."""
    return _as_frames(frames)["delta"].view("<u4")


def constant_delta(frames):
    """Return the single delta (as np.float32) if every frame has the same delta bit pattern, else None."""
    frames = _as_frames(frames)
    if len(frames) == 0:
        return None
    bits = delta_bits(frames)
    if np.all(bits == bits[0]):
        return frames["delta"][0]
    return None


def first_press(frames) -> dict:
    """Frame index of the first frame each key is down (None if never pressed)."""
    keys = _as_frames(frames)["keys"]
    out = {}
    for bit, name in KEY_NAMES.items():
        idx = np.flatnonzero(keys & bit)
        out[name] = int(idx[0]) if idx.size else None
    return out


def describe(frames) -> dict:
    frames = _as_frames(frames)
    bits = delta_bits(frames)
    uniq, counts = np.unique(bits, return_counts=True)
    deltas = {
        float(np.array(u, dtype="<u4").view("<f4")): int(c) for u, c in zip(uniq, counts)
    }
    cd = constant_delta(frames)
    unknown = int(np.count_nonzero(frames["keys"] & ~np.uint32(ALL_KEYS_MASK)))
    return {
        "frames": int(len(frames)),
        "constant_delta": None if cd is None else float(cd),
        "deltas": deltas,
        "total_ms": float(frames["delta"].astype(np.float64).sum()),
        "first_press": first_press(frames),
        "frames_with_unknown_bits": unknown,
    }


def keys_to_str(bits: int) -> str:
    names = [n for b, n in KEY_NAMES.items() if bits & b]
    return "+".join(names) if names else "-"


def _main(argv=None) -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(prog="python -m ballrain.tas", description="Describe legacy .tas / raw frame files")
    p.add_argument("files", nargs="+")
    a = p.parse_args(argv)
    for f in a.files:
        print(f, json.dumps(describe(load_any(f))))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
