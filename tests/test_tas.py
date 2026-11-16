import struct
import zlib

import numpy as np
import pytest

from ballrain import tas


def legacy_loader_accepts(data: bytes) -> np.ndarray:
    """Model of the real TASSupport legacy loader (TASRecord.cpp, x86: sizeof(size_t) == 4)."""
    assert len(data) >= 8
    (header,) = struct.unpack_from("<I", data, 0)
    payload = zlib.decompress(data[4:])  # uncompress(): zlib wrapper required
    assert header == len(payload)
    n = len(payload) // 8
    return np.frombuffer(payload[: n * 8], dtype=tas.FRAME_DTYPE)


def test_round_trip_all_records(tas_paths):
    for p in tas_paths:
        raw = p.read_bytes()
        a = tas.decode(raw)
        assert len(a) > 0, p
        enc = tas.encode(a)
        b = tas.decode(enc)
        assert b.tobytes() == a.tobytes(), p
        assert legacy_loader_accepts(enc).tobytes() == a.tobytes(), p
        (h,) = struct.unpack_from("<I", raw, 0)
        assert h == 8 * len(a), p
        assert raw[4:6] == enc[4:6] == b"\x78\xda", p


def test_known_delta_132_is_float32_of_1000_over_132(tas_paths):
    recs = [p for p in tas_paths if p.stem.endswith("_132")]
    if not recs:
        pytest.skip("no *_132 record")
    want = np.float32(1000 / 132).view("<u4")
    for p in recs:
        f = tas.load(p)
        assert tas.constant_delta(f) is not None, p
        assert tas.delta_bits(f)[0] == want, p


def test_save_load_files(tmp_path):
    f = tas.make_frames([1000 / 132] * 3, [0, tas.KEY_UP, tas.KEY_UP | tas.KEY_SHIFT])
    tas.save(tmp_path / "x.tas", f)
    tas.save_raw(tmp_path / "x.bin", f)
    assert tas.load(tmp_path / "x.tas").tobytes() == f.tobytes()
    assert (tmp_path / "x.bin").read_bytes() == f.tobytes()
    assert tas.load_raw(tmp_path / "x.bin").tobytes() == f.tobytes()
    assert tas.load_any(tmp_path / "x.bin").tobytes() == tas.load_any(tmp_path / "x.tas").tobytes()
    raw = (tmp_path / "x.tas").read_bytes()
    assert struct.unpack_from("<I", raw)[0] == 24
    legacy_loader_accepts(raw)


def test_frame_layout():
    f = tas.make_frames([1.5], [0x1FF])
    assert f.tobytes() == struct.pack("<fI", 1.5, 0x1FF)
    assert tas.FRAME_SIZE == 8


def test_empty_record_is_8_bytes_or_more():
    enc = tas.encode(tas.make_frames([], []))
    assert len(enc) >= 8
    assert len(tas.decode(enc)) == 0


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x00" * 7,
        struct.pack("<I", 16) + zlib.compress(b"\x00" * 8),  # header mismatch
        struct.pack("<I", 8) + zlib.compress(b"\x00" * 8)[2:],  # raw deflate, no zlib wrapper
        struct.pack("<I", 8) + zlib.compress(b"\x00" * 8)[:-3],  # truncated
        struct.pack("<I", 8) + zlib.compress(b"\x00" * 8) + b"junk",  # trailing bytes
        struct.pack("<I", 7) + zlib.compress(b"\x00" * 7),  # not a whole frame
    ],
)
def test_decode_rejects(data):
    with pytest.raises(tas.TASFormatError):
        tas.decode(data)


def test_raw_rejects_partial_frame():
    with pytest.raises(tas.TASFormatError):
        tas.frames_from_bytes(b"\x00" * 9)


def test_describe():
    keys = [0, 0, tas.KEY_Q, tas.KEY_SHIFT | tas.KEY_LEFT, tas.KEY_UP]
    f = tas.make_frames([1000 / 132] * 5, keys)
    d = tas.describe(f)
    assert d["frames"] == 5
    assert d["constant_delta"] == float(np.float32(1000 / 132))
    fp = d["first_press"]
    assert (fp["q"], fp["shift"], fp["left"], fp["up"], fp["down"]) == (2, 3, 3, 4, None)
    g = tas.make_frames([1.0, 2.0], [0, 0])
    assert tas.constant_delta(g) is None
    assert tas.describe(g)["deltas"] == {1.0: 1, 2.0: 1}


@pytest.mark.parametrize(
    "dtype",
    [
        [("delta", "<f8"), ("keys", "<u8")],  # same names, wider fields
        [("delta", "<f4"), ("keys", "<u8")],
        [("delta", "<f8"), ("keys", "<u4")],
        [("delta", "<f2"), ("keys", "<u2")],  # same names, narrower fields
        [("delta", ">f4"), ("keys", ">u4")],  # same widths, wrong byte order
        [("delta", "<f4"), ("keys", "<i4")],  # same widths, wrong kind
        np.dtype({"names": ["delta", "keys"], "formats": ["<f4", "<u4"], "offsets": [0, 4], "itemsize": 12}),
    ],
)
def test_wrong_width_dtype_is_rejected_not_cast(dtype):
    f = np.zeros(3, dtype=dtype)
    f["delta"] = 1000 / 132
    f["keys"] = tas.KEY_UP
    for fn in (tas.frames_to_bytes, tas.encode, tas.describe, tas.as_frames):
        with pytest.raises(TypeError):
            fn(f)


def test_plain_arrays_are_rejected():
    for bad in (np.zeros(4, dtype="<u8"), np.zeros((2, 2), dtype="<f4"), [1.0, 2.0]):
        with pytest.raises(TypeError):
            tas.frames_to_bytes(bad)


def test_exact_dtype_passes_through():
    f = tas.make_frames([1.5, 2.5], [1, 2])
    assert tas.as_frames(f).tobytes() == f.tobytes()
    assert tas.as_frames(f.reshape(2, 1)).shape == (2,)


def test_constant_delta_by_bits():
    f = tas.make_frames([0.0, -0.0], [0, 0])
    assert tas.constant_delta(f) is None  # 0.0 == -0.0 as floats, but the bit patterns differ
