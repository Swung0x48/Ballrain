"""The engine-state hash chain (ballrain.pool.engine_chain, protocol chain v3): the canonical records leave out what
differs between machines / runs for the same inputs (SCOR, SREN, WTCH, IVPO, GRPS, string cells) and keep everything
else; the byte layout is pinned (the native client implements the same: native/core/src/engine_chain.cpp,
cross-checked on recorded streams by tools/native/stream_digests.py)."""
from __future__ import annotations

import hashlib
import struct

from ballrain import protocol as BP
from ballrain.pool import engine_chain as E


def _obs_arrs(cells):
    """OBS1 ARRS payload from (array, row, col, type, data) cells."""
    out = struct.pack("<H", len(cells))
    for a, r, c, t, d in cells:
        out += struct.pack("<IHHBB", a, r, c, t, len(d)) + d
    return out


def _obs(scor=b"\x01" * 60, ball=b"\x02" * 230, frames=7, wtch=b"\x06" * 19, path=b"C:/Ballance/", points=5):
    arrs = _obs_arrs([(9, 0, 3, BP.ARR_STRING, path), (9, 0, 0, BP.ARR_INT, struct.pack("<i", points))])
    return BP.build_message(BP.OBS_MAGIC, frames, [("TIME", b"\x03" * 69), ("BALL", ball), ("SCOR", scor),
                                                    ("SREN", b"\x05" * 88), ("WTCH", wtch), ("ARRS", arrs)])


def _lvl_arrs(path=b"C:/Ballance/", lives=3):
    """LVL1 ARRS payload: one array (ckid 9, name 0), 1 row x 2 columns (STRING, INT)."""
    p = struct.pack("<I", 1) + struct.pack("<IIHH", 9, 0, 1, 2)
    p += struct.pack("<IBI", 1, BP.ARR_STRING, 0xFFFFFFFF) + struct.pack("<IBI", 2, BP.ARR_INT, 0xFFFFFFFF)
    p += bytes([len(path)]) + path + bytes([4]) + struct.pack("<i", lives)
    return p


def _lvl(ivpo=b"\x00" * 4, frames=3, grps=struct.pack("<I", 0), path=b"C:/Ballance/", lives=3):
    strs = struct.pack("<I", 3) + b"".join(struct.pack("<H", len(s)) + s for s in (b"GameSettings", b"Path", b"N"))
    return BP.build_message(BP.LVL_MAGIC, frames, [("STRS", strs), ("GRPS", grps), ("ARRS", _lvl_arrs(path, lives)),
                                                    ("IVPO", ivpo), ("DEPT", struct.pack("<I", 0))])


def _digest(*recs):
    d = E.EngineDigest()
    for kind, data in recs:
        if kind == "O":
            d.obs(data)
        else:
            d.level(kind.encode(), data)
    return d.take()


def test_masked_parts_do_not_count_and_the_rest_does():
    base = _digest(("O", _obs()), ("F", _lvl()))
    same = [
        _digest(("O", _obs(scor=b"\x09" * 60)), ("F", _lvl())),  # SCOR / SREN: the Energy pointer
        _digest(("O", _obs(wtch=b"\x07" * 19)), ("F", _lvl())),  # WTCH: 1 ulp between C runtimes
        _digest(("O", _obs(path=b"/data/local/tmp/game/")), ("F", _lvl())),  # string cells (the game path)
        _digest(("O", _obs()), ("F", _lvl(ivpo=b"\x07" * 4))),  # IVPO: deduplicated by address
        _digest(("O", _obs()), ("F", _lvl(grps=struct.pack("<I", 1) + b"\x05" * 12))),  # GRPS: the install's Sounds/
        _digest(("O", _obs()), ("F", _lvl(path=b"D:/Games/Ballance/"))),  # the level message's path cell
    ]
    assert all(x == base for x in same)
    differ = [
        _digest(("O", _obs(ball=b"\x08" * 230)), ("F", _lvl())),
        _digest(("O", _obs(frames=8)), ("F", _lvl())),
        _digest(("O", _obs(points=6)), ("F", _lvl())),  # a numeric cell delta counts
        _digest(("O", _obs()), ("F", _lvl(lives=4))),  # a numeric level-message cell counts
        _digest(("F", _lvl()), ("O", _obs())),  # order matters
        _digest(("O", _obs()), ("I", _lvl())),  # full vs increment
    ]
    assert all(x != base for x in differ)
    # an increment read: complete messages only (a trailing partial one is not consumed)
    two = _lvl() + _lvl(frames=4)
    assert _digest(("I", two + _lvl()[:30])) == _digest(("I", two))
    # the full level message is one message
    assert _digest(("F", _lvl() + _lvl(frames=9))) == _digest(("F", _lvl()))


def test_byte_layout():
    o = _obs()
    m = BP.split_message(o, 0, BP.OBS_MAGIC)
    rec = b"O" + struct.pack("<4sHHII", b"OBS1", m.version, m.flags, m.frames_done, 3)
    for s in m.sections:
        if s.tag in ("TIME", "BALL"):
            rec += struct.pack("<IHHI", BP.fourcc(s.tag), s.version, s.flags, len(s.payload)) + s.payload
        elif s.tag == "ARRS":  # the string cell keeps its header with length 0, the int cell is unchanged
            p = _obs_arrs([(9, 0, 3, BP.ARR_STRING, b""), (9, 0, 0, BP.ARR_INT, struct.pack("<i", 5))])
            rec += struct.pack("<IHHI", BP.fourcc("ARRS"), s.version, s.flags, len(p)) + p
    want = hashlib.sha256(b"ballrain-pool/engine-row/v3\0" + rec).digest()
    assert _digest(("O", o)) == want
    seed = hashlib.sha256(b"ballrain-pool/engine-chain/v3\0task").digest()
    assert E.chain_seed("task") == seed and E.chain_next(seed, want) == hashlib.sha256(seed + want).digest()
