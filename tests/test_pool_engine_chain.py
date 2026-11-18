"""The engine-state hash chain (ballrain.pool.engine_chain, protocol chain v2): the canonical records leave out the
address-carrying sections and keep everything else; the byte layout is pinned (the native client implements the
same: native/core/src/engine_chain.cpp, cross-checked on recorded streams by tools/native/stream_digests.py)."""
from __future__ import annotations

import hashlib
import struct

from ballrain import protocol as BP
from ballrain.pool import engine_chain as E


def _obs(scor=b"\x01" * 60, ball=b"\x02" * 230, frames=7):
    return BP.build_message(BP.OBS_MAGIC, frames, [("TIME", b"\x03" * 69), ("BALL", ball), ("SCOR", scor),
                                                    ("SREN", b"\x05" * 88)])


def _lvl(ivpo=b"\x00" * 4, frames=3):
    return BP.build_message(BP.LVL_MAGIC, frames, [("STRS", struct.pack("<I", 0)), ("IVPO", ivpo),
                                                    ("DEPT", struct.pack("<I", 0))])


def _digest(*recs):
    d = E.EngineDigest()
    for kind, data in recs:
        if kind == "O":
            d.obs(data)
        else:
            d.level(kind.encode(), data)
    return d.take()


def test_masked_sections_do_not_count_and_the_rest_does():
    base = _digest(("O", _obs()), ("F", _lvl()))
    assert _digest(("O", _obs(scor=b"\x09" * 60)), ("F", _lvl())) == base  # SCOR / SREN: the Energy pointer
    assert _digest(("O", _obs()), ("F", _lvl(ivpo=b"\x07" * 4))) == base  # IVPO: deduplicated by address
    assert _digest(("O", _obs(ball=b"\x08" * 230)), ("F", _lvl())) != base
    assert _digest(("O", _obs(frames=8)), ("F", _lvl())) != base
    assert _digest(("F", _lvl()), ("O", _obs())) != base  # order matters
    assert _digest(("O", _obs()), ("I", _lvl())) != base  # full vs increment
    # an increment read: complete messages only (a trailing partial one is not consumed)
    two = _lvl() + _lvl(frames=4)
    assert _digest(("I", two + _lvl()[:30])) == _digest(("I", two))
    # the full level message is one message
    assert _digest(("F", _lvl() + _lvl(frames=9))) == _digest(("F", _lvl()))


def test_byte_layout():
    o = _obs()
    m = BP.split_message(o, 0, BP.OBS_MAGIC)
    rec = b"O" + struct.pack("<4sHHII", b"OBS1", m.version, m.flags, m.frames_done, 2)
    for s in m.sections:
        if s.tag in ("TIME", "BALL"):
            rec += struct.pack("<IHHI", BP.fourcc(s.tag), s.version, s.flags, len(s.payload)) + s.payload
    want = hashlib.sha256(b"ballrain-pool/engine-row/v2\0" + rec).digest()
    assert _digest(("O", o)) == want
    seed = hashlib.sha256(b"ballrain-pool/engine-chain/v2\0task").digest()
    assert E.chain_seed("task") == seed and E.chain_next(seed, want) == hashlib.sha256(seed + want).digest()
