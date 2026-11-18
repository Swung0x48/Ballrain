"""The pool's hash chain over the raw engine state (docs/pool-prototype.md 3.2, chain v2).

A task's observation rows are features computed from what the headless engine sent: the OBS1 messages of the socket
and the LVL1 level messages it wrote (read at AgentSession's read points). The chain commits to those engine bytes,
not to the float32 features, so two machines that compute features with slightly different float rounding (another
CPU, compiler, BLAS: the native client vs the Python verifier) still agree on the chain; the verifier recomputes the
features from its own replay and compares the submitted values with tolerances (rollout.replay_task).

Row digest: d_i = sha256 over the canonical records of every engine message received after row i-1 was formed (row
0: from the process start, i.e. the launch, the hand-over and the intro) up to the moment row i is formed. Chain:
h(-1) = sha256("ballrain-pool/engine-chain/v2\\0" + task_id), h(i) = sha256(h(i-1) || d_i).

Canonical records (each starts with a kind byte; all integers little-endian):
  b"O"  one OBS1 message: magic, u16 version, u16 flags, u32 frames_done, u32 n, then n sections
        (u32 tag, u16 version, u16 flags, u32 length, payload), SCOR and SREN left out
  b"F"  the full level message as read: u32 n_messages, each message as above (LVL1), IVPO left out
  b"I"  the level increments consumed by one read of level.msg.inc (complete messages only), same layout
Left out because the engine writes addresses there (two runs of the same input differ, docs/headless.md 5 and
docs/native-client.md 4.1): SCOR / SREN (the Energy array's 4th column is a parameter pointer), IVPO (records are
deduplicated by IVP object address). The features do not read them (hs_block off; the HS block's follower count would
come from SCOR: an HS policy would need its own masking first).

The native client (native/core/src/engine_chain.cpp) implements the same records; tests/test_pool_engine_chain.py pins
the byte layout.
"""
from __future__ import annotations

import hashlib
import struct

from ballrain import protocol as BP

CHAIN_VERSION = 2
SEED_TAG = b"ballrain-pool/engine-chain/v2\0"
ROW_TAG = b"ballrain-pool/engine-row/v2\0"
SKIP_OBS = frozenset({"SCOR", "SREN"})
SKIP_LVL = frozenset({"IVPO"})
_SEC = struct.Struct("<IHHI")
_MSG = struct.Struct("<4sHHII")


def chain_seed(task_id: str) -> bytes:
    return hashlib.sha256(SEED_TAG + task_id.encode("utf-8")).digest()


def chain_next(prev: bytes, row_digest: bytes) -> bytes:
    return hashlib.sha256(bytes(prev) + bytes(row_digest)).digest()


def canonical_message(buf: bytes, offset: int, magic: int, skip) -> tuple[bytes, int]:
    """(canonical bytes, byte_len) of one OBS1 / LVL1 message at buf[offset:]."""
    m = BP.split_message(buf, offset, magic)
    kept = [s for s in m.sections if s.tag not in skip]
    out = [_MSG.pack(struct.pack("<I", m.magic), m.version, m.flags, m.frames_done, len(kept))]
    for s in kept:
        out.append(_SEC.pack(BP.fourcc(s.tag), s.version, s.flags, len(s.payload)))
        out.append(s.payload)
    return b"".join(out), m.byte_len


class EngineDigest:
    """Accumulates the canonical engine records between two rows (AgentSession feeds it, BallanceEnv owns it)."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._h = hashlib.sha256(ROW_TAG)
        self.records = 0

    def obs(self, msg: bytes) -> None:
        """One OBS1 message (the socket payload after its 24-byte header)."""
        c, _ = canonical_message(msg, 0, BP.OBS_MAGIC, SKIP_OBS)
        self._h.update(b"O" + c)
        self.records += 1

    def level(self, kind: bytes, data: bytes) -> None:
        """kind b"F": level.msg as read (one message); b"I": the complete increments one .inc read consumed."""
        parts, pos = [], 0
        while len(data) - pos >= BP.MSG_HDR.size:
            (bl,) = struct.unpack_from("<I", data, pos + 8)
            if pos + bl > len(data):
                break
            c, n = canonical_message(data, pos, BP.LVL_MAGIC, SKIP_LVL)
            parts.append(c)
            pos += n
            if kind == b"F":
                break
        self._h.update(kind + struct.pack("<I", len(parts)) + b"".join(parts))
        self.records += 1

    def take(self) -> bytes:
        """The digest of the records since the last take (or reset); starts the next row."""
        d = self._h.digest()
        self.reset()
        return d
