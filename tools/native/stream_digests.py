"""Engine-state digests (ballrain.pool.engine_chain) of a recorded stream (tools/native/golden.py format), one per
FEAT record: every OBS / level-message record since the previous FEAT. The native client's nc_golden --digests
computes the same from the same stream; the two outputs must be identical (the chain's byte layout in both languages).

    python tools/native/stream_digests.py <stream.bin> > py.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ballrain.pool.engine_chain import EngineDigest  # noqa: E402
from stream_diff import read_stream  # noqa: E402


def main(argv=None) -> int:
    path = (argv or sys.argv[1:])[0]
    d = EngineDigest()
    for k, p in read_stream(path):
        if k == 1:  # HELLO: a new process
            d.reset()
        elif k == 2:
            d.obs(p[24:])
        elif k == 4 and p[:1] == b"\1":
            d.level(b"F", p[1:])
        elif k == 5 and p[:1] == b"\1":
            # the consumed part of the tail: complete messages (the recording stores the whole unread tail)
            d.level(b"I", p[1:])
        elif k == 7:
            print(d.take().hex())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
