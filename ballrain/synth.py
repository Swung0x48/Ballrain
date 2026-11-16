"""Synthesis of off-route test inputs.

Random holds: an arrow combination is drawn uniformly from ARROW_COMBOS (no key, the four single
arrows and the four diagonals; opposing arrows are never combined) and held for a uniform random
number of frames in [hold_min, hold_max]; the final hold is truncated so the output length is exact.
Only arrow bits are ever set by the random part. Randomness: numpy PCG64 seeded with `seed`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import tas

ARROW_COMBOS = (
    0,
    tas.KEY_UP,
    tas.KEY_DOWN,
    tas.KEY_LEFT,
    tas.KEY_RIGHT,
    tas.KEY_UP | tas.KEY_LEFT,
    tas.KEY_UP | tas.KEY_RIGHT,
    tas.KEY_DOWN | tas.KEY_LEFT,
    tas.KEY_DOWN | tas.KEY_RIGHT,
)

DEFAULT_DELTA = np.float32(1000 / 132)
DEFAULT_IDLE = 140


def random_hold_keys(n: int, seed: int, hold_min: int = 5, hold_max: int = 60) -> np.ndarray:
    if n < 0 or not 1 <= hold_min <= hold_max:
        raise ValueError("need n >= 0 and 1 <= hold_min <= hold_max")
    rng = np.random.Generator(np.random.PCG64(seed))
    keys = np.empty(n, dtype=np.uint32)
    i = 0
    while i < n:
        combo = ARROW_COMBOS[int(rng.integers(len(ARROW_COMBOS)))]
        length = int(rng.integers(hold_min, hold_max + 1))
        keys[i:i + length] = combo
        i += length
    return keys


def off_route(frames, k: int, n: int, seed: int, hold_min: int = 5, hold_max: int = 60) -> np.ndarray:
    """First k frames of `frames`, then n frames of seeded random arrow holds at the prefix's delta."""
    frames = tas.as_frames(frames)
    if not 1 <= k <= len(frames):
        raise ValueError(f"k={k} outside 1..{len(frames)}")
    prefix = frames[:k]
    delta = tas.constant_delta(prefix)
    if delta is None:
        raise ValueError("prefix delta is not constant")
    tail = tas.make_frames(np.full(n, delta, dtype=np.float32), random_hold_keys(n, seed, hold_min, hold_max))
    return np.concatenate([prefix, tail])


def idle_then_random(n: int, seed: int, idle: int = DEFAULT_IDLE, delta=DEFAULT_DELTA,
                     hold_min: int = 5, hold_max: int = 60, q_frames=()) -> np.ndarray:
    """`idle` frames with no key, then n frames of seeded random arrow holds; constant f32 delta.

    q_frames: idle frames at which Q alone is pressed for one frame (level 1: Q ends the tutorial,
    as SR01_1_14_457_264 does at frame 140); each must lie inside the idle part.
    """
    if idle < 0:
        raise ValueError("idle must be >= 0")
    idle_keys = np.zeros(idle, dtype=np.uint32)
    for q in q_frames:
        if not 0 <= q < idle:
            raise ValueError(f"q frame {q} outside the idle part 0..{idle - 1}")
        idle_keys[q] = tas.KEY_Q
    keys = np.concatenate([idle_keys, random_hold_keys(n, seed, hold_min, hold_max)])
    return tas.make_frames(np.full(idle + n, np.float32(delta), dtype=np.float32), keys)


def write_outputs(frames, stem) -> tuple[Path, Path]:
    """Write <stem>.tas (legacy) and <stem>.bin (raw frames)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    t = tas.save(stem.with_name(stem.name + ".tas"), frames)
    r = tas.save_raw(stem.with_name(stem.name + ".bin"), frames)
    return t, r


def _main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m ballrain.synth")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("off-route", help="prefix of a .tas + random holds")
    a.add_argument("tas")
    a.add_argument("-k", type=int, required=True)
    b = sub.add_parser("idle-random", help="idle frames + random holds from level start")
    b.add_argument("--idle", type=int, default=DEFAULT_IDLE)
    for s in (a, b):
        s.add_argument("-n", type=int, required=True)
        s.add_argument("--seed", type=int, required=True)
        s.add_argument("-o", "--out", required=True, help="output stem; writes <out>.tas and <out>.bin")
    args = p.parse_args(argv)
    if args.cmd == "off-route":
        frames = off_route(tas.load_any(args.tas), args.k, args.n, args.seed)
    else:
        frames = idle_then_random(args.n, args.seed, args.idle)
    for f in write_outputs(frames, args.out):
        print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
