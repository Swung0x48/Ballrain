"""Automatic rollback guard for the learner (scripts/train_m2.py --auto-rollback).

r6 (2026-11-17) collapsed four times at the L1 platform exit: within one 100-version bin the reach-next-checkpoint rate
of the TAS starts just behind the frontier fell from ~0.75 to ~0.3 (deaths at the exit 5% -> 60%), with no warning in
the PPO statistics. The guard watches the episodes of TAS starts in a start-frame range, bins them by the policy version
they started with, and when the last `consecutive` complete bins (each with >= min_n episodes) all have a reach rate
below `low`, names the newest earlier bin with a rate >= `good`: the learner then reloads the checkpoint saved at that
bin's first version (so checkpoints must be kept at every bin start, --ckpt-keep-every = bin).

A bin [b * bin, (b + 1) * bin) is complete once the current version is >= its end + settle (episodes started with a
version finish later). After a rollback only bins that start at or after the rollback version count as bad (the
versions keep increasing), so the guard cannot fire again before `consecutive` new bins are complete; good bins of
before the rollback stay usable as targets.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ReachGuard:
    lo: int = 4100  # start-frame range of the watched TAS starts (inclusive)
    hi: int = 4299
    bin: int = 100  # policy versions per bin
    min_n: int = 20  # episodes for a bin to count
    low: float = 0.5  # rate below which a bin is bad
    good: float = 0.7  # rate at or above which a bin is a rollback target
    consecutive: int = 2  # bad complete bins in a row that trigger
    settle: int = 50  # versions after a bin's end before it counts as complete
    kinds: tuple = ("tas",)  # episode kinds watched (the fake game's smoke test uses "level")
    floor: int = 0  # bins starting before this version are never counted as bad (set at each rollback)
    stats: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0]))  # bin -> [n, successes]

    @classmethod
    def from_config(cls, cfg: dict) -> "ReachGuard":
        known = {"lo", "hi", "bin", "min_n", "low", "good", "consecutive", "settle", "kinds"}
        bad = set(cfg) - known
        if bad:
            raise ValueError(f"unknown auto-rollback keys {sorted(bad)}")
        cfg = dict(cfg)
        if "kinds" in cfg:
            cfg["kinds"] = tuple(cfg["kinds"])
        return cls(**cfg)

    def add(self, kind: str, start_frame: int, version_start: int, success: bool) -> None:
        if kind not in self.kinds or not (self.lo <= int(start_frame) <= self.hi) or version_start is None:
            return
        s = self.stats[int(version_start) // self.bin]
        s[0] += 1
        s[1] += int(bool(success))

    def rate(self, b: int) -> float | None:
        n, k = self.stats.get(b, (0, 0))
        return k / n if n >= self.min_n else None

    def complete_bins(self, version: int) -> list:
        return sorted(b for b in self.stats if (b + 1) * self.bin + self.settle <= version)

    def triggered(self, version: int) -> bool:
        done = [b for b in self.complete_bins(version) if self.rate(b) is not None]
        recent = [b for b in done if b * self.bin >= self.floor]
        if len(recent) < self.consecutive:
            return False
        last = recent[-self.consecutive:]
        return last[-1] - last[0] == self.consecutive - 1 and all(self.rate(b) < self.low for b in last)

    def targets(self, version: int) -> list:
        """When triggered: the first versions of the good bins before the bad ones, newest first (the learner takes the
        newest one whose checkpoint file exists); otherwise []."""
        if not self.triggered(version):
            return []
        done = [b for b in self.complete_bins(version) if self.rate(b) is not None]
        first_bad = [b for b in done if b * self.bin >= self.floor][-self.consecutive]
        return [b * self.bin for b in reversed(done) if b < first_bad and self.rate(b) >= self.good]

    def rolled_back(self, version: int) -> None:
        self.floor = int(version)

    def summary(self, version: int, k: int = 4) -> str:
        bins = self.complete_bins(version)[-k:]
        return ", ".join(f"v{b * self.bin}: {self.stats[b][1]}/{self.stats[b][0]}" for b in bins)
