"""Mirror the scalar curves of the training runs into a sanitized TensorBoard logdir for public viewing.

Only scalars are copied (no text summaries, configs, histograms or event-file host names), and runs
are renamed to neutral labels. Usage:
    uv run python tools/tb_public_mirror.py --src runtime/m2-train --dst runtime/tb-public --every 300
"""
from __future__ import annotations

import argparse
import re
import shutil
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

RUN_RE = re.compile(r"^(r\d+)(?:-mac)?$|^(r\d+)-sim2real$")


def run_label(top: str, sub: str) -> str | None:
    m = re.fullmatch(r"r(\d+)(-mac)?", top)
    if m:
        return f"r{m.group(1)}"
    m = re.fullmatch(r"r(\d+)-sim2real", top)
    if m:
        return f"r{m.group(1)}-eval-{sub}" if sub else f"r{m.group(1)}-eval"
    return None  # smoke tests, probes and scratch runs are not published


def collect(src: Path) -> dict[str, list[Path]]:
    runs: dict[str, list[Path]] = {}
    for ev in src.rglob("events.out.tfevents*"):
        rel = ev.relative_to(src).parts
        top = rel[0]
        sub = rel[1] if len(rel) > 2 and rel[1] not in ("tb",) else ""
        sub = re.sub(r"^tb-?", "", sub)
        label = run_label(top, sub)
        if label:
            runs.setdefault(label, []).append(ev)
    return runs


def mirror(src: Path, dst: Path) -> int:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    n = 0
    for label, files in sorted(collect(src).items()):
        w = SummaryWriter(str(tmp / label), filename_suffix=".public")
        for f in sorted(files):
            ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                for e in ea.Scalars(tag):
                    w.add_scalar(tag, e.value, e.step, walltime=e.wall_time)
                    n += 1
        w.close()
    # rename event files so no host name appears, then swap directories
    for ev in tmp.rglob("events.out.tfevents*"):
        ev.rename(ev.with_name(f"events.out.tfevents.{int(ev.stat().st_mtime)}.public"))
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="runtime/m2-train")
    p.add_argument("--dst", default="runtime/tb-public")
    p.add_argument("--every", type=float, default=0, help="seconds between refreshes (0 = once)")
    a = p.parse_args()
    while True:
        t = time.time()
        n = mirror(Path(a.src), Path(a.dst))
        print(time.strftime("%H:%M:%S"), f"mirrored {n} scalar points in {time.time() - t:.1f}s", flush=True)
        if a.every <= 0:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
