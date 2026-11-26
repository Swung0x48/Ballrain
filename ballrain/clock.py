"""Log timestamps as Unix seconds ("1795705562.123"): an instant, independent of the writer's time zone. Only a
display converts it to local time (the dashboard page does that in the viewer's browser).

Older logs begin with the writer's local time ("2026-11-26 22:37:31", no zone) or, briefly on 2026-11-26, local time
with its UTC offset ("2026-11-26 22:37:31+08:00"); to_epoch() reads all three, the first only with the writer's clock
offset supplied by the reader."""
from __future__ import annotations

import datetime as dt
import re
import time

TS_PATTERN = r"\d{9,11}(?:\.\d+)?|\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:[+-]\d\d:\d\d)?"
_EPOCH = re.compile(r"\d{9,11}(?:\.\d+)?")


def stamp(t: float | None = None) -> str:
    """Unix seconds with milliseconds: "1795705562.123"."""
    return f"{time.time() if t is None else t:.3f}"


def is_epoch(ts: str) -> bool:
    return bool(_EPOCH.fullmatch(ts))


def to_epoch(ts: str, naive_utc_offset_h: float | None = None) -> float | None:
    """Unix seconds of a log timestamp: an epoch stamp as is; a stamp with its UTC offset exactly; an old offset-free
    local stamp with the writer's UTC offset in hours (None = unknown: returns None)."""
    if is_epoch(ts):
        return float(ts)
    if re.fullmatch(r".*[+-]\d\d:\d\d", ts):
        return dt.datetime.fromisoformat(ts).timestamp()
    if naive_utc_offset_h is None:
        return None
    t = dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    return t.replace(tzinfo=dt.timezone(dt.timedelta(hours=naive_utc_offset_h))).timestamp()
