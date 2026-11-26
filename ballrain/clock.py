"""Log timestamps that carry their own UTC offset ("2026-11-26 22:37:31+08:00"), so a reader on another machine
gets the exact instant without guessing the writer's time zone. Older logs have no offset ("2026-11-26 22:37:31");
parse() returns a naive datetime for those and the reader has to supply the writer's clock offset itself."""
from __future__ import annotations

import datetime as dt
import re

TS_PATTERN = r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:[+-]\d\d:\d\d)?"


def stamp(now: dt.datetime | None = None) -> str:
    """The current local time with its UTC offset, to the second: "YYYY-MM-DD HH:MM:SS+hh:mm"."""
    return (now or dt.datetime.now().astimezone()).isoformat(sep=" ", timespec="seconds")


def parse(ts: str) -> dt.datetime:
    """A stamp() string -> aware datetime; an old offset-free string -> naive datetime (writer's local time)."""
    if re.fullmatch(r".*[+-]\d\d:\d\d", ts):
        return dt.datetime.fromisoformat(ts)
    return dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
