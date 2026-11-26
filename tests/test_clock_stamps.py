"""Log stamps with their UTC offset (ballrain.clock) and the dashboard's reading of old and new stamps."""
import datetime as dt
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "dashboard"))

from ballrain import clock  # noqa: E402
import data as D  # noqa: E402


def test_stamp_roundtrip():
    t = dt.datetime(2026, 11, 26, 22, 37, 31, tzinfo=dt.timezone(dt.timedelta(hours=8)))
    s = clock.stamp(t)
    assert s == "2026-11-26 22:37:31+08:00"
    assert clock.parse(s) == t
    assert clock.parse("2026-11-26 22:37:31").tzinfo is None
    assert D.LOG_RE.match(s + " u1 steps").group(1) == s
    assert D.LOG_RE.match("2026-11-26 22:37:31 u1 steps").group(1) == "2026-11-26 22:37:31"


class _Run:
    remote = True


def test_to_pc_offset_stamp_is_exact():
    s = "2026-11-26 22:37:31+08:00"
    want = dt.datetime(2026, 11, 26, 14, 37, 31, tzinfo=dt.timezone.utc).astimezone().replace(tzinfo=None)
    assert D.to_pc(s, _Run(), off_h=999.0) == want  # the measured offset is ignored when the stamp has its own
    assert D.to_pc("2026-11-26 22:37:31", _Run(), off_h=12.0) == dt.datetime(2026, 11, 26, 10, 37, 31)
