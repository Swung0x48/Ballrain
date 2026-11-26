"""Log stamps as Unix seconds (ballrain.clock) and the dashboard's reading of new and old stamps."""
import datetime as dt
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "dashboard"))

from ballrain import clock  # noqa: E402
import data as D  # noqa: E402


def test_stamp_is_epoch():
    s = clock.stamp(1795705562.1234)
    assert s == "1795705562.123"
    assert clock.is_epoch(s)
    assert clock.to_epoch(s) == 1795705562.123
    assert clock.to_epoch("2026-11-26 22:37:31+08:00") == dt.datetime(2026, 11, 26, 14, 37, 31, tzinfo=dt.timezone.utc).timestamp()
    assert clock.to_epoch("2026-11-26 22:37:31") is None
    assert clock.to_epoch("2026-11-26 22:37:31", 8) == clock.to_epoch("2026-11-26 22:37:31+08:00")


def test_log_re_reads_all_forms():
    for ts in ("1795705562.123", "2026-11-26 22:37:31+08:00", "2026-11-26 22:37:31"):
        assert D.LOG_RE.match(ts + " u1 steps 2").groups() == (ts, "u1 steps 2")


class _Run:
    remote = True


def test_to_pc():
    want = dt.datetime.fromtimestamp(1795705562.0)
    assert D.to_pc("1795705562.000", _Run(), off_h=999.0) == want  # an instant needs no offset
    assert D.to_pc("2026-11-26 22:37:31", _Run(), off_h=12.0) == dt.datetime(2026, 11, 26, 10, 37, 31)
