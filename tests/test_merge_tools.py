"""tools/merge helpers: teacher maps and hand-over ranges."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "merge"))
from common import parse_ranges, parse_teachers, teacher_key  # noqa: E402


def test_parse_teachers():
    t = parse_teachers(["3=ball.pt", "default=flight.pt"])
    assert t == {"3": "ball.pt", "default": "flight.pt"}
    with pytest.raises(SystemExit):
        parse_teachers(["3ball.pt"])
    with pytest.raises(SystemExit):
        parse_teachers(["3=a.pt", "3=b.pt"])


def test_teacher_key():
    t = {"3": "b", "default": "a"}
    assert teacher_key(3, t) == "3"
    assert teacher_key(1, t) == "default"
    assert teacher_key("flight", {"flight": "x"}) == "flight"
    assert teacher_key(2, {"3": "b"}) is None


def test_parse_ranges():
    assert parse_ranges("4000-5800:2,936-25300") == [(4000, 5800, 2.0), (936, 25300, 1.0)]
    with pytest.raises(SystemExit):
        parse_ranges("5800-4000")


def test_pick_teacher_by_route_s():
    from common import pick_teacher
    t = {"s:2172-9999": "gap", "3": "ball", "default": "main"}
    assert pick_teacher(4, 2200.0, t) == "s:2172-9999"
    assert pick_teacher(4, 2100.0, t) == "default"
    assert pick_teacher(3, 1500.0, t) == "3"
    assert pick_teacher(4, None, t) == "default"
    assert pick_teacher(4, float("nan"), t) == "default"
