"""Gate verdict rules (ballrain/gates.py) on synthetic traces / events / frame logs. No game launch."""
import os
import sys
from pathlib import Path

import pytest

from ballrain import gates, tas

ROT = b"-1 -0 -8.74227766e-08 0 1 0 8.74227766e-08 0 -1"


def line(frame, x=0.0, y=0.0, z=0.0, name=b"Ball_Wood"):
    return b"%d %s %s %.9g %.9g %.9g\r\n" % (frame, name, ROT, x, y, z)


def moving(frames, step=0.5):
    return b"".join(line(f, x=step * i) for i, f in enumerate(frames))


def write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def events_text(*evs, frames=None):
    lines = [f"{i} {n}" for i, n in evs]
    if frames is not None:
        lines.append(f"{frames} Summary frames={frames} wall_ms=10")
    return ("\n".join(lines) + "\n").encode()


@pytest.fixture
def case(tmp_path):
    """A G1 record that completes: 200 frames, LevelFinish at frame 150."""
    rec = tas.make_frames([1000 / 132] * 200, [0] * 100 + [tas.KEY_UP] * 100)
    tr = moving(range(105, 150))
    c = {
        "ref": write(tmp_path / "ref.txt", tr),
        "bio": write(tmp_path / "bio.txt", tr),
        "events": write(tmp_path / "events.txt", events_text(
            (0, "PreLoadLevel"), (104, "StartLevel"), (110, "BallNavActive"), (150, "LevelFinish"), frames=150)),
        "framelog": tas.save_raw(tmp_path / "framelog.bin", rec[:150]),
        "record": rec,
        "stored": write(tmp_path / "trace-orig.txt", tr),
        "dir": tmp_path,
    }
    return c


def judge(c, name="SR07_2_05_840_264", stored="default"):
    return gates.judge_g1(name, c["ref"], c["bio"], c["events"], c["framelog"], c["record"],
                          c["stored"] if stored == "default" else stored)


def test_g1_pass(case):
    j = judge(case)
    assert j["pass"] is True and j["verdict"] == "PASS" and j["failed_checks"] == []
    assert j["stored"]["complete"] and j["stored"]["pass"]


def test_g1_bio_differs_fails(case):
    write(case["bio"], moving(range(105, 150), step=0.5000001))
    j = judge(case)
    assert j["pass"] is False and "bio vs ref differs" in j["verdict"]


def test_g1_bio_prefix_fails(case):
    write(case["bio"], moving(range(105, 140)))
    j = judge(case)
    assert j["pass"] is False and "prefix-equal" in j["verdict"]


def test_g1_empty_traces_fail(case):
    write(case["bio"], b"")
    write(case["ref"], b"")
    j = judge(case, stored=None)
    assert j["pass"] is False
    assert "ref sanity" in j["failed_checks"] and "bio sanity" in j["failed_checks"]
    assert j["cmp_bio_vs_ref"]["status"] == "both-empty"


def test_g1_complete_stored_must_match(case):
    write(case["stored"], moving(range(105, 149)) + line(149, x=99.0))  # same terminal frame, other line
    j = judge(case)
    assert j["stored"]["complete"] and j["stored"]["pass"] is False
    assert j["pass"] is False and "ref vs stored differs" in j["verdict"]


def test_g1_incomplete_stored_is_information_only(case):
    write(case["stored"], moving(range(105, 130)))  # stored run stopped early
    j = judge(case)
    assert not j["stored"]["complete"] and j["stored"]["pass"] is None
    assert "different terminal frame" in j["stored"]["relation"] and "prefix-equal" in j["stored"]["relation"]
    assert j["pass"] is True


def test_g1_no_stored_trace(case):
    j = judge(case, stored=None)
    assert j["stored"] is None and j["pass"] is True
    j = judge(case, stored=case["dir"] / "missing.txt")
    assert j["stored"] is None and j["pass"] is True


def test_g1_needs_ball_nav_active(case):
    write(case["events"], events_text((0, "PreLoadLevel"), (150, "LevelFinish"), frames=150))
    j = judge(case)
    assert j["pass"] is False and "no BallNavActive" in j["failed_checks"]


@pytest.mark.parametrize("stop", ["GameOver", "RecordEnd"])
def test_g1_needs_level_finish(case, stop):
    write(case["events"], events_text((110, "BallNavActive"), (150, stop), frames=150))
    j = judge(case)
    assert j["pass"] is False and f"stop {stop} != LevelFinish" in j["failed_checks"]


def test_g1_framelog_must_be_record_prefix(case):
    bad = case["record"][:150].copy()
    bad["keys"][3] = tas.KEY_DOWN
    tas.save_raw(case["framelog"], bad)
    j = judge(case)
    assert j["pass"] is False and "framelog MISMATCH" in j["failed_checks"]


def test_g1_framelog_empty_or_missing_fails(case):
    tas.save_raw(case["framelog"], case["record"][:0])
    write(case["events"], events_text((110, "BallNavActive"), (150, "LevelFinish"), frames=0))
    assert judge(case)["framelog"]["pass"] is False
    case["framelog"].unlink()
    assert judge(case)["framelog"]["relation"] == "MISSING"


def test_g1_framelog_length_must_match_summary(case):
    write(case["events"], events_text((110, "BallNavActive"), (150, "LevelFinish"), frames=151))
    assert judge(case)["framelog"]["pass"] is False


@pytest.mark.parametrize("name", sorted(gates.G1_EXCLUDED))
def test_g1_excluded_is_neither_pass_nor_fail(case, name):
    j = judge(case, name=name)
    assert j["pass"] is None and j["verdict"].startswith("excluded: ")
    assert j["main_conditions_pass"] is True  # conditions still evaluated and kept
    write(case["bio"], b"")
    j = judge(case, name=name)
    assert j["pass"] is None and j["main_conditions_pass"] is False


def test_exclusion_list_is_the_user_decision():
    assert set(gates.G1_EXCLUDED) == {"SR08_1_59_626_264", "SR09_1_29_321_264", "HS13_8484_4_17_833"}


# --------------------------------------------------------------------------------------- G0

def test_sanity_pass_and_displacement():
    s = gates.sanity(moving(range(10), step=0.5), [(5, "BallNavActive")])
    assert s["pass"] and s["max_displacement"] == pytest.approx(4.5)


def test_sanity_uses_translation_not_rotation():
    # rotation changes, position fixed: no displacement
    a = b"1 Ball_Wood 1 0 0 0 1 0 0 0 1 5 5 5\r\n"
    b = b"2 Ball_Wood -1 0 0 0 -1 0 0 0 1 5 5 5\r\n"
    assert gates.max_displacement(a + b) == 0.0
    # the optional " | core ..." diagnostic suffix is ignored
    c = b"3 Ball_Wood 1 0 0 0 1 0 0 0 1 5 7 5 | 0x1p+0 0x0p+0\r\n"
    assert gates.max_displacement(a + c) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "data, events, failed",
    [
        (b"", [(5, "BallNavActive")], {"non_empty", "displacement"}),
        (moving(range(10), step=0.1), [(5, "BallNavActive")], {"displacement"}),  # 0.9 units only
        (moving(range(10), step=0.5), [(5, "CamNavActive")], {"ball_nav_active"}),
        (moving(range(10), step=0.5), [], {"ball_nav_active"}),
    ],
)
def test_sanity_failures(data, events, failed):
    s = gates.sanity(data, events)
    assert not s["pass"]
    assert {k for k, v in s["checks"].items() if not v} == failed


def test_sanity_reference_side_has_no_events(tmp_path):
    s = gates.sanity(write(tmp_path / "t.txt", moving(range(10))))
    assert s["pass"] and s["ball_nav_active"] is None and "ball_nav_active" not in s["checks"]
    assert not gates.sanity(tmp_path / "missing.txt")["pass"]


# --------------------------------------------------------------------------------------- G3

def test_g3_inexact_reload_is_fallback(tmp_path):
    a = write(tmp_path / "a.txt", moving(range(105, 150)))
    stale = write(tmp_path / "b.txt", b"0 Ball_Wood 1 0 0 0 1 0 0 0 1 0 0 0\r\n" + moving(range(105, 150)))
    ok, verdict = gates.g3_verdict(gates.compare(a, stale), gates.compare(a, stale))
    assert not ok and verdict == "fallback: one process per episode"
    ok, verdict = gates.g3_verdict(gates.compare(a, a), gates.compare(a, a))
    assert ok and verdict.startswith("EQUAL")


def test_g3_multi_load_verdict(tmp_path):
    a = write(tmp_path / "a.txt", moving(range(105, 150)))
    stale = write(tmp_path / "b.txt", b"0 Ball_Wood 1 0 0 0 1 0 0 0 1 0 0 0\r\n" + moving(range(105, 150)))
    same, diff = gates.compare(a, a), gates.compare(a, stale)
    assert gates.g3_verdict_loads(same, [same, same], [same, same])[0]
    ok, verdict = gates.g3_verdict_loads(same, [same, diff], [same, diff])  # load 3 drifts
    assert not ok and verdict == gates.G3_FALLBACK
    assert not gates.g3_verdict_loads(diff, [same], [diff])[0]  # load 1 itself differs from fresh
    with pytest.raises(ValueError):
        gates.g3_verdict_loads(same, [], [])


def test_summary_counts():
    cases = {
        "G1/A": {"pass": True, "verdict": "PASS", "export_supplement": {"pass": True}},
        "G1/B": {"pass": None, "verdict": "excluded: x", "export_supplement": {"pass": False, "cmp": {"status": "differs"}}},
        "G1/C": {"pass": False, "verdict": "FAIL: y", "export_supplement": {"error": "boom", "pass": False}},
        "G1/D": {"pass": False, "error": "E", "verdict": "error: E"},
        "G3/X": {"pass": False, "verdict": gates.G3_FALLBACK},
    }
    s = gates.summarize(cases)
    g1 = s["G1"]
    assert (g1["pass"], g1["fail"], g1["excluded"], g1["error"]) == (1, 1, 1, 1)
    assert g1["excluded_cases"] == ["B"]
    assert g1["export_supplement"] == {"pass": 1, "fail": 1, "error": 1,
                                       "failed": ["B: differs", "C: error boom"]}
    assert s["G3"]["verdict"] == gates.G3_FALLBACK
    assert gates.summarize({"G3/X": {"pass": False, "error": "missing", "verdict": "error"}})["G3"]["verdict"] == "error"


# ---------------------------------------------------------------------------- record cleanup

def test_remove_installed(tmp_path):
    records = tmp_path / "TASRecords"
    records.mkdir()
    src = write(tmp_path / "m0_x.tas", b"abc")
    dst = write(records / "m0_x.tas", b"abc")
    assert gates.remove_installed(src, dst) and not dst.exists() and src.exists()
    # a record that lives in TASRecords itself is never removed
    own = write(records / "SR07.tas", b"abc")
    assert not gates.remove_installed(own, own) and own.exists()
    # different content: not ours, left alone
    other = write(records / "m0_y.tas", b"xyz")
    assert not gates.remove_installed(write(tmp_path / "m0_y.tas", b"abc"), other) and other.exists()
    assert not gates.remove_installed(tmp_path / "missing.tas", other) and other.exists()


def test_remove_installed_never_deletes_the_same_file_under_another_spelling(tmp_path):
    records = tmp_path / "TASRecords"
    records.mkdir()
    rec = write(records / "SR07.tas", b"abc")
    # a hard link is one file with two paths: string comparison says "different", samefile does not
    link = tmp_path / "SR07.tas"
    os.link(rec, link)
    assert not gates.remove_installed(link, rec) and rec.exists()
    # a path with redundant components
    alt = tmp_path / "TASRecords" / "." / "SR07.tas"
    assert not gates.remove_installed(alt, rec) and rec.exists()
    # a Windows 8.3 short spelling of the directory, where the volume provides one
    if sys.platform == "win32":
        import ctypes

        buf = ctypes.create_unicode_buffer(32768)
        if ctypes.windll.kernel32.GetShortPathNameW(str(rec), buf, len(buf)) and buf.value != str(rec):
            assert not gates.remove_installed(Path(buf.value), rec) and rec.exists()


@pytest.mark.parametrize("refs,agent,ok", [
    (["a", "a", "a"], "a", True),
    (["a", "a", "b"], "a", True),       # one reference deviation: the majority decides
    (["a", "b", "a"], "a", True),
    (["a", "a", "b"], "b", False),      # agent equals the deviating replay only
    (["a", "b", "c"], "a", False),      # no majority
    (["a", None, None], "a", False),    # missing traces never count
    (["a", "a", None], "a", True),
    (["a", "a"], "a", False),           # 3 replays required
    (["a", "a", "a"], None, False),
])
def test_m1_majority_verdict(refs, agent, ok):
    got, why = gates.majority_verdict(refs, agent)
    assert got is ok and why.startswith("PASS" if ok else "FAIL")


def test_m1_identical_files_and_events(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"x\n")
    b.write_bytes(b"x\n")
    assert gates.identical_files(a, b) and not gates.identical_files(a, tmp_path / "missing")
    b.write_bytes(b"y\n")
    assert not gates.identical_files(a, b)
    e = tmp_path / "events.txt"
    e.write_text("3 StartLevel\n13205 LevelFinish\n13205 Summary frames=13205 wall_ms=1908\n")
    assert gates.events_without_summary(e) == ["3 StartLevel", "13205 LevelFinish"]
