import pytest

from ballrain import trace

M = b" -1 -0 -8.74227766e-08 0 1 0 8.74227766e-08 0 -1 -27.1306877 14.8949604 -0.304026186"


def line(frame, name=b"Ball_Wood", tail=M, eol=b"\r\n"):
    return str(frame).encode() + b" " + name + tail + eol


def tr(frames, **kw):
    return b"".join(line(f, **kw) for f in frames)


def test_identical():
    a = tr(range(105, 120))
    d = trace.compare(a, a)
    assert d.status == "identical" and d.ok
    assert d.same_end and d.lines_a == d.lines_b == 15
    assert (d.first_frame_a, d.last_frame_a) == (105, 119)


def test_differs_reports_first_line_and_frame():
    a = tr(range(105, 120))
    b = tr(range(105, 110)) + line(110, tail=M.replace(b"14.8949604", b"14.8949605")) + tr(range(111, 120))
    d = trace.compare(a, b)
    assert d.status == "differs" and not d.ok
    assert d.first_diff_index == 5 and d.first_diff_frame_a == d.first_diff_frame_b == 110
    assert d.common_prefix == 5
    assert d.line_a == line(110) and d.line_b != d.line_a
    assert not d.eol_only
    assert "frame a=110" in d.summary()


def test_prefix_equal_is_not_a_pass():
    a = tr(range(105, 120))
    b = tr(range(105, 112))
    for x, y in ((a, b), (b, a)):
        d = trace.compare(x, y)
        assert d.status == "prefix-equal" and not d.ok
        assert abs(d.lines_a - d.lines_b) == 8
        assert d.first_diff_index == 7
        assert not d.same_end
        assert "lengths differ by 8" in d.summary()
    d = trace.compare(a, b, require_same_end=False)
    assert d.status == "prefix-equal" and d.ok


def test_both_empty_is_not_a_pass():
    d = trace.compare(b"", b"")
    assert d.status == "both-empty" and not d.ok
    assert not trace.compare(b"", b"", require_same_end=False).ok


def test_one_empty():
    d = trace.compare(b"", tr([105]))
    assert d.status == "one-empty" and not d.ok and d.lines_a == 0 and d.lines_b == 1
    assert d.first_diff_frame_b == 105 and d.common_prefix == 0
    assert "never a pass" in d.summary()


@pytest.mark.parametrize("require_same_end", [True, False])
def test_one_empty_never_passes_either_order(require_same_end):
    full = tr(range(105, 120))
    for a, b in ((b"", full), (full, b"")):
        d = trace.compare(a, b, require_same_end=require_same_end)
        assert d.status == "one-empty" and not d.ok
        assert d.common_prefix == 0


def test_prefix_equal_needs_a_common_line():
    # A non-empty common prefix passes only with require_same_end=False ...
    d = trace.compare(tr([105, 106]), tr([105]), require_same_end=False)
    assert d.status == "prefix-equal" and d.common_prefix == 1 and d.ok
    # ... and a first-line mismatch is never a prefix.
    d = trace.compare(tr([105, 106]), tr([106]), require_same_end=False)
    assert d.status == "differs" and not d.ok


def test_cli_empty_file_fails_even_with_allow_length_diff(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_bytes(tr(range(10)))
    b.write_bytes(b"")
    for x, y in ((a, b), (b, a)):
        assert trace._main([str(x), str(y), "--allow-length-diff"]) == 1
        assert trace._main([str(x), str(y)]) == 1
    a.write_bytes(b"")
    assert trace._main([str(a), str(b), "--allow-length-diff"]) == 1


def test_single_line_short_traces():
    assert trace.compare(tr([105]), tr([105])).status == "identical"
    assert trace.compare(tr([105]), tr([106])).status == "differs"


def test_eol_difference_is_a_difference():
    d = trace.compare(tr(range(3), eol=b"\r\n"), tr(range(3), eol=b"\n"))
    assert d.status == "differs" and not d.ok and d.eol_only and d.first_diff_index == 0


def test_unterminated_last_line():
    full = tr(range(5))
    cut = full[:-10]
    d = trace.compare(full, cut)
    assert d.status == "differs" and d.first_diff_index == 4 and d.unterminated_b
    d = trace.compare(full[:-2], full[:-2])
    assert d.status == "identical" and d.unterminated_a


def test_same_frames_different_ends():
    a = tr([1, 2, 3])
    b = tr([1, 2, 3]) + tr([3])
    d = trace.compare(a, b)
    # same terminal frame and final line, but a duplicated frame: still a failure
    assert d.status == "prefix-equal" and d.same_end and not d.ok


def test_ball_name_difference():
    d = trace.compare(tr([7]), tr([7], name=b"Ball_Paper"))
    assert d.status == "differs"


def test_cli(tmp_path, capsys):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_bytes(tr(range(10)))
    b.write_bytes(tr(range(10)))
    assert trace._main([str(a), str(b)]) == 0
    b.write_bytes(tr(range(9)))
    assert trace._main([str(a), str(b)]) == 1
    assert "prefix-equal" in capsys.readouterr().out
    assert trace._main([str(a), str(b), "--allow-length-diff"]) == 0
