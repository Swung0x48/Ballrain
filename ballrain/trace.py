"""Byte-exact comparison of per-frame ball traces.

A trace line is "<frameIndex> <ballName> <12 x %.9g>" followed by the writer's line terminator
(the reference TASSupport writes in text mode, i.e. CRLF). Lines are compared as raw bytes
INCLUDING the terminator; nothing is normalised. An EOL-only difference is reported as a hint
but is still a difference.

Statuses:
  identical      same number of lines, every line byte-identical, both non-empty
  differs        some line within the common prefix differs (first one reported)
  prefix-equal   the shorter trace is a byte-identical prefix of the longer one; lengths differ;
                 both traces are non-empty
  one-empty      exactly one trace has no line (e.g. one side stalled): never a pass
  both-empty     neither trace has a line (e.g. a stalled run): never a pass
Only "identical" passes when require_same_end=True (the default). With require_same_end=False
"prefix-equal" also passes, but only with a common prefix of at least one line; the status and the
length difference are still reported. An empty trace is never a prefix that passes.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path


def read_lines(path_or_bytes) -> list[bytes]:
    """Split raw bytes into lines, keeping each line's terminator. A final unterminated line is kept as is."""
    data = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else Path(path_or_bytes).read_bytes()
    # Split strictly on \n (bytes.splitlines would also split on lone \r, \x0b, \x0c, ...).
    parts = bytes(data).split(b"\n")
    lines = [p + b"\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _strip_eol(line: bytes) -> bytes:
    if line.endswith(b"\r\n"):
        return line[:-2]
    if line.endswith(b"\n"):
        return line[:-1]
    return line


def frame_of(line: bytes):
    """Frame number = first whitespace-separated token of the line, or None if it is not an integer."""
    tok = _strip_eol(line).split(b" ", 1)[0]
    try:
        return int(tok)
    except ValueError:
        return None


@dataclass
class TraceDiff:
    status: str
    ok: bool
    lines_a: int
    lines_b: int
    common_prefix: int  # number of leading byte-identical lines
    first_diff_index: int | None = None  # 0-based line index of the first difference (or first extra line)
    first_diff_frame_a: int | None = None
    first_diff_frame_b: int | None = None
    line_a: bytes | None = None
    line_b: bytes | None = None
    eol_only: bool = False  # the first differing lines differ only in their terminator
    first_frame_a: int | None = None
    first_frame_b: int | None = None
    last_frame_a: int | None = None
    last_frame_b: int | None = None
    same_end: bool = False  # both traces end on the same terminal frame with the same final line
    unterminated_a: bool = False  # final line lacks a terminator (writer killed mid-line?)
    unterminated_b: bool = False
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        out = [
            f"status: {self.status}  ({'PASS' if self.ok else 'FAIL'})",
            f"lines: a={self.lines_a} b={self.lines_b}  common prefix={self.common_prefix}",
            f"frames: a={self.first_frame_a}..{self.last_frame_a}  b={self.first_frame_b}..{self.last_frame_b}"
            f"  same terminal frame: {self.same_end}",
        ]
        if self.status == "differs":
            out.append(
                f"first difference at line {self.first_diff_index} "
                f"(frame a={self.first_diff_frame_a} b={self.first_diff_frame_b})"
                + ("  [line terminators differ only]" if self.eol_only else "")
            )
            out.append(f"  a: {self.line_a!r}")
            out.append(f"  b: {self.line_b!r}")
        elif self.status == "prefix-equal":
            longer = "a" if self.lines_a > self.lines_b else "b"
            out.append(
                f"prefix-equal, lengths differ by {abs(self.lines_a - self.lines_b)} "
                f"({longer} is longer; first extra line {self.first_diff_index}: "
                f"{self.line_a if longer == 'a' else self.line_b!r})"
            )
        elif self.status == "one-empty":
            empty = "a" if self.lines_a == 0 else "b"
            out.append(f"{empty} is empty, the other has {max(self.lines_a, self.lines_b)} lines (never a pass)")
        elif self.status == "both-empty":
            out.append("both traces are empty (never a pass)")
        out.extend(self.notes)
        return "\n".join(out)


def compare(a, b, require_same_end: bool = True) -> TraceDiff:
    """Compare two traces (paths or bytes) byte-exactly."""
    la, lb = read_lines(a), read_lines(b)
    n = min(len(la), len(lb))
    i = 0
    while i < n and la[i] == lb[i]:
        i += 1

    d = TraceDiff(status="", ok=False, lines_a=len(la), lines_b=len(lb), common_prefix=i)
    if la:
        d.first_frame_a, d.last_frame_a = frame_of(la[0]), frame_of(la[-1])
        d.unterminated_a = not la[-1].endswith(b"\n")
    if lb:
        d.first_frame_b, d.last_frame_b = frame_of(lb[0]), frame_of(lb[-1])
        d.unterminated_b = not lb[-1].endswith(b"\n")
    d.same_end = bool(la and lb and la[-1] == lb[-1] and d.last_frame_a == d.last_frame_b)

    if not la and not lb:
        d.status = "both-empty"
    elif not la or not lb:
        d.status = "one-empty"
        d.first_diff_index = 0
        if la:
            d.line_a, d.first_diff_frame_a = la[0], frame_of(la[0])
        else:
            d.line_b, d.first_diff_frame_b = lb[0], frame_of(lb[0])
    elif i < n:
        d.status = "differs"
        d.first_diff_index = i
        d.line_a, d.line_b = la[i], lb[i]
        d.first_diff_frame_a, d.first_diff_frame_b = frame_of(la[i]), frame_of(lb[i])
        d.eol_only = _strip_eol(la[i]) == _strip_eol(lb[i])
    elif len(la) != len(lb):
        d.status = "prefix-equal"
        d.first_diff_index = n
        if len(la) > n:
            d.line_a, d.first_diff_frame_a = la[n], frame_of(la[n])
        else:
            d.line_b, d.first_diff_frame_b = lb[n], frame_of(lb[n])
    else:
        d.status = "identical"

    if d.unterminated_a:
        d.notes.append("note: last line of a has no terminator")
    if d.unterminated_b:
        d.notes.append("note: last line of b has no terminator")

    prefix_ok = d.status == "prefix-equal" and d.common_prefix > 0 and d.lines_a > 0 and d.lines_b > 0
    d.ok = d.status == "identical" or (not require_same_end and prefix_ok)
    return d


def _main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m ballrain.trace", description="Byte-exact trace comparison")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument(
        "--allow-length-diff",
        action="store_true",
        help="also pass when one trace is a byte-identical prefix of the other (status is still reported)",
    )
    args = p.parse_args(argv)
    d = compare(args.a, args.b, require_same_end=not args.allow_length_diff)
    print(d.summary())
    return 0 if d.ok else 1


if __name__ == "__main__":
    sys.exit(_main())
