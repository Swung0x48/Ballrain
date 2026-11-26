"""tools/dashboard/data.read_csv: a growing CSV is read incrementally and equals a full read."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "dashboard"))
import data as D  # noqa: E402


def _write(p, rows, mode="w"):
    with open(p, mode, encoding="utf-8", newline="") as f:
        if mode == "w":
            f.write("update,end,s_end\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")


def test_growing_file_is_appended_and_matches_a_full_read(tmp_path):
    p = tmp_path / "episodes.csv"
    _write(p, [(1, "stall", 410.5), (2, "LevelFinish", 2100.0)])
    a = D.read_csv(p, usecols=["update", "end"])
    assert list(a["update"]) == [1, 2]
    _write(p, [(3, "BallOff", 12.0), (4, "stall", 400.0)], mode="a")
    with open(p, "a", encoding="utf-8") as f:
        f.write("5,sta")  # a row still being written
    b = D.read_csv(p, usecols=["update", "end"])
    assert list(b["update"]) == [1, 2, 3, 4] and list(b.columns) == ["update", "end"]
    with open(p, "a", encoding="utf-8") as f:
        f.write("ll,1.0\n")
    c = D.read_csv(p, usecols=["update", "end"])
    full = pd.read_csv(p, usecols=["update", "end"])
    assert c.reset_index(drop=True).equals(full)


def test_rewritten_file_is_read_again(tmp_path):
    p = tmp_path / "updates.csv"
    _write(p, [(1, "x", 1.0)])
    assert len(D.read_csv(p)) == 1
    with open(p, "w", encoding="utf-8") as f:
        f.write("update,end,s_end,extra\n9,y,2.0,7\n8,z,3.0,6\n")
    d = D.read_csv(p)
    assert list(d["update"]) == [9, 8] and "extra" in d.columns
