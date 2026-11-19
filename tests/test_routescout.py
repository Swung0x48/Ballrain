import json
from pathlib import Path

import pytest
from ballrain.routescout.contract import parser, difficulty, select, UNVERIFIED
from ballrain.routescout.__main__ import main


def test_cli():
    p = parser()
    a = p.parse_args(["plan", "--level", "2", "--out", "run"])
    assert not a.verify and a.spawn is None and a.instance == "hlw-91"
    for args in (
        ["--instance", "train-00"],
        ["--spawn", "0,nan,0"],
        ["--spawn", "1,2,3", "--from-checkpoint", "1"],
    ):
        with pytest.raises(SystemExit):
            p.parse_args(["plan", "--level", "2", *args])
    assert p.parse_args(["verify", "run"]).out == Path("run")


def test_custom_fails_without_engine(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["plan", "--map", "custom.nmo", "--out", str(tmp_path)])
    assert "file loader" in capsys.readouterr().err and not list(tmp_path.iterdir())


def test_difficulty_bounds():
    assert difficulty([[1, 100]], ["wood"], 0)["score"] == 0
    r = difficulty([[1, 1], [2, 1]] * 16, ["wood", "paper", "stone", "wood"], 1)
    assert r["score"] == sum(r["components"].values()) == 100


def test_frontiers():
    def row(i, sr, hs, d):
        return dict(
            id=i,
            family=i,
            sr_ms=sr,
            hs={"score_delta": hs},
            difficulty={"score": d},
            complete=True,
            checkpoints=[[1, sr]],
            distance_to_goal=0,
        )

    result = select(
        [
            row("a", 10, 2, 10),
            row("b", 12, 8, 10),
            row("c", 10, 2, 10),
            row("d", 12, 1, 20),
        ],
        6,
    )
    assert {r["id"] for r in result if r["pareto"]["sr"]} == {"a", "c"}
    assert {r["id"] for r in result if r["pareto"]["hs"]} == {"b"}


def test_candidate_schema():
    from types import SimpleNamespace as NS
    import numpy as np
    from ballrain.routescout.engine import Trace

    ball = dict(pos=np.zeros(3), speed=np.zeros(3), speed_change=np.zeros(3))
    s = NS(
        obs=NS(ball=ball, body=[], fric={"partner": np.array([])}),
        frames_done=20,
        tracker=NS(
            level=NS(groups={}),
            ball_type=lambda: 0,
            sr_ms=12.0,
            hs_partial=lambda: 8,
            checkpoints=0,
        ),
    )
    t = Trace(s, None)
    t.keys = [[1, 6]]
    s.frames_done += 6
    r = t.candidate(dict(from_checkpoint=0, target=1, build="pin"), [], np.ones(3))
    assert r["verification"] == {"status": UNVERIFIED} and not r["complete"]
    assert (
        r["ball_sequence"] == ["wood"]
        and r["hs"]["score_delta"] == 0
        and r["keys"] == [[1, 6]]
    )
    assert {
        "id",
        "family",
        "segments",
        "checkpoints",
        "centre_path",
        "transformer_events",
        "mechanisms",
        "difficulty",
        "sr_ms",
        "game_ms",
        "witness",
        "search_build",
    } <= json.loads(json.dumps(r)).keys()


def test_engine_smoke(tmp_path):
    p = parser().parse_args(["plan", "--level", "2"])
    if not (p.build.is_file() and (p.root / "base.cmo").is_file()):
        pytest.skip("no local spare headless engine configured")
    main(
        [
            "plan",
            "--level",
            "2",
            "--instance",
            "hlw-98",
            "--budget-s",
            "4",
            "--out",
            str(tmp_path),
        ]
    )
    rows = json.loads((tmp_path / "candidates.json").read_text())["candidates"]
    assert rows and all(r["verification"]["status"] == UNVERIFIED for r in rows)
    assert (tmp_path / "top-view.png").read_bytes().startswith(b"\x89PNG")
