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


def test_frontier_is_not_a_candidate_family():
    import random
    import numpy as np
    from ballrain.routescout import frontier as f

    cells = {}
    a, b, c = [f.cell(np.array([x, 2, 1]), 0, 1, "spawn") for x in (0, 4, 8)]
    assert f.store(cells, a, None, [[1, 8]], "A", 20)
    assert f.store(cells, b, a, [[8, 12]], "B", 22)  # detour away from goal
    assert f.store(cells, c, b, [[2, 4]], "C", 14)
    assert not f.store(cells, b, c, [], "cycle", 3)
    assert f.tape(cells, c) == [[1, 8], [8, 12], [2, 4]]
    assert f.choose(cells, random.Random(1), "other") is None
    cells[b]["diverged"] = True
    assert b not in {f.choose(cells, random.Random(i), "spawn") for i in range(20)}


def test_frontier_exits_exclude_void_and_walls():
    from types import SimpleNamespace
    from ballrain.routescout.frontier import exits
    import numpy as np

    def march(p, d, **kw):
        return dict(edge_s=1 if d[0] < 0 else 99, obst_s=2 if d[1] else 99, y_last=4.5)

    assert exits(
        SimpleNamespace(march=march), np.array([1.0, 3.0, 2.0]), 0, 2, "s"
    ) == {"s:0,2,1,1,0": [5.0, 4.5, 2.0]}


def test_unknown_checkpoint_keeps_coverage_without_fabricating_a_distance():
    from ballrain.routescout.contract import order

    r = dict(
        id="partial",
        guide_remaining=None,
        distance_to_goal=None,
        checkpoints=[],
        complete=False,
        sr_ms=10,
        hs={"score_delta": 0},
        family="f",
        difficulty={"score": 0},
    )
    assert select([r], 2) == [r] and order(r)


def test_unknown_checkpoint_keeps_engine_order():
    from types import SimpleNamespace
    import numpy as np
    from ballrain.routescout.engine import checkpoint_goals

    m = np.eye(4)
    m[3, :3] = [12, -3, 20]
    arrays = SimpleNamespace(
        rows=lambda name: 2, get=lambda name, i, j: m if (i, j) == (1, 1) else None
    )
    goals = checkpoint_goals(SimpleNamespace(arrays=arrays), 2)
    assert goals[0] is None and np.array_equal(goals[1], m[3, :3])
    with pytest.raises(ValueError, match="outside map"):
        checkpoint_goals(SimpleNamespace(arrays=arrays), 3)


def test_frontier_restore_requires_the_whole_witness():
    from types import SimpleNamespace as NS
    from ballrain.routescout import frontier as f

    cells = {}
    f.store(cells, "a", None, [[2, 4]], "ok", 1)
    tr = NS(step=lambda k, n: True, keys=[[2, 4]], witness=NS(hexdigest=lambda: "ok"))
    assert f.restore(tr, cells, "a", lambda: False)
    assert f.restore(tr, cells, "a", lambda: True) is None
    tr.witness = NS(hexdigest=lambda: "wrong")
    assert not f.restore(tr, cells, "a", lambda: False)
    tr.keys = [[2, 3]]
    tr.witness = NS(hexdigest=lambda: "ok")
    assert not f.restore(tr, cells, "a", lambda: False)


def test_field_guide_goes_around_void_instead_of_following_goal_chord():
    from types import SimpleNamespace as NS
    import numpy as np
    from ballrain.routescout.floor import FloorGuide
    from ballrain.fastreach import TL

    # Two walkways separated by a void wider than any hypothetical jump.
    tiles = np.arange(8 * 5).reshape(8, 5)
    rest = np.full((tiles.size, TL, TL, 1), np.nan)
    for (i, j), tile in np.ndenumerate(tiles):
        if i in (0, 7) or j == 4:
            rest[tile] = 2.0
    field = NS(
        tiles=tiles,
        Rr=rest,
        Rf=np.ones_like(rest, dtype=int),
        Rn=np.zeros((*rest.shape, 2)),
        ox=0.0,
        oz=0.0,
    )
    grid = FloorGuide(field)
    start = np.array([0.0, 2.0, 0.0])
    goal = np.array([28.0, 2.0, 0.0])
    assert grid.path(start, None) is None
    path = grid.path(start, goal)[0]
    assert path[:, 2].max() >= 16 and np.allclose(path[-1], goal)


def test_conservative_proposal_avoids_open_edge():
    from types import SimpleNamespace as NS
    import random
    import numpy as np
    from ballrain.routescout.floor import waypoint
    from ballrain.routescout.guidance import ribbon

    def march(p, d, **kw):
        return dict(edge_s=0 if d[0] > 0.01 else 20, obst_s=20, obst_h=0)

    field = NS(support=lambda p, tol: (2.0, None, 1), march=march)
    rules = NS(gate=lambda *args: None)
    p = np.array([0.0, 2.0, 0.0])
    line = ribbon([p, p + [20, 0, 0]])
    end = waypoint(field, rules, p, line, 0, 4, random.Random(1), 0)
    assert end[0] < 0.01 and np.linalg.norm(end - p) >= 3
