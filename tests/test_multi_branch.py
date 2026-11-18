"""Distribution, real bundled phase geometry, single-credit ledger, explicit migration."""

import copy, json, pickle
import numpy as np
import pytest
from ballrain import path
from ballrain.branch import load_branch
from ballrain.branch_assets import read_bundle, validate, install
from ballrain.multi_branch import manifest
from scripts import train_m2 as train

BASE = path.REFPATH_DIR.parent
MAIN = BASE / "refpaths-v2"
ALT = BASE / "refpaths-alt/L02-multi"
OLD = BASE / "refpaths-alt/L02-stone"
SHA = path.file_sha256(MAIN / "L02.npz")


@pytest.fixture
def multi():
    return load_branch(path.ReferencePath.load(MAIN / "L02.npz"), 2, ALT, MAIN)


@pytest.mark.parametrize(
    "mutation",
    ["traversal", "duplicate", "region", "unknown", "phase gap", "nan", "bool"],
)
def test_bad_manifest(mutation):
    doc = json.loads((ALT / "branches.json").read_bytes())
    entry = doc["branches"][0]
    p = entry["phases"][0]
    if mutation == "traversal":
        entry["id"] = "../bad"
    if mutation == "duplicate":
        doc["branches"].append(entry)
    if mutation == "region":
        p["when"] = {"ball": 0}
    if mutation == "unknown":
        p["when"]["script"] = "unused"
    if mutation == "phase gap":
        p["local_s"][1] -= 1
    if mutation == "nan":
        p["when"]["box"][0][0] = float("nan")
    if mutation == "bool":
        p["when"]["ball"] = True
    with pytest.raises(path.RouteError):
        manifest(json.dumps(doc))


def test_distributed_digest_and_install():
    b = pickle.loads(pickle.dumps(read_bundle(2, ALT)))
    key = validate(b, 2, SHA)
    tmp = install(b, 2, SHA, key)
    assert read_bundle(2, tmp.name) == b
    original = read_bundle(2, OLD)
    assert b["members"][-1] == original
    assert (
        validate(original, 2, SHA)
        == "f7faf3cf9c9621f55f6c0de3f56dd6c2813a0cec33ca22672cbd6c4ba8e9649b"
    )
    bad = copy.deepcopy(b)
    bad["members"].reverse()
    with pytest.raises(path.RouteError):
        install(bad, 2, SHA, key)
    b["members"][0]["drive_npz"] += b"!"
    with pytest.raises(path.RouteError):
        validate(b, 2, SHA)


@pytest.mark.parametrize("key", ["back-to-wood", "east-platform"])
def test_paid_span_and_no_second_payout(multi, key):
    e, f = multi.by_id[key]
    sec = f.sector
    phases = e["phases"]
    ms, bs = f.stations
    start = ms[1]
    end = ms[-2]
    # Start on the default TAS, then physically traverse each installed polyline.
    _, pr = multi.update(f.main.point_at(start), sec, 0)
    assert not pr["branch"]
    credits = [pr["s"]]
    seen = set()
    for phase in phases:
        lo, hi = phase["local_s"]
        ball = phase["when"]["ball"]
        for s in np.linspace(lo, hi, 1 + int(np.ceil((hi - lo) * 4))):
            _, pr = multi.update(f.alt.point_at(s), sec, ball)
            credits.append(pr["s"])
            seen.add(pr["phase"])
    assert set(range(len(phases))) <= seen and key in multi.done
    assert np.all(np.diff(credits) >= -1e-8)
    assert credits[-1] - credits[0] == pytest.approx(end - start, rel=0, abs=1e-8)
    # Retrace the merge repeatedly; neither credit nor elapsed clock can be minted twice.
    for _ in range(12):
        for station in (end - 0.8, end + 1):
            multi.update(f.main.point_at(station), sec, ball)
        if _ == 0:
            initial = (multi.credit, multi.clock_ms)
        assert initial == (multi.credit, multi.clock_ms)
    multi.reset()
    assert not multi.done and multi.phase == 0 and multi.key is None
    cond = phases[0]["when"]
    pt = np.mean(cond["box"], axis=0)
    assert multi.update(pt, sec, cond["ball"])[1]["branch_id"] == key


def test_paper_after_plate_and_wrong_sector_keep_main(multi):
    e, f = multi.by_id["east-platform"]
    assert not multi.update(f.alt.point_at(f.hi), 3, 1)[1]["branch"]
    center = np.mean(e["phases"][0]["when"]["box"], axis=0)
    assert not multi.update(center, 2, 1)[1]["branch"]
    assert not multi.update(center, 3, 0)[1]["branch"]
    assert multi.update(center, 3, 1)[1]["phase"] == 0
    # A stone observation near the plate cannot teleport outbound paper to the return phase.
    assert multi.update(f.alt.point_at(f.hi), 3, 2)[1]["phase"] == 0


def test_old_bundle_and_checkpoint_repin(tmp_path):
    args = train.parse(
        [
            "--run",
            "unit",
            "--level",
            "2",
            "--path-dir",
            str(MAIN),
            "--branch-dir",
            str(OLD),
        ]
    )
    old = train.make_env_config(args, tmp_path)
    ck = train.checkpoint_config(old)
    args.branch_dir = str(ALT)
    with pytest.raises(path.RouteError, match="checkpoint pin"):
        train.make_env_config(args, tmp_path, ck)
    args.branch_update = True
    new = train.make_env_config(args, tmp_path, ck)
    assert new.features.branch_sha256 != old.features.branch_sha256
    args.branch_update = False
    args.branch_dir = None
    assert (
        train.make_env_config(
            args, tmp_path, train.checkpoint_config(new)
        ).branch_assets
        == new.branch_assets
    )
    assert train.parse_rest_starts("8-8@branch:a:4,9-9@branch:3") == [
        [8, 8, {"id": "a", "branch_s": 4.0}],
        [9, 9, {"branch_s": 3.0}],
    ]


def test_v1_worker_refused_before_v2_template():
    from types import SimpleNamespace
    from threading import Condition
    from unittest.mock import Mock
    from ballrain.train.remote import PROTOCOL, RemoteHub

    hub = object.__new__(RemoteHub)
    hub.template = {"env": SimpleNamespace(branch_assets=read_bundle(2, ALT))}
    hub.stats = {"disconnects": 0}
    hub.log, hub._cond = Mock(), Condition()
    conn = Mock()
    conn.recv.return_value = dict(
        type="hello", protocol=PROTOCOL, branch_assets_version=1, branch_rules_version=1
    )
    hub._serve(conn)
    assert "multi_branch_v2" in conn.send.call_args.args[0]["error"]


def test_native_gate_boundaries():
    from ballrain.multi_branch import matches

    g = dict(ball=2, s=[3, 4], box=[[0, 0, 0], [2, 2, 2]])
    assert matches(g, np.zeros(3), 1, 2, 3.0) and matches(g, np.full(3, 2.0), 9, 2, 4.0)
    assert not matches(g, np.zeros(3), 1, 2, 4.001)


def test_nested_south_loop_matches_v1_progress_and_clocks(multi):
    legacy = load_branch(multi.main, 2, OLD, MAIN)
    for station in np.linspace(legacy.lo - 3, legacy.hi + 3, 600):
        ball = 2 if legacy.lo <= station <= legacy.hi else 1
        pos = legacy.alt.point_at(station)
        _, a = multi.update(pos, legacy.sector, ball)
        _, b = legacy.update(pos, legacy.sector, ball)
        for field in ("branch", "s", "local_s", "ref_remaining_ms", "ref_elapsed_ms"):
            assert a[field] == b[field]


def test_early_wood_uses_verified_material_and_location(multi):
    entry, f = multi.by_id["back-to-wood"]
    multi.update(np.mean(entry["phases"][0]["when"]["box"], axis=0), 2, 2)
    _, wrong = multi.update(np.array([716.0, 32.0, -268.0]), 2, 0)
    assert wrong["phase"] == 0
    _, right = multi.update(np.mean(entry["phases"][-1]["when"]["box"], axis=0), 2, 0)
    assert right["phase"] == len(entry["phases"]) - 1
