"""Multi-branch routes over the pool (wire v2, docs/pool-prototype.md 15): the learner's v2 envelope is exported with
its own pin and the multi_branch_v2 requirement, shipped as one deterministic branch pack (manifest_json + m{i:02d}_*),
installed in the layout ballrain.branch_assets.read_bundle reads, and read back as the env's EnvConfig.branch_assets;
the task's rest carries the member id. v1 stays as it was (tests/test_pool_branch.py)."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from ballrain import branch_assets as BA
from ballrain.pool import protocol as P
from ballrain.pool import task as T
from ballrain.pool import weights as W

REPO = Path(__file__).resolve().parents[1]
MULTI = REPO / "benchmarks" / "refpaths-alt" / "L02-multi"
MAIN = REPO / "benchmarks" / "refpaths-v2" / "L02.npz"
pytestmark = pytest.mark.skipif(not (MULTI / "branches.json").is_file() or not MAIN.is_file(),
                                reason="the multi-branch assets are not in this checkout")


def v2():
    return BA.read_bundle(2, MULTI)


def main_sha():
    return T.file_sha256(MAIN)


def pin(b=None):
    return BA.validate(b or v2(), 2, main_sha())


def ids():
    return [e["id"] for e in json.loads((MULTI / "branches.json").read_bytes())["branches"]]


def v2_state(b=None):
    from test_pool_branch import l2_state
    st = l2_state(branch_assets=b or v2())
    st["env_config"]["features"]["branch_sha256"] = pin(b)
    return st


def test_export_pins_the_v2_envelope_and_requires_multi_branch(tmp_path):
    from ballrain.pool import capabilities as C
    m = W.export_state(v2_state(), tmp_path, "r8")
    assert m["features"]["branch_sha256"] == pin()
    assert m["requires"] == ["multi_branch_v2"]  # L02-multi: default rules in every member
    assert C.describe("multi_branch_v2")[0] == "0.5.0"
    # the Python client does not run v2: it never gets these tasks (no client-side Python work)
    from ballrain.pool.client_api import CLIENT_CAPABILITIES
    assert C.missing(m["requires"], CLIENT_CAPABILITIES) == ["multi_branch_v2"]
    msg = C.refusal("0.4.0", C.missing(m["requires"], ["task_format3", "branch_v1", "branch_rules_v1"]))
    assert msg["code"] == "client_too_old" and "多分线" in msg["message"] and msg["params"]["min_version"] == "0.5.0"
    # a member with other rules: branch_rules_v1 as well
    b = v2()
    desc = json.loads(b["members"][0]["desc_json"])
    desc["rules"]["drive_gain"] = 0.9
    b["members"][0] = dict(b["members"][0], desc_json=json.dumps(desc).encode())
    b["members"][0]["sha256"] = {k: hashlib.sha256(b["members"][0][k]).hexdigest() for k in T.BRANCH_PACK_ARRAYS}
    assert W.export_state(v2_state(b), tmp_path / "r", "r8")["requires"] == ["multi_branch_v2", "branch_rules_v1"]


def test_export_refuses_a_wrong_pin_or_a_broken_envelope(tmp_path):
    st = v2_state()
    st["env_config"]["features"]["branch_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pin"):
        W.export_state(st, tmp_path, "r8")
    b = v2()
    b["members"] = b["members"][:-1]  # fewer members than manifest entries
    with pytest.raises(ValueError, match="mismatch"):
        T.branch_bundle(b, 2, main_sha())
    with pytest.raises(ValueError, match="members"):
        T.branch_bundle(dict(v2(), members=[]), 2, main_sha())
    with pytest.raises(ValueError, match="version"):
        T.branch_bundle(dict(v2(), version=3), 2, main_sha())
    with pytest.raises(ValueError, match="single-line"):
        T.branch_bundle(dict(v2(), members=[v2()]), 2, main_sha())
    b = v2()
    b["members"][1] = dict(b["members"][1], drive_npz=b["members"][0]["drive_npz"])  # a drive of another member
    b["members"][1]["sha256"] = dict(b["members"][1]["sha256"], drive_npz=hashlib.sha256(b["members"][1]["drive_npz"]).hexdigest())
    with pytest.raises(ValueError):
        T.branch_bundle(b, 2, main_sha())
    with pytest.raises(ValueError):  # the main line pairing is checked in every member
        T.branch_bundle(v2(), 2, "1" * 64)


def test_v2_pack_round_trip(tmp_path):
    ms = main_sha()
    sha, pack = T.export_branch_pack(v2(), 2, tmp_path / "assets", ms)
    assert T.file_sha256(pack) == sha and T.export_branch_pack(v2(), 2, tmp_path / "again", ms)[0] == sha
    with np.load(pack, allow_pickle=False) as z:
        assert set(z.files) == set(T.branch_pack_v2_arrays(3))
        assert z["manifest_json"].tobytes() == (MULTI / "branches.json").read_bytes()
        assert z["m01_line_npz"].tobytes() == (MULTI / ids()[1] / "L02.npz").read_bytes()
    d = T.install_branch_pack(pack, tmp_path / "cache", 2, sha, pin(), ms)
    assert (d / "branches.json").read_bytes() == (MULTI / "branches.json").read_bytes()
    for i in ids():
        for name in ("L02.npz", "L02.json", "drive.npz"):
            assert (d / i / name).read_bytes() == (MULTI / i / name).read_bytes()
    assert BA.validate(BA.read_bundle(2, d), 2, ms) == pin()  # what the learner's own reader makes of it
    assert T.install_branch_pack(pack, tmp_path / "cache", 2, sha, pin(), ms) == d  # idempotent
    b = T.read_branch_dir(d, 2, pin(), ms)
    assert b["version"] == 2 and b["manifest"] == (MULTI / "branches.json").read_bytes() and len(b["members"]) == 3
    tmp = BA.install(b, 2, ms, pin())  # the env's install accepts it
    assert sorted(p.name for p in Path(tmp.name).iterdir()) == sorted(ids() + ["branches.json"])
    with pytest.raises(ValueError, match="not the policy's"):
        T.install_branch_pack(pack, tmp_path / "c2", 2, sha, "0" * 64, ms)
    with pytest.raises(T.TaskMismatch):
        T.read_branch_dir(d, 2, "0" * 64, ms)


def _repack(tmp_path, arrays, name="p.npz"):
    p = tmp_path / name
    T._write_npz(p, arrays)
    return p, T.file_sha256(p)


def test_v2_pack_refuses_other_array_sets_and_swapped_members(tmp_path):
    ms = main_sha()
    _, pack = T.export_branch_pack(v2(), 2, tmp_path / "assets", ms)
    with np.load(pack, allow_pickle=False) as z:
        a = {k: z[k].copy() for k in z.files}
    for bad in ({**a, "extra": np.zeros(3, np.uint8)},
                {k: v for k, v in a.items() if k != "m02_drive_npz"},
                {**a, **{f"m03_{k}": a[f"m00_{k}"] for k in T.BRANCH_PACK_ARRAYS}}):
        p, s = _repack(tmp_path, bad)
        with pytest.raises(ValueError):
            T.install_branch_pack(p, tmp_path / "c", 2, s, pin(), ms)
    sw = dict(a)  # members swapped: each one valid, the pin (manifest order) is not
    for k in T.BRANCH_PACK_ARRAYS:
        sw[f"m00_{k}"], sw[f"m01_{k}"] = a[f"m01_{k}"], a[f"m00_{k}"]
    p, s = _repack(tmp_path, sw, "sw.npz")
    with pytest.raises(ValueError):
        T.install_branch_pack(p, tmp_path / "c", 2, s, pin(), ms)


def test_v1_pack_is_unchanged(tmp_path):
    from test_pool_branch import bundle
    sha, pack = T.export_branch_pack(bundle(), 2, tmp_path, main_sha())
    with np.load(pack, allow_pickle=False) as z:
        assert set(z.files) == set(T.BRANCH_PACK_ARRAYS)


def test_task_rest_takes_a_member_id():
    t = {"task_id": "a" * 32, "run": "r8", "policy_sha": "b" * 64, "level": 2, "record": "SR02_x",
         "record_sha256": "c" * 64, "start_frame": 14730, "ball_order_seed": 1, "engine": "x87soft",
         "chunk_steps": 256, "max_steps": 4096, "max_frames": 60000, "deadline_s": 3600, "issued_at": 0,
         "kind": "sample", "greedy": False, "sample_seed": 3,
         "assets": {"starts": "d" * 64, "path": "e" * 64, "branch": "f" * 64},
         "rest": {"target": None, "branch_s": 870.0, "branch_id": "back-to-wood"}}
    assert P.check(P.TASK, t)["rest"]["branch_id"] == "back-to-wood"
    for bad in ("Back", "1abc", "a_b", "", "a" * 33):
        with pytest.raises(P.ProtocolError):
            P.check(P.TASK, dict(t, rest=dict(t["rest"], branch_id=bad)))


def test_gateway_draws_member_rest_starts():
    from test_pool_branch import _Starts
    from ballrain.pool.server_tasks import PolicySource, Published
    ps = PolicySource("unused", "r8", "unused", [])
    st = _Starts([0, 100, 200])
    pub = Published(manifest={}, policy_sha="b" * 64, run="r8", record="SR02_x", record_sha256="c" * 64,
                    starts_sha="d" * 64, path_sha="e" * 64, level=2, valid=st.valid, curriculum=None, source="state",
                    published_at=0, starts=st, branch_sha="f" * 64,
                    draws={"rest_starts": [[9000, 9000, {"branch_s": 870.0, "id": "back-to-wood"}]], "p_rest": 1.0})
    assert ps.pick_start(pub) == (9000, {"target": None, "branch_s": 870.0, "branch_id": "back-to-wood"})
    pub.draws["rest_starts"] = [[9000, 9000, {"branch_s": 1975.0}]]  # legacy syntax: no id, as in v1
    assert ps.pick_start(pub) == (9000, {"target": None, "branch_s": 1975.0})


def test_verifier_env_from_a_v2_task(tmp_path):
    """What the verifier replays with: build_env_config hands the installed v2 pack to the env as the v2 envelope and
    the task's rest (with the member id) as start_rest; the env installs the routes and starts that member's drive."""
    from ballrain import fakegame
    from ballrain.env import BallanceEnv
    from ballrain.instances import runtime_dir
    rd = runtime_dir() / "datasets" / "tas-v1" / "SR02_1_28_992_264"
    if not (rd / "starts.npz").is_file():
        pytest.skip("the SR02 dataset record is not here (BALLRAIN_RUNTIME)")
    st = v2_state()
    st["env_config"]["start_record"] = rd.name
    m = W.export_state(st, tmp_path / "w", "r8")
    m = W.read_manifest(tmp_path / "w" / f"{m['policy_sha']}.json")
    ms = main_sha()
    ssha, spack = T.export_starts_pack(rd, tmp_path / "assets")
    psha, stored = T.export_path_asset(MAIN, tmp_path / "assets")
    bsha, bpack = T.export_branch_pack(v2(), 2, tmp_path / "assets", ms)
    sdir = T.install_starts_pack(spack, tmp_path / "cache", ssha)
    pdir = T.install_path_asset(stored, tmp_path / "cache", 2, psha)
    bdir = T.install_branch_pack(bpack, tmp_path / "cache", 2, bsha, m["features"]["branch_sha256"], ms)
    meta = json.loads((rd / "meta.json").read_text(encoding="utf-8"))
    k = int(T.valid_start_frames(sdir, m["env"].get("exclude_starts"))[-1])
    t = T.Task(task_id=T.new_task_id(), run="r8", policy_sha=m["policy_sha"], level=2, record=rd.name,
               record_sha256=meta["record_sha256"], start_frame=k, ball_order_seed=7, engine="x87soft",
               assets={"starts": ssha, "path": psha, "branch": bsha}, issued_at=0,
               rest={"target": None, "branch_s": 870.0, "branch_id": "back-to-wood"})
    lg = T.LocalGame(engine="x87soft", binary="SimTool", root="Ballance")
    cfg = T.build_env_config(t, m, lg, sdir, pdir, tmp_path / "out", branch_dir=bdir)
    assert cfg.branch_assets["version"] == 2 and cfg.features.branch_sha256 == pin()
    assert cfg.start_rest == {"target": None, "branch_s": 870.0, "branch_id": "back-to-wood"}
    e = BallanceEnv(cfg, launcher=fakegame.FakeLauncher(backend="headless"))
    try:
        assert (Path(e._branch_temp.name) / "branches.json").is_file()
        assert e._draw_start() == k and e._rest_next == {"branch_s": 870.0, "id": "back-to-wood"}
    finally:
        e.close()
    with pytest.raises(T.TaskMismatch):  # a v2 policy without its pack
        T.build_env_config(t, m, lg, sdir, pdir, tmp_path / "out")


def test_learner_link_says_multi_branch_v2():
    import inspect
    from ballrain.pool import learner_link
    assert '"multi_branch_v2": True' in inspect.getsource(learner_link.LearnerLink.connect)
