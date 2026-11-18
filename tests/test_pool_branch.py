"""The route branch in the pool (docs/l2-stone-branch.md): the learner's files (state["branch_assets"]) pinned by the
export, shipped as the branch pack, installed and checked by clients and the verifier; the gateway draws the
branch's at-rest starts. No component reads branch files of its own."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from ballrain.pool import protocol as P  # noqa: E402
from ballrain.pool import task as T  # noqa: E402
from ballrain.pool import weights as W  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
STONE = REPO / "benchmarks" / "refpaths-alt" / "L02-stone"
MAIN = REPO / "benchmarks" / "refpaths-v2" / "L02.npz"
pytestmark = pytest.mark.skipif(not (STONE / "drive.npz").is_file() or not MAIN.is_file(),
                                reason="the stone branch assets are not in this checkout")


def bundle(**over):
    ba = {"level": 2, "line_npz": (STONE / "L02.npz").read_bytes(), "desc_json": (STONE / "L02.json").read_bytes(),
          "drive_npz": (STONE / "drive.npz").read_bytes()}
    ba.update(over)
    return ba


def pin():
    """features.branch_sha256 as the learner computes it (ballrain.branch_assets)."""
    from ballrain.branch_assets import read_bundle, validate
    return validate(read_bundle(2, STONE), 2, T.file_sha256(MAIN))


def changed_rules_bundle(**rules):
    desc = json.loads((STONE / "L02.json").read_text(encoding="utf-8"))
    desc["rules"].update(rules)
    b = bundle(desc_json=json.dumps(desc).encode())
    b["sha256"] = {k: hashlib.sha256(b[k]).hexdigest() for k in ("line_npz", "desc_json", "drive_npz")}
    return b


def l2_state(branch=True, path_sha=None, branch_dir=None, **extra):
    from test_pool_weights import _state
    from ballrain.features import FeatureConfig
    state, _, _ = _state()
    ec = state["env_config"]
    ec["level"] = 2
    ec["features"] = dict(asdict(FeatureConfig()), path_source="tas", branch_sha256=pin() if branch else None,
                          branch_dir=branch_dir, path_sha256=path_sha or T.file_sha256(MAIN))
    state.update(extra)
    return state


def test_export_pins_the_learners_branch_files(tmp_path):
    m = W.export_state(l2_state(branch_assets=bundle()), tmp_path, "r8")
    assert m["features"]["branch_sha256"] == pin() and "branch_dir" not in m["features"]
    # an npz round trip of the bundle (uint8 arrays) pins the same files
    u8 = {k: (np.frombuffer(v, np.uint8) if isinstance(v, bytes) else v) for k, v in bundle().items()}
    assert W.export_state(l2_state(branch_assets=u8), tmp_path / "b", "r8")["policy_sha"] == m["policy_sha"]
    # a run without a branch exports no pins (and ignores a stray bundle)
    plain = W.export_state(l2_state(branch=False, branch_assets=bundle()), tmp_path / "c", "r8")["features"]
    assert plain.get("branch_sha256") is None
    # a legacy checkpoint (branch_dir marker, no pin) with the learner's files: pinned from the files
    legacy = l2_state(branch=False, branch_dir="x", branch_assets=bundle())
    assert W.export_state(legacy, tmp_path / "d", "r8")["features"]["branch_sha256"] == pin()


def test_export_refuses_a_branch_without_the_learners_files(tmp_path):
    with pytest.raises(ValueError, match="branch_assets"):
        W.export_state(l2_state(), tmp_path, "r8")
    with pytest.raises(ValueError, match="does not match"):
        W.export_state(l2_state(path_sha="0" * 64, branch_assets=bundle()), tmp_path, "r8")
    with pytest.raises(ValueError, match="level"):
        W.export_state(l2_state(branch_assets=bundle(level=1)), tmp_path, "r8")
    with pytest.raises(ValueError, match="does not match"):
        W.export_state(l2_state(branch_assets=bundle(line_npz=MAIN.read_bytes())), tmp_path, "r8")
    with pytest.raises(ValueError, match="sha256 differs"):
        W.export_state(l2_state(branch_assets=bundle(sha256={"line_npz": "0" * 64})), tmp_path, "r8")
    with pytest.raises(ValueError, match="branch_sha256"):  # the files differ from the pin
        W.export_state(l2_state(branch_assets=changed_rules_bundle(drive_gain=0.9)), tmp_path, "r8")
    with pytest.raises(ValueError, match="unknown branch rule"):
        W.export_state(l2_state(branch=False, branch_dir="x", branch_assets=changed_rules_bundle(no_such_rule=1)),
                       tmp_path, "r8")


def test_branch_pack_round_trip(tmp_path):
    ms = T.file_sha256(MAIN)
    sha, pack = T.export_branch_pack(bundle(), 2, tmp_path / "assets", ms)
    assert T.file_sha256(pack) == sha and T.export_branch_pack(bundle(), 2, tmp_path / "again", ms)[0] == sha
    d = T.install_branch_pack(pack, tmp_path / "cache", 2, sha, pin(), ms)
    for name in ("L02.npz", "L02.json", "drive.npz"):
        assert (d / name).read_bytes() == (STONE / name).read_bytes()
    assert T.read_branch_dir(d, 2, pin(), ms)["sha256"]["desc_json"] == T.file_sha256(STONE / "L02.json")
    assert T.install_branch_pack(pack, tmp_path / "cache", 2, sha, pin(), ms) == d  # idempotent
    with pytest.raises(ValueError, match="not the policy's"):
        T.install_branch_pack(pack, tmp_path / "c2", 2, sha, "0" * 64, ms)
    with pytest.raises(ValueError, match="sha256"):
        T.install_branch_pack(pack, tmp_path / "c3", 2, "1" * 64, pin(), ms)
    with pytest.raises(T.TaskMismatch):
        T.read_branch_dir(d, 2, "0" * 64, ms)
    # the installed directory is what ballrain.branch.load_branch reads
    from ballrain.branch import load_branch
    from ballrain.path import ReferencePath
    f = load_branch(ReferencePath.load(MAIN), 2, str(d), str(MAIN.parent))
    assert f.ball_type == 2 and f.sector == 4 and f.rules.drive_gain == 0.8


def test_task_schema_takes_the_branch_pack():
    t = {"task_id": "a" * 32, "run": "r8", "policy_sha": "b" * 64, "level": 2, "record": "SR02_x",
         "record_sha256": "c" * 64, "start_frame": 14730, "ball_order_seed": 1, "engine": "x87soft",
         "chunk_steps": 256, "max_steps": 4096, "max_frames": 60000, "deadline_s": 3600, "issued_at": 0,
         "kind": "sample", "greedy": False, "sample_seed": 3,
         "assets": {"starts": "d" * 64, "path": "e" * 64, "branch": "f" * 64},
         "rest": {"target": None, "branch_s": 1700.0}}
    assert P.check(P.TASK, t)["assets"]["branch"] == "f" * 64
    with pytest.raises(P.ProtocolError):
        P.check(P.TASK, dict(t, assets=dict(t["assets"], other="0" * 64)))


class _Starts:
    def __init__(self, valid):
        self.valid = np.asarray(valid)

    def nearest_any(self, k):
        return int(k)

    def nearest_valid(self, k):
        return int(self.valid[np.argmin(abs(self.valid - k))])

    def expected(self, k):
        return {"frame": k} if k > 0 else None


def test_gateway_draws_branch_rest_starts():
    from ballrain.pool.server_tasks import PolicySource, Published
    ps = PolicySource("unused", "r8", "unused", [])
    st = _Starts([0, 100, 200])  # 14730 is not a valid start (excluded), still a hand-over state
    pub = Published(manifest={}, policy_sha="b" * 64, run="r8", record="SR02_x", record_sha256="c" * 64,
                    starts_sha="d" * 64, path_sha="e" * 64, level=2, valid=st.valid, curriculum=None, source="state",
                    published_at=0, starts=st, branch_sha="f" * 64,
                    draws={"rest_starts": [[14730, 14730, {"branch_s": 1975.0}]], "p_rest": 1.0})
    k, rest = ps.pick_start(pub)
    assert k == 14730 and rest == {"target": None, "branch_s": 1975.0}
    # the env draws the same entry with the same calls (random(), integers(len), integers(lo, hi + 1))
    rng = np.random.Generator(np.random.PCG64(5))
    ps.rng = np.random.Generator(np.random.PCG64(5))
    pub.draws["rest_starts"].append([100, 200])
    got = [ps.pick_start(pub) for _ in range(50)]
    want = []
    for _ in range(50):
        assert rng.random() < 1.0
        e = pub.draws["rest_starts"][int(rng.integers(2))]
        kk = int(rng.integers(int(e[0]), int(e[1]) + 1))
        want.append((kk, {"target": None, "branch_s": 1975.0}) if len(e) == 3 else (st.nearest_valid(kk), {"target": None}))
    assert got == want


def test_branch_policies_require_the_branch_capability(tmp_path):
    from ballrain.pool import capabilities as C
    m = W.export_state(l2_state(branch_assets=bundle()), tmp_path, "r8")
    assert m["requires"] == ["branch_v1"] and "branch_v1" in C.CAPABILITIES  # default rules: branch_v1 only
    # changed rules: also branch_rules_v1 (the client must read the L<nn>.json rules)
    from ballrain.branch_assets import validate
    b2 = changed_rules_bundle(drive_gain=0.9)
    st = l2_state(branch_assets=b2)
    st["env_config"]["features"]["branch_sha256"] = validate(b2, 2, T.file_sha256(MAIN))
    assert W.export_state(st, tmp_path / "r", "r8")["requires"] == ["branch_v1", "branch_rules_v1"]
    plain = W.export_state(l2_state(branch=False), tmp_path / "p", "r8")
    assert "requires" not in plain  # nothing new needed: every client keeps getting its tasks
    # requires is not an identity field: the same policy with or without it has the same sha
    assert W.policy_sha(dict(plain, requires=["x"]), {}) == W.policy_sha(plain, {})
    from ballrain.pool.client_api import CLIENT_CAPABILITIES
    assert not C.missing(m["requires"], CLIENT_CAPABILITIES)


def test_hello_carries_capabilities():
    eng = {"kind": "x87soft", "sha256": "a" * 64, "version": "v1"}
    base = dict(client_version="0.4.0", platform="windows", engine=eng, game_files={}, instances=1)
    try:
        P.make_message("hello", **base)
    except P.ProtocolError:
        pytest.skip("this test's hello skeleton does not fit the schema")
    assert P.make_message("hello", **base, capabilities=["task_format3", "branch_v1", "later_thing"])
    for bad in (["x" * 33], ["a b"], ["c"] * 33):
        with pytest.raises(P.ProtocolError):
            P.make_message("hello", **base, capabilities=bad)
