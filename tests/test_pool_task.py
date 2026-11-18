"""ballrain.pool.task: the task schema, sha-addressed start-state / reference-line assets, game-file hashing and the
client's BallanceEnv built from a task (FakeLauncher; the r6 manifest on the SR01 record when present)."""
import json
from pathlib import Path

import numpy as np
import pytest

from ballrain.curriculum import STARTS_VERSION, TasStarts
from ballrain.instances import REPO_ROOT
from ballrain.pool import protocol as P, task as T

SHA = "ab" * 32


def _task(**over):
    d = dict(task_id=T.new_task_id(), run="r6", policy_sha=SHA, level=1, record="SR01_test", record_sha256="cd" * 32,
             start_frame=120, ball_order_seed=T.new_ball_order_seed(), engine="x87soft",
             assets={"starts": "ef" * 32, "path": "01" * 32}, issued_at=1000)
    d.update(over)
    return T.Task(**d)


def test_task_round_trip_and_validation():
    t = _task()
    d = t.to_dict()
    msg = P.parse_message("task", P.encode_message("task", task=d))
    t2 = T.Task.from_dict(msg["task"])
    assert t2 == t and t2.ball_order == f"retail:{t.ball_order_seed}" and t2.deadline == 1000 + t.deadline_s
    for bad in (dict(d, engine="wine"), dict(d, level=0), dict(d, ball_order_seed=2 ** 32), dict(d, start_frame=-1),
                dict(d, record="../../x"), dict(d, assets={"starts": SHA}), dict(d, kind="mine"), dict(d, greedy=0),
                dict(d, chunk_steps=P.MAX_CHUNK_STEPS + 1), dict(d, extra=1)):
        with pytest.raises(P.ProtocolError):
            T.Task.from_dict(bad)
    assert len({T.new_task_id() for _ in range(100)}) == 100


# ---------------------------------------------------------------------------------------------- assets

def _record_dir(root: Path, name="SR01_test", n_frames=400, n_valid=50) -> Path:
    d = root / "dataset" / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    frames = np.zeros(n_frames, T.RECORD_FRAME_DTYPE)
    frames["delta"] = np.float32(1000 / 264)
    frames["keys"] = rng.integers(0, 16, n_frames)
    np.savez(d / "raw.npz", record_frames=frames, obs_bytes=np.zeros(4, np.uint8), obs_offsets=np.zeros(2, np.int64))
    fd = np.sort(rng.choice(np.arange(10, n_frames - 10), n_valid, replace=False)).astype(np.int64)
    np.savez(d / "starts.npz", version=np.array(STARTS_VERSION), frames_done=fd, sector=np.ones(n_valid, np.int64),
             sr_double=fd * 3.7878, points=np.zeros(n_valid, np.int64), lives=np.full(n_valid, 3, np.int64),
             pos=rng.normal(size=(n_valid, 3)), checkpoints=np.zeros(n_valid, np.int64),
             levelfinish_frame=np.array(n_frames - 5), sr_final=np.array(1234.5))
    np.savez(d / "camera.npz", quarter=np.zeros(n_frames + 1, np.int8), dev=np.zeros(n_frames + 1, np.float32))
    (d / "meta.json").write_text(json.dumps({"dataset_version": 1, "record": name + ".tas", "record_sha256": "cd" * 32,
                                             "level": 1, "levelfinish_frame": n_frames - 5, "sr_double_final": 1234.5,
                                             "source": "C:/Users/someone/TASRecords/x.tas"}), encoding="utf-8")
    return d


def test_starts_pack_round_trip(tmp_path):
    rd = _record_dir(tmp_path)
    sha, pack = T.export_starts_pack(rd, tmp_path / "assets")
    assert pack.name == f"{sha}.npz" and T.file_sha256(pack) == sha
    assert T.export_starts_pack(rd, tmp_path / "assets2")[0] == sha  # deterministic bytes
    assert b"someone" not in pack.read_bytes()  # the dataset's own meta (local paths) is not shipped
    d = T.install_starts_pack(pack, tmp_path / "cache", sha)
    assert d.name == "SR01_test" and (d / ".complete").is_file()
    assert T.install_starts_pack(pack, tmp_path / "cache", sha) == d  # idempotent
    a, b = TasStarts(str(rd)), TasStarts(str(d))
    np.testing.assert_array_equal(a.valid, b.valid)
    np.testing.assert_array_equal(a.frames, b.frames)
    assert (a.levelfinish_frame, a.sr_final) == (b.levelfinish_frame, b.sr_final)
    v = T.valid_start_frames(d, [[0, 100]])
    assert v[0] == 0 and (v[1:] >= 100).all() and set(v[1:]) <= set(a.valid)
    with pytest.raises(ValueError, match="sha256"):
        T.install_starts_pack(pack, tmp_path / "cache2", "0" * 64)


def test_assets_refuse_bad_content(tmp_path):
    rd = _record_dir(tmp_path)
    _, pack = T.export_starts_pack(rd, tmp_path / "assets")
    arrays = dict(np.load(pack, allow_pickle=False))
    # an extra object array: refused before anything is unpickled
    evil = tmp_path / "evil.npz"
    np.savez(evil, **arrays, payload=np.array([object()], dtype=object))
    with pytest.raises(ValueError):
        T.install_starts_pack(evil, tmp_path / "c1", T.file_sha256(evil))
    # wrong dtype for a known array
    bad = dict(arrays, starts_frames_done=arrays["starts_frames_done"].astype(np.float64))
    badf = tmp_path / "bad.npz"
    T.save_npz_deterministic(badf, bad)
    with pytest.raises(ValueError, match="expected"):
        T.install_starts_pack(badf, tmp_path / "c2", T.file_sha256(badf))
    # start frames not increasing
    fd = arrays["starts_frames_done"].copy()
    fd[3] = fd[2]
    T.save_npz_deterministic(badf, dict(arrays, starts_frames_done=fd))
    with pytest.raises(ValueError, match="increasing"):
        T.install_starts_pack(badf, tmp_path / "c3", T.file_sha256(badf))
    # meta with a path-like record name
    meta = json.loads(arrays["meta_json"].tobytes())
    T.save_npz_deterministic(badf, dict(arrays, meta_json=np.frombuffer(
        P.dumps(dict(meta, record="../../etc")), np.uint8).copy()))
    with pytest.raises(P.ProtocolError):
        T.install_starts_pack(badf, tmp_path / "c4", T.file_sha256(badf))


def _path_asset(tmp_path, level=1) -> Path:
    n = 100
    p = tmp_path / "refpaths" / f"L{level:02d}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, points=np.random.default_rng(1).normal(size=(n, 3)).astype(np.float32), sector=np.ones(n, np.uint8),
             t_ms=np.arange(n, dtype=np.float64), meta=np.array(json.dumps({"level": level, "source": "tas"})))
    return p


def test_path_asset_install_and_lookup(tmp_path):
    src = _path_asset(tmp_path)
    sha, stored = T.export_path_asset(src, tmp_path / "assets")
    assert sha == T.file_sha256(src) and stored.name == f"{sha}.npz"
    assert T.find_path_asset(1, sha, [tmp_path / "nowhere", src.parent]) == src
    with pytest.raises(FileNotFoundError):
        T.find_path_asset(1, "0" * 64, [src.parent])
    d = T.install_path_asset(stored, tmp_path / "cache", 1, sha)
    assert (d / "L01.npz").is_file() and T.file_sha256(d / "L01.npz") == sha
    with pytest.raises(ValueError, match="level"):
        T.install_path_asset(stored, tmp_path / "cache", 2, sha)
    with pytest.raises(ValueError, match="sha256"):
        T.install_path_asset(stored, tmp_path / "cache", 1, "0" * 64)
    # a pickled meta object is refused (np.load(allow_pickle=False) raises before unpickling)
    evil = tmp_path / "evil.npz"
    z = dict(np.load(src, allow_pickle=False))
    np.savez(evil, **dict(z, meta=np.array({"level": 1}, dtype=object)))
    with pytest.raises(ValueError):
        T.install_path_asset(evil, tmp_path / "cache", 1, T.file_sha256(evil))


def test_hash_game_files(tmp_path):
    root = tmp_path / "Ballance"
    (root / "Bin").mkdir(parents=True)
    (root / "Bin" / "VxMath.dll").write_bytes(b"vx")
    (root / "base.cmo").write_bytes(b"base")
    (root / "Text").mkdir()
    (root / "Text" / "a.txt").write_bytes(b"a")
    (root / "3D Entities" / "sub").mkdir(parents=True)
    (root / "3D Entities" / "sub" / "x.nmo").write_bytes(b"x")
    h = T.hash_game_files(root, "x87soft")
    assert set(h) == set(P.GAME_FILE_LIST)
    assert h["base.cmo"] == T.file_sha256(root / "base.cmo") and h["Database.tdb"] == "0" * 64
    assert h["Text/"] == P.tree_digest({"a.txt": T.file_sha256(root / "Text" / "a.txt")})
    assert h["3D Entities/"] == P.tree_digest({"sub/x.nmo": T.file_sha256(root / "3D Entities" / "sub" / "x.nmo")})
    h2 = T.hash_game_files(root, "2002")
    assert h2["BuildingBlocks/physics_RT.dll"] == "0" * 64
    P.validate("hello", P.make_message("hello", client_version="0.3.0", platform="linux",
                                       engine={"kind": "x87soft", "sha256": "c" * 64, "version": "1"},
                                       game_files=h, instances=2))
    with pytest.raises(ValueError):
        T.hash_game_files(root, "wine")


def test_local_game_headless_config():
    lg = T.LocalGame(engine="x87soft", binary="bin/SimTool", root="game")
    hc = lg.headless("retail:7", {"legacy_edit": True, "boot_ticks": None})
    assert hc["ball_order"] == "retail:7" and hc["legacy_edit"] is True
    from ballrain.headless import HeadlessConfig
    HeadlessConfig(**hc)
    with pytest.raises(ValueError):
        lg.headless("retail:7", {"binary": "/tmp/evil"})
    with pytest.raises(ValueError):
        T.LocalGame(engine="x87soft", binary="b", root="r", extra_env={"BALLRAIN_HL_BALL_ORDER": "ckid"}).headless("retail:1")
    with pytest.raises(ValueError):
        lg.headless("retail")  # the seed is the task's, never drawn by the launcher


# ---------------------------------------------------------------------------------------------- env from a task

R6 = REPO_ROOT / "runtime" / "m2-train-mac" / "r6"
SR01 = REPO_ROOT / "runtime" / "datasets" / "tas-v1" / "SR01_1_14_457_264"
PATH_DIRS = [REPO_ROOT / "benchmarks" / "refpaths", REPO_ROOT / "runtime" / "refpaths-alt"]


@pytest.mark.skipif(not (R6.is_dir() and any(R6.glob("ckpt_*.pt")) and (SR01 / "starts.npz").is_file()),
                    reason="r6 checkpoint or the SR01 dataset record not here")
def test_env_from_task_matches_the_learner_config(tmp_path):
    torch = pytest.importorskip("torch")
    from dataclasses import asdict
    from ballrain import fakegame
    from ballrain.pool import weights as W
    ck = sorted(R6.glob("ckpt_*.pt"))[-1]
    try:  # the newest file may be mid-pull
        state = torch.load(ck, weights_only=False, map_location="cpu")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{ck.name} not readable: {e}")
    m = W.export_state(state, tmp_path / "w", "r6")
    m = W.read_manifest(tmp_path / "w" / f"{m['policy_sha']}.json")
    try:
        line = T.find_path_asset(1, m["features"]["path_sha256"], PATH_DIRS)
    except FileNotFoundError:
        pytest.skip("the run's reference line is not on this machine")
    ssha, pack = T.export_starts_pack(SR01, tmp_path / "assets")
    psha, stored = T.export_path_asset(line, tmp_path / "assets")
    sdir = T.install_starts_pack(pack, tmp_path / "cache", ssha)
    pdir = T.install_path_asset(stored, tmp_path / "cache", 1, psha)
    valid = T.valid_start_frames(sdir, m["env"]["exclude_starts"])
    meta = json.loads((SR01 / "meta.json").read_text(encoding="utf-8"))
    t = _task(policy_sha=m["policy_sha"], record=SR01.name, record_sha256=meta["record_sha256"],
              start_frame=int(valid[len(valid) // 2]), ball_order_seed=99, assets={"starts": ssha, "path": psha})
    lg = T.LocalGame(engine="x87soft", binary="SimTool", root="Ballance")
    cfg = T.build_env_config(t, m, lg, sdir, pdir, tmp_path / "out")
    # the learner's own configuration, apart from the start (fixed, no random draws) and the local paths
    ec, c = state["env_config"], asdict(cfg)
    norm = lambda v: [norm(x) for x in v] if isinstance(v, (list, tuple)) else v  # noqa: E731
    assert {k for k in ec["features"] if norm(ec["features"][k]) != norm(c["features"][k])} == {"path_dir"}
    assert ec["reward"] == c["reward"]
    assert {k for k in W.ENV_FIELDS if k in ec and norm(ec[k]) != norm(c[k])} == {"start_record"}
    assert c["start_frame"] == t.start_frame and c["curriculum"] is None and c["p_rest"] == c["p_rehearse"] == 0
    assert c["headless"]["ball_order"] == "retail:99" and c["backend"] == "headless"
    e = T.build_env(t, m, lg, sdir, pdir, tmp_path / "out", launcher=fakegame.FakeLauncher(backend="headless"))
    try:
        assert list(e.actions) == m["actions"]["keys"] and e.observation_space.shape == (m["arch"]["n_obs"],)
    finally:
        e.close()
    # mismatching pieces are refused before anything runs
    with pytest.raises(T.TaskMismatch):
        T.build_env_config(_task(policy_sha="0" * 64), m, lg, sdir, pdir, tmp_path / "out")
    with pytest.raises(T.TaskMismatch):
        T.build_env_config(t, m, T.LocalGame(engine="2002", binary="b", root="r"), sdir, pdir, tmp_path / "out")
    # k = 0: from level load (no snapping to the first valid start frame)
    t0 = T.Task(**dict(t.__dict__, start_frame=0))
    assert T.build_env_config(t0, m, lg, sdir, pdir, tmp_path / "out").start_frame is None
    e0 = T.build_env(t0, m, lg, sdir, pdir, tmp_path / "out", launcher=fakegame.FakeLauncher(backend="headless"))
    try:
        assert e0._draw_start() == 0
    finally:
        e0.close()
    excluded = int(m["env"]["exclude_starts"][0][0]) if m["env"]["exclude_starts"] else None
    if excluded is not None:
        bad = T.Task(**dict(t.__dict__, start_frame=excluded + 1))
        with pytest.raises(T.TaskMismatch, match="valid start frame"):
            T.build_env(bad, m, lg, sdir, pdir, tmp_path / "out", launcher=fakegame.FakeLauncher(backend="headless"))
