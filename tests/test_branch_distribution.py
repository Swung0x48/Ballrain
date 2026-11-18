"""Offline learner -> bridge -> spawned EnvConfig and checkpoint contract."""
import pickle
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ballrain import path
from ballrain.env import BallanceEnv
from ballrain.train.remote import RemoteBridge, RemoteHub, PROTOCOL
from scripts import train_m2 as train


@pytest.fixture
def setup(tmp_path):
    source = tmp_path / "learner-only"
    shutil.copytree(path.REFPATH_DIR.parent / "refpaths-alt/L02-stone", source)
    args = train.parse(["--run", "unit", "--level", "2", "--path-dir", "benchmarks/refpaths-v2",
                        "--branch-dir", str(source)])
    return args, train.make_env_config(args, tmp_path)


def test_remote_no_branch_checkout(setup, tmp_path, monkeypatch):
    args, cfg = setup
    worker_repo = tmp_path / "worker"
    main = worker_repo / args.path_dir / "L02.npz"
    main.parent.mkdir(parents=True)
    shutil.copyfile(path.REPO_ROOT / args.path_dir / main.name, main)
    monkeypatch.setattr(path, "REPO_ROOT", worker_repo)
    Path(args.branch_dir).rename(tmp_path / "unavailable")
    bridge = RemoteBridge(("localhost", 1), b"k", ["unit"], "retail", lambda _: tmp_path)
    cfg.start_record = cfg.curriculum = None
    bridge.template = pickle.loads(pickle.dumps(dict(env=cfg, seed=1, worker={})))
    bridge.wid_base = 100
    spec = pickle.loads(pickle.dumps(bridge._spec(0, "unit", 0)))
    from ballrain.train.worker import Worker
    env = Worker(spec, None, None).env
    try:
        meta = path.ReferencePath.load(main).meta
        monkeypatch.setattr(path, "level_anchors", lambda _: meta)
        tracker = SimpleNamespace(level_no=2, level=SimpleNamespace(version=1), arrays=Mock(rows=lambda _: 0))
        env.fx.reference_path(tracker)
        assert env.fx.branch is not None
        assert (Path(env.fx.branch_root) / "drive.npz").read_bytes() == cfg.branch_assets["drive_npz"]
    finally:
        env.close()


def test_checkpoint_resume(setup, tmp_path):
    args, cfg = setup
    torch = pytest.importorskip("torch")
    dest = tmp_path / "ck.pt"
    torch.save(train.checkpoint_config(cfg), dest)
    ck = torch.load(dest, weights_only=False)
    assert ck['branch_assets'] == cfg.branch_assets and 'branch_assets' not in ck['env_config']
    assert train.make_env_config(args, tmp_path, ck).branch_assets == cfg.branch_assets
    desc = Path(args.branch_dir) / 'L02.json'
    desc.write_bytes(desc.read_bytes() + b' ')
    with pytest.raises(path.RouteError, match="checkpoint pin"):
        train.make_env_config(args, tmp_path, ck)
    desc.unlink()
    args.branch_dir = None
    assert train.checkpoint_config(train.make_env_config(args, tmp_path, ck)) == ck


@pytest.mark.parametrize('key', ['desc_json', 'line_npz', 'drive_npz'])
def test_corrupt_payload_refused(setup, key):
    _, cfg = setup
    cfg.branch_assets[key] += b'!'
    with pytest.raises(path.RouteError, match='sha256'):
        BallanceEnv(cfg)


def test_old_bridge_refused_before_config(setup):
    hub = object.__new__(RemoteHub)
    hub.template = {'env': setup[1]}
    hub.stats = {'disconnects': 0}
    hub.log, hub._cond = Mock(), threading.Condition()
    conn = Mock()
    conn.recv.return_value = {'type': 'hello', 'protocol': PROTOCOL}
    hub._serve(conn)
    conn.send.assert_called_once_with({'type': 'error', 'error': 'distributed branch requires branch_assets_version=1'})


def test_old_spawned_worker_cannot_unpickle_branch(setup, monkeypatch):
    from ballrain import env
    wire = pickle.dumps(setup[1])
    monkeypatch.delattr(env, '_restore_branch_env_v1')
    with pytest.raises(AttributeError, match='_restore_branch_env_v1'):
        pickle.loads(wire)


def test_live_pool_state_keeps_checkpoint_contract(setup):
    import numpy as np
    from ballrain.pool.learner_link import state_from_params
    cfg = setup[1]
    state = state_from_params({'env': cfg}, {'model': {'pi.0.weight': np.zeros((9, 32))}, 'version': 1, 'rms': {}})
    assert state['branch_assets'] == cfg.branch_assets
    assert state['env_config'] == train.checkpoint_config(cfg)['env_config']
