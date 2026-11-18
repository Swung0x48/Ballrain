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


def edit_rules(args, rules):
    import json
    desc = Path(args.branch_dir) / 'L02.json'
    data = json.loads(desc.read_bytes())
    if rules is None:
        del data['rules']
    else:
        data['rules'] = rules
    desc.write_bytes(json.dumps(data).encode())


def test_rules_change_worker_stall_and_asset_pin(setup, tmp_path):
    from ballrain.branch_rules import DEFAULT_RULES
    from ballrain.train.worker import stall_age
    args, default_cfg = setup
    for overrides, expected_age in [(None, 0), ({'stall_pause_speed': 2}, 30)]:
        edit_rules(args, overrides)
        cfg = train.make_env_config(args, tmp_path)
        assert cfg.features.branch_sha256 != default_cfg.features.branch_sha256
        cfg.start_record = cfg.curriculum = None
        env = BallanceEnv(pickle.loads(pickle.dumps(cfg)))
        try:
            rules = env.branch_rules
            if not overrides:
                assert rules == DEFAULT_RULES
            for branch, age in [(True, expected_age), (False, 30)]:
                ep = SimpleNamespace(stall_branch=branch, mark_s=0, mark_frame=0, move_frame=0)
                assert stall_age(ep, 30, 0, branch, 1.5, 5, 3, 100, 50, rules) == age
        finally:
            env.close()


@pytest.mark.parametrize('raw,match', [({'stall_pasue_speed': 0}, 'unknown branch rule'),
                                        ({'rest_require_support': 1}, 'invalid branch rule'),
                                        ({'stall_speed_source': 'velocity'}, 'invalid branch rule'),
                                        ({'rest_control_repeat': 0}, 'invalid branch rule'),
                                        ({'rest_stop_speed': float('nan')}, 'invalid branch rule'),
                                        ([], 'must be an object')])
def test_invalid_rules_rejected_before_dispatch(setup, tmp_path, raw, match):
    args, _ = setup
    edit_rules(args, raw)
    with pytest.raises(path.RouteError, match=match):
        train.make_env_config(args, tmp_path)


def test_changed_rules_require_capability_and_spawn_reader(setup, tmp_path, monkeypatch):
    from ballrain import env
    args, _ = setup
    edit_rules(args, {'rest_fallback': 'error'})
    cfg = train.make_env_config(args, tmp_path)
    hub = object.__new__(RemoteHub)
    hub.template, hub.stats = {'env': cfg}, {'disconnects': 0}
    hub.log, hub._cond = Mock(), threading.Condition()
    conn = Mock()
    conn.recv.return_value = {'type': 'hello', 'protocol': PROTOCOL, 'branch_assets_version': 1}
    hub._serve(conn)
    assert conn.send.call_args.args[0]['type'] == 'error'
    assert 'branch_rules_version' in conn.send.call_args.args[0]['error']
    wire = pickle.dumps(cfg)
    monkeypatch.delattr(env, '_restore_branch_env_rules_v1')
    with pytest.raises(AttributeError, match='_restore_branch_env_rules_v1'):
        pickle.loads(wire)


@pytest.mark.parametrize('overrides,accept', [({}, False), ({'rest_require_support': False}, True),
                                             ({'rest_require_support': False, 'rest_stop_speed': 0.2}, False),
                                             ({'rest_require_support': False, 'rest_position_tolerance': 0.5}, False)])
def test_rest_rules_at_handover(setup, tmp_path, monkeypatch, overrides, accept):
    import numpy as np
    from ballrain import env as env_module
    from ballrain.branch import load_branch
    args, _ = setup
    edit_rules(args, overrides)
    cfg = train.make_env_config(args, tmp_path)
    cfg.start_record = cfg.curriculum = None
    env = BallanceEnv(cfg)
    try:
        root = Path(env.fx.branch_root)
        main = path.ReferencePath.load(path.asset_path(2, path.resolve_path_dir(cfg.features.path_dir)))
        route = load_branch(main, 2, root, cfg.features.path_dir)
        with np.load(root / 'drive.npz') as drive:
            k, record = int(drive['prefix_frame']), str(drive['record_sha256'])
            station = float(drive['samples'][0, 1])
        ball = {'pos': route.alt.point_at(station) + [1, 0, 0]}
        tracker = Mock(obs=SimpleNamespace(ball=ball), sector=route.sector, flags=SimpleNamespace(ball_off=False))
        tracker.ball_type.return_value = route.ball_type
        p = dict(session=Mock(ended=False, tracker=tracker), start={'record_sha256': record},
                 recorder=SimpleNamespace(meta=SimpleNamespace(start_state={})))
        fx = Mock(branch=route)
        fx.compute.return_value = None, dict(supported=False, path={'branch': True}, vel=np.array([0.4, 0, 0]))
        monkeypatch.setattr(env_module, 'FeatureExtractor', Mock(return_value=fx))
        monkeypatch.setattr(env, '_drive', Mock())
        monkeypatch.setattr(env, '_brake', Mock())
        if accept:
            env._branch_rest(p, k, station)
            assert p['start']['branch_s'] == station
        else:
            with pytest.raises(RuntimeError, match='at rest'):
                env._branch_rest(p, k, station)
        assert env._drive.call_args.args[-1] == env._brake.call_args.args[-1] == env.branch_rules
    finally:
        env.close()
