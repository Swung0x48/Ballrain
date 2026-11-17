"""ballrain.pool.weights: checkpoint -> plain npz + manifest, the checked loader, and the numpy policy forward against
the torch ActorCritic (synthetic always; the r6 learner checkpoint on recorded SR01 observations when present)."""
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ballrain.env import EnvConfig  # noqa: E402
from ballrain.instances import REPO_ROOT  # noqa: E402
from ballrain.pool import weights as W  # noqa: E402
from ballrain.train import ppo  # noqa: E402


def _state(hidden=(64, 32), act="tanh", n_obs=40, n_act=9, seed=0, **env):
    torch.manual_seed(seed)
    model = ppo.ActorCritic(n_obs, n_act, hidden, act)
    with torch.no_grad():  # move off the orthogonal init (small output gain) so the probabilities are not ~uniform
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    rng = np.random.default_rng(seed)
    rms = {"mean": rng.normal(size=n_obs), "var": rng.random(n_obs) * 4 + 1e-3, "count": 1e6}
    ec = asdict(EnvConfig(level=1, world_actions=True, action_repeat=2, start_record="SR01_test",
                          out_root="C:/Users/someone/secret", **env))
    args = {"activation": act, "hidden": ",".join(map(str, hidden)), "chunk": 256, "gamma": 0.9985,
            "route_probes": '{"a": [1, 2, 3, 4]}', "gates": "", "split_extra": "[[1, 2, 3.5]]", "init_from": "/home/x/y.pt",
            "stall_progress": 99.0}
    return {"model": model.state_dict(), "rms": rms, "n_obs": n_obs, "n_act": n_act, "args": args, "env_config": ec,
            "counters": {"update": 12, "steps": 3456}}, model, rms


def _torch_probs(model, rms, raw):
    x = ppo.normalize(raw, rms)
    with torch.no_grad():
        logits, v = model(torch.as_tensor(x, dtype=torch.float32))
        return torch.softmax(logits, -1).numpy(), torch.log_softmax(logits, -1).numpy(), v.numpy()


def _export_load(state, tmp_path):
    m = W.export_state(state, tmp_path, "r-test")
    npz, mp = tmp_path / f"{m['policy_sha']}.npz", tmp_path / f"{m['policy_sha']}.json"
    return m, npz, mp, W.load_policy(npz, mp, expect_sha=m["policy_sha"])


@pytest.mark.parametrize("act", ["tanh", "relu"])
def test_numpy_policy_matches_torch(tmp_path, act):
    state, model, rms = _state(act=act)
    m, npz, mp, pol = _export_load(state, tmp_path)
    raw = np.random.default_rng(5).normal(size=(500, 40)) * 3 + rms["mean"]
    p_t, lp_t, v_t = _torch_probs(model, rms, raw)
    x = pol.normalize(raw)
    np.testing.assert_array_equal(x, ppo.normalize(raw, rms))
    lp = pol.log_probs(x)
    assert np.abs(np.exp(lp) - p_t).max() < 1e-5
    assert np.abs(lp - lp_t).max() < 1e-4
    assert np.abs(pol.value(x) - v_t).max() < 1e-4
    assert (np.argmax(lp, -1) == np.argmax(lp_t, -1)).mean() > 0.99
    # act: a valid action with its own log-prob; greedy = argmax
    rng = np.random.default_rng(0)
    a, logp, v = pol.act(x[0], rng)
    assert 0 <= a < 9 and logp == pytest.approx(float(lp[0, a]), abs=1e-5) and v == pytest.approx(float(v_t[0]), abs=1e-4)
    assert pol.act(x[1], rng, greedy=True)[0] == int(np.argmax(lp[1]))
    counts = np.bincount([pol.act(x[2], rng)[0] for _ in range(4000)], minlength=9) / 4000
    assert np.abs(counts - np.exp(lp[2])).max() < 0.04


def test_manifest_contents_and_public_safety(tmp_path):
    state, _, _ = _state()
    m, npz, mp, pol = _export_load(state, tmp_path)
    assert m["arch"] == {"kind": "mlp_actor_critic", "n_obs": 40, "n_act": 9, "hidden": [64, 32], "activation": "tanh"}
    assert m["actions"]["keys"] == [0, 1, 2, 4, 8, 5, 9, 6, 10] and m["actions"]["names"][5] == "up+left"
    assert m["env"]["start_record"] == "SR01_test" and m["env"]["action_repeat"] == 2
    assert m["features"]["path_dir"] is None
    assert m["rollout"]["route_probes"] == {"a": [1, 2, 3, 4]} and m["rollout"]["split_extra"] == [[1, 2, 3.5]]
    assert m["rollout"]["gates"] == {} and m["rollout"]["chunk"] == 256
    assert m["rollout"]["stall_progress"] == 5.0  # train_m2 does not pass it to WorkerSpec: the default is what ran
    assert set(m["engine_options"]) == set(W.HEADLESS_FIELDS)
    text = mp.read_text(encoding="utf-8")
    for leak in ("someone", "secret", "/home/", "out_root", "init_from", "C:/"):
        assert leak not in text
    with pytest.raises(ValueError, match="path"):
        W.assert_public({"a": ["ok", "D:\\repos\\x"]})
    with pytest.raises(ValueError, match="path"):
        W.assert_public({"a": "/Users/x"})
    # the weights file is a plain npz: only .npy members, no pickled objects, deterministic bytes
    with zipfile.ZipFile(npz) as z:
        assert all(n.endswith(".npy") for n in z.namelist())
    with np.load(npz, allow_pickle=False) as z:
        assert set(z.files) == set(m["arrays"])
    m2 = W.export_state(state, tmp_path / "again", "r-test")
    assert m2["policy_sha"] == m["policy_sha"] and m2["npz_sha256"] == m["npz_sha256"]


def test_loader_refuses_tampering(tmp_path):
    state, _, _ = _state()
    m, npz, mp, _ = _export_load(state, tmp_path)
    with pytest.raises(ValueError, match="policy_sha"):
        W.load_policy(npz, mp, expect_sha="0" * 64)
    # a byte changed in the file
    data = bytearray(npz.read_bytes())
    data[len(data) // 2] ^= 1
    bad = tmp_path / "bad.npz"
    bad.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256"):
        W.load_policy(bad, mp)
    # a consistent file and manifest whose arrays differ from the policy_sha
    arrays = dict(np.load(npz, allow_pickle=False))
    arrays["pi.0.bias"] = arrays["pi.0.bias"] + 1
    from ballrain.pool.task import save_npz_deterministic, file_sha256
    alt = tmp_path / "alt.npz"
    save_npz_deterministic(alt, arrays)
    m_alt = dict(m, npz_sha256=file_sha256(alt), npz_bytes=alt.stat().st_size)
    with pytest.raises(ValueError, match="policy_sha"):
        W.load_policy(alt, m_alt)
    # the manifest's configuration edited: policy_sha no longer matches
    m_cfg = json.loads(mp.read_text(encoding="utf-8"))
    m_cfg["env"]["action_repeat"] = 1
    with pytest.raises(ValueError, match="policy_sha"):
        W.load_policy(npz, m_cfg)
    # an object array (pickled inside the npy) is refused without unpickling
    obj = dict(arrays, pi_extra=np.array([{"x": 1}], dtype=object))
    objf = tmp_path / "obj.npz"
    np.savez(objf, **obj)
    m_obj = dict(m, npz_sha256=file_sha256(objf), npz_bytes=objf.stat().st_size,
                 arrays=dict(m["arrays"], pi_extra={"dtype": "|O", "shape": [1]}))
    with pytest.raises(ValueError):
        W.load_policy(objf, m_obj)
    with pytest.raises(ValueError, match="object"):
        save_npz_deterministic(tmp_path / "o2.npz", obj)
    # wrong shapes / non-finite values / unknown architecture
    arrays2 = dict(np.load(npz, allow_pickle=False))
    arrays2["v.0.weight"] = arrays2["v.0.weight"].copy()
    arrays2["v.0.weight"][0, 0] = np.nan
    nanf = tmp_path / "nan.npz"
    save_npz_deterministic(nanf, arrays2)
    with pytest.raises(ValueError, match="non-finite"):
        W.load_policy(nanf, dict(m, npz_sha256=file_sha256(nanf), npz_bytes=nanf.stat().st_size))
    with pytest.raises(ValueError, match="architecture"):
        W.load_policy(npz, dict(m, arch=dict(m["arch"], activation="gelu")))
    with pytest.raises(ValueError):
        W.read_manifest(b'{"format": "something-else"}')


def test_export_refuses_unsupported(tmp_path):
    state, _, _ = _state(esc_action=True)
    with pytest.raises(ValueError, match="Esc"):
        W.export_state(state, tmp_path, "r")
    state, _, _ = _state()
    state["model"]["extra.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected"):
        W.export_state(state, tmp_path, "r")


# ---------------------------------------------------------------------------------------------- the real learner

R6 = REPO_ROOT / "runtime" / "m2-train-mac" / "r6"
SR01 = REPO_ROOT / "runtime" / "datasets" / "tas-v1" / "SR01_1_14_457_264"


def _recorded_features(feature_cfg: dict, n: int = 300) -> np.ndarray:
    """Training observations of the SR01 TAS record (GameTracker + FeatureExtractor in order, as
    tests/test_dataset_regression.py feeds the tracker), every 50th frame."""
    from ballrain import dataset, features as FX, protocol as BP
    from ballrain.state import GameTracker
    fc = dict(feature_cfg)
    for k in ("far_waypoints", "ref_speed_at"):
        fc[k] = tuple(fc.get(k) or ())
    fx = FX.FeatureExtractor(FX.FeatureConfig(**fc))
    t = GameTracker(dataset.level_of(SR01.name))
    lm, _ = BP.parse_level_message((SR01 / "level.msg").read_bytes())
    t.merge_level(lm)
    if (SR01 / "level.msg.inc").is_file():
        for msg in BP.parse_level_stream((SR01 / "level.msg.inc").read_bytes())[0]:
            t.merge_level(msg)
    out = []
    for i, o in enumerate(dataset.load_observations(SR01 / "raw.npz")):
        t.update(o)
        if i % 50 == 0 and i > 400:
            try:
                v, _ = fx.compute(t)
            except Exception:  # noqa: BLE001  (frames without control / route: skip)
                continue
            out.append(v)
            if len(out) >= n:
                break
    return np.asarray(out, np.float32)


@pytest.mark.skipif(not (R6.is_dir() and any(R6.glob("ckpt_*.pt")) and (SR01 / "raw.npz").is_file()),
                    reason="r6 checkpoint or the SR01 dataset record not here")
def test_r6_checkpoint_on_recorded_observations(tmp_path):
    ck = sorted(R6.glob("ckpt_*.pt"))[-1]
    try:  # the learner's own file (trusted); the newest may be mid-pull
        state = torch.load(ck, weights_only=False, map_location="cpu")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{ck.name} not readable: {e}")
    from ballrain.pool.task import find_path_asset
    fcfg = dict(state["env_config"]["features"])
    try:
        line = find_path_asset(1, fcfg["path_sha256"], [REPO_ROOT / "benchmarks" / "refpaths",
                                                         REPO_ROOT / "runtime" / "refpaths-alt"])
    except FileNotFoundError:
        pytest.skip("the run's reference line is not on this machine")
    fcfg["path_dir"] = str(line.parent)
    m = W.export_state(state, tmp_path, "r6")
    pol = W.load_policy(tmp_path / f"{m['policy_sha']}.npz", tmp_path / f"{m['policy_sha']}.json", m["policy_sha"])
    a = state["args"]
    model = ppo.ActorCritic(state["n_obs"], state["n_act"], tuple(int(h) for h in str(a.get("hidden", "256,256")).split(",")),
                            a.get("activation", "tanh"))
    model.load_state_dict(state["model"])
    raw = _recorded_features(fcfg)
    assert raw.shape[0] >= 50 and raw.shape[1] == state["n_obs"] == m["arch"]["n_obs"]
    p_t, lp_t, v_t = _torch_probs(model, state["rms"], raw)
    x = pol.normalize(raw)
    lp = pol.log_probs(x)
    assert np.abs(np.exp(lp) - p_t).max() < 1e-5
    assert np.abs(pol.value(x) - v_t).max() < 1e-3
