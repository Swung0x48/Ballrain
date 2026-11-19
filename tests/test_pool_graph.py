"""policy_graph_v1: the network as data (ballrain/pool/weights.py build_graph / graph_forward; native PolicyGraph)."""
import json

import numpy as np
import pytest
import torch

from ballrain.pool import weights as W
from ballrain.train import ppo
from test_pool_weights import _state


def _ball_type_state(seed=3):
    state, model, rms = _state(n_obs=40, seed=seed)
    torch.manual_seed(seed)
    bt = ppo.ActorCritic(40, 9, (64, 32), "tanh", actor_heads="ball_type")
    with torch.no_grad():
        for p in bt.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    state["model"] = bt.state_dict()
    state["args"] = dict(state["args"], actor_heads="ball_type")
    return state, bt, rms


@pytest.mark.parametrize("ball_type", [False, True])
def test_graph_matches_the_arch_path(tmp_path, ball_type):
    state, model, rms = _ball_type_state() if ball_type else _state(n_obs=40)
    m = W.export_state(state, tmp_path, "r-test")
    assert m["graph"] == W.build_graph(m["arch"])
    npz = tmp_path / f"{m['policy_sha']}.npz"
    pol = W.load_policy(npz, m)
    legacy = W.load_policy(npz, {k: v for k, v in m.items() if k != "graph"})  # a manifest of before the graph
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2, (500, 40)).astype(np.float32)
    x[:, 1:4] = 0
    for i in range(500):  # every head: no bit, then each of the three
        if i % 4:
            x[i, i % 4] = 1.0
    assert np.array_equal(pol.logits(x), legacy.logits(x))
    assert np.array_equal(pol.value(x), legacy.value(x))
    with torch.no_grad():
        lg, v = model(torch.as_tensor(x))
    assert np.abs(pol.logits(x) - lg.numpy()).max() < 1e-4


def test_graph_bound_to_the_identity(tmp_path):
    state, _, _ = _ball_type_state()
    m = W.export_state(state, tmp_path, "r-test")
    npz = tmp_path / f"{m['policy_sha']}.npz"
    g = json.loads(json.dumps(m["graph"]))
    sel = next(n for n in g["nodes"] if n["op"] == "select_linear")
    sel["heads"] = sel["heads"][::-1]  # the heads swapped: same arrays, another network
    with pytest.raises(ValueError, match="architecture"):
        W.load_policy(npz, dict(m, graph=g))
    # an arch of kind "graph" pins its graph by graph_sha256 (an identity field)
    arch = dict(m["arch"], kind="graph", graph_sha256="0" * 64)
    with pytest.raises(ValueError, match="architecture"):
        W.load_policy(npz, dict(m, arch=arch))


def test_check_graph_refuses_bad_graphs(tmp_path):
    state, _, _ = _state(n_obs=40)
    m = W.export_state(state, tmp_path, "r-test")
    arrays = dict(np.load(tmp_path / f"{m['policy_sha']}.npz"))
    g = m["graph"]
    assert set(W.check_graph(g, arrays, 40, 9)) | {"rms_mean", "rms_var"} == set(arrays)
    for bad, msg in [({**g, "version": 2}, "version"),
                     ({**g, "nodes": [{"op": "softplus", "in": "x", "out": "y"}]}, "unknown op"),
                     ({**g, "nodes": [{"op": "linear", "in": "nope", "out": "y", "w": "pi.0.weight", "b": "pi.0.bias"}]},
                      "before defined"),
                     ({**g, "nodes": [{"op": "linear", "in": "x", "out": "y", "w": "pi.2.weight", "b": "pi.2.bias"}]},
                      "shape"),
                     ({**g, "logits": "x"}, "output")]:
        with pytest.raises(ValueError, match=msg):
            W.check_graph(bad, arrays, 40, 9)
