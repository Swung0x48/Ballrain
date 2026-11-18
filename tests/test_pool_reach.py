"""f7 policies in the pool: the export keeps the reach block's pin and gates its tasks by a capability that names the
block version (docs/f7-native-port.md 6)."""
from dataclasses import asdict

import pytest

pytest.importorskip("torch")

from ballrain.pool import weights as W  # noqa: E402


def f7_state(**feats):
    from test_pool_weights import _state
    from ballrain.features import FeatureConfig
    from ballrain.reach_features import N_REACH
    state, _, _ = _state(n_obs=40 + N_REACH)
    ec = state["env_config"]
    ec["features"] = dict(asdict(FeatureConfig(reach=True, camera="world")), **feats)
    return state


def test_f7_export_pins_the_block_and_requires_its_capability(tmp_path):
    from ballrain.reach_features import reach_version
    m = W.export_state(f7_state(), tmp_path, "r8")
    f = m["features"]
    assert f["reach"] is True and f["version"] == "f7" and f["reach_version"] == reach_version()
    assert m["requires"] == [W.reach_capability(reach_version())]
    cap = m["requires"][0]
    assert "." not in cap and len(cap) <= 32 and cap.startswith("reach_")


def test_f7_export_refuses_another_block_version(tmp_path):
    with pytest.raises(ValueError, match="reach block"):
        W.export_state(f7_state(reach_version="fastreach4.1-block1"), tmp_path, "r8")


def test_f6_export_has_no_reach_fields(tmp_path):
    from test_pool_weights import _state
    state, _, _ = _state()
    m = W.export_state(state, tmp_path, "r6")
    assert "reach" not in m["features"] and "reach_version" not in m["features"] and "requires" not in m
