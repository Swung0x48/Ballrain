"""tools/export_model.py: an f7 checkpoint (the reach block) gets the sanitized .pt and a README without the plain
export (the pool format and the native client compute f6 only); a refusal of the plain export writes nothing."""
import importlib.util
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ballrain.env import EnvConfig  # noqa: E402
from ballrain.features import FeatureConfig, make_layout  # noqa: E402
from ballrain.train import ppo  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def tool(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("export_model", ROOT / "tools" / "export_model.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "REPO", tmp_path / "repo")
    monkeypatch.setattr(m, "_private_names", lambda: [])
    return m


def checkpoint(path, fc, activation="tanh"):
    n_obs = sum(k for _, k in make_layout(fc))
    model = ppo.ActorCritic(n_obs, 9, (16,), "tanh")
    ec = asdict(EnvConfig(level=1, world_actions=True, action_repeat=2, start_record="SR01_test", features=fc))
    ec["out_root"] = "runtime/x"
    torch.save({"model": model.state_dict(), "rms": {"mean": np.zeros(n_obs), "var": np.ones(n_obs), "count": 1.0},
                "n_obs": n_obs, "n_act": 9, "args": {"activation": activation, "hidden": "16"}, "env_config": ec,
                "counters": {"update": 1, "steps": 1}, "opt": {}}, path)


def test_f7_checkpoint_exports_without_plain(tmp_path, monkeypatch):
    m = tool(tmp_path, monkeypatch)
    ck = tmp_path / "f7.pt"
    checkpoint(ck, FeatureConfig(camera="world", reach=True))
    m.main([str(ck), "m7", "--run", "r-test", "--update", "1"])
    out = tmp_path / "repo" / "models" / "m7"
    # plus the reference line the model observes when its path_dir resolves here (bundled since the refpaths change)
    assert sorted(p.name for p in out.iterdir() if p.name != "refpaths") == ["README.md", "m7.pt"]
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "plain/" in readme and "f7" in readme and FeatureConfig(reach=True).reach_version in readme
    saved = torch.load(out / "m7.pt", weights_only=False)
    assert saved["env_config"]["features"]["reach"] is True and "opt" not in saved


def test_f6_checkpoint_exports_plain(tmp_path, monkeypatch):
    m = tool(tmp_path, monkeypatch)
    ck = tmp_path / "f6.pt"
    checkpoint(ck, FeatureConfig(camera="world"))
    m.main([str(ck), "m6", "--run", "r-test", "--update", "1"])
    out = tmp_path / "repo" / "models" / "m6"
    assert (out / "m6.pt").is_file() and len(list((out / "plain").glob("*.npz"))) == 1


def test_a_refused_plain_export_writes_nothing(tmp_path, monkeypatch):
    m = tool(tmp_path, monkeypatch)
    ck = tmp_path / "bad.pt"
    checkpoint(ck, FeatureConfig(camera="world"), activation="gelu")  # export_state refuses the activation
    with pytest.raises(SystemExit, match="nothing written"):
        m.main([str(ck), "mb", "--run", "r-test", "--update", "1"])
    assert not (tmp_path / "repo" / "models" / "mb").exists()
