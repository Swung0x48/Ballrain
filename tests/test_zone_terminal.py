"""Terminal zones (WorkerSpec.zones entries with "terminal": true): the life ends as a failure where the zone fires
(end "zone_terminal", terminated, the zone penalty booked). Without the key the worker is bit-identical to the code
before the option (ballrain at commit BASE, the parent of the option, exported with git archive), on the fake game. Uses tests/slow_fail_run.py."""
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BASE = "b4632001ad2ddcf40ec109c8f3acc3442e23f7af"  # main before the terminal option


def _run(root: Path, record: Path, out: Path, n: int, extra: dict) -> dict:
    p = subprocess.run([sys.executable, str(REPO / "tests" / "slow_fail_run.py"), str(root), str(record), str(out), str(n),
                        json.dumps(extra)], capture_output=True, text=True, encoding="utf-8", timeout=900, cwd=str(REPO),
                       env={**os.environ, "BALLRAIN_TORCH_POLICY": "1"})
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def record(tmp_path_factory):
    from ballrain.pool import dev
    return dev.fake_record(tmp_path_factory.mktemp("zt-record"))


@pytest.fixture(scope="module")
def head_root(tmp_path_factory):
    """ballrain at BASE (before the terminal option) in a temporary root."""
    root = tmp_path_factory.mktemp("zt-head")
    tar = subprocess.run(["git", "archive", "--format=tar", BASE, "ballrain"], cwd=str(REPO), capture_output=True,
                         check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(root)
    return root


@pytest.mark.skipif(sys.platform != "win32", reason="ballrain.fakegame's shared memory needs Windows")
def test_off_identical_to_head_and_terminal_ends_the_life(record, head_root, tmp_path):
    n = 20
    far = {"zones": {"far": {"box": [1e8, 1e8 + 1, 1e8, 1e8 + 1], "y_below": -1e8, "penalty": 3.0}}}
    head = _run(head_root, record, tmp_path / "head", n, far)
    off = _run(REPO, record, tmp_path / "off", n, far)
    assert off == head  # zones without "terminal": every chunk and episode identical to the code before the option
    never = _run(REPO, record, tmp_path / "never", n, {"zones": {"far": {**far["zones"]["far"], "terminal": True}}})
    assert never == off  # terminal but never entered
    every = {"zones": {"all": {"box": [-1e9, 1e9, -1e9, 1e9], "y_below": 1e9, "penalty": 3.0, "terminal": True}}}
    on = _run(REPO, record, tmp_path / "on", n, every)
    eps = [e for c in on["chunks"] for e in c["episodes"]]
    zt = [e for e in eps if e["end"] == "zone_terminal"]
    assert zt and len(zt) >= len(eps) - 2  # every life ends in the zone (a chunk-boundary one may be open)
    assert all(e["terms"].get("zone_penalty") == -3.0 and not e["finish"] and e["steps"] <= 2 for e in zt)
