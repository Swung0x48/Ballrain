"""Tests that launch the real game copies. Skipped unless BALLRAIN_LIVE=1 is set.

They run the G0 sanity block of scripts/m0_gates.py (reference copy with the real TASSupport, then
BallrainIO in the training copy) and write under <runtime>/m0-results like the driver does.
"""
import os
import sys

import pytest

from ballrain.instances import REPO_ROOT

pytestmark = pytest.mark.skipif(os.environ.get("BALLRAIN_LIVE") != "1",
                                reason="launches the game; set BALLRAIN_LIVE=1 to run")


def test_g0_sanity_live():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import m0_gates  # needs runtime/ref and runtime/train-00

    assert m0_gates.main(["--gates", "G0"]) == 0
    results = m0_gates.json.loads(m0_gates.RESULTS.read_text(encoding="utf-8"))
    g0 = results["summary"]["G0"]
    assert g0["pass"] == len(m0_gates.G0_RECORDS) and g0["fail"] == g0["error"] == 0, g0
