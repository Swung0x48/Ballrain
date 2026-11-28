"""EnvConfig.collect_hold_arrows against ballrain.fakegame: off, the auto-collect macro injects exactly what it did before
(Esc, idle while paused, Esc, every macro frame without arrows); on, the frames the game simulates (the pause Esc's
frames before the toggle, the unpause Esc's frames after it) carry the arrows of the step before the macro, and no paused
frame does. No game is launched."""
import json
import sys

import numpy as np
import pytest

from ballrain import fakegame, tas
from ballrain.env import BallanceEnv, EnvConfig

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="named shared memory / events need Windows")

UP = 1


def _run(tmp_path, **kw):
    """Hold UP through one auto-collect and on to the end of the run; the exported frames and the macro's info."""
    env = BallanceEnv(EnvConfig(level=2, out_root=str(tmp_path), action_repeat=4, mode="hs", auto_collect=True, **kw),
                      launcher=fakegame.FakeLauncher())
    env.reset()
    macro, steps = None, 0
    for _ in range(2000):
        _, _, term, trunc, info = env.step(UP)
        steps += 1
        if macro is None and info.get("auto_collect", {}).get("result") == "done":
            macro = info["auto_collect"]
        if term or trunc:
            break
    env.close()
    meta = json.loads(open(info["run_meta"], encoding="utf-8").read())
    frames = tas.load(tmp_path / "train-00" / meta["run_id"] / meta["tas_path"])
    return frames, macro, meta


def test_off_is_the_macro_as_before(tmp_path):
    fa, ma, _ = _run(tmp_path / "a")
    fb, mb, _ = _run(tmp_path / "b", collect_hold_arrows=False)
    assert ma is not None and ma["unpause"] == "unpaused"
    assert np.array_equal(fa["keys"], fb["keys"]) and np.array_equal(fa["delta"], fb["delta"])
    esc = np.flatnonzero(fa["keys"] & tas.KEY_ESC)
    assert len(esc) >= 2 and len(esc) % 2 == 0  # one pause + one unpause Esc per collect
    n = EnvConfig().esc_frames
    for e in esc:  # every Esc action: Esc alone, then n - 1 frames with no key
        assert int(fa["keys"][e]) == tas.KEY_ESC and not fa["keys"][e + 1:e + n].any()
    for p, u in zip(esc[::2], esc[1::2]):  # no arrow on any macro frame
        assert not (fa["keys"][p:u + n] & ~np.uint32(tas.KEY_ESC)).any()


def test_on_holds_the_arrows_on_the_simulated_esc_frames_only(tmp_path):
    f0, _, _ = _run(tmp_path / "off")
    f1, m1, _ = _run(tmp_path / "on", collect_hold_arrows=True)
    assert m1 is not None and m1["pause"] == "paused" and m1["unpause"] == "unpaused"
    n = EnvConfig().esc_frames
    esc = np.flatnonzero(f1["keys"] & tas.KEY_ESC)
    assert len(esc) >= 2 and len(esc) % 2 == 0
    for e0, e1 in zip(esc[::2], esc[1::2]):
        up = int(f1["keys"][e0 - 1]) & ~tas.KEY_ESC  # the arrows of the step before the macro
        assert up
        assert int(f1["keys"][e0]) == tas.KEY_ESC | up and int(f1["keys"][e0 + 1]) == up  # before the pause toggle
        assert not f1["keys"][e0 + 2:e1].any()  # paused frames: no key, as off
        assert int(f1["keys"][e1]) == tas.KEY_ESC and int(f1["keys"][e1 + 1]) == 0  # unpause press, still paused
        assert all(int(k) == up for k in f1["keys"][e1 + 2:e1 + n])  # simulated frames after the toggle: arrows held
    # everything before the first macro is the same as with the flag off
    assert np.array_equal(f0["keys"][:esc[0]], f1["keys"][:esc[0]])
