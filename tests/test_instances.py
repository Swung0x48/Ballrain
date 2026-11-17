"""Launcher plumbing tests. Nothing here starts the game or writes into runtime/: fake copies live in
tmp_path, and process handling is exercised on a copy of cmd.exe named FakePlayer.exe."""
import hashlib
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from ballrain import instances as ins
from ballrain import tas

win = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")

CFG = (
    "# Configuration File for Mod: TAS Support - 0.3.12\n\n# Miscellaneous\nMisc {\n\n"
    "\t# Enable TAS Features\n\tB Enable 1\n\n"
    "\t# Skip render until the given frame to speed up TAS playing\n\tI SkipRenderUntil 0\n\n"
    "\t# Automatically exit game when TAS playing finished\n\tB ExitOnFinish 0\n\n"
    "\t# Automatically load TAS record on game startup\n\tS AutoLoadTAS \n\n"
    "\t# Automatically load given level on game startup\n\tI AutoLoadLevel 0\n\n"
    "\t# Compatibility mode for older TAS records (Restart game to take effect)\n\tB LegacyMode 0\n\n"
    "\t# Key for stop playing\n\tK StopKey 61\n\n}\n\n"
).encode()


@dataclass(frozen=True)
class FakeCopy(ins.GameCopy):
    @property
    def exe(self) -> Path:
        return self.bin_dir / "FakePlayer.exe"


def make_copy(root: Path, cls=FakeCopy, mods=()):
    (root / "Bin").mkdir(parents=True)
    (root / "BuildingBlocks").mkdir()
    (root / "BuildingBlocks" / "physics_RT.dll").write_bytes(b"fake physics")
    (root / "ModLoader" / "Configs").mkdir(parents=True)
    (root / "ModLoader" / "Mods").mkdir()
    (root / "ModLoader" / "TASRecords").mkdir()
    (root / "ModLoader" / "Configs" / "TASSupport.cfg").write_bytes(CFG)
    for m in mods:
        (root / "ModLoader" / "Mods" / m).write_bytes(b"")
    if sys.platform == "win32":
        shutil.copyfile(Path(os.environ["SystemRoot"]) / "System32" / "cmd.exe", root / "Bin" / "FakePlayer.exe")
    return cls(root.resolve())


@pytest.fixture
def fake(tmp_path, monkeypatch):
    c = make_copy(tmp_path / "rt" / "fake")
    monkeypatch.setattr(ins, "PHYSICS_RT_SHA256", hashlib.sha256(b"fake physics").hexdigest())
    return c


def test_physics_hash_of_real_copies():
    for name in (ins.REFERENCE_COPY, ins.TRAINING_COPY):
        p = ins.runtime_dir() / name
        if not (p / "BuildingBlocks" / "physics_RT.dll").exists():
            pytest.skip(f"{p} not present")
        assert ins.verify_physics(ins.get_copy(name)) == ins.PHYSICS_RT_SHA256
    assert ins.PHYSICS_RT_SHA256.startswith("e72e4afc")


def test_physics_mismatch(fake):
    (fake.physics_dll).write_bytes(b"other")
    with pytest.raises(ins.InstanceError):
        ins.verify_physics(fake)


def test_get_copy_rejects_escape(tmp_path):
    (tmp_path / "rt").mkdir()
    with pytest.raises(ins.InstanceError):
        ins.get_copy("..", tmp_path / "rt")
    with pytest.raises(ins.InstanceError):
        ins.get_copy("missing", tmp_path / "rt")


def test_foreign_scripts(fake, tmp_path):
    assert ins.find_foreign_script_paths(fake) == []
    inside = f'@echo off\r\nset "BMMO_TAS_TRACE={fake.root}\\trace.txt"\r\ncd /d "{fake.bin_dir}"\r\nstart "" Player.exe\r\n'
    (fake.root / "ok.cmd").write_text(inside)
    assert ins.find_foreign_script_paths(fake) == []
    ins.verify_no_foreign_scripts(fake)
    other = tmp_path / "rt" / "other" / "Bin"
    (fake.bin_dir / "bad.bat").write_text(f'cd /d "{other}"\r\nset "X=C:\\Elsewhere\\t.txt"\r\n')
    bad = ins.find_foreign_script_paths(fake)
    assert [p for _, p in bad] == [str(other), "C:\\Elsewhere\\t.txt"]
    with pytest.raises(ins.InstanceError):
        ins.verify_no_foreign_scripts(fake)
    # a sibling whose name merely starts with the copy's name is still foreign
    (fake.bin_dir / "bad.bat").write_text(f'cd /d "{fake.root}2\\Bin"\r\n')
    assert len(ins.find_foreign_script_paths(fake)) == 1


def test_edit_cfg_preserves_everything_else(fake):
    vals = ins.set_tas_config(fake, AutoLoadTAS="SR01_1_14_457_264", AutoLoadLevel=1, LegacyMode=1,
                              ExitOnFinish=1, SkipRenderUntil=999999)
    assert vals == {"AutoLoadTAS": "SR01_1_14_457_264", "AutoLoadLevel": "1", "LegacyMode": "1",
                    "ExitOnFinish": "1", "SkipRenderUntil": "999999"}
    new = fake.tas_cfg.read_bytes()
    expect = (CFG.replace(b"\tS AutoLoadTAS \n", b"\tS AutoLoadTAS SR01_1_14_457_264\n")
              .replace(b"I AutoLoadLevel 0", b"I AutoLoadLevel 1")
              .replace(b"B LegacyMode 0", b"B LegacyMode 1")
              .replace(b"B ExitOnFinish 0", b"B ExitOnFinish 1")
              .replace(b"I SkipRenderUntil 0", b"I SkipRenderUntil 999999"))
    assert new == expect


def test_edit_cfg_crlf_and_errors():
    crlf = CFG.replace(b"\n", b"\r\n")
    out = ins.edit_cfg_bytes(crlf, {"AutoLoadLevel": 7})
    assert out == crlf.replace(b"I AutoLoadLevel 0\r\n", b"I AutoLoadLevel 7\r\n")
    with pytest.raises(ins.InstanceError):
        ins.edit_cfg_bytes(CFG.replace(b"\tI AutoLoadLevel 0\n", b""), {"AutoLoadLevel": 1})
    with pytest.raises(ins.InstanceError):
        ins.edit_cfg_bytes(CFG.replace(b"I AutoLoadLevel", b"S AutoLoadLevel"), {"AutoLoadLevel": 1})
    with pytest.raises(KeyError):
        ins.edit_cfg_bytes(CFG, {"Record": 1})
    with pytest.raises(ValueError):
        ins.edit_cfg_bytes(CFG, {"LegacyMode": 2})
    with pytest.raises(ValueError):
        ins.edit_cfg_bytes(CFG, {"AutoLoadTAS": "a\nb"})


def test_launch_script_text(fake):
    env = {k: None for k in ins.MANAGED_VARS}
    env["BMMO_TAS_TRACE"] = str(fake.root / "t.txt")
    s = ins.launch_script_text(fake, env)
    lines = s.split("\r\n")
    assert lines[0] == "@echo off"
    for k in ins.MANAGED_VARS:
        assert (f'set "{k}={fake.root / "t.txt"}"' if k == "BMMO_TAS_TRACE" else f'set "{k}="') in lines
    assert lines[-3:] == [f'cd /d "{fake.bin_dir}"', 'start "" FakePlayer.exe', ""]
    with pytest.raises(ins.InstanceError):
        ins.launch_script_text(fake, {"BALLRAIN_TRACE": "C:\\a%b"})


def test_full_env_strips_inherited(monkeypatch):
    monkeypatch.setenv("BMMO_TAS_CORE", "1,2")
    monkeypatch.setenv("BALLRAIN_LOADS", "9")
    env = ins.full_env({"BALLRAIN_MODE": "replay", "BALLRAIN_LOADS": None})
    assert "BMMO_TAS_CORE" not in env and "BALLRAIN_LOADS" not in env
    assert env["BALLRAIN_MODE"] == "replay"


def test_ballrain_env_and_paths(tmp_path):
    p = ins.ballrain_output_paths(tmp_path, 3)
    assert p[0]["trace"] == tmp_path.resolve() / "trace.txt"
    assert p[2]["framelog"] == tmp_path.resolve() / "framelog.bin.3"
    assert p[1]["events"].name == "events.txt.2"
    env = ins.ballrain_env(tmp_path / "f.bin", 7, tmp_path, loads=3, skip_render=False)
    assert env["BALLRAIN_MODE"] == "replay" and env["BALLRAIN_LEVEL"] == "7" and env["BALLRAIN_LOADS"] == "3"
    assert env["BALLRAIN_SKIP_RENDER"] is None and env["BALLRAIN_EXIT"] == "1"
    assert env["BMMO_TAS_TRACE"] is None
    assert all(Path(env[k]).is_absolute() for k in ("BALLRAIN_FRAMES", "BALLRAIN_TRACE", "BALLRAIN_FRAMELOG", "BALLRAIN_EVENTS"))
    with pytest.raises(ValueError):
        ins.ballrain_env(tmp_path / "f.bin", 14, tmp_path)


def test_install_record(fake, tmp_path):
    f = tas.make_frames([1000 / 132] * 4, [0, 1, 2, 3])
    src = tas.save(tmp_path / "X_1.tas", f)
    assert ins.install_record(fake, src) == "X_1"
    assert (fake.records_dir / "X_1.tas").read_bytes() == src.read_bytes()
    assert ins.install_record(fake, src) == "X_1"  # same content: fine
    tas.save(src, f[:2])
    with pytest.raises(ins.InstanceError):
        ins.install_record(fake, src)
    (tmp_path / "bad.tas").write_bytes(b"\x00" * 12)
    with pytest.raises(tas.TASFormatError):
        ins.install_record(fake, tmp_path / "bad.tas")


def test_run_copies_refuse_wrong_mod_layout(tmp_path, monkeypatch):
    rt = tmp_path / "rt"
    make_copy(rt / "ref", ins.GameCopy)
    make_copy(rt / "train", ins.GameCopy, mods=["TASSupport.bmodp"])
    make_copy(rt / "train2", ins.GameCopy)
    f = tas.save(tmp_path / "a.tas", tas.make_frames([1.0], [0]))
    with pytest.raises(ins.InstanceError, match="TASSupport"):
        ins.run_reference(f, 1, tmp_path / "t.txt", copy="ref", runtime=rt)
    with pytest.raises(ins.InstanceError, match="TASSupport"):
        ins.run_ballrain(f, 1, tmp_path / "o", copy="train", runtime=rt)
    with pytest.raises(ins.InstanceError, match="BallrainIO"):
        ins.run_ballrain(f, 1, tmp_path / "o", copy="train2", runtime=rt)


def test_refuses_when_already_running(fake, monkeypatch):
    monkeypatch.setattr(ins, "running_in_copy", lambda c: [4242])
    with pytest.raises(ins.InstanceError, match="already running"):
        ins.preflight(fake)
    called = []
    with pytest.raises(ins.InstanceError, match="already running"):
        ins.run_copy(fake, {}, {}, 5, recipe=lambda c, e: called.append(1) or 0)
    assert called == []


def test_recipes_registry():
    assert ins.LAUNCH_RECIPE is ins.launch_explorer_cmd
    assert set(ins.LAUNCH_RECIPES) == {"explorer_cmd", "direct"}


@win
def test_process_handle_verifies_path():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(ins.InstanceError):
            ins.ProcessHandle(p.pid, Path(sys.executable).with_name("not-python.exe"))
        assert p.poll() is None  # the wrong-path handle never touched it
    finally:
        p.kill()
        p.wait()


@win
def test_run_copy_normal_exit_direct_recipe(fake):
    out = fake.root / "out.txt"
    out.write_text("stale")
    keep = []

    def recipe(copy, env):  # direct creation, cwd = Bin, writes the 'trace' then exits
        assert env["BALLRAIN_TRACE"] == str(out)
        q = subprocess.Popen([str(copy.exe), "/c", "echo %BALLRAIN_TRACE%>" + str(out)], cwd=str(copy.bin_dir),
                             env=ins.full_env(env), stdin=subprocess.DEVNULL)
        keep.append(q)
        return q.pid

    env = {k: None for k in ins.MANAGED_VARS}
    env["BALLRAIN_TRACE"] = str(out)
    r = ins.run_copy(fake, env, {"trace": out}, timeout=30, recipe=recipe)
    assert r.ok and not r.timed_out and not r.killed and r.exit_code == 0
    assert out.read_text().strip() == str(out)
    assert (fake.root / "out.txt.prev").read_text() == "stale"


@win
def test_run_copy_timeout_kills_only_own_pid(fake):
    out = fake.root / "never.txt"

    def recipe(copy, env):
        q = subprocess.Popen([str(copy.exe), "/c", "for /l %i in (0,0,1) do rem"], cwd=str(copy.bin_dir),
                             stdin=subprocess.DEVNULL)
        keep.append(q)
        return q.pid

    keep = []
    t = time.monotonic()
    r = ins.run_copy(fake, {}, {"trace": out}, timeout=1.5, recipe=recipe)
    assert r.timed_out and r.killed and not r.ok
    assert r.missing_outputs == ["trace"]
    assert time.monotonic() - t < 15
    assert ins.running_in_copy(fake) == []


@win
def test_direct_recipe_starts_in_bin(fake, monkeypatch):
    # launch_direct itself, on the fake exe (cmd.exe copy exits at once on empty stdin)
    pid = ins.launch_direct(fake, {"BALLRAIN_MODE": "replay"})
    with ins.ProcessHandle(pid, fake.exe) as h:
        assert h.wait(10)


def test_windows_visible_switch(monkeypatch):
    monkeypatch.delenv(ins.VISIBLE_VAR, raising=False)
    assert ins.windows_visible() is False
    for v, want in (("1", True), ("yes", True), ("0", False), ("", False), (" ", False)):
        monkeypatch.setenv(ins.VISIBLE_VAR, v)
        assert ins.windows_visible() is want
    monkeypatch.setenv(ins.VISIBLE_VAR, "1")
    assert ins.windows_visible(False) is False and ins.windows_visible(True) is True
    assert ins.VISIBLE_VAR not in ins.full_env({})  # launcher-only: never passed to the game


@win
def test_direct_recipe_hides_windows_by_default(fake, monkeypatch):
    seen = []

    class FakePopen:
        def __init__(self, args, **kw):
            seen.append(kw)
            self.pid = 4242

    monkeypatch.setattr(ins.subprocess, "Popen", FakePopen)
    monkeypatch.delenv(ins.VISIBLE_VAR, raising=False)
    try:
        ins.launch_direct(fake, {})
        si = seen[-1]["startupinfo"]
        assert si.dwFlags & ins.subprocess.STARTF_USESHOWWINDOW and si.wShowWindow == ins.SW_HIDE
        assert seen[-1]["cwd"] == str(fake.bin_dir)
        ins.launch_direct(fake, {}, visible=True)
        assert seen[-1]["startupinfo"] is None
        monkeypatch.setenv(ins.VISIBLE_VAR, "1")
        ins.launch_direct(fake, {})
        assert seen[-1]["startupinfo"] is None
    finally:
        ins._LAUNCHED.pop(4242, None)


def test_set_splash(fake, monkeypatch):
    assert ins.splash_state(fake) == "missing"
    with pytest.raises(ins.InstanceError, match="missing"):
        ins.set_splash(fake, False)
    (fake.bin_dir / "splash.bmp").write_bytes(b"BM")
    assert ins.set_splash(fake, False) == "off"
    assert not (fake.bin_dir / "splash.bmp").exists() and (fake.bin_dir / "splash.bmp.off").read_bytes() == b"BM"
    assert ins.set_splash(fake, False) == "off"
    assert ins.set_splash(fake, True) == "on"
    monkeypatch.setattr(ins, "running_in_copy", lambda c: [4242])
    with pytest.raises(ins.InstanceError, match="running"):
        ins.set_splash(fake, False)
    assert ins.splash_state(fake) == "on"


def _cmd_recipe(command_for, keep):
    """Stub recipe: run the fake exe (a cmd.exe copy) with a /c command built from env, cwd = Bin."""
    def recipe(copy, env):
        q = subprocess.Popen(f'"{copy.exe}" /c {command_for(copy, env)}', cwd=str(copy.bin_dir),
                             env=ins.full_env(env), stdin=subprocess.DEVNULL)
        keep.append(q)
        return q.pid
    return recipe


@win
def test_run_reference_end_to_end(fake, tmp_path, monkeypatch):
    (fake.mods_dir / "TASSupport.bmodp").write_bytes(b"")
    fake.modloader_log.write_text("old log")
    os.utime(fake.modloader_log, (1, 1))
    monkeypatch.setattr(ins, "get_copy", lambda name, runtime=None: fake)
    f = tas.make_frames([1000 / 264] * 10, [0] * 10)
    src = tas.save(tmp_path / "SRX_1.tas", f)
    trace = tmp_path / "out" / "trace-SRX.txt"
    keep = []
    recipe = _cmd_recipe(lambda c, e: f'echo 105 Ball_Wood> "{e["BMMO_TAS_TRACE"]}" & echo new log> "{c.modloader_log}"', keep)
    r = ins.run_reference(src, 3, trace, timeout=30, recipe=recipe)
    assert r["result"].ok and r["trace"] == trace.resolve()
    assert trace.read_text().strip() == "105 Ball_Wood"
    assert r["modloader_log"].read_text().strip() == "new log"
    assert r["cfg"] == {"AutoLoadTAS": "SRX_1", "AutoLoadLevel": "3", "LegacyMode": "1",
                        "ExitOnFinish": "1", "SkipRenderUntil": "999999"}
    assert (fake.records_dir / "SRX_1.tas").read_bytes() == src.read_bytes()

    # a run that writes nothing: the trace is reported missing and the stale log is not passed off as new
    fake.modloader_log.write_text("old log")
    os.utime(fake.modloader_log, (1, 1))
    r = ins.run_reference(src, 3, trace, timeout=30, recipe=_cmd_recipe(lambda c, e: "rem", keep))
    assert r["result"].missing_outputs == ["trace"] and not r["result"].ok
    assert r["modloader_log"] is None
    assert (trace.with_name(trace.name + ".prev")).read_text().strip() == "105 Ball_Wood"


@win
def test_run_ballrain_end_to_end(fake, tmp_path, monkeypatch):
    (fake.mods_dir / "BallrainIO.bmodp").write_bytes(b"")
    monkeypatch.setattr(ins, "get_copy", lambda name, runtime=None: fake)
    f = tas.make_frames([1000 / 132] * 6, [0, 0, 1, 1, 8, 8])
    src = tas.save(tmp_path / "case.tas", f)
    out = tmp_path / "run"
    keep = []

    def writes_load1_only(c, e):
        assert e["BALLRAIN_LOADS"] == "2" and e["BALLRAIN_LEVEL"] == "2" and e["BALLRAIN_SKIP_RENDER"] == "1"
        assert e["BALLRAIN_EXIT"] == "1" and e["BALLRAIN_MODE"] == "replay"
        return " & ".join(f'echo x> "{e[k]}"' for k in ("BALLRAIN_TRACE", "BALLRAIN_FRAMELOG", "BALLRAIN_EVENTS"))

    r = ins.run_ballrain(src, 2, out, loads=2, timeout=30, recipe=_cmd_recipe(writes_load1_only, keep))
    assert (out / "frames.bin").read_bytes() == f.tobytes()
    assert r["frames"] == (out / "frames.bin").resolve()
    assert [d["trace"].name for d in r["loads"]] == ["trace.txt", "trace.txt.2"]
    res = r["result"]
    assert not res.ok and not res.timed_out
    assert sorted(res.missing_outputs) == ["events.2", "framelog.2", "trace.2"]


def test_create_training_copies_and_sync_mod(tmp_path, monkeypatch):
    rt = tmp_path / "rt"
    src = make_copy(rt / "train-00", cls=ins.GameCopy, mods=("BallrainIO.bmodp",))
    (src.modloader_dir / "ModLoader.log").write_text("log")
    (src.root / ins.LAUNCH_SCRIPT_NAME).write_text("@echo off\r\n")
    (src.root / "BuildingBlocks" / "extra.dll").write_bytes(b"abc")
    monkeypatch.setattr(ins, "PHYSICS_RT_SHA256", hashlib.sha256(b"fake physics").hexdigest())
    res = ins.create_training_copies(3, runtime=rt)
    assert res == {"train-01": "created", "train-02": "created"}
    c2 = ins.get_copy("train-02", rt)
    assert not (c2.modloader_dir / "ModLoader.log").exists() and not (c2.root / ins.LAUNCH_SCRIPT_NAME).exists()
    assert ins.verify_copy_matches(src, c2, hash_all=True) == []
    assert ins.create_training_copies(3, runtime=rt) == {"train-01": "exists", "train-02": "exists"}
    (c2.physics_dll).write_bytes(b"other physics")
    with pytest.raises(ins.InstanceError):
        ins.create_training_copies(3, runtime=rt)
    (c2.physics_dll).write_bytes(b"fake physics")
    (c2.root / "BuildingBlocks" / "extra.dll").write_bytes(b"abd")
    with pytest.raises(ins.InstanceError):  # binaries are hash-checked against the source
        ins.create_training_copies(3, runtime=rt)
    (c2.root / "BuildingBlocks" / "extra.dll").write_bytes(b"abc")
    (c2.mods_dir / "BallrainIO.bmodp").write_bytes(b"old")  # the mod is not compared: it is synced
    assert ins.create_training_copies(3, runtime=rt)["train-02"] == "exists"
    assert len(set(ins.mod_hashes(["train-00", "train-02"], rt).values())) == 2
    build = tmp_path / "BallrainIO.bmodp"
    build.write_bytes(b"new build")
    out = ins.sync_mod(build, ["train-00", "train-01", "train-02"], runtime=rt)
    assert set(out.values()) == {hashlib.sha256(b"new build").hexdigest()}
    assert ins.create_training_copies(3, runtime=rt)["train-02"] == "exists"


def test_training_copies_on_disk():
    names = [n for n in ins.training_copy_names(8) if (ins.runtime_dir() / n / "BuildingBlocks" / "physics_RT.dll").exists()]
    if not names:
        pytest.skip("no training copies")
    for n in names:
        gc = ins.get_copy(n)
        assert ins.verify_physics(gc) == ins.PHYSICS_RT_SHA256
        assert not gc.has_tassupport() and gc.has_ballrainio()


def test_agent_vars_are_managed():
    for k in ("BALLRAIN_SHM", "BALLRAIN_PARENT_PID", "BALLRAIN_LEVELMSG", "BALLRAIN_PREFIX", "BALLRAIN_SR_TAIL"):
        assert k in ins.MANAGED_VARS
