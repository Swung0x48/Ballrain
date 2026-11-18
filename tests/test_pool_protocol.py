"""ballrain.pool.protocol / reasons: strict control messages, the binary chunk codec, the observation hash chain, and
the pool's pickle-free rule (docs/pool-prototype.md)."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ballrain.pool import protocol as P
from ballrain.pool.reasons import MESSAGES as REASON_TEXT, TOKEN_FATAL, Reason, http_status, payload, render

POOL_DIR = Path(P.__file__).resolve().parent
REPO = POOL_DIR.parents[1]
N_OBS, N_ACT = 208, 9
TID = "0123456789abcdef0123456789abcdef"


# ============================================================================================== helpers

def _task_obs(rows, seed=0):
    return np.random.default_rng(seed).normal(size=(rows, N_OBS)).astype(np.float32)


def _chunk(obs, chain, seq, step0, steps, final=False, seed=1, **over):
    """Encoded chunk seq of a task whose float32 observation rows are obs and chain values chain."""
    rng = np.random.default_rng(seed + seq)
    rows = obs[step0:step0 + steps + 1]
    prev = P.chain_seed(TID) if step0 == 0 else bytes(chain[step0 - 1])
    flags = np.zeros(steps, np.uint8)
    if final:
        flags[-1] = P.FLAG_TERMINATED
    header = dict(task_id=TID, seq=seq, step0=step0, steps=steps, n_obs=N_OBS, n_act=N_ACT, policy_sha="a" * 64,
                  chain_prev=prev.hex(), final=final, end="LevelFinish" if final else None)
    arrays = dict(actions=rng.integers(0, N_ACT, steps), rewards=rng.normal(size=steps),
                  logp=-rng.random(steps) * 3, values=rng.normal(size=steps + 1), flags=flags, obs=rows,
                  obs_hash=chain[step0:step0 + steps + 1])
    header.update(over.pop("header", {}))
    arrays.update(over)
    return P.encode_chunk(header, arrays), rows


def _digests(obs):
    """Stand-in engine digests of the rows (engine_chain row digests are 32 bytes; their content does not matter here)."""
    return [hashlib.sha256(b"engine row" + r.astype("<f4").tobytes()).digest() for r in obs]


@pytest.fixture
def task_chunks():
    obs = _task_obs(2 * 32 + 1)
    chain = P.chain_from_digests(_digests(obs), P.chain_seed(TID))
    b0, r0 = _chunk(obs, chain, 0, 0, 32)
    b1, r1 = _chunk(obs, chain, 1, 32, 32, final=True)
    return obs, chain, (b0, r0), (b1, r1)


def _split(data: bytes):
    """Encoded chunk -> (header dict, body bytes) without checks (to build tampered chunks)."""
    _, _, _, hlen = P._PREAMBLE.unpack_from(data, 0)
    h = json.loads(data[P._PREAMBLE.size:P._PREAMBLE.size + hlen])
    return h, data[P._PREAMBLE.size + hlen:-32]


def _seal(header: dict, body: bytes, raw_header: bytes | None = None, fmt=P.CHUNK_FORMAT) -> bytes:
    """A chunk with a valid trailing digest around any header / body (so the inner checks are what refuses it)."""
    hj = raw_header if raw_header is not None else P.dumps(header)
    out = P._PREAMBLE.pack(P.CHUNK_MAGIC, fmt, 0, len(hj)) + hj + body
    return out + hashlib.sha256(out).digest()


def _sections(header, body):
    out, off = {}, 0
    for s in header["arrays"]:
        out[s["name"]] = (s, body[off:off + s["nbytes"]])
        off += s["nbytes"]
    return out


def _rebuild(header, parts):
    """header / body from an ordered list of (section, stored bytes)."""
    h = dict(header, arrays=[s for s, _ in parts])
    return h, b"".join(b for _, b in parts)


def _bad(data, reason=Reason.BAD_CHUNK, match=None):
    with pytest.raises(P.ProtocolError) as e:
        P.decode_chunk(data)
    assert e.value.reason == reason, e.value
    if match:
        assert re.search(match, str(e.value)), str(e.value)
    return e.value


# ============================================================================================== codec

def test_chunk_round_trip_and_chain(task_chunks):
    obs, chain, (b0, r0), (b1, r1) = task_chunks
    c0, c1 = P.decode_chunk(b0, n_act=N_ACT, n_obs=N_OBS), P.decode_chunk(b1)
    assert c0.steps == 32 and c1.header["final"] and c1.header["end"] == "LevelFinish"
    assert c0.arrays["actions"].dtype == np.uint8 and c0.arrays["obs"].dtype == np.float16
    assert c0.arrays["obs"].shape == (33, N_OBS) and c0.arrays["obs_hash"].shape == (33, 32)
    np.testing.assert_array_equal(c0.arrays["obs"], r0.astype(np.float16))
    assert not c0.arrays["rewards"].flags.writeable
    # continuity
    assert P.check_chain_link(c0, c1) is None
    # chain definition (v2): h(i) = sha256(h(i-1) || engine digest of row i), seeded with the task id
    d = _digests(obs)
    assert bytes(chain[0]) == hashlib.sha256(P.chain_seed(TID) + d[0]).digest()
    assert bytes(chain[1]) == hashlib.sha256(bytes(chain[0]) + d[1]).digest()
    assert P.chain_seed(TID) == hashlib.sha256(b"ballrain-pool/engine-chain/v3\0" + TID.encode()).digest()
    # the float16 copy is checked with a tolerance against the verifier's own float32 features
    assert P.obs_close(c1.arrays["obs"][3], r1[3])
    near = r1[3] + np.float32(5e-5)  # another CPU's rounding: within OBS_TOL beyond the float16 step
    assert P.obs_close(c1.arrays["obs"][3], near)
    far = r1[3].copy()
    far[5] += np.float32(0.05)
    assert not P.obs_close(c1.arrays["obs"][3], far)
    # near 10 the float16 spacing is 2**-7: the client's value just above the rounding midpoint lands one float16 step
    # away from where the verifier's own value (just below it) rounds; tolerated, a few steps are not
    own = np.full(N_OBS, 9.99608, np.float32)
    client = np.float16(own + np.float32(4e-5))
    assert np.float16(own)[0] != client[0] and P.obs_close(client, own)
    assert not P.obs_close(np.float16(own) + np.float16(0.03), own)
    assert P.value_close(1.0, 1.0 + 5e-5, P.REWARD_TOL) and not P.value_close(1.0, 1.001, P.REWARD_TOL)
    assert P.value_close(float("nan"), float("nan"), P.ARC_TOL) and not P.value_close(float("nan"), 1.0, P.ARC_TOL)


def test_chain_link_detects_breaks(task_chunks):
    obs, chain, (b0, _), _ = task_chunks
    c0 = P.decode_chunk(b0)
    other = _task_obs(65, seed=7)
    b1x, _ = _chunk(other, P.chain_from_digests(_digests(other), P.chain_seed(TID)), 1, 32, 32, final=True)
    assert P.check_chain_link(c0, P.decode_chunk(b1x)) is not None
    b2, _ = _chunk(obs, chain, 2, 32, 32, final=True)
    assert P.check_chain_link(c0, P.decode_chunk(b2)) == "not consecutive"
    b1p, _ = _chunk(obs, chain, 1, 32, 32, final=True, header={"policy_sha": "b" * 64})
    assert P.check_chain_link(c0, P.decode_chunk(b1p)) == "policy changed inside a task"


def test_optional_arrays_and_canonical_header(task_chunks):
    obs, chain, _, _ = task_chunks
    b, _ = _chunk(obs, chain, 0, 0, 32, sector=np.ones(32), frames=np.full(32, 2), sr_ms=np.arange(32.0),
                  arc=np.full(32, np.nan), env_rewards=np.zeros(32))
    c = P.decode_chunk(b)
    assert c.arrays["frames"].dtype == np.dtype("<u2") and np.isnan(c.arrays["arc"]).all()
    with pytest.raises(ValueError, match="unknown chunk arrays"):
        _chunk(obs, chain, 0, 0, 32, extra=np.zeros(32))
    with pytest.raises(ValueError, match="shape"):
        _chunk(obs, chain, 0, 0, 32, actions=np.zeros(31))


def test_tampered_bytes_rejected(task_chunks):
    _, _, (b0, _), _ = task_chunks
    # any flipped byte anywhere: the trailing digest
    for i in (0, 5, 20, len(b0) // 2, len(b0) - 40, len(b0) - 1):
        t = bytearray(b0)
        t[i] ^= 0x41
        _bad(bytes(t))
    _bad(b0[:-1])
    _bad(b0 + b"\0")
    _bad(b"")
    _bad("not bytes")


def test_tampered_inner_sections_rejected(task_chunks):
    """Tampering behind a recomputed digest: the inner checks refuse it."""
    _, _, (b0, _), _ = task_chunks
    h, body = _split(b0)
    secs = _sections(h, body)
    order = [n for n in (s["name"] for s in h["arrays"])]
    # a section's content changed (its own sha256 is still the old one)
    s, raw = secs["rewards"]
    raw2 = bytearray(raw)
    raw2[0] ^= 1
    parts = [(secs[n][0], bytes(raw2) if n == "rewards" else secs[n][1]) for n in order]
    _bad(_seal(*_rebuild(h, parts)), match="sha256 mismatch")
    # a section's content changed and its sha256 updated: decodes (the codec is not the authority; replay is)
    s2 = dict(s, sha256=hashlib.sha256(bytes(raw2)).hexdigest())
    parts = [(s2, bytes(raw2)) if n == "rewards" else secs[n] for n in order]
    P.decode_chunk(_seal(*_rebuild(h, parts)))
    # stored size lies
    parts = [(dict(secs[n][0], nbytes=secs[n][0]["nbytes"] + 1), secs[n][1]) if n == "actions" else secs[n]
             for n in order]
    _bad(_seal(*_rebuild(h, parts)), match="add up")
    # a section shorter than its shape (size and sha consistent)
    short = secs["actions"][1][:-1]
    parts = [(dict(secs[n][0], nbytes=len(short), sha256=hashlib.sha256(short).hexdigest()), short)
             if n == "actions" else secs[n] for n in order]
    _bad(_seal(*_rebuild(h, parts)), match="expected")
    # duplicate, missing and unknown sections
    _bad(_seal(*_rebuild(h, [secs[n] for n in order] + [secs["actions"]])), match="twice")
    _bad(_seal(*_rebuild(h, [secs[n] for n in order if n != "logp"])), match="missing sections")
    unk = (dict(secs["actions"][0], name="evil"), secs["actions"][1])
    _bad(_seal(*_rebuild(h, [secs[n] for n in order] + [unk])))
    # obs must be zstd (raw not allowed)
    raw_obs = P.zstd_decompress_exact(secs["obs"][1], 33 * N_OBS * 2)
    parts = [(dict(secs[n][0], codec="raw", nbytes=len(raw_obs)), raw_obs) if n == "obs" else secs[n] for n in order]
    _bad(_seal(*_rebuild(h, parts)), match="codec")
    # trailing body bytes outside every section
    _bad(_seal(h, body + b"x"), match="add up")


def test_zstd_bomb_and_garbage_rejected(task_chunks):
    _, _, (b0, _), _ = task_chunks
    h, body = _split(b0)
    secs = _sections(h, body)
    order = [s["name"] for s in h["arrays"]]
    want = 33 * N_OBS * 2
    for blob in (P.zstd_compress(b"\0" * (want * 50)),  # expands far past the expected size
                 P.zstd_compress(b"\0" * (want - 2)),  # too short
                 P.zstd_compress(b"\0" * want) + P.zstd_compress(b"\0"),  # a second frame
                 b"\x28\xb5\x2f\xfd garbage"):
        parts = [(dict(secs[n][0], nbytes=len(blob)), blob) if n == "obs" else secs[n] for n in order]
        _bad(_seal(*_rebuild(h, parts)))
    with pytest.raises(P.ProtocolError):
        P.zstd_decompress_exact(P.zstd_compress(b"a" * 10_000_000), 100)


def _with_array(h, body, name, arr):
    secs = _sections(h, body)
    raw = np.ascontiguousarray(arr).tobytes()
    s = secs[name][0]
    stored = P.zstd_compress(raw) if s["codec"] == "zstd" else raw
    parts = [(dict(s, nbytes=len(stored), sha256=hashlib.sha256(raw).hexdigest()), stored) if n == name else secs[n]
             for n in (x["name"] for x in h["arrays"])]
    return _seal(*_rebuild(h, parts))


def test_content_checks(task_chunks):
    _, _, (b0, _), (b1, _) = task_chunks
    h, body = _split(b0)
    c = P.decode_chunk(b0)
    acts = c.arrays["actions"].copy()
    acts[3] = N_ACT
    _bad(_with_array(h, body, "actions", acts), match="n_act")
    r = c.arrays["rewards"].copy()
    r[0] = np.nan
    _bad(_with_array(h, body, "rewards", r), match="non-finite")
    v = c.arrays["values"].copy()
    v[-1] = np.inf
    _bad(_with_array(h, body, "values", v), match="non-finite")
    o = c.arrays["obs"].copy()
    o[2, 2] = np.inf
    _bad(_with_array(h, body, "obs", o), match="non-finite")
    lp = c.arrays["logp"].copy()
    lp[5] = 0.5
    e = _bad(_with_array(h, body, "logp", lp), Reason.LOGP_INVALID)
    assert e.payload()["params"]["step"] == 5 and "第 5 步" in e.payload()["message"]
    fl = c.arrays["flags"].copy()
    fl[0] = 4
    _bad(_with_array(h, body, "flags", fl), match="flag bits")
    fl = c.arrays["flags"].copy()
    fl[0] = 3
    _bad(_with_array(h, body, "flags", fl), match="both")
    fl[0] = P.FLAG_TRUNCATED
    _bad(_with_array(h, body, "flags", fl), match="truncation before")
    # final chunk must end its last step; end set exactly on the final one
    h1, body1 = _split(b1)
    f1 = P.decode_chunk(b1).arrays["flags"].copy()
    f1[-1] = 0
    _bad(_with_array(h1, body1, "flags", f1), match="final chunk")
    _bad(_seal(dict(h, end="GameOver"), body), match="end is set")
    with pytest.raises(P.ProtocolError):
        P.decode_chunk(b0, n_act=N_ACT + 1)


def test_header_rejected(task_chunks):
    _, _, (b0, _), _ = task_chunks
    h, body = _split(b0)
    for bad in (dict(h, steps=0), dict(h, steps=P.MAX_CHUNK_STEPS + 1), dict(h, steps=True), dict(h, seq=-1),
                dict(h, task_id="../x"), dict(h, policy_sha="A" * 64), dict(h, protocol=2), dict(h, type="hello"),
                dict(h, end="pwned"), dict(h, extra=1), {k: v for k, v in h.items() if k != "n_obs"},
                dict(h, n_obs=P.MAX_N_OBS + 1), dict(h, final=1)):
        _bad(_seal(bad, body))
    raw = P.dumps(h)
    _bad(_seal(h, body, raw_header=raw[:-1] + b',"seq":0}'), match="duplicate")
    _bad(_seal(h, body, raw_header=raw.replace(b'"seq":0', b'"seq":NaN')))
    _bad(_seal(h, body, raw_header=b"\xff" + raw[1:]))
    _bad(_seal(h, body, fmt=P.CHUNK_FORMAT + 1), match="format")
    bad_magic = bytearray(b0[:-32])
    bad_magic[0:4] = b"XXXX"
    _bad(bytes(bad_magic) + hashlib.sha256(bytes(bad_magic)).digest(), match="magic")


def test_oversize_rejected(task_chunks):
    _, _, (b0, _), _ = task_chunks
    with pytest.raises(P.ProtocolError) as e:
        P.decode_chunk(b0, max_bytes=len(b0) - 1)
    assert e.value.reason == Reason.CHUNK_TOO_LARGE and http_status(e.value.reason) == 413
    _bad(b"\0" * (P.MAX_CHUNK_BYTES + 1), Reason.CHUNK_TOO_LARGE)
    # a header length past the limit (or past the data)
    h, body = _split(b0)
    pre = P._PREAMBLE.pack(P.CHUNK_MAGIC, P.CHUNK_FORMAT, 0, P.MAX_CHUNK_HEADER_BYTES + 1)
    blob = pre + b"{" * 100 + body
    _bad(blob + hashlib.sha256(blob).digest(), match="header length")
    big = dict(h, stats={"frames": 1}, padding="x" * P.MAX_CHUNK_HEADER_BYTES)
    _bad(_seal(big, body))
    # control messages: size and nesting checked before parsing
    with pytest.raises(P.ProtocolError, match="bytes >"):
        P.loads_strict(b" " * (P.MAX_JSON_BYTES + 1))
    with pytest.raises(P.ProtocolError, match="nesting"):
        P.loads_strict(b"[" * 30000 + b"]" * 30000)
    with pytest.raises(P.ProtocolError, match="nesting"):
        P.loads_strict(b'{"a":' * 20 + b"1" + b"}" * 20)
    assert P.loads_strict(b'{"a":"[[[[[[[[[[[[[[[[[[[[[[[["}') == {"a": "[" * 24}
    # too many sections of a valid name is caught by the list bound
    secs = [{"name": "actions", "codec": "raw", "nbytes": 0, "sha256": "0" * 64}] * (len(P.CHUNK_ARRAYS) + 1)
    _bad(_seal(dict(h, arrays=secs), b""))


# ============================================================================================== messages

def _hello(**over):
    d = dict(client_version="0.3.0", platform="windows",
             engine={"kind": "2002", "sha256": "c" * 64, "version": "1.0"},
             game_files={n: "d" * 64 for n in P.GAME_FILE_LIST_2002}, instances=4)
    d.update(over)
    return d


def test_messages_round_trip_and_strictness():
    b = P.encode_message("hello", **_hello())
    m = P.parse_message("hello", b)
    assert m["protocol"] == P.PROTOCOL_VERSION and m["type"] == "hello" and m["instances"] == 4
    d = json.loads(b)
    for bad in (dict(d, instances=True), dict(d, instances=0), dict(d, instances=65), dict(d, instances="4"),
                dict(d, platform="amiga"), dict(d, extra=1), {k: v for k, v in d.items() if k != "engine"},
                dict(d, type="task"), dict(d, client_version="1 0"), dict(d, engine=dict(d["engine"], sha256="x")),
                dict(d, game_files={"C:/Windows/x": "d" * 64}), dict(d, game_files={"../x": "d" * 64}),
                dict(d, client_version="a\x00b"), dict(d, instances=4.0), None, [], "hello"):
        with pytest.raises(P.ProtocolError) as e:
            P.validate("hello", bad)
        assert e.value.reason == Reason.BAD_MESSAGE
    with pytest.raises(P.ProtocolError) as e:
        P.validate("hello", dict(d, protocol=99))
    assert e.value.reason == Reason.PROTOCOL_UNSUPPORTED and "99" in e.value.payload()["message"]
    with pytest.raises(P.ProtocolError):
        P.parse_message("hello", b'{"protocol":1,"protocol":1}')
    with pytest.raises(P.ProtocolError):
        P.parse_message("hello", b.replace(b'"instances":4', b'"instances":Infinity'))
    with pytest.raises(KeyError):
        P.validate("nope", {})


def test_every_message_kind_has_a_valid_example():
    user = {"forum_id": 42, "username": "player", "nickname": "球球"}
    reason = payload(Reason.NO_TASK)
    task = {"task_id": TID, "kind": "sample", "run": "r6", "policy_sha": "a" * 64, "level": 1, "record": "SR01_x",
            "record_sha256": "b" * 64, "start_frame": 100, "ball_order_seed": 7, "sample_seed": 12345, "engine": "x87soft",
            "chunk_steps": 256, "max_steps": 4096, "max_frames": 60000, "deadline_s": 3600, "issued_at": 1,
            "greedy": False, "assets": {"starts": "c" * 64, "path": "d" * 64}}
    stats = {"steps_accepted": 1, "steps_pending": 0, "steps_verified": 1, "chunks_accepted": 1, "chunks_rejected": 0}
    examples = {
        "login_start": dict(client_version="0.3.0"),
        "login_started": dict(login_id="L" * 16, poll_secret="s" * 32, browser_url="https://example.invalid/pool/login", user_code="ABCD-EF23",
                              expires_in_s=600, poll_s=2),
        "login_poll": dict(login_id="L" * 16, poll_secret="s" * 32),
        "login_status": dict(status="ok", token="t" * 43, user=user),
        "hello": _hello(),
        "hello_ok": dict(session_id="S" * 16, max_instances=8, heartbeat_s=30, task_poll_s=5, user=user,
                         leaderboard=True, server_time=1),
        "task": dict(task=task),
        "no_task": dict(reason=reason, retry_s=30),
        "task_end": dict(task_id="t" * 32, chunks=2, reason="failed: start: the game did not connect"),
        "task_end_ok": dict(task_id="t" * 32, state="closed"),
        "chunk_result": dict(task_id=TID, seq=0, status="accepted", verify="pending", steps=256),
        "heartbeat": dict(session_id="S" * 16, running=True, steps_per_s=200.0,
                          instances=[{"slot": 0, "state": "running", "steps": 10, "task_id": TID}]),
        "heartbeat_ok": dict(server_time=1, max_instances=8),
        "me": dict(user=user, leaderboard=True, token_state="active", stats=stats),
        "me_update": dict(leaderboard=False),
        "leaderboard": dict(updated=1, entries=[{"rank": 1, "nickname": "球球", "steps_verified": 10}]),
        "error": dict(reason=payload(Reason.TOKEN_INVALID)),
    }
    assert set(examples) == set(P.MESSAGES)
    for kind, fields in examples.items():
        m = P.parse_message(kind, P.encode_message(kind, **fields))
        assert m["type"] == kind
    # the user object never carries an email
    with pytest.raises(P.ProtocolError):
        P.validate("me", dict(P.make_message("me", **examples["me"]), user=dict(user, email="a@b.c")))


# ============================================================================================== reasons

def test_reasons_render_in_chinese():
    assert set(REASON_TEXT) == set(Reason)
    for r in Reason:
        msg = render(r)
        assert msg and re.search(r"[\u4e00-\u9fff]", msg), r
        assert "{" not in msg and "}" not in msg
    p = payload(Reason.REPLAY_MISMATCH, task_id="t1", chunk=3, step=1234)
    assert p == {"code": "replay_mismatch", "message": "任务 t1 的第 3 段数据在第 1234 步重放结果与提交不一致。",
                 "params": {"task_id": "t1", "chunk": 3, "step": 1234}}
    P.check(P.REASON, p)
    assert "注册未满 30 天（已注册 3 天）" in render(Reason.ACCOUNT_TOO_NEW, min_days=30, days=3)
    assert "管理员" in render(Reason.TOKEN_REVOKED, note="测试")
    # parameters cannot inject format fields or control characters
    s = render(Reason.TOKEN_REVOKED, note="{__class__}\x1b[2J" + "x" * 500)
    assert "\x1b" not in s and len(s) < 300 and "{__class__}" in s
    assert render(Reason.CHUNK_TOO_LARGE) .count("?") == 2
    assert Reason.TOKEN_REVOKED_REPLAY in TOKEN_FATAL and http_status(Reason.RATE_LIMITED) == 429
    assert http_status(Reason.BAD_CHUNK) == 400 and http_status(Reason.TOKEN_REVOKED) == 403


# ============================================================================================== game files

def test_tree_digest_is_canonical():
    a = P.tree_digest({"A\\b.txt": "1" * 64, "c.txt": "2" * 64})
    b = P.tree_digest({"c.txt": "2" * 64, "a/B.TXT": "1" * 64})
    assert a == b and len(a) == 64


# ============================================================================================== pickle-free

def test_pool_sources_are_pickle_free():
    """No pickle, no allow_pickle=True, no eval / exec, no module-level torch import in ballrain/pool and tools/pool."""
    files = sorted(POOL_DIR.glob("*.py")) + sorted((REPO / "tools" / "pool").glob("**/*.py"))
    assert files
    for f in files:
        src = f.read_text(encoding="utf-8")
        code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
        assert not re.search(r"^\s*(import|from)\s+(c?pickle|dill|cloudpickle|marshal|shelve)\b", code, re.M), f
        assert not re.search(r"pickle\.(loads?|Unpickler)", code), f
        assert "allow_pickle=True" not in code, f
        assert not re.search(r"(?<![\w.])(eval|exec)\(", code), f
        assert not re.search(r"^(import|from)\s+torch\b", code, re.M), f"{f}: torch at module level"
        for m in re.finditer(r"torch\.load\(", code):
            assert f.name == "weights.py", f"{f}: torch.load outside weights.export_checkpoint"


def test_client_modules_do_not_import_torch():
    code = ("import sys; import ballrain.pool, ballrain.pool.protocol, ballrain.pool.reasons, ballrain.pool.weights, "
            "ballrain.pool.task; print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_decoder_never_unpickles(task_chunks, monkeypatch):
    import pickle
    calls = []
    monkeypatch.setattr(pickle, "loads", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(pickle, "load", lambda *a, **k: calls.append(a))
    _, _, (b0, _), _ = task_chunks
    P.decode_chunk(b0)
    # a pickle payload in place of a section is just bytes: sha / size checks refuse it
    h, body = _split(b0)
    evil = pickle.dumps(np.zeros(3))
    secs = _sections(h, body)
    parts = [(dict(secs[n][0], nbytes=len(evil), sha256=hashlib.sha256(evil).hexdigest()), evil) if n == "actions"
             else secs[n] for n in (s["name"] for s in h["arrays"])]
    _bad(_seal(*_rebuild(h, parts)))
    assert not calls


@pytest.mark.skipif(P._zstd()[0] != "stdlib", reason="stdlib compression.zstd (Python >= 3.14) not available")
def test_zstandard_fallback_matches_stdlib():
    zstandard = pytest.importorskip("zstandard")
    from compression import zstd
    data = np.arange(5000, dtype="<f2").tobytes()
    blob = zstd.compress(data)
    with zstandard.ZstdDecompressor().stream_reader(blob, read_across_frames=False) as r:
        assert r.read(len(data) + 1) == data
