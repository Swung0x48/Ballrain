"""f8 contact / rail blocks (ballrain.features: FeatureConfig(version="f8"); docs/observations.md f8): f6 / f7 layouts,
values and serialisation unchanged; the blocks appended after the reach block; contact merging and slot order;
left / right rails on a hand-built double rail; workers on older code refuse an f8 template; recorded data (SR01 /
SR02 TAS records) with the r8 configuration: the f8 vector starts with the f7 vector bit for bit."""
import hashlib
import pickle
import pickletools
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pytest

from ballrain import dataset, features as FX, fastreach as FR, instances, protocol as P
from ballrain.env import EnvConfig
from ballrain.fakegame import box_mesh
from ballrain.path import SYNTHETIC_LEVEL_MARKER
from ballrain.state import GameTracker

R8 = dict(camera="world", path_dir="benchmarks/refpaths-v2", far_waypoints=(20.0, 30.0, 45.0),
          ref_speed_at=(0.0, 15.0, 30.0), hs_block=False)
DATA = instances.runtime_dir() / "datasets" / f"tas-v{dataset.DATASET_VERSION}"
# FeatureConfig's fields before f8: f8 adds none (pool manifests, checkpoints, the remote template and the native
# parser, which refuses unknown fields, see exactly the f7 configuration)
FIELDS = ["version", "camera", "pos_scale", "vel_scale", "angvel_scale", "absolute_position", "height_grid",
          "height_spacing", "height_clip", "height_up", "fan_dirs", "fan_step", "fan_max", "fan_rule", "fan_step_drop",
          "fan_drop", "fan_up", "mesh_edge", "rail_radius", "path_source", "path_dir", "path_sha256", "branch_dir",
          "path_lateral_scale", "path_back", "path_ahead", "path_margin", "path_reacquire", "waypoints",
          "waypoint_spacing", "far_waypoints", "ref_speed_at", "bodies", "body_radius", "dynamic_geometry", "counters",
          "hs_block", "follower_sources", "checkpoint_frames_scale", "clip", "reach", "reach_version", "branch_sha256"]
N_F8 = FX.CONTACT_SLOTS * len(FX.CONTACT_SLOT_FIELDS) + len(FX.CONTACT_AGG_FIELDS) + 2 * len(FX.RAIL_SIDE_FIELDS)


def test_f6_f7_unchanged_and_f8_appends_after_the_reach_block():
    assert [f.name for f in fields(FX.FeatureConfig)] == FIELDS
    assert FX.layout_size(FX.FeatureConfig()) == 204
    f7, f8 = FX.FeatureConfig(**R8, reach=True), FX.FeatureConfig(**R8, reach=True, version="f8")
    assert FX.layout_size(f7) == 267 and f7.version == "f7" and not f7.contacts  # the r8 run
    assert f8.version == "f8" and f8.contacts and f8.reach_version == f7.reach_version
    l7, l8 = FX.make_layout(f7), FX.make_layout(f8)
    assert l8[:len(l7)] == l7 and l8[len(l7):] == [("contacts", 32), ("contact_agg", 5), ("rails_lr", 12)]
    assert N_F8 == 49 and FX.layout_size(f8) == 316
    # f8 without the reach block: f6 + the blocks
    f8n = FX.FeatureConfig(**R8, version="f8")
    assert f8n.version == "f8" and f8n.reach_version is None and FX.layout_size(f8n) == 208 + 49
    # labels survive the dicts workers and checkpoints carry; an f7 label without reach is still f6
    assert FX.FeatureConfig(**asdict(f8)).version == "f8" and FX.FeatureConfig(**asdict(f7)).version == "f7"
    assert FX.FeatureConfig(**dict(asdict(f7), reach=False)).version == "f6"
    assert FX.FeatureConfig().to_dict() == {k: v for k, v in asdict(FX.FeatureConfig()).items()
                                            if k not in ("branch_dir", "branch_sha256")}


class _OldCode(pickle.Unpickler):
    """An unpickler that resolves globals like code from before f8 (no _restore_features_f8)."""

    def find_class(self, module, name):
        if module == "ballrain.features" and name == "_restore_features_f8":
            raise AttributeError(f"Can't get attribute {name!r} on <module {module!r}>")
        return super().find_class(module, name)


def _globals(data: bytes) -> set:
    out, strings = set(), []
    for op, arg, _ in pickletools.genops(data):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
            strings.append(arg)
        elif op.name == "STACK_GLOBAL":
            out.add(tuple(strings[-2:]))
        elif op.name == "GLOBAL":
            out.add(tuple(arg.split(" ")))
    return out


def test_workers_on_older_code_refuse_an_f8_template():
    """The learner's worker template (EnvConfig, pickled to spawned and remote workers) with f8 features names a
    restore function older code lacks: unpickling fails instead of building the f7 layout from the same fields.
    f6 / f7 configurations pickle exactly as before (plain dataclass state)."""
    import io
    f7 = FX.FeatureConfig(**R8, reach=True)
    f8 = FX.FeatureConfig(**R8, reach=True, version="f8")
    for env in (EnvConfig(features=f8), EnvConfig(features=f8, branch_assets=None)):
        data = pickle.dumps({"env": env, "seed": 1})
        assert ("ballrain.features", "_restore_features_f8") in _globals(data)
        back = pickle.loads(data)["env"].features
        assert back == f8 and back.contacts and FX.layout_size(back) == 316
        with pytest.raises(AttributeError, match="_restore_features_f8"):
            _OldCode(io.BytesIO(data)).load()
    import copy
    assert copy.deepcopy(f8) == f8
    d7 = pickle.dumps(EnvConfig(features=f7))
    assert ("ballrain.features", "_restore_features_f8") not in _globals(d7)
    assert _OldCode(io.BytesIO(d7)).load().features == f7


def test_remote_hub_refuses_bridges_without_features_f8():
    """The remote hub sends an f8 template only to bridges whose hello says features_f8 (older bridges and the
    pool's learner link get an error and never see the template)."""
    import threading
    from multiprocessing.connection import Client

    from ballrain.train.remote import PROTOCOL, RemoteHub
    import queue
    for contacts, ok in ((True, False), (True, True), (False, False)):
        fc = FX.FeatureConfig(**R8, version="f8" if contacts else "f6")
        hub = RemoteHub(("127.0.0.1", 0), b"k", queue.Queue(), {"env": EnvConfig(features=fc), "worker": {}},
                        log=lambda *_: None)
        try:
            c = Client(hub._listener.address, authkey=b"k")
            hello = {"type": "hello", "protocol": PROTOCOL, "host": "h", "session": 1, "n_workers": 1}
            if ok:
                hello["features_f8"] = True
            c.send(hello)
            m = c.recv()
            if contacts and not ok:
                assert m["type"] == "error" and "features_f8" in m["error"]
            else:
                assert m["type"] == "config"
            c.close()
        finally:
            hub.close()


def _mind(rows):
    m = np.zeros(len(rows), P.MIND_DT)
    for k, (p, g, n) in enumerate(rows):
        m["partner"][k], m["gap"][k], m["plane"][k], m["self_status"][k] = p, g, (*n, 0), FX.IVP_ST_BALL
    return m


def _unit(*v):
    v = np.asarray(v, float)
    return tuple(v / np.linalg.norm(v))


def test_contact_slots_merge_facets_and_keep_each_tube_in_its_slot():
    """A double rail (one entity, two tubes, each touched through several facet entries) and the floor: three
    contacts; the right tube always in slot 0, the floor in slot 1, the left tube in slot 2, whatever the entry order,
    gap noise, which tube is nearer, or other contacts coming and going."""
    RAIL, FLOOR = 7, 3
    right = np.array([1.0, 0, 0])
    rows = []
    for s in (1, -1):  # normal from the tube towards the ball: the left tube's normal points right (+x)
        for k, (g, tilt) in enumerate(((0.020, 0.0), (0.020, 0.0), (0.05, 0.12), (0.09, -0.15), (0.3, 0.3))):
            rows.append((RAIL, g, _unit(s * 0.69, 0.72, tilt)))
    rows += [(FLOOR, 0.6, (0, 1, 0)), (FLOOR, 0.6, (0, 1, 0)), (FLOOR, 2.5, (0, 1, 0))]
    m = _mind(rows)
    N = m["plane"][:, :3].astype(float)
    cl = FX.merge_contacts(m, N)
    assert sorted((p, round(g, 3)) for g, p, _ in cl) == [(FLOOR, 0.6), (RAIL, 0.02), (RAIL, 0.02)]
    sl = FX.contact_slots(cl, right)
    assert [e[1] if e else None for e in sl] == [RAIL, FLOOR, RAIL, None]
    assert sl[0][2][0] < 0 < sl[2][2][0]  # the right tube (normal to -x) in slot 0, the left one in slot 2
    rng = np.random.default_rng(0)
    for _ in range(40):
        perm = rng.permutation(len(rows))
        mp = m[perm].copy()
        mp["gap"] = mp["gap"] + rng.uniform(0, 0.08, len(mp)).astype(np.float32)  # either tube may be nearer
        if rng.random() < 0.5:  # the floor leaves the window
            mp["gap"][mp["partner"] == FLOOR] = 2.5
        sp = FX.contact_slots(FX.merge_contacts(mp, mp["plane"][:, :3].astype(float)), right)
        assert sp[0][1] == RAIL and sp[2][1] == RAIL and sp[0][2][0] < 0 < sp[2][2][0]
    # several contacts in one direction: the nearest one holds the slot (a wall ahead: "other")
    walls = [(20 + k, 1.0 + 0.1 * k, (0, 0, -1)) for k in range(3)]
    mw = _mind(rows + walls)
    sw = FX.contact_slots(FX.merge_contacts(mw, mw["plane"][:, :3].astype(float)), right)
    assert [e[1] for e in sw] == [RAIL, FLOOR, RAIL, 20]
    # the old nearest-only reading flips between the two tubes on the same noise
    picks = set()
    for _ in range(40):
        g = m["gap"] + rng.uniform(0, 0.004, len(m)).astype(np.float32)
        picks.add(np.sign(N[int(np.argmin(g)), 0]))
    assert picks == {-1.0, 1.0}


RAILS, FLOOR_ID = 11, 10


def double_rail_tracker(spacing=3.4):
    """A floor at y = 0 and two rails (boxes 0.3 wide, 0.3 high, along z) at x = -spacing / 2, +spacing / 2."""
    lm = P.LevelMessage(0)
    I43 = np.zeros((4, 3), np.float32)
    lm.entities = [P.Entity(FLOOR_ID, "Floor", 41, 0, True, I43, 2, 2, b""),
                   P.Entity(RAILS, "Rail", 41, 0, True, I43, 3, 3, b""),
                   P.Entity(21, SYNTHETIC_LEVEL_MARKER, 41, 0, False, I43, -1, -1, b"")]
    lm.groups = [P.Group(1, "Phys_Floors", np.array([FLOOR_ID], np.uint32)),
                 P.Group(2, "Phys_FloorRails", np.array([RAILS], np.uint32))]
    lm.phys = [P.PhysRecord(0, 0, FLOOR_ID, False, True), P.PhysRecord(1, 0, RAILS, False, True)]
    eye = np.eye(4, dtype=np.float32)
    fv = np.array([[-10, -3, 0], [10, -3, 0], [10, -3, 40], [-10, -3, 40]], np.float32)
    ff = np.array([[0, 2, 1], [0, 3, 2]], np.uint32)
    rv, rf = box_mesh((0.15, 0.15, 20.0))
    m1, m2 = eye.copy(), eye.copy()
    m1[3, :3] = (-spacing / 2, 0, 20)
    m2[3, :3] = (spacing / 2, 0, 20)
    lm.meshes = [P.Mesh(FLOOR_ID, 100, 1, eye, fv, ff), P.Mesh(RAILS, 101, 1, m1, rv, rf), P.Mesh(RAILS, 102, 1, m2, rv, rf)]
    lm.depth = [(99, np.array([-100, -60, -100], np.float32), np.array([100, -50, 100], np.float32))]
    lm.arrays = [P.ArrayDef(500, "ResetPoints", [P.ArrayColumn("Resetpoint", 5, "Matrix")], [[eye.tobytes()]]),
                 P.ArrayDef(501, "Checkpoints", [P.ArrayColumn("Matrix", 5, "Matrix"), P.ArrayColumn("Object", 5, None)],
                            [[_m44((0, 2, 40)).tobytes(), b"\0\0\0\0"]]),
                 P.ArrayDef(502, "Energy", [P.ArrayColumn(n, 1, None) for n in ("Points", "Lifes", "S", "L", "T", "LifeBonus")],
                            [[np.int32(v).tobytes() for v in (950, 3, 1000, 3, 500, 200)]])]
    t = GameTracker(1)
    t.merge_level(lm)
    t.flags.ball_nav = True
    t._cam = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], np.float32)
    return t


def _m44(pos):
    m = np.eye(4, dtype=np.float32)
    m[3, :3] = pos
    return m


def _obs(t, pos, mind_rows):
    o = P.Observation(frames_done=100)
    b = np.zeros(1, P.BALL_DT)[0]
    b["pos"], b["found"], b["type"], b["extra_radius"] = pos, 1, 0, 2.0
    o.ball = b
    o.mind = _mind(mind_rows) if mind_rows else o.mind
    c = np.zeros(1, P.CAMS_DT)[0]
    for i in range(5):
        c["cams"][i]["matrix"] = t._cam
    o.cams = c
    t.obs = o


def test_double_rail_shows_both_rails_and_both_contacts():
    t = double_rail_tracker()
    fx = FX.FeatureExtractor(FX.FeatureConfig(camera="world", version="f8", fan_dirs=0, height_grid=0, bodies=0,
                                              path_source="checkpoints"))
    s = fx.slices()
    # the ball (radius 2) resting on both rails (tops at y = 0.15, inner edges at x = +-1.55): centre height from the
    # edges; contact normals 50 degrees from vertical (up-facing: n_y 0.63)
    e = 1.7 - 0.15
    y = 0.15 + np.sqrt(4 - e ** 2)
    nl, nr = _unit(e, y - 0.15, 0), _unit(-e, y - 0.15, 0)  # from the left / right rail edge to the centre
    _obs(t, (0, y, 10), [(RAILS, 0.01, nr), (RAILS, 0.012, nl), (RAILS, 0.04, nl), (FLOOR_ID, 1.5, (0, 1, 0))])
    v, info = fx.compute(t)
    assert len(v) == FX.layout_size(fx.cfg) and np.abs(v[s["contacts"].start:]).max() <= 1
    c = v[s["contacts"]].reshape(4, 8)
    # route right = +x (checkpoint line along +z): the right rail (normal to -x) first, the floor, the left rail
    assert c[:, 0].tolist() == [1, 1, 1, 0] and c[:, 6].tolist() == [1, 0, 1, 0]
    assert c[0, 5] < 0 < c[2, 5] and c[1, 5] == 0
    agg = v[s["contact_agg"]]
    assert agg[:3] == pytest.approx([0, 1, 0], abs=1e-6) and agg[3] == 0.5 and agg[4] == 1.0
    rl = v[s["rails_lr"]].reshape(2, 6)
    assert rl[:, 0].tolist() == [1, 1]
    assert rl[0, 5] * 6 == pytest.approx(-e, abs=1e-6) and rl[1, 5] * 6 == pytest.approx(e, abs=1e-6)
    assert rl[0, 2] < 0 < rl[1, 2]  # world frame: left rail at -x
    assert info["rails_lr"][0][2] == info["rails_lr"][1][2] == RAILS
    # crown ride on the right rail: the same rail on both sides, offsets continuous across the crown
    lat = []
    for x in (1.6, 1.65, 1.7, 1.75, 1.8):
        _obs(t, (x, 2.15, 10), [(RAILS, 0.0, (0, 1, 0))])
        v, info = fx.compute(t)
        rl = v[s["rails_lr"]].reshape(2, 6)
        assert rl[:, 0].tolist() == [1, 1]
        lat.append(rl[:, 5] * 6)
        assert rl[:, 3] * 6 == pytest.approx([-2.0, -2.0], abs=1e-6)  # the crown 2 below the centre on both sides
    lat = np.array(lat)
    assert np.all(np.abs(np.diff(lat, axis=0)) <= 0.05 + 1e-9)
    # nothing around: empty slots are zero
    _obs(t, (30, 5, 10), [])
    v, _ = fx.compute(t)
    assert not v[s["contacts"].start:].any()


def test_f6_f7_values_unchanged_on_a_hand_built_level():
    """Same tracker, same frames: the f8 vector is the f6 vector plus the blocks (and with f8 off the vector is what
    it was: f6 path untouched)."""
    t = double_rail_tracker()
    base = dict(camera="world", fan_dirs=4, height_grid=3, bodies=2, path_source="checkpoints")
    f6, f8 = FX.FeatureExtractor(FX.FeatureConfig(**base)), FX.FeatureExtractor(FX.FeatureConfig(**base, version="f8"))
    for x in (-1.0, 0.0, 0.5, 1.7):
        _obs(t, (x, 2.1, 10), [(RAILS, 0.05, (0, 1, 0))])
        a, _ = f6.compute(t)
        b, _ = f8.compute(t)
        assert np.array_equal(b[:len(a)], a) and len(b) == len(a) + N_F8


def _replay_pair(name, limit):
    d = DATA / name
    if not (d / "raw.npz").is_file():
        pytest.skip(f"dataset record {name} not built")
    if not FR.HAVE_NUMBA:
        pytest.skip("the reach block needs numba")
    f7 = FX.FeatureExtractor(FX.FeatureConfig(**R8, reach=True))
    f8 = FX.FeatureExtractor(FX.FeatureConfig(**R8, reach=True, version="f8"))
    fd = FX.FeatureExtractor(FX.FeatureConfig())
    A, B, C = [], [], []
    for i, _, t in dataset.OfflineReplay(d):
        if i >= limit:
            break
        A.append(f7.compute(t)[0])
        B.append(f8.compute(t)[0])
        C.append(fd.compute(t)[0])
    return np.stack(A), np.stack(B), np.stack(C)


# digests of the first 4000 frames' default-configuration vectors (FeatureConfig(): f6, 204 entries) at the commit
# before f8, on this platform's numpy (float32 bytes)
DEFAULT_DIGEST = {"SR01_1_14_457_264": "21dbc2ca7cc843f2", "SR02_1_28_992_264": "8ebbd13e1048f228"}


@pytest.mark.parametrize("name", ["SR01_1_14_457_264", "SR02_1_28_992_264"])
def test_recorded_f8_starts_with_the_f7_vector(name):
    A, B, C = _replay_pair(name, 4000)
    assert A.shape[1] == 267 and B.shape[1] == 316
    assert np.array_equal(B[:, :267], A)
    assert np.abs(B[:, 267:]).max() <= 1.0
    assert B[:, 267:].any()
    assert hashlib.sha256(C.tobytes()).hexdigest()[:16] == DEFAULT_DIGEST[name]


R8_CKPT = instances.runtime_dir() / "m2-train-mac" / "r8" / "latest.pt"


def test_r8_checkpoint_grows_to_f8_with_identical_outputs():
    """--init-from <r8> --init-full on an f8 config (scripts/train_m2.py load_policy -> ballrain.train.grow.load_grown):
    first-layer input columns of actor and critic for the 49 new entries are zero; Adam exp_avg of those columns
    zero, exp_avg_sq seeded with the row mean of the old columns (step counts kept); observation statistics of the new
    entries mean 0 / var 1 with the checkpoint's count; the policy (ball-type heads) and value at the switch equal r8's
    for real observations, whatever the new block holds."""
    import torch

    from ballrain.train.grow import load_grown
    from ballrain.train.ppo import ActorCritic, RunningMeanStd, normalize
    if not R8_CKPT.is_file():
        pytest.skip("no r8 checkpoint here")
    ck = torch.load(R8_CKPT, weights_only=False, map_location="cpu")
    n7 = int(ck["n_obs"])
    f8 = asdict(FX.FeatureConfig(**dict(ck["env_config"]["features"], version="f8")))
    n8 = FX.layout_size(FX.FeatureConfig(**f8))
    assert (n7, n8) == (267, 316)
    hidden = tuple(int(x) for x in ck["args"]["hidden"].split(","))
    old = ActorCritic(n7, 9, hidden, ck["args"]["activation"], ck["args"]["actor_heads"])
    old.load_state_dict(ck["model"])
    new = ActorCritic(n8, 9, hidden, ck["args"]["activation"], ck["args"]["actor_heads"])
    opt = torch.optim.Adam(new.parameters(), lr=7.5e-5, eps=1e-5)
    rms = RunningMeanStd(n8)
    assert load_grown(new, opt, rms, ck, n8, f8, load_opt=True)
    for k in ("pi.0.weight", "v.0.weight"):
        w = new.state_dict()[k]
        assert torch.equal(w[:, :n7], ck["model"][k]) and not w[:, n7:].any()
    names = [n for n, _ in new.named_parameters()]
    st = opt.state_dict()["state"]
    for i, n in enumerate(names):
        if n in ("pi.0.weight", "v.0.weight"):
            o = ck["opt"]["state"][i]
            assert torch.equal(st[i]["exp_avg"][:, :n7], o["exp_avg"]) and not st[i]["exp_avg"][:, n7:].any()
            assert torch.equal(st[i]["exp_avg_sq"][:, :n7], o["exp_avg_sq"])
            assert torch.equal(st[i]["exp_avg_sq"][:, n7:], o["exp_avg_sq"].mean(1, keepdim=True).expand(-1, n8 - n7))
            assert float(st[i]["step"]) == float(o["step"])
    assert np.array_equal(rms.mean[:n7], ck["rms"]["mean"]) and not rms.mean[n7:].any()
    assert np.array_equal(rms.var[:n7], ck["rms"]["var"]) and (rms.var[n7:] == 1).all()
    assert rms.count == ck["rms"]["count"]
    # observations: recorded f7 / f8 vectors when the dataset is here, else random ones in the observed ranges
    try:
        A, B, _ = _replay_pair("SR02_1_28_992_264", 3000)
        A, B = A[::7], B[::7]
    except pytest.skip.Exception:
        rng = np.random.default_rng(0)
        A = rng.normal(ck["rms"]["mean"], np.sqrt(ck["rms"]["var"]), (400, n7)).astype(np.float32)
        B = np.concatenate([A, rng.uniform(-1, 1, (400, n8 - n7)).astype(np.float32)], 1)
    a7, a8 = normalize(A, ck["rms"]), normalize(B, rms.state_dict())
    assert np.array_equal(a8[:, :n7], a7) and np.array_equal(a8[:, n7:], B[:, n7:])  # new entries pass unscaled
    # float32, one row at a time (the workers' path) and as one batch (the learner's): the zero columns add exactly
    # 0, but a 316-long dot product is summed in other blocks than a 267-long one, so the last bits may differ
    # (measured on the r8 checkpoint over all 24395 SR02 states: one row at a time max |d logit| 4.3e-6 on logits up
    # to ~20, max |d value| 4.8e-7; batched 2.4e-5 / 1.8e-6; the greedy action never changes)
    for batch in (False, True):
        with torch.no_grad():
            if batch:
                la, va = old(torch.as_tensor(a7))
                lb, vb = new(torch.as_tensor(a8))
            else:
                la, va = (torch.cat(z) for z in zip(*(old(torch.as_tensor(a7[i:i + 1])) for i in range(len(a7)))))
                lb, vb = (torch.cat(z) for z in zip(*(new(torch.as_tensor(a8[i:i + 1])) for i in range(len(a8)))))
        dl = float((la - lb).abs().max()) / (1 + float(la.abs().max()))
        dv = float(((va - vb).abs() / (1 + va.abs())).max())
        assert dl <= 1e-5 and dv <= 1e-5, (batch, dl, dv)
        assert torch.equal(la.argmax(-1), lb.argmax(-1))
    # float64: the grown network is the old one to rounding of the old one's own sums
    o64 = ActorCritic(n7, 9, hidden, ck["args"]["activation"], ck["args"]["actor_heads"]).double()
    n64 = ActorCritic(n8, 9, hidden, ck["args"]["activation"], ck["args"]["actor_heads"]).double()
    o64.load_state_dict(old.state_dict()), n64.load_state_dict(new.state_dict())
    with torch.no_grad():
        l6, v6 = o64(torch.as_tensor(a7, dtype=torch.float64))
        l8, v8 = n64(torch.as_tensor(a8, dtype=torch.float64))
    assert float((l6 - l8).abs().max()) <= 1e-12 and float((v6 - v8).abs().max()) <= 1e-12


def test_train_m2_contacts_flag_and_init_from_bundle(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import train_m2
    assert train_m2.parse(["--run", "x"]).contacts is False
    a = train_m2.parse(["--run", "x", "--reach", "--contacts", "--world-actions", "--level", "2"])
    fc = train_m2.make_env_config(a, Path("unused")).features
    n7 = FX.layout_size(FX.FeatureConfig(**dict(asdict(fc), version="f7")))
    assert fc.reach and fc.contacts and fc.version == "f8" and train_m2.obs_layout(fc)[0] == n7 + N_F8
    a = train_m2.parse(["--run", "x", "--reach", "--world-actions", "--level", "2"])
    assert train_m2.make_env_config(a, Path("unused")).features.version == "f7"
    src = (train_m2.REPO / "scripts" / "train_m2.py").read_text(encoding="utf-8")
    assert 'saved.get("contacts")' in src and "init=ck0" in src


def test_r9_from_r8_carries_every_argument_and_the_bundle():
    """scripts/derive_run_args.py on the r8 checkpoint: the new run's arguments equal r8's (rehearsal, rest lists,
    curriculum, rewards ...) except the run identity, the start (--init-from, --init-full), --contacts and the
    pool; its EnvConfig equals r8's except the feature label and out_root; the branch bundle is r8's."""
    import sys
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import derive_run_args as D
    import train_m2
    if not R8_CKPT.is_file():
        pytest.skip("no r8 checkpoint here")
    ck = torch.load(R8_CKPT, weights_only=False, map_location="cpu")
    argv, unknown = D.r9_argv(ck, "r9", "r8/latest.pt", contacts=True, no_pool=True)
    assert not unknown
    a = vars(train_m2.parse(argv))
    changed = {k for k in a if a[k] != ck["args"].get(k)}
    assert changed <= {"run", "resume", "init_from", "init_full", "contacts", "pool_ingest"}, changed
    assert a["init_full"] and a["contacts"] and not a["pool_ingest"] and not a["resume"]
    for k in ("rest_starts", "rehearse_starts", "rehearse_always", "p_rest", "p_rehearse", "counter_rand", "reward",
              "rehearse_behind_frontier", "actor_heads", "world_actions", "reach", "path_dir", "lr", "start_back"):
        assert a[k] == ck["args"][k], k
    diffs = D.compare_env(ck, argv)
    assert {d[0] for d in diffs} == {"out_root", "features.version"}
    env = train_m2.make_env_config(train_m2.parse(argv), Path("unused"), None, init=ck)
    assert env.branch_assets == ck["branch_assets"]
    assert env.features.branch_sha256 == ck["env_config"]["features"]["branch_sha256"]
