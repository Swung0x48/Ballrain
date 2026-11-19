"""Mastered-state anchor (scripts/train_m2.py --anchor-mastered, ballrain.train.anchor): start families, rolling success
with hysteresis, teacher snapshots, balanced weighting, the ppo_update term, the worker's per-row start tags, and the
learner end to end (--fake): off = as before, on = engages once a family is mastered and continues on --resume."""
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ballrain import instances
from ballrain.env import BallanceEnv, EnvConfig
from ballrain.train import anchor as AN
from ballrain.train.ppo import ActorCritic, ppo_update

REPO = instances.REPO_ROOT
sys.path.insert(0, str(REPO / "scripts"))
import train_m2  # noqa: E402


def _key(tag, kinds=AN.KINDS):
    return AN.family_key(tag, band_frames=200, base_frame=936, kinds=kinds)


# ============================================================================================== families

def test_family_keys():
    rest = {"src": "rest", "entry": [13900, 13900, {"branch_s": 1447.0, "id": "stone-arrival"}]}
    assert _key({"kind": "tas_rest", "frame": 13900, "sector": 3, "source": rest}) == \
        "rest:13900-13900@branch:stone-arrival:1447"
    # a moving start that fell back to rest belongs to the same entry
    assert _key({"kind": "branch_moving_fallback", "frame": 13900, "sector": 3, "source": rest}) == \
        "rest:13900-13900@branch:stone-arrival:1447"
    assert _key({"kind": "tas_rest", "source": {"src": "rest", "entry": [11080, 11080, 921.0, -348.6, 0.0]}}) == \
        "rest:11080-11080@921:-348.6:0"
    assert _key({"kind": "tas_rest", "source": {"src": "rest", "entry": [11080, 11080]}}) == "rest:11080-11080"
    # the same --rest-starts text parses to the same key (duplicated entries for weighting are one family)
    for e in train_m2.parse_rest_starts("14535-14535@branch:south-loop:1523.604744,14730-14730@933.8:-57:0:1200,"
                                        "8200-8200@branch-move:back-to-wood:917.5"):
        assert AN.rest_entry_label(e) in ("14535-14535@branch:south-loop:1523.604744", "14730-14730@933.8:-57:0:1200",
                                          "8200-8200@branch-move:back-to-wood:917.5")
    assert _key({"kind": "tas", "frame": 15000, "source": {"src": "rehearse", "entry": [14521, 17561]}}) == \
        "rehearse:14521-17561"
    assert _key({"kind": "tas", "frame": 1136, "source": {"src": "curriculum"}}) == "band:1136-1335"
    assert _key({"kind": "tas", "frame": 1135, "source": {"src": "curriculum"}}) == "band:936-1135"
    assert _key({"kind": "tas", "frame": 0, "source": {"src": "curriculum"}}) == "band:k0"
    assert _key({"kind": "respawn", "frame": 9000, "sector": 3, "source": None}) == "respawn:s3"
    assert _key({"kind": "level", "frame": 0, "source": None}) == "level"
    # unknown source (an older worker), malformed tags, kinds left out: no family
    assert _key({"kind": "tas", "frame": 5000, "source": None}) is None
    assert _key({"kind": "tas_rest", "source": {"src": "rest"}}) is None
    assert _key(None) is None and _key("x") is None
    assert _key({"kind": "respawn", "sector": 2}, kinds=("rest",)) is None


def test_draw_start_records_the_source_without_extra_draws():
    def env(**kw):
        e = object.__new__(BallanceEnv)
        e.cfg = EnvConfig(**kw)
        e.starts = SimpleNamespace(nearest_valid=lambda k: int(k), nearest_any=lambda k: int(k))
        e.curriculum = None
        e._start_rng = np.random.default_rng(0)
        return e
    e = env(rest_starts=[[1940, 1940, 3.5, -170.0, 1.5], [100, 200]], p_rest=1.0)
    ref = np.random.default_rng(0)
    for _ in range(20):
        k = e._draw_start()
        src = e._source_next
        assert src["src"] == "rest" and src["entry"] in ([1940, 1940, 3.5, -170.0, 1.5], [100, 200])
        # the same generator draws as the code before the tag: p_rest, the entry, (target jitter x 2 | frame)
        ref.random()
        j = int(ref.integers(2))
        if j == 0:
            ref.uniform(-1.5, 1.5), ref.uniform(-1.5, 1.5), ref.integers(1940, 1941)
        else:
            assert k == int(ref.integers(100, 201))
    assert e._start_rng.bit_generator.state == ref.bit_generator.state
    e = env(rehearse_starts=[[500, 600]], rehearse_always=[[700, 700]], p_rehearse=1.0)
    for _ in range(10):
        k = e._draw_start()
        assert e._source_next == {"src": "rehearse", "entry": [500, 600] if k < 700 else [700, 700]}
    e = env()
    e.curriculum = SimpleNamespace(sample=lambda rng: 4242)
    assert e._draw_start() == 4242 and e._source_next == {"src": "curriculum"}
    e = env(start_frame=0)
    assert e._draw_start() == 0 and e._source_next is None


# ============================================================================================== statistics

def _feed(st, key, outcomes, version=0, update=0):
    for s in outcomes:
        st.add_life(key, bool(s), version, update)


def test_mastery_needs_min_lives_and_threshold():
    st = AN.MasteredAnchor(window=10, min_lives=8, threshold=0.9, grace=5)
    _feed(st, "f", [1] * 7)
    assert not st.families["f"].anchored  # 7 lives < min_lives
    _feed(st, "f", [1])
    assert st.families["f"].anchored and st.drain_events()[0][:2] == ("mastered", "f")
    st2 = AN.MasteredAnchor(window=10, min_lives=8, threshold=0.9, grace=5)
    _feed(st2, "g", [1, 1, 1, 1, 0, 1, 1, 1, 1, 0])  # 0.8 over 10 lives
    assert not st2.families["g"].anchored


def test_hysteresis_grace_and_recovery():
    st = AN.MasteredAnchor(window=10, min_lives=10, threshold=0.9, grace=3)
    _feed(st, "f", [1] * 10)
    f = st.families["f"]
    _feed(st, "f", [0])  # 0.9: still at the threshold
    assert f.anchored and f.below == 0
    _feed(st, "f", [0, 0])  # 0.8, 0.7: below, inside the grace
    assert f.anchored and f.below == 2
    _feed(st, "f", [1])  # 0.7 (the 1 replaces a 1): the third life below
    assert f.anchored and f.below == 3
    _feed(st, "f", [1])  # the fourth life below the threshold: released
    assert not f.anchored and f.teacher is None
    assert [e[0] for e in st.drain_events()] == ["mastered", "released"]
    # a recovery inside the grace resets the count
    st = AN.MasteredAnchor(window=4, min_lives=4, threshold=0.75, grace=5)
    _feed(st, "f", [1, 1, 1, 1, 0, 0, 1, 1])  # 0.75, 0.5, 0.5, 0.5: below 3
    f = st.families["f"]
    assert f.anchored and f.below == 3
    _feed(st, "f", [1])  # 0.75: recovered
    assert f.anchored and f.below == 0
    # once released, mastering again needs the threshold over the window again
    st = AN.MasteredAnchor(window=4, min_lives=4, threshold=0.75, grace=0)
    _feed(st, "f", [1, 1, 1, 1, 0, 0])
    assert not st.families["f"].anchored
    _feed(st, "f", [1])  # [1, 0, 0, 1]
    assert not st.families["f"].anchored
    _feed(st, "f", [1, 1])  # [0, 1, 1, 1]
    assert st.families["f"].anchored


def _snap(tag):
    calls = []

    def fn():
        calls.append(tag)
        return {"tag": tag}
    return fn, calls


def test_snapshot_per_family_reuse_inside_the_window_and_kept_during_grace():
    st = AN.MasteredAnchor(window=4, min_lives=4, threshold=0.75, grace=10)
    _feed(st, "a", [1, 1, 1, 1], version=10)
    fn, calls = _snap("u12")
    st.refresh(12, fn)
    assert st.anchored() == {"a": 12} and calls == ["u12"]
    # b's mastering window was played by policies from version 11 on: the stored u12 snapshot is inside it
    _feed(st, "b", [1, 1, 1, 1], version=11)
    fn, calls = _snap("u15")
    st.refresh(15, fn)
    assert st.anchored() == {"a": 12, "b": 12} and calls == []
    # c's window starts after u12: a new snapshot
    _feed(st, "c", [1, 1, 1, 1], version=14)
    fn, calls = _snap("u20")
    st.refresh(20, fn)
    assert st.anchored() == {"a": 12, "b": 12, "c": 20} and calls == ["u20"] and sorted(st.snapshots) == [12, 20]
    # a drifts: it keeps its own snapshot through the grace (never a snapshot of the drifting policy)
    _feed(st, "a", [0, 0], version=25)
    fn, calls = _snap("u30")
    st.refresh(30, fn)
    assert st.anchored()["a"] == 12 and calls == [] and st.families["a"].below == 1  # 0.75, then 0.5
    # released families give their snapshot back
    st2 = AN.MasteredAnchor(window=2, min_lives=2, threshold=1.0, grace=0)
    _feed(st2, "x", [1, 1], version=1)
    st2.refresh(2, _snap("s")[0])
    _feed(st2, "x", [0])
    assert st2.anchored() == {} and st2.snapshots == {}


def test_snapshot_cap_retires_healthy_and_drops_unhealthy():
    st = AN.MasteredAnchor(window=2, min_lives=2, threshold=1.0, grace=100, max_teachers=2)
    for i, key in enumerate("abc"):
        _feed(st, key, [1, 1], version=10 * i + 5)
        st.refresh(10 * i + 9, _snap(key)[0])
    # at the cap the oldest all-healthy snapshot (a's, u9) is retired: a moves to the new one
    assert sorted(st.snapshots) == [19, 29] and st.anchored() == {"a": 29, "b": 19, "c": 29}
    assert st.counts["retired"] == 1
    # b and c now below the threshold (in their grace): no snapshot can be retired, the oldest is dropped
    _feed(st, "b", [0], version=30)
    _feed(st, "c", [0], version=30)
    _feed(st, "a", [0], version=30)
    _feed(st, "d", [1, 1], version=31)
    st.refresh(40, _snap("d")[0])
    assert st.counts["dropped"] == 1 and not st.families["b"].anchored
    assert len(st.snapshots) <= 2 and st.anchored()["d"] == 40


def test_ratchet_moves_forward_only_on_a_full_newer_window():
    for ratchet in (False, True):
        st = AN.MasteredAnchor(window=4, min_lives=4, threshold=0.75, grace=10, ratchet=ratchet)
        _feed(st, "a", [1, 1, 1, 0], version=1)  # mastered at 0.75 by versions 1
        st.refresh(2, _snap("u2")[0])
        assert st.anchored() == {"a": 2} and st.families["a"].teacher_rate == 0.75
        _feed(st, "a", [1, 1, 1], version=3)  # window [0, 1, 1, 1] still holds a version-1 life
        st.refresh(5, _snap("u5")[0])
        assert st.anchored() == {"a": 2}
        _feed(st, "a", [1], version=4)  # all four lives by versions > 2 at 1.0 >= 0.75
        fn, calls = _snap("u6")
        st.refresh(6, fn)
        assert st.anchored() == ({"a": 6} if ratchet else {"a": 2})
        assert sorted(st.snapshots) == ([6] if ratchet else [2])  # the old snapshot is given back
    # a newer full window below the teacher-time rate does not move it
    st = AN.MasteredAnchor(window=4, min_lives=4, threshold=0.75, grace=10, ratchet=True)
    _feed(st, "a", [1, 1, 1, 1], version=1)
    st.refresh(2, _snap("u2")[0])
    _feed(st, "a", [1, 0, 1, 1], version=3)  # 0.75 < 1.0
    st.refresh(6, _snap("u6")[0])
    assert st.anchored() == {"a": 2}


def test_explicit_teacher_and_lost_snapshots():
    st = AN.MasteredAnchor(window=2, min_lives=2, threshold=1.0, explicit=True)
    _feed(st, "a", [1, 1])
    st.refresh(5, lambda: pytest.fail("no snapshots with an explicit teacher"))
    assert st.anchored() == {"a": "explicit"} and st.snapshots == {}
    # explicit -> automatic on resume: a healthy family gets the current policy, an unhealthy one is released
    st2 = AN.MasteredAnchor(window=4, min_lives=2, threshold=0.75, grace=10)
    st2.load_state_dict(st.state_dict())
    _feed(st2, "b", [1, 1, 1, 1])
    st2.families["b"].pending = False  # (as if its teacher had been dropped)
    _feed(st2, "a", [0, 0])  # 0.5 -> below, in grace
    st2.refresh(9, _snap("now")[0])
    assert st2.anchored() == {"b": 9} and not st2.families["a"].anchored


def test_state_round_trip():
    st = AN.MasteredAnchor(window=5, min_lives=3, threshold=0.6, grace=4, drop_scale=2.0)
    _feed(st, "a", [1, 1, 1, 0], version=3)
    _feed(st, "b", [0, 1], version=4)
    st.refresh(7, lambda: {"w": torch.zeros(2)})
    d = st.state_dict()
    st2 = AN.MasteredAnchor(window=5, min_lives=3, threshold=0.6, grace=4, drop_scale=2.0)
    st2.load_state_dict(d)
    assert st2.state_dict()["families"] == d["families"] and list(st2.snapshots) == [7]
    assert st2.anchored() == st.anchored() == {"a": 7}
    assert st2.weight("a") == pytest.approx(1 + 2.0 * (1.0 - 0.75))


# ============================================================================================== weights / loss

def test_balanced_row_weights():
    keys = ["big"] * 1000 + ["rare"] * 20 + [None] * 50 + ["tiny"] * 4 + ["off"] * 30
    anchored = {"big": 1, "rare": 1, "tiny": 1}
    w, groups = AN.row_weights(keys, anchored, min_rows=16)
    assert set(groups) == {"big", "rare", "tiny"}
    per = {k: float(w[g].sum()) for k, g in groups.items()}
    assert per["big"] == pytest.approx(1 / 3) and per["rare"] == pytest.approx(1 / 3)
    assert per["tiny"] == pytest.approx(1 / 3 * 4 / 16)  # under min_rows: a part of a share
    assert w[1020:1070].sum() == 0 and w[-30:].sum() == 0
    # the weighted sum of per-row values = the mean over families of each family's mean
    kl = np.random.default_rng(0).random(len(keys))
    w, groups = AN.row_weights(keys, anchored, min_rows=1)
    assert float((w * kl).sum()) == pytest.approx(np.mean([kl[g].mean() for g in groups.values()]))
    w2, _ = AN.row_weights(keys, anchored, weight_of=lambda k: 3.0 if k == "rare" else 1.0, min_rows=1)
    assert float(w2[groups["rare"]].sum()) == pytest.approx(1.0)
    assert AN.row_weights([None, "x"], {})[0].sum() == 0


def _setup(seed=0, n=512, n_obs=8, n_act=5):
    torch.manual_seed(seed)
    pi = ActorCritic(n_obs, n_act, (32, 32))
    teacher = ActorCritic(n_obs, n_act, (32, 32))
    obs = np.random.default_rng(seed).normal(size=(n, n_obs)).astype(np.float32)
    with torch.no_grad():
        lt = torch.log_softmax(teacher(torch.as_tensor(obs))[0], -1).numpy()
        lp = torch.log_softmax(pi(torch.as_tensor(obs))[0], -1).numpy()
    batch = {"obs": obs, "actions": np.zeros(n, np.int64), "logp": lp[:, 0].astype(np.float32),
             "adv": np.random.default_rng(seed + 1).normal(size=n).astype(np.float32),
             "ret": np.zeros(n, np.float32)}
    return pi, obs, lt, batch


def _kl(pi, obs, lt, rows):
    with torch.no_grad():
        lp = torch.log_softmax(pi(torch.as_tensor(obs))[0], -1).numpy()
    return float((np.exp(lt) * (lt - lp)).sum(-1)[rows].mean())


ARGS = dict(epochs=4, minibatch=128, clip=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5)


def test_off_is_bit_identical():
    pi, obs, lt, batch = _setup(3)
    res = []
    for kw in ({}, {"anchor_logp": None, "anchor_weight": None, "anchor_coef": 0.5},
               {"anchor_logp": lt, "anchor_weight": np.zeros(len(obs), np.float32), "anchor_coef": 0.5},
               {"anchor_logp": lt, "anchor_weight": np.full(len(obs), 1 / len(obs), np.float32), "anchor_coef": 0.0}):
        m = ActorCritic(8, 5, (32, 32))
        m.load_state_dict(pi.state_dict())
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(7)
        stats = ppo_update(m, opt, batch, gen=g, **ARGS, **kw)
        res.append((m.state_dict(), opt.state_dict()["state"], g.get_state(), stats))
    for sd, ost, g, stats in res[1:]:
        assert all(torch.equal(sd[k], res[0][0][k]) for k in sd)
        assert all(torch.equal(ost[i][k], res[0][1][i][k]) for i in ost for k in ost[i] if torch.is_tensor(ost[i][k]))
        assert torch.equal(g, res[0][2])
        assert stats == res[0][3] and not any(k.startswith("anchor") for k in stats)


def test_anchor_term_is_not_diluted_for_a_rare_family():
    """Two families, the rare one 6 % of the rows, each with its own teacher (as a per-family snapshot): the balanced
    term pulls the rare family towards its teacher much further than a row mean with the same total coefficient."""
    keys = ["big"] * 480 + ["rare"] * 32
    rare = np.arange(480, 512)
    res = {}
    for mode in ("balanced", "row_mean"):
        pi, obs, lt, batch = _setup(1)
        _, _, lt_rare, _ = _setup(5)  # another teacher for the rare family
        lt = lt.copy()
        lt[rare] = lt_rare[rare]
        batch["adv"][:] = 0
        if mode == "balanced":
            w, _ = AN.row_weights(keys, {"big": 1, "rare": 1}, min_rows=16)
        else:
            w = np.full(512, 1 / 512, np.float32)
        before = _kl(pi, obs, lt, rare)
        opt = torch.optim.Adam(pi.parameters(), lr=1e-3)
        st = ppo_update(pi, opt, batch, epochs=4, minibatch=128, clip=0.2, vf_coef=0.0, ent_coef=0.0,
                        max_grad_norm=1.0, anchor_logp=lt, anchor_weight=w, anchor_coef=1.0)
        assert st["anchor_n"] == 512 and st["anchor_coef"] == 1.0 and np.isfinite(st["anchor_kl"])
        res[mode] = before - _kl(pi, obs, lt, rare)
    assert res["balanced"] > 2 * res["row_mean"] > 0, res


def test_anchor_and_teacher_terms_coexist():
    pi, obs, lt, batch = _setup(2)
    opt = torch.optim.Adam(pi.parameters(), lr=1e-3)
    w = np.zeros(len(obs), np.float32)
    w[:64] = 1 / 64
    st = ppo_update(pi, opt, batch, **ARGS, teacher_logp=lt, teacher_mask=np.arange(len(obs)) >= 256,
                    teacher_coef=0.1, anchor_logp=lt, anchor_weight=w, anchor_coef=0.1)
    assert np.isfinite(st["teacher_kl"]) and np.isfinite(st["anchor_kl"]) and st["anchor_n"] == 64


def test_row_families_maps_rows_and_excludes_untagged():
    tags = [{"kind": "tas_rest", "frame": 13900, "sector": 3, "source": {"src": "rest", "entry": [13900, 13900]}},
            {"kind": "respawn", "frame": 9999, "sector": 4, "source": None}]
    c1 = {"rewards": np.zeros(5), "start_tags": tags, "start_ref": np.array([0, 0, 0, 1, 1], np.int16),
          "past_cp": np.array([0, 0, 1, 0, 0], bool)}
    pool = {"rewards": np.zeros(3), "pool": True}
    old = {"rewards": np.zeros(2)}  # a chunk of an older worker: no tags
    bad = {"rewards": np.zeros(2), "start_tags": tags, "start_ref": np.array([0], np.int16)}  # wrong length
    keys = AN.row_families([c1, pool, old, bad], _key)
    assert keys == ["rest:13900-13900", "rest:13900-13900", None, "respawn:s4", "respawn:s4"] + [None] * 7
    keys = AN.row_families([c1], _key, past_cp_counts=True)  # --success finish: the whole life
    assert keys[2] == "rest:13900-13900"


def test_worker_tags_rows_with_their_life():
    from ballrain.train.worker import Worker
    w = object.__new__(Worker)
    w.spec = SimpleNamespace(start_tags=True)
    ep1, ep2 = SimpleNamespace(d={"start_tag": {"k": 1}}), SimpleNamespace(d={"start_tag": {"k": 2}})
    buf = {"start_tags": [], "start_ref": np.full(4, -1, np.int16), "past_cp": np.zeros(4, bool)}
    for t, (ep, past) in enumerate([(ep1, False), (ep1, True), (ep2, False), (ep2, False)]):
        Worker._tag_row(buf, t, ep, past)
    assert buf["start_tags"] == [{"k": 1}, {"k": 2}] and buf["start_ref"].tolist() == [0, 0, 1, 1]
    assert buf["past_cp"].tolist() == [False, True, False, False]
    buf2 = {"start_tags": [], "start_ref": np.full(1, -1, np.int16), "past_cp": np.zeros(1, bool)}
    Worker._tag_row(buf2, 0, ep2, False)  # the life goes on in the next chunk: registered there again
    assert buf2["start_tags"] == [{"k": 2}] and buf2["start_ref"].tolist() == [0]


def test_teacher_layout_rules_match_the_teacher_option():
    """The --teacher rules (now shared with the anchor's teachers and stored snapshots): f7 / f8 blocks appended at the
    end and the plain-reach -> mech1 allowance pass, any other feature difference or action count does not."""
    from dataclasses import asdict

    from ballrain.features import CONTACT_FEATURE_VERSION, FeatureConfig, layout_size
    plain, f7 = FeatureConfig(), FeatureConfig(reach=True)
    mech = FeatureConfig(reach=True)
    mech.reach_version = f"{f7.reach_version}-mech1-abc"
    f8 = FeatureConfig(reach=True, version=CONTACT_FEATURE_VERSION)
    f8.reach_version = mech.reach_version
    n = {k: layout_size(v) for k, v in (("plain", plain), ("f7", f7), ("mech", mech), ("f8", f8))}
    E = AN.teacher_layout_error
    assert E(asdict(f8), asdict(f8), n["f8"], n["f8"], 9, 9) is None
    assert E(asdict(mech), asdict(f7), n["f7"], n["mech"], 9, 9) is None  # plain reach under mechanisms
    assert E(asdict(f8), asdict(f7), n["f7"], n["f8"], 9, 9) is None  # f7 teacher, f8 student
    assert E(asdict(mech), asdict(plain), n["plain"], n["mech"], 9, 9) is None  # no reach block
    assert "observations differ" in E(asdict(f8), asdict(f8), n["f8"], n["f8"], 8, 9)
    other = FeatureConfig(far_waypoints=(20.0,))
    assert "far_waypoints" in E(asdict(plain), asdict(other), layout_size(other), n["plain"], 9, 9)
    src = (REPO / "scripts" / "train_m2.py").read_text(encoding="utf-8")
    assert src.count("teacher_layout_error(") == 3  # --teacher, --anchor-teacher, stored snapshots on resume


# ============================================================================================== options / learner

def test_options_and_saved_args():
    a = train_m2.parse(["--run", "x"])
    assert not a.anchor_mastered and a.anchor_teacher == "" and a.anchor_threshold == 0.9 and a.anchor_window == 40
    assert a.anchor_min_lives == 40 and a.anchor_grace == 80 and a.anchor_coef == 0.1 and a.anchor_max_teachers == 8
    assert a.anchor_rows == "life" and not a.anchor_ratchet
    d = train_m2.saved_args(a)
    assert not any(k.startswith("anchor_") for k in d) and set(d) == {k for k in vars(a) if not k.startswith("anchor_")}
    a = train_m2.parse(["--run", "x", "--anchor-mastered", "--anchor-threshold", "0.85"])
    d = train_m2.saved_args(a)
    assert d["anchor_mastered"] and d["anchor_threshold"] == 0.85 and "anchor_grace" in d
    # anchor statistics never enter updates.csv (a new column would start updates.<i>.csv on a live run)
    assert not any("anchor" in c for c in train_m2.UPDATE_COLS)


DRIVER = r"""
import sys
sys.path.insert(0, sys.argv[1])
import train_m2
raise SystemExit(train_m2.main(sys.argv[2:]))
"""


def _learner(tmp_path, record, run, *extra, steps=3072):
    import os
    env = dict(os.environ, BALLRAIN_RUNTIME=str(tmp_path / "rt"))
    args = [sys.executable, "-c", DRIVER, str(REPO / "scripts"), "--run", run, "--fake", "--workers", "1",
            "--total-steps", str(steps), "--batch-steps", "512", "--chunk", "64", "--minibatch", "256", "--epochs", "1",
            "--log-every", "1", "--ckpt-every", "1", "--ckpt-keep-every", "1", "--record", str(record), *extra]
    p = subprocess.run(args, env=env, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       encoding="utf-8", timeout=600)
    return p.returncode, p.stdout, tmp_path / "rt" / "m2-train" / run


@pytest.fixture(scope="module")
def record(tmp_path_factory):
    if sys.platform != "win32":
        pytest.skip("ballrain.fakegame's shared memory needs Windows")
    from ballrain.pool import dev
    return dev.fake_record(tmp_path_factory.mktemp("anchor-record"))


def test_learner_off_writes_what_it_wrote_before(tmp_path, record):
    code, out, run = _learner(tmp_path, record, "off")
    assert code == 0, out[-3000:]
    ck = torch.load(run / "latest.pt", weights_only=False)
    assert set(ck) == {"model", "opt", "rms", "curriculum", "counters", "best", "args", "n_obs", "n_act",
                       "branch_assets", "env_config"}
    assert not any(k.startswith("anchor") for k in ck["args"])
    import json
    assert not any(k.startswith("anchor") for k in json.loads((run / "config.json").read_text(encoding="utf-8")))
    assert not (run / "anchor.csv").exists() and "anchor" not in out


def test_learner_anchor_engages_and_resumes(tmp_path, record):
    opts = ["--anchor-mastered", "--anchor-window", "4", "--anchor-min-lives", "2", "--anchor-threshold", "0.0",
            "--anchor-coef", "0.5"]
    code, out, run = _learner(tmp_path, record, "on", *opts)
    assert code == 0, out[-3000:]
    assert "anchor: mastered level" in out and "anchor: teacher level (snapshot u" in out
    import csv
    rows = list(csv.DictReader(open(run / "anchor.csv", encoding="utf-8")))
    engaged = [r for r in rows if r["rows"] and int(r["rows"]) > 0]
    assert engaged and all(float(r["n"]) > 0 and r["kl"] != "" for r in engaged)
    ck = torch.load(run / "latest.pt", weights_only=False)
    an = ck["anchor"]
    assert an["families"]["level"]["anchored"] and an["snapshots"]
    sid = an["families"]["level"]["teacher"]
    assert set(an["snapshots"][sid]["actor"]) == {k for k in ck["model"] if not k.startswith("v.")}
    code, out2, run = _learner(tmp_path, record, "on", "--resume", *opts, steps=4608)
    assert code == 0, out2[-3000:]
    assert "resumed: " in out2 and "mastered, 1 snapshots" in out2 and "anchor: mastered level" not in out2
    ck2 = torch.load(run / "latest.pt", weights_only=False)
    assert ck2["anchor"]["families"]["level"]["lives"] > an["families"]["level"]["lives"]
    # resumed with the anchor off: the state is not carried on, the checkpoint is a plain one again
    code, out3, run = _learner(tmp_path, record, "on", "--resume", steps=5120)
    assert code == 0, out3[-3000:]
    assert "not carried on" in out3 and "anchor" not in torch.load(run / "latest.pt", weights_only=False)
