#include "ballrain/wire.h"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace ballrain {

static const uint32_t OBS_MAGIC = 0x3153424Fu;  // "OBS1"
static const uint32_t ACT_MAGIC = 0x31544341u;  // "ACT1"
static const uint32_t LVL_MAGIC = 0x314C564Cu;  // "LVL1"

const char *end_reason_name(uint32_t r) {
    static const char *n[] = {"none", "LevelFinish", "GameOver", "frame_limit", "python_stop", "reset_or_exit",
                              "timeout", "error"};
    return r < 8 ? n[r] : "?";
}

const char *event_name(uint16_t c) {
    static const char *n[] = {"?", "PreLoadLevel", "PostLoadLevel", "StartLevel", "CamNavActive", "CamNavInactive",
                              "BallNavActive", "BallNavInactive", "CounterActive", "CounterInactive", "PauseLevel",
                              "UnpauseLevel", "BallOff", "Dead", "PreSubLife", "PostSubLife", "PreCheckpoint",
                              "PostCheckpoint", "ExtraPoint", "PreLifeUp", "PostLifeUp", "LevelFinish", "GameOver",
                              "PreEndLevel", "PostEndLevel", "Physicalize", "Unphysicalize", "ActiveBallChanged",
                              "PreResetLevel", "PreExitLevel", "RunEnd"};
    return c <= 30 ? n[c] : "?";
}

uint64_t event_mask(std::initializer_list<uint16_t> codes) {
    uint64_t m = 0;
    for (auto c : codes) {
        if (c >= 64) throw Error("event code outside 0..63");
        m |= uint64_t(1) << c;
    }
    return m;
}

bool Observation::has_event(uint16_t code) const {
    for (auto &e : events)
        if (e.code == code) return true;
    return false;
}

// ---------------------------------------------------------------------------------------------- messages

namespace {

struct Section {
    uint32_t tag;
    uint16_t version, flags;
    const char *p;
    size_t n;
};

struct Message {
    uint32_t magic;
    uint16_t version, flags;
    uint32_t byte_len, frames_done;
    std::vector<Section> sections;
};

Message split_message(const char *buf, size_t avail, uint32_t expect_magic) {
    if (avail < 24) throw ProtocolError("message shorter than its header");
    Reader r(buf, avail);
    Message m;
    m.magic = r.u32();
    m.version = r.u16();
    m.flags = r.u16();
    m.byte_len = r.u32();
    m.frames_done = r.u32();
    uint32_t count = r.u32();
    r.u32();
    if (m.magic != expect_magic) throw ProtocolError("bad magic " + tag_str(m.magic));
    if (m.byte_len < 24 || m.byte_len > avail) throw ProtocolError("byte_len outside buffer");
    size_t end = m.byte_len, pos = 24;
    for (uint32_t i = 0; i < count; ++i) {
        if (pos + 12 > end) throw ProtocolError("section header beyond message end");
        Reader h(buf + pos, 12);
        Section s;
        s.tag = h.u32();
        s.version = h.u16();
        s.flags = h.u16();
        uint32_t plen = h.u32();
        pos += 12;
        if (pos + plen > end) throw ProtocolError("section " + tag_str(s.tag) + " payload beyond message end");
        s.p = buf + pos;
        s.n = plen;
        m.sections.push_back(s);
        pos += align4(plen);
    }
    return m;
}

void rd_time(Reader &r, TimeSec &t) {
    t.env_time = r.f64(), t.next_psi = r.f64(), t.last_psi = r.f64();
    t.mgr_delta_ms = r.f32(), t.mgr_step_s = r.f32(), t.time_factor = r.f32(), t.delta_ms = r.f32();
    t.sr_double = r.f64();
    t.sr_float = r.f32();
    t.sr_running = r.u8();
    t.frames_covered = r.u32();
}
void rd_ball(Reader &r, BallSec &b) {
    b.ckid = r.u32();
    b.type = r.u8(), b.found = r.u8();
    r.arr(b.ck_matrix, 12);
    r.arr(b.pos, 3), r.arr(b.q_last, 4), r.arr(b.q_next, 4);
    b.tpsi = r.f64();
    r.arr(b.speed, 3), r.arr(b.speed_change, 3), r.arr(b.rot_speed, 3), r.arr(b.rot_speed_change, 3), r.arr(b.rot_axis, 3);
    b.current_speed = r.f32(), b.abs_omega = r.f32();
    b.core_flags = r.u32(), b.core_state = r.u32();
    b.extra_radius = r.f32();
}
void rd_fric(Reader &r, FricItem &f) {
    f.partner = r.u32();
    f.self_status = r.u8(), f.partner_status = r.u8(), f.cp_status = r.u8();
    r.u8();
    f.gap = r.f32(), f.friction = r.f32();
    r.arr(f.lt_normal, 3);
    r.arr(f.lt_point, 3);
    f.lt_valid = r.u8();
}
void rd_mind(Reader &r, MindItem &m) {
    m.partner = r.u32();
    m.self_status = r.u8(), m.partner_status = r.u8();
    r.u16();
    m.gap = r.f32();
    r.arr(m.plane, 4);
    m.sum_extra_radius = r.f32();
    m.ledge = r.u32();
    m.tri_index = r.u16();
    m.edge_index = r.i8();
    m.n_tris = r.u8();
    r.arr(m.tri, 9), r.arr(m.tri_adj, 9);
}
void rd_body(Reader &r, BodyItem &b) {
    b.ckid = r.u32(), b.core_state = r.u32();
    r.arr(b.pos, 3), r.arr(b.q, 4), r.arr(b.q_cfo, 4);
    r.arr(b.shift, 3), r.arr(b.speed, 3), r.arr(b.rot_speed, 3), r.arr(b.ck_matrix, 12);
}
void rd_evt(Reader &r, EvtItem &e) { e.frame = r.u32(), e.code = r.u16(), e.param = r.u16(); }
void rd_wtch(Reader &r, WtchItem &w) {
    w.ckid = r.u32();
    w.visible = r.u8();
    r.arr(w.pos, 3);
}
void rd_hs(Reader &r, HsFields &h) {
    h.hs_valid = r.u8();
    r.arr(h.types, 6);
    h.bonus_type = r.u8();
    r.arr(h.energy, 6);
    h.level_bonus = r.u32();
    h.bml_hs = r.i32(), h.hs = r.i32();
}

// list section: count (u16 / u32), items of `size` bytes; newer section versions may append item fields
template <typename T, typename F>
std::vector<T> rd_list(const Section &s, size_t size, bool u32count, F fn) {
    Reader r(s.p, s.n);
    size_t csz = u32count ? 4 : 2;
    if (s.n < csz) throw ProtocolError(tag_str(s.tag) + ": no count");
    size_t n = u32count ? r.u32() : r.u16();
    size_t body = s.n - csz, stride = size;
    if (s.version > 1 && n) {
        stride = body / n;
        if (stride < size) throw ProtocolError(tag_str(s.tag) + ": item stride too small");
    }
    if (n * stride > body) throw ProtocolError(tag_str(s.tag) + ": items exceed payload");
    std::vector<T> out(n);
    for (size_t i = 0; i < n; ++i) {
        Reader it(s.p + csz + i * stride, size);
        fn(it, out[i]);
    }
    return out;
}

}  // namespace

Observation parse_observation(const char *data, size_t n) {
    Message m = split_message(data, n, OBS_MAGIC);
    if (m.version != OBS_VERSION) throw ProtocolError("OBS version unsupported");
    Observation o;
    o.frames_done = m.frames_done;
    o.flags = m.flags;
    for (auto &s : m.sections) {
        std::string t = tag_str(s.tag);
        Reader r(s.p, s.n);
        if (t == "TIME") {
            if (s.n < 57) throw ProtocolError("TIME: short payload");
            TimeSec ts;
            rd_time(r, ts);
            o.time = ts;
        } else if (t == "BALL") {
            if (s.n < 230) throw ProtocolError("BALL: short payload");
            BallSec b;
            rd_ball(r, b);
            o.ball = b;
        } else if (t == "CAMS") {
            if (s.n < 264) throw ProtocolError("CAMS: short payload");
            CamsSec c;
            for (auto &cam : c.cams) cam.ckid = r.u32(), r.arr(cam.matrix, 12);
            c.attached = r.u32();
            o.cams = c;
        } else if (t == "SCOR") {
            if (s.n < 56) throw ProtocolError("SCOR: short payload");
            ScorSec sc;
            rd_hs(r, sc.hs);
            sc.ball_offs = r.u32(), sc.last_balloff = r.u32(), sc.last_ballnav = r.u32();
            o.scor = sc;
            if (s.n >= 60) {
                ScorExt e;
                e.followers = r.u16(), e.sources = r.u16();
                o.scor_ext = e;
            }
        } else if (t == "SREN") {
            if (s.n < 80) throw ProtocolError("SREN: short payload");
            SrenSec sr;
            sr.sr_double = r.f64();
            sr.sr_f32sum = r.f32(), sr.sr_bml = r.f32();
            sr.tail_frames = r.u32();
            sr.level_finished = r.u8(), sr.counter_inactive_seen = r.u8(), sr.sr_running = r.u8();
            r.u8();
            sr.sr_double_at_finish = r.f64();
            sr.sr_f32sum_at_finish = r.f32();
            rd_hs(r, sr.hs);
            o.sren = sr;
            if (s.n >= 88) {
                SrenExt e;
                e.followers_at_finish = r.u16(), e.followers_end = r.u16(), e.post_frames = r.u32();
                o.sren_ext = e;
            }
        } else if (t == "HASH") {
            if (s.n < 12) throw ProtocolError("HASH: short payload");
            HashSec h;
            h.trace_hash = r.u64(), h.frames = r.u32();
            o.hash = h;
        } else if (t == "GLCT") {
            if (s.n < 30) throw ProtocolError("GLCT: short payload");
        } else if (t == "FRIC") {
            o.fric = rd_list<FricItem>(s, 53, false, rd_fric);
        } else if (t == "MIND") {
            o.mind = rd_list<MindItem>(s, 112, false, rd_mind);
        } else if (t == "BODY") {
            o.body = rd_list<BodyItem>(s, 180, false, rd_body);
        } else if (t == "EVTS") {
            o.events = rd_list<EvtItem>(s, 8, false, rd_evt);
        } else if (t == "WTCH") {
            o.wtch = rd_list<WtchItem>(s, 17, false, rd_wtch);
        } else if (t == "TRAC") {
            rd_list<int>(s, 52, true, [](Reader &, int &) {});
        } else if (t == "ARRS") {
            size_t cnt = r.u16();
            for (size_t i = 0; i < cnt; ++i) {
                ArrCell c;
                c.array = r.u32(), c.row = r.u16(), c.col = r.u16(), c.col_type = r.u8();
                uint8_t ln = r.u8();
                if (ln > r.left()) throw ProtocolError("ARRS cell beyond payload");
                c.data = r.raw(ln);
                o.arrs.push_back(std::move(c));
            }
        } else if (t == "PHYS") {
            size_t cnt = r.u16();
            o.phys.resize(cnt);
            r.arr(o.phys.data(), cnt);
        } else {
            o.unknown.push_back(t);
        }
    }
    return o;
}

Bytes encode_action(const Action &a, uint32_t py_seq) {
    if (a.repeat < 1) throw Error("repeat must be >= 1");
    Bytes out;
    put<uint32_t>(out, ACT_MAGIC);
    put<uint16_t>(out, ACT_VERSION);
    put<uint16_t>(out, a.flags);
    put<uint32_t>(out, py_seq);
    put<uint32_t>(out, uint32_t(a.frames.size()));
    put<uint32_t>(out, a.repeat);
    put<uint32_t>(out, uint32_t(a.stop_mask & 0xFFFFFFFFu));
    put<uint32_t>(out, uint32_t(a.stop_mask >> 32));
    put<uint32_t>(out, a.obs_mask);
    put<uint32_t>(out, 0);
    for (auto &f : a.frames) put<float>(out, f.delta), put<uint32_t>(out, f.keys);
    return out;
}

// ---------------------------------------------------------------------------------------------- level messages

namespace {

struct LReader {
    Reader r;
    const std::vector<std::string> &strings;
    std::optional<std::string> s() {
        uint32_t i = r.u32();
        if (i == NO_STRING) return std::nullopt;
        return lookup(i);
    }
    std::string lookup(uint32_t i) {
        if (i >= strings.size()) throw ProtocolError("string index outside table");
        return strings[i];
    }
};

void rd_ents(LReader &L, LevelMessage &lm) {
    uint32_t n = L.r.u32();
    for (uint32_t k = 0; k < n; ++k) {
        Entity e;
        e.ckid = L.r.u32();
        uint32_t name_i = L.r.u32();
        if (e.ckid != 0 && name_i != NO_STRING) e.name = L.lookup(name_i);
        e.class_id = L.r.u32(), e.parent = L.r.u32();
        e.visible = L.r.u8() != 0;
        L.r.arr(e.matrix, 12);
        e.contact_id = L.r.i32(), e.coll_id = L.r.i32();
        e.floor = L.r.raw(L.r.u16());
        lm.entities.push_back(std::move(e));
    }
}
void rd_grps(LReader &L, LevelMessage &lm) {
    uint32_t n = L.r.u32();
    for (uint32_t k = 0; k < n; ++k) {
        Group g;
        g.ckid = L.r.u32();
        g.name = L.s();
        g.members.resize(L.r.u32());
        L.r.arr(g.members.data(), g.members.size());
        lm.groups.push_back(std::move(g));
    }
}
void rd_arrs(LReader &L, LevelMessage &lm) {
    uint32_t n = L.r.u32();
    for (uint32_t k = 0; k < n; ++k) {
        ArrayDef a;
        a.ckid = L.r.u32();
        a.name = L.s();
        uint16_t rows = L.r.u16(), ncols = L.r.u16();
        for (uint16_t c = 0; c < ncols; ++c) {
            ArrayColumn col;
            col.name = L.s();
            col.type = L.r.u8();
            col.param_type = L.s();
            a.columns.push_back(std::move(col));
        }
        a.cells.resize(rows);
        for (uint16_t rr = 0; rr < rows; ++rr)
            for (uint16_t c = 0; c < ncols; ++c) a.cells[rr].push_back(L.r.raw(L.r.u8()));
        lm.arrays.push_back(std::move(a));
    }
}
void rd_phys(LReader &L, LevelMessage &lm) {
    uint32_t n = L.r.u32();
    for (uint32_t k = 0; k < n; ++k) {
        PhysRecord p;
        p.seq = L.r.u32(), p.frame = L.r.u32(), p.ckid = L.r.u32();
        p.unphysicalize = L.r.u8() != 0, p.fixed = L.r.u8() != 0;
        p.friction = L.r.f32(), p.elasticity = L.r.f32(), p.mass = L.r.f32();
        p.collision_group = L.s();
        p.frozen = L.r.u8() != 0, p.collision = L.r.u8() != 0, p.calc_mass_center = L.r.u8() != 0;
        p.linear_damping = L.r.f32(), p.rot_damping = L.r.f32();
        p.surface = L.s();
        L.r.arr(p.mass_center, 3);
        p.convex.resize(L.r.u16());
        L.r.arr(p.convex.data(), p.convex.size());
        uint16_t nb = L.r.u16();
        for (uint16_t b = 0; b < nb; ++b) {
            PhysRecord::Ball ball;
            L.r.arr(ball.center, 3);
            ball.radius = L.r.f32();
            p.balls.push_back(ball);
        }
        p.concave.resize(L.r.u16());
        L.r.arr(p.concave.data(), p.concave.size());
        lm.phys.push_back(std::move(p));
    }
}
void rd_mesh(LReader &L, LevelMessage &lm) {
    uint32_t n = L.r.u32();
    for (uint32_t k = 0; k < n; ++k) {
        Mesh m;
        m.ckid = L.r.u32(), m.mesh = L.r.u32();
        m.kind = L.r.u8();
        L.r.arr(m.matrix, 16);
        m.vertices.resize(size_t(3) * L.r.u32());
        L.r.arr(m.vertices.data(), m.vertices.size());
        m.faces.resize(size_t(3) * L.r.u32());
        L.r.arr(m.faces.data(), m.faces.size());
        lm.meshes.push_back(std::move(m));
    }
}
void rd_ivpo(LReader &L, LevelMessage &lm) {
    uint32_t n = L.r.u32();
    for (uint32_t k = 0; k < n; ++k) {
        IvpObject o;
        o.ckid = L.r.u32();
        o.object_type = L.r.i32();
        o.unmoveable = L.r.u8() != 0;
        o.extra_radius = L.r.f32();
        o.material = L.r.raw(32);
        lm.ivp.push_back(std::move(o));
    }
}
void rd_dept(LReader &L, LevelMessage &lm) {
    uint32_t n = L.r.u32();
    for (uint32_t k = 0; k < n; ++k) {
        DepthBox d;
        d.ckid = L.r.u32();
        L.r.arr(d.lo, 3), L.r.arr(d.hi, 3);
        lm.depth.push_back(d);
    }
}
void rd_wlst(LReader &L, LevelMessage &lm) {
    uint32_t n = L.r.u32();
    for (uint32_t k = 0; k < n; ++k) {
        uint32_t ck = L.r.u32();
        lm.watch.emplace_back(ck, L.s());
    }
}
void rd_emsh(LReader &L, LevelMessage &lm) {
    uint32_t n = L.r.u32();
    for (uint32_t k = 0; k < n; ++k) {
        uint32_t e = L.r.u32(), m = L.r.u32();
        lm.entity_meshes.emplace_back(e, m);
    }
}

}  // namespace

LevelMessage parse_level_message(const char *data, size_t n, size_t *len) {
    Message m = split_message(data, n, LVL_MAGIC);
    LevelMessage lm;
    lm.frames_done = m.frames_done;
    for (auto &s : m.sections)
        if (tag_str(s.tag) == "STRS") {
            Reader r(s.p, s.n);
            uint32_t cnt = r.u32();
            for (uint32_t i = 0; i < cnt; ++i) {
                uint16_t l = r.u16();
                Bytes b = r.raw(l);  // latin-1, kept as bytes
                lm.strings.push_back(b);
            }
            break;
        }
    for (auto &s : m.sections) {
        std::string t = tag_str(s.tag);
        if (t == "STRS") continue;
        LReader L{Reader(s.p, s.n), lm.strings};
        try {
            if (t == "ENTS") rd_ents(L, lm);
            else if (t == "GRPS") rd_grps(L, lm);
            else if (t == "ARRS") rd_arrs(L, lm);
            else if (t == "PHYS") rd_phys(L, lm);
            else if (t == "MESH") rd_mesh(L, lm);
            else if (t == "IVPO") rd_ivpo(L, lm);
            else if (t == "DEPT") rd_dept(L, lm);
            else if (t == "WLST") rd_wlst(L, lm);
            else if (t == "EMSH") rd_emsh(L, lm);
            else if (t == "LTAB") { /* camera ids, contact blocks: not used by the features */ }
            else lm.unknown.push_back(t);
        } catch (const ProtocolError &e) {
            throw ProtocolError("LVL1 " + t + ": " + e.what());
        }
    }
    if (len) *len = m.byte_len;
    return lm;
}

std::vector<LevelMessage> parse_level_stream(const char *data, size_t n, size_t *used) {
    std::vector<LevelMessage> out;
    size_t pos = 0;
    while (n - pos >= 24) {
        uint32_t bl;
        std::memcpy(&bl, data + pos + 8, 4);
        if (pos + bl > n) break;
        size_t len = 0;
        out.push_back(parse_level_message(data + pos, n - pos, &len));
        pos += len;
    }
    if (used) *used = pos;
    return out;
}

// ---------------------------------------------------------------------------------------------- cells

bool CellValue::to_int(int64_t &out) const {
    switch (kind) {
        case Int: out = i; return true;
        case Bool: out = i; return true;
        case Float:
            if (!std::isfinite(f)) return false;
            out = int64_t(std::trunc(f));
            return true;
        default: return false;
    }
}

static std::string lower(const std::optional<std::string> &s) {
    std::string t = s ? *s : std::string();
    for (auto &c : t) c = char(std::tolower(static_cast<unsigned char>(c)));
    return t;
}

CellValue decode_cell(const Bytes &data, uint8_t col_type, const std::optional<std::string> &param_type) {
    CellValue v;
    if (col_type == ARR_INT || col_type == ARR_OBJECT) {
        v.kind = CellValue::Int;
        if (data.size() >= 4) {
            if (col_type == ARR_INT) {
                int32_t x;
                std::memcpy(&x, data.data(), 4);
                v.i = x;
            } else {
                uint32_t x;
                std::memcpy(&x, data.data(), 4);
                v.i = x;
            }
        }
        return v;
    }
    if (col_type == ARR_FLOAT) {
        v.kind = CellValue::Float;
        if (data.size() >= 4) {
            float x;
            std::memcpy(&x, data.data(), 4);
            v.f = x;
        }
        return v;
    }
    if (col_type == ARR_STRING) {
        v.kind = CellValue::String;
        size_t e = data.size();
        while (e && data[e - 1] == '\0') --e;
        v.s = data.substr(0, e);
        return v;
    }
    if (col_type == ARR_PARAMETER) {
        std::string pt = lower(param_type);
        if (data.size() == 64 || pt == "matrix") {
            if (data.size() < 64) {  // numpy would raise on the reshape
                v.kind = CellValue::None;
                return v;
            }
            v.kind = CellValue::Matrix;
            std::memcpy(v.m, data.data(), 64);
            return v;
        }
        if (pt == "boolean" || pt == "bool") {
            v.kind = CellValue::Bool;
            int32_t x = 0;
            if (data.size() >= 4) std::memcpy(&x, data.data(), 4);
            v.i = x != 0;
            return v;
        }
        if ((pt == "float" || pt == "time" || pt == "percentage") && data.size() >= 4) {
            v.kind = CellValue::Float;
            float x;
            std::memcpy(&x, data.data(), 4);
            v.f = x;
            return v;
        }
        if (data.size() == 4) {
            v.kind = CellValue::Int;
            int32_t x;
            std::memcpy(&x, data.data(), 4);
            v.i = x;
            return v;
        }
        v.kind = CellValue::Raw;
        v.s = data;
        return v;
    }
    v.kind = CellValue::Raw;
    v.s = data;
    return v;
}

}  // namespace ballrain
