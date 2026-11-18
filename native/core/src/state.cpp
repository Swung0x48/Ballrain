#include "ballrain/state.h"

#include <algorithm>
#include <atomic>
#include <cmath>

namespace ballrain {

const char *const G_FLOORS = "Phys_Floors";
const char *const G_RAILS = "Phys_FloorRails";
const char *const G_STOPPER = "Phys_FloorStopper";
const char *const G_DEPTH = "DepthTestCubes";

// ---------------------------------------------------------------------------------------------- ArrayStore

void ArrayStore::define(const ArrayDef &a, bool seed_cells) {
    if (a.name) names_[*a.name] = a.ckid;
    columns_[a.ckid] = a.columns;
    int &nr = nrows_[a.ckid];
    nr = std::max(nr, int(a.cells.size()));
    if (seed_cells)
        for (size_t r = 0; r < a.cells.size(); ++r)
            for (size_t c = 0; c < a.cells[r].size(); ++c) {
                uint64_t k = key(a.ckid, uint32_t(r), uint32_t(c));
                if (!cells_.count(k)) cells_[k] = Cell{a.columns[c].type, a.cells[r][c]};  // obs deltas are newer
            }
}

void ArrayStore::apply(const std::vector<ArrCell> &cells) {
    for (auto &c : cells) {
        cells_[key(c.array, c.row, c.col)] = Cell{c.col_type, c.data};
        int &nr = nrows_[c.array];
        nr = std::max(nr, int(c.row) + 1);
    }
}

std::optional<uint32_t> ArrayStore::ckid(const std::string &name) const {
    auto it = names_.find(name);
    if (it == names_.end()) return std::nullopt;
    return it->second;
}

int ArrayStore::rows(const std::string &name) const {
    auto k = ckid(name);
    if (!k) return 0;
    auto it = nrows_.find(*k);
    return it == nrows_.end() ? 0 : it->second;
}

std::optional<CellValue> ArrayStore::get(const std::string &name, int row, int col) const {
    auto k = ckid(name);
    if (!k) return std::nullopt;
    auto it = cells_.find(key(*k, uint32_t(row), uint32_t(col)));
    if (it == cells_.end()) return std::nullopt;
    std::optional<std::string> pt;
    auto ci = columns_.find(*k);
    if (ci != columns_.end() && col < int(ci->second.size())) pt = ci->second[size_t(col)].param_type;
    return decode_cell(it->second.data, it->second.type, pt);
}

int64_t ArrayStore::get_int(const std::string &name, int row, int col, int64_t dflt) const {
    auto v = get(name, row, col);
    if (!v) return dflt;
    int64_t out;
    return v->to_int(out) ? out : dflt;
}

// ---------------------------------------------------------------------------------------------- LevelData

void LevelData::merge(const LevelMessage &lm) {
    for (auto &e : lm.entities) {
        if (e.ckid == 0) continue;
        entities[e.ckid] = e;
        if (e.name) by_name.emplace(*e.name, e.ckid);  // setdefault: the first one wins
    }
    for (auto &g : lm.groups) {
        if (!g.name) continue;
        auto old = groups.find(*g.name);
        if (old != groups.end())
            for (auto m : old->second.members) {
                auto it = ent_groups.find(m);
                if (it != ent_groups.end()) it->second.erase(*g.name);
            }
        groups[*g.name] = g;
        for (auto m : g.members) ent_groups[m].insert(*g.name);
    }
    std::vector<const PhysRecord *> ps;
    for (auto &p : lm.phys) ps.push_back(&p);
    std::stable_sort(ps.begin(), ps.end(), [](auto *a, auto *b) { return a->seq < b->seq; });
    for (auto *p : ps) {
        phys[p->seq] = *p;
        auto cur = phys_by_ent.find(p->ckid);
        if (cur == phys_by_ent.end() || p->seq >= cur->second.seq) phys_by_ent[p->ckid] = *p;
    }
    std::set<uint32_t> physicalized;
    for (auto &p : lm.phys)
        if (!p.unphysicalize && p.frame == lm.frames_done) physicalized.insert(p.ckid);
    for (auto &m : lm.meshes) {
        auto key = std::make_pair(m.ckid, m.mesh);
        auto it = meshes.find(key);
        if (it == meshes.end()) mesh_order[m.ckid].push_back(m.mesh);
        MeshEntry &e = meshes[key];
        e.mesh = m;
        e.serial = ++serial_;
        if (physicalized.count(m.ckid)) {
            auto old = phys_meshes.find(key);
            if (old == phys_meshes.end() || lm.frames_done >= old->second.first)
                phys_meshes[key] = std::make_pair(lm.frames_done, m);
        }
    }
    for (auto &o : lm.ivp) ivp[o.ckid] = o;
    if (!lm.depth.empty()) depth = lm.depth;
    if (!lm.watch.empty()) watch = lm.watch;
    ++version;
}

const std::string *LevelData::name(uint32_t ckid) const {
    auto it = entities.find(ckid);
    if (it == entities.end() || !it->second.name) return nullptr;
    return &*it->second.name;
}

bool LevelData::in_group(uint32_t ckid, const std::string &group) const {
    auto it = ent_groups.find(ckid);
    return it != ent_groups.end() && it->second.count(group);
}

int32_t LevelData::contact_id(uint32_t ckid) const {
    auto it = entities.find(ckid);
    return it == entities.end() ? -1 : it->second.contact_id;
}

bool LevelData::is_physicalized(uint32_t ckid) const {
    auto it = phys_by_ent.find(ckid);
    return it != phys_by_ent.end() && !it->second.unphysicalize;
}

bool LevelData::is_fixed(uint32_t ckid) const {
    auto it = phys_by_ent.find(ckid);
    return it != phys_by_ent.end() && it->second.fixed;
}

std::vector<const MeshEntry *> LevelData::meshes_of(uint32_t ckid) const {
    std::vector<const MeshEntry *> out;
    auto it = mesh_order.find(ckid);
    if (it == mesh_order.end()) return out;
    for (auto m : it->second) out.push_back(&meshes.at(std::make_pair(ckid, m)));
    return out;
}

std::vector<uint32_t> LevelData::static_entities() const {
    std::vector<uint32_t> out;
    for (auto &kv : phys_by_ent)
        if (!kv.second.unphysicalize && kv.second.fixed && !in_group(kv.first, G_STOPPER)) out.push_back(kv.first);
    std::sort(out.begin(), out.end());
    return out;
}

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    auto b = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

uint64_t LevelData::static_key() const {
    uint64_t h = 0xcbf29ce484222325ull;
    for (auto c : static_entities()) {
        std::vector<const MeshEntry *> ms = meshes_of(c);
        std::stable_sort(ms.begin(), ms.end(), [](auto *a, auto *b) { return a->mesh.mesh < b->mesh.mesh; });
        h = fnv(h, &c, 4);
        for (auto *e : ms) {
            h = fnv(h, e->mesh.matrix, sizeof e->mesh.matrix);
            h = fnv(h, e->mesh.vertices.data(), e->mesh.vertices.size() * 4);
            h = fnv(h, e->mesh.faces.data(), e->mesh.faces.size() * 4);
        }
    }
    return h;
}

bool LevelData::entity_position(const std::string &nm, double out[3]) const {
    auto it = by_name.find(nm);
    if (it == by_name.end()) return false;
    const Entity &e = entities.at(it->second);
    for (int k = 0; k < 3; ++k) out[k] = e.matrix[9 + k];
    return true;
}

bool LevelData::physical_position(const std::string &nm, double out[3]) const {
    auto it = by_name.find(nm);
    if (it == by_name.end()) return false;
    uint32_t ck = it->second;
    std::vector<const Mesh *> ms;
    for (auto &kv : phys_meshes)
        if (kv.first.first == ck) ms.push_back(&kv.second.second);
    if (ms.empty())
        for (auto *e : meshes_of(ck)) ms.push_back(&e->mesh);
    bool any = false;
    double lo[3], hi[3];
    for (auto *m : ms) {
        size_t nv = m->vertices.size() / 3;
        if (!nv) continue;
        const float *M = m->matrix;
        for (size_t i = 0; i < nv; ++i) {
            const float *v = &m->vertices[3 * i];
            for (int c = 0; c < 3; ++c) {
                double w = double(v[0]) * M[0 * 4 + c] + double(v[1]) * M[1 * 4 + c] + double(v[2]) * M[2 * 4 + c] +
                           double(M[3 * 4 + c]);
                if (!any || w < lo[c]) lo[c] = (!any ? w : std::min(lo[c], w));
                if (!any || w > hi[c]) hi[c] = (!any ? w : std::max(hi[c], w));
            }
            any = true;
        }
    }
    if (!any) return entity_position(nm, out);
    for (int c = 0; c < 3; ++c) out[c] = 0.5 * (lo[c] + hi[c]);
    return true;
}

// ---------------------------------------------------------------------------------------------- GameTracker

static std::atomic<uint64_t> g_tracker_serial{0};

GameTracker::GameTracker(int level) : level_no(level), serial(++g_tracker_serial) {}

void GameTracker::merge_level(const LevelMessage &lm) {
    level.merge(lm);
    for (auto &a : lm.arrays) arrays.define(a);
}

static int ball_name_type(const std::string *n) {
    if (!n) return -1;
    if (*n == "Ball_Wood") return 0;
    if (*n == "Ball_Paper") return 1;
    if (*n == "Ball_Stone") return 2;
    return -1;
}

bool GameTracker::is_ball_record(uint32_t seq) const {
    auto it = level.phys.find(seq);
    return it != level.phys.end() && ball_name_type(level.name(it->second.ckid)) >= 0;
}

void GameTracker::update(Observation o) {
    frames_done = o.frames_done;
    arrays.apply(o.arrs);
    for (auto &w : o.wtch) watch[w.ckid] = {int(w.visible), {w.pos[0], w.pos[1], w.pos[2]}};
    Flags &f = flags;
    for (auto &e : o.events) {
        uint32_t n = e.frame;
        uint16_t code = e.code;
        events.push_back({n, code, e.param});
        switch (code) {
            case Ev::StartLevel: f.started = true; break;
            case Ev::Unphysicalize:
            case Ev::Physicalize:
                if (is_ball_record(e.param))
                    trafo_ = (code == Ev::Unphysicalize && f.started && !f.ball_off && !f.finished);
                break;
            case Ev::BallNavActive:
                f.ball_nav = true, f.ball_off = false;
                trafo_ = false;
                if (!control_start_frame) control_start_frame = n;
                break;
            case Ev::BallNavInactive: f.ball_nav = false; break;
            case Ev::CamNavActive: f.cam_nav = true; break;
            case Ev::CamNavInactive: f.cam_nav = false; break;
            case Ev::CounterActive: f.counter = true; break;
            case Ev::CounterInactive: f.counter = false; break;
            case Ev::PauseLevel: f.paused = true, paused_since = n; break;
            case Ev::UnpauseLevel: f.paused = false, paused_since.reset(); break;
            case Ev::BallOff:
                f.ball_off = true;
                trafo_ = false;
                ++deaths;
                break;
            case Ev::PreCheckpoint: last_checkpoint_frame = n; break;
            case Ev::PostCheckpoint: ++checkpoints; break;
            case Ev::ExtraPoint: ++extra_points; break;
            case Ev::PostLifeUp: ++life_ups; break;
            case Ev::LevelFinish:
                f.finished = true;
                trafo_ = false;
                levelfinish_frame = n;
                break;
            case Ev::GameOver: f.game_over = true; break;
            default: break;
        }
    }
    obs = std::move(o);
}

int GameTracker::points() const { return int(arrays.get_int("Energy", 0, 0, 0)); }
int GameTracker::lives() const { return int(arrays.get_int("Energy", 0, 1, 0)); }
int GameTracker::life_bonus() const { return int(arrays.get_int("Energy", 0, 5, 200)); }

int GameTracker::sector() const {
    auto v = arrays.get("IngameParameter", 0, 1);
    int64_t s = 0;
    if (v) {
        if (!v->to_int(s)) s = 0;
    }
    return s > 0 ? int(s) : 1 + checkpoints;
}

std::optional<uint32_t> GameTracker::active_ball() const {
    int64_t v = arrays.get_int("CurrentLevel", 0, 1, 0);
    if (!v) return std::nullopt;
    return uint32_t(v);
}

std::optional<double> GameTracker::sr_ms() const {
    if (!obs || !obs->time) return std::nullopt;
    return obs->time->sr_double;
}

int GameTracker::ball_type() const {
    if (obs && obs->ball && obs->ball->type <= 2) return obs->ball->type;
    auto ab = active_ball();
    if (!ab) return 255;
    int t = ball_name_type(level.name(*ab));
    return t >= 0 ? t : 255;
}

bool GameTracker::control_valid() const {
    return flags.ball_nav && !flags.ball_off && !trafo_ && obs && obs->ball && obs->ball->found;
}

// P_Extra_Point_Ball([0-6])(\d{3})?
static bool follower_name(const std::string &n, std::string &inst, int &k) {
    static const std::string pre = "P_Extra_Point_Ball";
    if (n.compare(0, pre.size(), pre) != 0) return false;
    std::string rest = n.substr(pre.size());
    if (rest.empty() || rest[0] < '0' || rest[0] > '6') return false;
    k = rest[0] - '0';
    std::string tail = rest.substr(1);
    if (tail.empty()) {
        inst.clear();
        return true;
    }
    if (tail.size() != 3) return false;
    for (char c : tail)
        if (c < '0' || c > '9') return false;
    inst = tail;
    return true;
}

const std::map<uint32_t, std::pair<std::string, int>> &GameTracker::follower_slots() const {
    if (fslots_version_ != level.version) {
        fslots_.clear();
        for (auto &w : level.watch) {
            const std::string *n = level.name(w.first);
            std::string inst;
            int k;
            if (follower_name(n ? *n : std::string(), inst, k)) fslots_[w.first] = {inst, k};
        }
        fslots_version_ = level.version;
    }
    return fslots_;
}

FollowerState GameTracker::follower_state() const {
    std::map<std::string, int> mains;
    std::vector<std::pair<std::string, std::array<float, 3>>> smalls;
    // Python iterates the slots dict in watch-list order; the result only depends on the multiset
    for (auto &kv : follower_slots()) {
        auto w = watch.find(kv.first);
        if (w == watch.end()) continue;
        if (kv.second.second == 0) mains[kv.second.first] = w->second.first;
        else if (w->second.first) smalls.emplace_back(kv.second.first, w->second.second);
    }
    FollowerState st;
    for (auto &s : smalls) {
        auto m = mains.find(s.first);
        if (m != mains.end() && m->second == 0) {
            st.per_source[s.first] += 1;
            st.positions.push_back({double(s.second[0]), double(s.second[1]), double(s.second[2])});
        }
    }
    for (auto &kv : st.per_source) st.total += kv.second;
    return st;
}

int GameTracker::followers() const {
    if (obs && obs->scor_ext) return obs->scor_ext->followers;
    return follower_state().total;
}

std::optional<double> GameTracker::nearest_follower(const double pos[3]) const {
    auto st = follower_state();
    if (st.positions.empty()) return std::nullopt;
    double best = 0;
    bool any = false;
    for (auto &p : st.positions) {
        double dx = p[0] - pos[0], dy = p[1] - pos[1], dz = p[2] - pos[2];
        double d = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (!any || d < best) best = d, any = true;
    }
    return best;
}

uint32_t GameTracker::frames_since_checkpoint() const {
    std::optional<uint32_t> ref = last_checkpoint_frame ? last_checkpoint_frame : control_start_frame;
    if (!ref) return 0;
    return frames_done > *ref ? frames_done - *ref : 0;
}

uint32_t GameTracker::paused_frames() const {
    if (!paused_since) return 0;
    return frames_done > *paused_since ? frames_done - *paused_since : 0;
}

}  // namespace ballrain
