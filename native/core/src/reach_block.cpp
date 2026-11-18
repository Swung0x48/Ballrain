#include "ballrain/reach_block.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <set>
#include <tuple>

#include "ballrain/features.h"
#include "ballrain/sha256.h"
#include "ballrain/util.h"

namespace ballrain {

std::string reach_version() {
    return "fastreach" + std::to_string(fastreach::BUILD_VERSION) + "." + std::to_string(fastreach::QUERY_VERSION) +
           "-block" + std::to_string(REACH_BLOCK_VERSION);
}

// ---------------------------------------------------------------------------------------------- constants (fastreach)
static const double DEATH_MIN_EXTENT = 20.0, R_BALL = 2.0;
static const double CONSTS[6] = {20.0, 66.0, 1.4, 0.5, 1.95, 0.1};  // GRAVITY PSI_HZ ROLL_INERTIA UP_FACE MOUNT_CAP MOUNT_FREE
static const double FAN_LOCAL_LO[3] = {-1.0, 0.0, -1.0}, FAN_LOCAL_HI[3] = {1.0, 16.0, 1.0};
static const double FAN_FORCE = 0.1;
static const char *const FAN_ENTITY = "P_Modul_18_Kollisionsquader";
static const double UNKNOWN_R = 8.0;
static const char *const BALL_NAMES[3] = {"Ball_Wood", "Ball_Paper", "Ball_Stone"};  // state.BALL_NAME_TYPES

static void put_f8(Sha256 &h, double v) {
    unsigned char b[8];
    std::memcpy(b, &v, 8);  // little-endian hosts only (x86, arm64)
    h.update(b, 8);
}

// ---------------------------------------------------------------------------------------------- canonical geometry

CanonicalGeometry canonical_geometry(const GameTracker &t, const LevelGeometry &geo) {
    const LevelData &lvl = t.level;
    const TriangleSet &T = geo.tris;
    size_t n = T.size();
    auto row = [&](size_t i) {
        return std::array<double, 9>{T.A[i][0], T.A[i][1], T.A[i][2], T.B[i][0], T.B[i][1], T.B[i][2],
                                     T.C[i][0], T.C[i][1], T.C[i][2]};
    };
    std::set<int64_t> rail_own(geo.rails.owner.begin(), geo.rails.owner.end());
    // entities: np.unique(owner) (sorted), each keyed by (name or "", its lexicographically smallest triangle)
    std::vector<int64_t> uo(T.owner.begin(), T.owner.end());
    std::sort(uo.begin(), uo.end());
    uo.erase(std::unique(uo.begin(), uo.end()), uo.end());
    std::map<int64_t, size_t> uidx;
    for (size_t i = 0; i < uo.size(); ++i) uidx[uo[i]] = i;
    std::vector<std::array<double, 9>> first(uo.size());
    std::vector<bool> have(uo.size(), false);
    for (size_t i = 0; i < n; ++i) {
        size_t u = uidx[T.owner[i]];
        auto r = row(i);
        if (!have[u] || r < first[u]) first[u] = r, have[u] = true;
    }
    std::vector<std::string> names(uo.size());
    for (size_t i = 0; i < uo.size(); ++i) {
        const std::string *nm = lvl.name(uint32_t(uo[i]));
        names[i] = nm ? *nm : "";
    }
    std::vector<size_t> order(uo.size());
    std::iota(order.begin(), order.end(), 0);
    // sorted(range(len(uo)), key=lambda i: ekey[i]): a stable sort by (name, triangle)
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return std::tie(names[a], first[a]) < std::tie(names[b], first[b]);
    });
    std::vector<int64_t> rank(uo.size());
    for (size_t k = 0; k < order.size(); ++k) rank[order[k]] = int64_t(k);
    std::vector<int64_t> r(n);
    for (size_t i = 0; i < n; ++i) r[i] = rank[uidx[T.owner[i]]];
    // np.lexsort((C.z, ..., A.x, rank)): by rank, then the 9 coordinates; stable
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<std::array<double, 9>> rows(n);
    for (size_t i = 0; i < n; ++i) rows[i] = row(i);
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        if (r[a] != r[b]) return r[a] < r[b];
        return rows[a] < rows[b];
    });
    CanonicalGeometry g;
    g.A.reserve(3 * n), g.B.reserve(3 * n), g.C.reserve(3 * n), g.rank.reserve(n), g.rail.reserve(n);
    for (size_t i : idx) {
        for (int k = 0; k < 3; ++k) g.A.push_back(rows[i][k]), g.B.push_back(rows[i][3 + k]), g.C.push_back(rows[i][6 + k]);
        g.rank.push_back(r[i]);
        g.rail.push_back(rail_own.count(T.owner[i]) ? 1 : 0);
    }
    return g;
}

// ---------------------------------------------------------------------------------------------- fans, zones, key

std::vector<double> reach_fan_obbs(const LevelData &lvl) {
    std::vector<double> out;
    const size_t fl = std::strlen(FAN_ENTITY);
    for (auto &kv : lvl.by_name) {  // std::map: sorted by name (sorted(level.by_name.items()))
        const std::string &name = kv.first;
        if (name.compare(0, fl, FAN_ENTITY) != 0) continue;
        std::string tail = name.substr(fl);
        bool digits = !tail.empty() && std::all_of(tail.begin(), tail.end(), [](char c) { return c >= '0' && c <= '9'; });
        if (!(tail.empty() || digits)) continue;
        auto it = lvl.entities.find(kv.second);
        if (it == lvl.entities.end()) continue;
        const float *m = it->second.matrix;
        double M[4][3];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 3; ++j) M[i][j] = double(m[3 * i + j]);
        double sc[3], U[3][3], e[3];
        for (int i = 0; i < 3; ++i) sc[i] = std::sqrt((M[i][0] * M[i][0] + M[i][1] * M[i][1]) + M[i][2] * M[i][2]);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) U[i][j] = M[i][j] / sc[i];
        double half[3], mid[3];
        for (int i = 0; i < 3; ++i) half[i] = 0.5 * (FAN_LOCAL_HI[i] - FAN_LOCAL_LO[i]), mid[i] = 0.5 * (FAN_LOCAL_LO[i] + FAN_LOCAL_HI[i]);
        for (int i = 0; i < 3; ++i) e[i] = half[i] * sc[i];
        for (int j = 0; j < 3; ++j)  // 0.5 (lo + hi) @ A + M[3]
            out.push_back(((mid[0] * M[0][j] + mid[1] * M[1][j]) + mid[2] * M[2][j]) + M[3][j]);
        for (int i = 0; i < 3; ++i) out.push_back(e[i]);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) out.push_back(U[i][j]);
        for (int j = 0; j < 3; ++j)  // |U|^T @ e
            out.push_back((std::abs(U[0][j]) * e[0] + std::abs(U[1][j]) * e[1]) + std::abs(U[2][j]) * e[2]);
    }
    return out;
}

std::string reach_fields_key(const GameTracker &t, const CanonicalGeometry &g) {
    Sha256 s;
    for (const auto *x : {&g.A, &g.B, &g.C})
        for (double v : *x) put_f8(s, v);
    for (int64_t v : g.rank) {
        unsigned char b[8];
        std::memcpy(b, &v, 8);
        s.update(b, 8);
    }
    if (!g.rail.empty()) s.update(g.rail.data(), g.rail.size());
    std::string static_key = "canon-" + s.hexdigest();
    // FastReach.cache_key(static_key, level.depth, (), fans=fan_boxes(level))
    Sha256 h;
    for (const DepthBox &d : t.level.depth) {
        double lo[3] = {double(d.lo[0]), double(d.lo[1]), double(d.lo[2])};
        double hi[3] = {double(d.hi[0]), double(d.hi[1]), double(d.hi[2])};
        if (std::min(hi[0] - lo[0], hi[2] - lo[2]) >= DEATH_MIN_EXTENT)
            for (double v : {lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]}) put_f8(h, v);
    }
    h.update("|", 1);
    h.update("|fans", 5);
    for (double v : reach_fan_obbs(t.level)) put_f8(h, v);
    put_f8(h, DEATH_MIN_EXTENT), put_f8(h, R_BALL);
    for (double v : CONSTS) put_f8(h, v);
    for (double v : FAN_LOCAL_LO) put_f8(h, v);
    for (double v : FAN_LOCAL_HI) put_f8(h, v);
    put_f8(h, FAN_FORCE);
    return static_key + ":" + h.hexdigest().substr(0, 16) + ":" + std::to_string(fastreach::BUILD_VERSION);
}

std::string reach_fields_name(int level, const std::string &key) {
    char n[16];
    std::snprintf(n, sizeof n, "L%02d-", level);
    return n + Sha256::hex(key).substr(0, 16) + ".npz";
}

std::vector<double> reach_zone_boxes(const LevelData &lvl) {
    // FastReach.zone_boxes(level, exclude_names=BALL_NAME_TYPES) in phys_by_ent order, each (lo - 4, hi + 4)
    std::vector<double> out;
    std::vector<uint32_t> cks;
    for (auto &kv : lvl.phys_by_ent) cks.push_back(kv.first);
    std::sort(cks.begin(), cks.end());  // the order only permutes the boxes (any order gives the same zone test)
    for (uint32_t ck : cks) {
        const PhysRecord &p = lvl.phys_by_ent.at(ck);
        if (p.unphysicalize || p.fixed || lvl.in_group(ck, G_STOPPER)) continue;
        const std::string *nm = lvl.name(ck);
        if (nm && (*nm == BALL_NAMES[0] || *nm == BALL_NAMES[1] || *nm == BALL_NAMES[2])) continue;
        // LevelData.mesh_world_bbox: the exports with a Physicalize record of the entity, else its meshes
        std::vector<const Mesh *> ms;
        for (auto &pm : lvl.phys_meshes)
            if (pm.first.first == ck) ms.push_back(&pm.second.second);
        if (ms.empty())
            for (auto *e : lvl.meshes_of(ck)) ms.push_back(&e->mesh);
        bool any = false;
        double lo[3], hi[3];
        for (const Mesh *m : ms) {
            size_t nv = m->vertices.size() / 3;
            if (!nv) continue;
            const float *M = m->matrix;  // (4, 4) row-vector: V @ M[:3, :3] + M[3, :3]
            for (size_t k = 0; k < nv; ++k) {
                double v[3] = {double(m->vertices[3 * k]), double(m->vertices[3 * k + 1]), double(m->vertices[3 * k + 2])};
                for (int j = 0; j < 3; ++j) {
                    double w = ((v[0] * double(M[j]) + v[1] * double(M[4 + j])) + v[2] * double(M[8 + j])) + double(M[12 + j]);
                    if (!any) lo[j] = hi[j] = w;
                    else lo[j] = std::min(lo[j], w), hi[j] = std::max(hi[j], w);
                }
                any = true;
            }
        }
        if (!any) continue;
        for (int j = 0; j < 3; ++j) out.push_back(lo[j] - 4.0);
        for (int j = 0; j < 3; ++j) out.push_back(hi[j] + 4.0);
    }
    return out;
}

// ---------------------------------------------------------------------------------------------- scaling

static std::array<double, N_REACH> scale_vector() {
    // reach_features._scale_vector over FEATURE_NAMES: 19 head entries, then 8 directions x 5
    std::array<double, N_REACH> s;
    s.fill(1.0);
    const double T = 1.0 / (double(fastreach::N_PSI) / 132.0), L = 1.0 / 20.0, V = 1.0 / 20.0;
    s[4] = T;   // air_t_first
    s[9] = T;   // air_t_near
    s[5] = L;   // air_first_dy
    s[6] = L;   // air_near_r
    s[7] = L;   // air_near_f
    s[8] = L;   // air_near_dy
    for (int d = 0; d < 8; ++d) {
        int b = 19 + 5 * d;  // edge, brake_margin, gap_deficit, obst, obst_deficit
        s[b] = L, s[b + 1] = L, s[b + 2] = V, s[b + 3] = L, s[b + 4] = V;
    }
    return s;
}

void reach_scale(const double raw[N_REACH], double out[N_REACH]) {
    static const std::array<double, N_REACH> S = scale_vector();
    for (int i = 0; i < N_REACH; ++i) {
        double v = raw[i] * S[i];
        if (std::isnan(v)) v = 0.0;
        else if (std::isinf(v)) v = v > 0 ? 1.0 : -1.0;
        out[i] = std::min(std::max(v, -1.0), 1.0);
    }
}

// ---------------------------------------------------------------------------------------------- the block

ReachBlock::ReachBlock(std::string fields_dir, double body_radius, const std::string &expect_version)
    : dir_(std::move(fields_dir)), body_radius_(body_radius) {
    if (!expect_version.empty() && expect_version != reach_version())
        throw Error("FeatureConfig.reach_version " + expect_version + ": this client's reach block is " + reach_version() +
                    " (the policy was trained on another block: update the client)");
}

const ReachBlock::LevelState &ReachBlock::level_state(const GameTracker &t, const LevelGeometry &geo) {
    if (state_tracker_ != t.serial || state_version_ != t.level.version || state_.key.empty()) {
        state_.key = reach_fields_key(t, canonical_geometry(t, geo));
        state_.zones = reach_zone_boxes(t.level);
        state_tracker_ = t.serial, state_version_ = t.level.version;
    }
    return state_;
}

std::shared_ptr<const fastreach::Fields> ReachBlock::fields(int level, const std::string &key) {
    for (auto it = cache_.begin(); it != cache_.end(); ++it)
        if (it->first == key) {
            cache_.splice(cache_.begin(), cache_, it);
            return cache_.front().second;
        }
    if (dir_.empty()) throw Error("the reach block needs its fields (no fields directory)");
    std::string path = dir_ + "/" + reach_fields_name(level, key);
    if (!file_exists(path)) throw Error("no reach fields for this geometry: " + reach_fields_name(level, key));
    auto f = fastreach::Fields::load(path);
    if (f->key != key) throw Error("reach fields " + reach_fields_name(level, key) + " were built for another key");
    cache_.emplace_front(key, f);
    while (cache_.size() > 2) cache_.pop_back();
    return f;
}

void ReachBlock::compute(const GameTracker &t, const LevelGeometry &geo, const V3 &pos, const V3 &vel, int quarter,
                         int ball_type, double out[N_REACH], double raw[N_REACH], std::string *key) {
    bool finite = true;
    for (int k = 0; k < 3; ++k) finite = finite && std::isfinite(pos[k]) && std::isfinite(vel[k]);
    if (!finite) {  // perceive refuses it: zeros (raw NaN)
        for (int i = 0; i < N_REACH; ++i) out[i] = 0.0, raw[i] = std::nan("");
        if (key) key->clear();
        return;
    }
    const LevelState &st = level_state(t, geo);
    auto f = fields(t.level_no, st.key);
    double p[3] = {pos[0], pos[1], pos[2]}, v[3] = {vel[0], vel[1], vel[2]};
    fastreach::perceive(*f, ws_, st.zones, ball_type, quarter, p, v, raw);
    // reach_features.moving_bodies + FastReach.perceive: a moving body within UNKNOWN_R of the ball = unknown zone
    if (t.obs) {
        for (const BodyItem &b : t.obs->body) {
            if (t.level.in_group(b.ckid, G_STOPPER)) continue;
            double d[3] = {b.pos[0] - p[0], b.pos[1] - p[1], b.pos[2] - p[2]};
            double n = std::sqrt((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]);
            if (n <= body_radius_ && n < UNKNOWN_R) raw[2] = 1.0;
        }
    }
    if (raw[2] > 0.5) raw[1] = 0.0;  // unknown_zone wins over doomed
    reach_scale(raw, out);
    if (key) *key = st.key;
}

}  // namespace ballrain
