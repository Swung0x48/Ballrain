#include "ballrain/features.h"

#include <algorithm>
#include <limits>
#include <set>

namespace ballrain {

static const double PI = 3.141592653589793;  // math.pi
static const double GEOM_EPS = 0.05, NO_CONTACT_GAP = 2.0, SUPPORT_GAP = 1.0;
static const char *SURFACE_SLOTS[] = {"none", "stone", "wood", "metal", "other"};
static const uint8_t IVP_ST_BALL = 3;
static const char *SYNTHETIC_LEVEL_MARKER = "Ballrain_SyntheticLevel";

static double norm3(const double *d) { return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]); }
static double clipd(double v, double lo, double hi) { return std::min(std::max(v, lo), hi); }

// features.to_frame
static V3 to_frame(const V3 &v, const V3 &r, const V3 &f) {
    return {v[0] * r[0] + v[2] * r[2], v[1], v[0] * f[0] + v[2] * f[2]};
}

// ---------------------------------------------------------------------------------------------- config

FeatureConfig FeatureConfig::from_json(const Json &j) {
    FeatureConfig c;
    auto tuple = [](const Json &v) {
        std::vector<double> out;
        if (v.is_array())
            for (auto &x : v.items()) out.push_back(x.as_double());
        return out;
    };
    for (auto &kv : j.members()) {
        const std::string &k = kv.first;
        const Json &v = kv.second;
        if (k == "version") c.version = v.as_string();
        else if (k == "camera") c.camera = v.as_string();
        else if (k == "pos_scale") c.pos_scale = v.as_double();
        else if (k == "vel_scale") c.vel_scale = v.as_double();
        else if (k == "angvel_scale") c.angvel_scale = v.as_double();
        else if (k == "absolute_position") c.absolute_position = v.as_bool();
        else if (k == "height_grid") c.height_grid = int(v.as_int());
        else if (k == "height_spacing") c.height_spacing = v.as_double();
        else if (k == "height_clip") c.height_clip = v.as_double();
        else if (k == "height_up") c.height_up = v.as_double();
        else if (k == "fan_dirs") c.fan_dirs = int(v.as_int());
        else if (k == "fan_step") c.fan_step = v.as_double();
        else if (k == "fan_max") c.fan_max = v.as_double();
        else if (k == "fan_rule") c.fan_rule = v.as_string();
        else if (k == "fan_step_drop") c.fan_step_drop = v.as_double();
        else if (k == "fan_drop") c.fan_drop = v.as_double();
        else if (k == "fan_up") c.fan_up = v.as_double();
        else if (k == "mesh_edge") c.mesh_edge = v.as_bool();
        else if (k == "rail_radius") c.rail_radius = v.as_double();
        else if (k == "path_source") c.path_source = v.as_string();
        else if (k == "path_dir") c.path_dir = v.is_null() ? "" : v.as_string();
        else if (k == "path_sha256") c.path_sha256 = v.is_null() ? "" : v.as_string();
        else if (k == "path_lateral_scale") c.path_lateral_scale = v.as_double();
        else if (k == "path_back") c.path_back = v.as_double();
        else if (k == "path_ahead") c.path_ahead = v.as_double();
        else if (k == "path_margin") c.path_margin = v.as_double();
        else if (k == "path_reacquire") c.path_reacquire = v.as_double();
        else if (k == "waypoints") c.waypoints = int(v.as_int());
        else if (k == "waypoint_spacing") c.waypoint_spacing = v.as_double();
        else if (k == "far_waypoints") c.far_waypoints = tuple(v);
        else if (k == "ref_speed_at") c.ref_speed_at = tuple(v);
        else if (k == "bodies") c.bodies = int(v.as_int());
        else if (k == "body_radius") c.body_radius = v.as_double();
        else if (k == "dynamic_geometry") c.dynamic_geometry = v.as_bool();
        else if (k == "counters") c.counters = v.as_bool();
        else if (k == "hs_block") c.hs_block = v.as_bool();
        else if (k == "follower_sources") c.follower_sources = int(v.as_int());
        else if (k == "checkpoint_frames_scale") c.checkpoint_frames_scale = v.as_double();
        else if (k == "clip") c.clip = v.as_double();
        else throw Error("unknown feature config field " + k);
    }
    if (c.version != "f6") throw Error("feature version " + c.version + " is not supported (f6)");
    if (c.fan_rule != "step" && c.fan_rule != "contact") throw Error("unknown fan_rule " + c.fan_rule);
    if (c.camera != "world" && c.camera != "orient_ref" && c.camera != "orient") throw Error("unknown camera " + c.camera);
    return c;
}

// ---------------------------------------------------------------------------------------------- geometry

LevelGeometry::LevelGeometry(const GameTracker &t) {
    const LevelData &lvl = t.level;
    std::vector<std::vector<V3>> verts;  // world vertices, kept alive for the items
    std::vector<std::pair<int64_t, const Mesh *>> mesh_items, rail_items;
    std::set<int64_t> rail_set;
    for (auto ck : lvl.static_entities())
        for (auto *e : lvl.meshes_of(ck)) {
            mesh_items.emplace_back(ck, &e->mesh);
            if (lvl.in_group(ck, G_RAILS)) rail_items.emplace_back(ck, &e->mesh), rail_set.insert(ck);
        }
    auto build = [&](const std::vector<std::pair<int64_t, const Mesh *>> &src) {
        std::vector<std::vector<V3>> vs(src.size());
        std::vector<TriangleSet::Item> items;
        for (size_t i = 0; i < src.size(); ++i) {
            const Mesh &m = *src[i].second;
            size_t nv = m.vertices.size() / 3;
            vs[i].resize(nv);
            for (size_t k = 0; k < nv; ++k) vs[i][k] = ck_to_world(&m.vertices[3 * k], m.matrix, 4);
        }
        for (size_t i = 0; i < src.size(); ++i)
            items.push_back({src[i].first, &vs[i], src[i].second->faces.data(), src[i].second->faces.size() / 3});
        return TriangleSet::from_items(items, 4.0);
    };
    tris = build(mesh_items);
    rails = build(rail_items);
    edges = BoundaryEdges(tris, [&](int64_t o) { return !rail_set.count(o); });
    key = lvl.static_key();
}

LevelGeometry &FeatureExtractor::geometry(const GameTracker &t) {
    if (!geo_ || geo_tracker_ != t.serial || geo_version_ != t.level.version) {
        uint64_t k = t.level.static_key();
        if (!geo_ || geo_->key != k) geo_ = std::make_unique<LevelGeometry>(t);
        geo_tracker_ = t.serial;
        geo_version_ = t.level.version;
    }
    return *geo_;
}

static double mesh_bound(const Mesh &m) {
    double best = 0;
    size_t nv = m.vertices.size() / 3;
    if (!nv) return 0.0;
    for (size_t i = 0; i < nv; ++i) {
        double x = m.vertices[3 * i], y = m.vertices[3 * i + 1], z = m.vertices[3 * i + 2];
        double s = x * x + y * y + z * z;
        if (i == 0 || s > best) best = s;
    }
    return std::sqrt(best);
}

// features.dynamic_triangles (moving bodies near center, culled to the rays' (x, z) footprint)
static TriangleSet dynamic_triangles(const GameTracker &t, const V3 &center, double radius, double bx0, double bz0,
                                     double bx1, double bz1, std::vector<std::vector<V3>> &keep) {
    TriangleSet out;
    if (!t.obs) {
        out.finish(0);
        return out;
    }
    const LevelData &lvl = t.level;
    double blo0 = bx0 - 1e-6, blo1 = bz0 - 1e-6, bhi0 = bx1 + 1e-6, bhi1 = bz1 + 1e-6;
    for (auto &b : t.obs->body) {
        uint32_t ck = b.ckid;
        if (lvl.in_group(ck, G_STOPPER) || !lvl.is_physicalized(ck)) continue;
        double d[3] = {b.pos[0] - center[0], b.pos[1] - center[1], b.pos[2] - center[2]};
        if (norm3(d) > radius) continue;
        double R[3][3], Rc[3][3], shift[3] = {b.shift[0], b.shift[1], b.shift[2]};
        quat_matrix(b.q, R);
        quat_matrix(b.q_cfo, Rc);
        for (auto *e : lvl.meshes_of(ck)) {
            const Mesh &m = e->mesh;
            double r = mesh_bound(m) + norm3(shift);
            if (b.pos[0] + r < blo0 || b.pos[0] - r > bhi0 || b.pos[2] + r < blo1 || b.pos[2] - r > bhi1) continue;
            size_t nv = m.vertices.size() / 3;
            std::vector<V3> V(nv);
            for (size_t k = 0; k < nv; ++k) V[k] = body_to_world(&m.vertices[3 * k], R, Rc, shift, b.pos);
            size_t nf = m.faces.size() / 3;
            for (size_t f = 0; f < nf; ++f) {
                const uint32_t *F = &m.faces[3 * f];
                const V3 &A = V[F[0]], &B = V[F[1]], &C = V[F[2]];
                double xmax = std::max(std::max(A[0], B[0]), C[0]), xmin = std::min(std::min(A[0], B[0]), C[0]);
                double zmax = std::max(std::max(A[2], B[2]), C[2]), zmin = std::min(std::min(A[2], B[2]), C[2]);
                if (xmax >= blo0 && xmin <= bhi0 && zmax >= blo1 && zmin <= bhi1) {
                    out.A.push_back(A), out.B.push_back(B), out.C.push_back(C);
                    out.owner.push_back(ck);
                }
            }
        }
    }
    (void)keep;
    out.finish(0);
    return out;
}

// ---------------------------------------------------------------------------------------------- extractor

FeatureExtractor::FeatureExtractor(FeatureConfig c) : cfg_(std::move(c)) {
    auto &L = layout_;
    L = {{"valid", 1}, {"ball_type", 3}, {"vel_cam", 3}, {"vel_path", 3}, {"speed", 1}, {"angvel", 3},
         {"cam_yaw_ref", 2}, {"cam_yaw", 2}, {"cam_rotating", 1}};
    if (cfg_.absolute_position) L.push_back({"pos_world", 3});
    for (auto &p : std::vector<std::pair<std::string, int>>{
             {"geom_contact", 1}, {"fric_contact", 1}, {"n_fric", 1}, {"n_mind", 1}, {"min_gap", 1}, {"normal_cam", 3},
             {"surface", 5}, {"partner_moving", 1}, {"partner_rail", 1}, {"rel_vel_partner", 3}})
        L.push_back(p);
    if (cfg_.mesh_edge) L.push_back({"edge_mesh", 1});
    if (cfg_.fan_dirs) L.push_back({"edge_fan", cfg_.fan_dirs});
    L.push_back({"rail_dist", 1});
    L.push_back({"rail_dir", 2});
    if (cfg_.height_grid) {
        L.push_back({"heights", cfg_.height_grid * cfg_.height_grid});
        L.push_back({"height_hit", cfg_.height_grid * cfg_.height_grid});
    }
    for (auto &p : std::vector<std::pair<std::string, int>>{{"path_progress", 1}, {"path_remaining", 1},
                                                             {"path_lateral", 1}, {"path_tangent_cam", 2}})
        L.push_back(p);
    L.push_back({"waypoints", 3 * cfg_.waypoints});
    if (cfg_.bodies) L.push_back({"bodies", 7 * cfg_.bodies});
    if (cfg_.counters)
        for (auto *n : {"lives", "points", "sr_s", "sector", "transforming", "ball_nav", "depth_margin"}) L.push_back({n, 1});
    if (cfg_.hs_block) {
        L.push_back({"followers", 1});
        L.push_back({"follower_sources", cfg_.follower_sources});
        L.push_back({"follower_near", 1});
        L.push_back({"paused", 1});
        L.push_back({"since_checkpoint", 1});
    }
    if (!cfg_.far_waypoints.empty()) L.push_back({"far_waypoints", 3 * int(cfg_.far_waypoints.size())});
    if (!cfg_.ref_speed_at.empty()) L.push_back({"ref_speed", int(cfg_.ref_speed_at.size())});
    int o = 0;
    for (auto &p : L) off_[p.first] = o, o += p.second;
    size_ = o;
}

std::vector<std::string> FeatureExtractor::names() const {
    std::vector<std::string> out;
    for (auto &p : layout_) {
        if (p.second == 1) out.push_back(p.first);
        else
            for (int i = 0; i < p.second; ++i) out.push_back(p.first + "[" + std::to_string(i) + "]");
    }
    return out;
}

int FeatureExtractor::offset(const std::string &block) const {
    auto it = off_.find(block);
    if (it == off_.end()) throw Error("no feature block " + block);
    return it->second;
}

static bool is_synthetic(const GameTracker &t) { return t.level.by_name.count(SYNTHETIC_LEVEL_MARKER) > 0; }

static std::string asset_file(const std::string &dir, int level) {
    char n[16];
    std::snprintf(n, sizeof n, "L%02d.npz", level);
    if (dir.empty()) return std::string();
    char last = dir.back();
    return dir + ((last == '/' || last == '\\') ? "" : "/") + n;
}

bool FeatureExtractor::needs_asset(const GameTracker &t) const {
    if (cfg_.path_source != "tas" || is_synthetic(t) || cfg_.path_dir.empty()) return false;
    return file_exists(asset_file(cfg_.path_dir, t.level_no));
}

void FeatureExtractor::check_route(const GameTracker &t) const {
    if ((!path_ || path_->source != "tas") && needs_asset(t))
        throw RouteError("level " + std::to_string(t.level_no) +
                         ": ball in play without its reference line; refusing to run without the route");
}

const ReferencePath *FeatureExtractor::reference_path(const GameTracker &t) {
    int rows = t.arrays.rows("Checkpoints");
    if (have_key_ && key_tracker_ == t.serial && key_version_ == t.level.version && key_rows_ == rows)
        return path_.get();
    std::shared_ptr<const ReferencePath> path;
    bool tas_pending = false;
    if (cfg_.path_source == "tas" && !is_synthetic(t)) {
        // ballrain.path.load_tas_path(level, tracker, path_dir, strict=True, sha256=path_sha256). Python resolves an
        // unset path_dir to its committed benchmarks/refpaths; the client has no such default, and quietly running on
        // the checkpoint polyline instead is the r6 failure path.py guards against: refuse.
        if (cfg_.path_dir.empty())
            throw RouteError("level " + std::to_string(t.level_no) +
                             ": path_source \"tas\" without a path_dir (install the reference line and set it)");
        std::string file = asset_file(cfg_.path_dir, t.level_no);
        if (!asset_loaded_) {
            asset_ = file.empty() ? nullptr : load_path_asset(file);
            asset_loaded_ = true;
        }
        if (!asset_) {
            if (!cfg_.path_dir.empty()) throw RouteError("reference line " + file + " missing");
        } else {
            if (!cfg_.path_sha256.empty() && asset_->sha256 != cfg_.path_sha256)
                throw RouteError("reference line " + file + " has sha256 " + asset_->sha256.substr(0, 12) +
                                 "..., the configuration expects " + cfg_.path_sha256.substr(0, 12) + "...");
            Anchors a = level_anchors(t);
            if (anchors_match(asset_->line->meta, a)) {
                path = asset_->line;
            } else if (anchors_ready(asset_->line->meta, a)) {
                throw RouteError("reference line " + file + " does not match the running level");
            }
        }
        tas_pending = !path && needs_asset(t);
    }
    if (!path && !tas_pending) path = ReferencePath::from_tracker(t);
    bool new_tracker = !have_key_ || key_tracker_ != t.serial;
    bool same_line = path && path_ && path->source == path_->source && path->P == path_->P;
    if (new_tracker || !same_line) {
        if (path)
            follower_ = std::make_unique<PathFollower>(path, cfg_.path_back, cfg_.path_ahead, cfg_.path_margin,
                                                       cfg_.path_reacquire);
        else
            follower_.reset();
    }
    path_ = path;
    have_key_ = true;
    key_tracker_ = t.serial, key_version_ = t.level.version, key_rows_ = rows;
    return path_.get();
}

// features._closest_on: nearest point of entity ck's collision mesh to p within r
static std::optional<V3> closest_on(const GameTracker &t, LevelGeometry &geo, uint32_t ck, const V3 &p, double r) {
    const LevelData &lvl = t.level;
    std::optional<V3> best;
    double bd = 0;
    auto consider = [&](const V3 &a, const V3 &b, const V3 &c) {
        V3 q = closest_point_triangle(p, a, b, c, nullptr);
        double e[3] = {q[0] - p[0], q[1] - p[1], q[2] - p[2]};
        double d = norm3(e);
        if (!best || d < bd) best = q, bd = d;
    };
    if (lvl.is_fixed(ck)) {
        std::vector<int64_t> idx;
        geo.tris.candidates(p, r, idx);
        for (auto i : idx)
            if (geo.tris.owner[size_t(i)] == int64_t(ck)) consider(geo.tris.A[size_t(i)], geo.tris.B[size_t(i)], geo.tris.C[size_t(i)]);
    } else {
        if (!t.obs) return std::nullopt;
        const BodyItem *body = nullptr;
        for (auto &b : t.obs->body)
            if (b.ckid == ck) {
                body = &b;
                break;
            }
        if (!body) return std::nullopt;
        double R[3][3], Rc[3][3], shift[3] = {body->shift[0], body->shift[1], body->shift[2]};
        quat_matrix(body->q, R);
        quat_matrix(body->q_cfo, Rc);
        for (auto *e : lvl.meshes_of(ck)) {
            const Mesh &m = e->mesh;
            size_t nv = m.vertices.size() / 3;
            std::vector<V3> V(nv);
            for (size_t k = 0; k < nv; ++k) V[k] = body_to_world(&m.vertices[3 * k], R, Rc, shift, body->pos);
            for (size_t f = 0; f + 2 < m.faces.size(); f += 3) consider(V[m.faces[f]], V[m.faces[f + 1]], V[m.faces[f + 2]]);
        }
    }
    if (!best || bd > r) return std::nullopt;
    return best;
}

std::vector<V3> FeatureExtractor::mind_normals(const GameTracker &t, const V3 &pos) {
    const auto &mind = t.obs->mind;
    std::vector<V3> N(mind.size());
    for (size_t j = 0; j < mind.size(); ++j) N[j] = {mind[j].plane[0], mind[j].plane[1], mind[j].plane[2]};
    std::map<uint32_t, std::optional<V3>> near;
    for (size_t j = 0; j < mind.size(); ++j) {
        if (mind[j].self_status == IVP_ST_BALL) continue;
        uint32_t ck = mind[j].partner;
        auto it = near.find(ck);
        if (it == near.end())
            it = near.emplace(ck, closest_on(t, geometry(t), ck, pos, double(mind[j].gap) + 4.0)).first;
        if (it->second) {
            const V3 &q = *it->second;
            double dot = N[j][0] * (pos[0] - q[0]) + N[j][1] * (pos[1] - q[1]) + N[j][2] * (pos[2] - q[2]);
            if (dot < 0)
                for (auto &x : N[j]) x = -x;
        }
    }
    return N;
}

std::vector<float> FeatureExtractor::compute(const GameTracker &t, FeatureInfo *info_out) {
    const FeatureConfig &c = cfg_;
    std::vector<double> F(size_t(size_), 0.0);
    auto at = [&](const char *block) -> double * { return &F[size_t(off_.at(block))]; };
    FeatureInfo info;
    const Observation *obs = t.obs ? &*t.obs : nullptr;
    // frame of the directional features
    right_ = {1, 0, 0}, fwd_ = {0, 0, 1};
    if (c.camera != "world" && obs && obs->cams) {
        const float *m = obs->cams->cams[c.camera == "orient_ref" ? 0 : 1].matrix;
        V3 f{m[6], 0.0, m[8]};
        double n = norm3(f.data());
        if (n > 1e-9) f = {f[0] / n, f[1] / n, f[2] / n};
        else f = {0.0, 0.0, 1.0};
        right_ = {f[2], 0.0, -f[0]}, fwd_ = f;
    }
    const V3 &right = right_, &fwd = fwd_;
    const BallSec *ball = obs && obs->ball ? &*obs->ball : nullptr;
    bool have_ball = ball && ball->found && ball->type <= 2;
    bool valid = have_ball && !t.transforming();
    *at("valid") = valid ? 1.0 : 0.0;
    int bt = t.ball_type();
    if (bt >= 0 && bt <= 2) at("ball_type")[bt] = 1;
    if (obs && obs->cams) {
        const float *m0 = obs->cams->cams[0].matrix, *m1 = obs->cams->cams[1].matrix;
        double y0 = std::atan2(double(m0[6]), double(m0[8])), y1 = std::atan2(double(m1[6]), double(m1[8]));
        double *cy = at("cam_yaw_ref");
        cy[0] = std::cos(y0), cy[1] = std::sin(y0);
        double *cz = at("cam_yaw");
        cz[0] = std::cos(y1), cz[1] = std::sin(y1);
        *at("cam_rotating") = std::abs(std::remainder(y0 - y1, 2 * PI)) > 1e-3 ? 1.0 : 0.0;
    }
    const ReferencePath *path = reference_path(t);
    if (valid) check_route(t);
    V3 pos{}, vel{};
    if (have_ball) {
        pos = {ball->pos[0], ball->pos[1], ball->pos[2]};
        for (int k = 0; k < 3; ++k) vel[k] = double(ball->speed[k]) + double(ball->speed_change[k]);
        info.have_ball = true, info.pos = pos, info.vel = vel;
        V3 vc = to_frame(vel, right, fwd);
        double *p = at("vel_cam");
        for (int k = 0; k < 3; ++k) p[k] = vc[k] / c.vel_scale;
        *at("speed") = norm3(vel.data()) / c.vel_scale;
        double *av = at("angvel");
        float sc = float(c.angvel_scale);
        for (int k = 0; k < 3; ++k) av[k] = double((ball->rot_speed[k] + ball->rot_speed_change[k]) / sc);  // float32
        if (c.absolute_position) {
            double *pw = at("pos_world");
            for (int k = 0; k < 3; ++k) pw[k] = pos[k] / c.pos_scale;
        }
        double radius = double(ball->extra_radius);
        if (radius == 0.0) radius = 2.0;
        std::optional<V3> contact, support;
        contacts(t, F, info, pos, vel, radius, contact, support);
        info.supported = support.has_value();
        terrain(t, F, info, pos, radius, contact, support);
        if (path) {
            Projection pr = follower_->update(pos, t.sector());
            info.path = pr;
            info.path_source = path->source;
            *at("path_progress") = pr.s / std::max(path->length, 1e-9);
            *at("path_remaining") = (path->length - pr.s) / (20 * c.pos_scale);
            *at("path_lateral") = pr.dist / c.path_lateral_scale;
            V3 tc = to_frame(pr.tangent, right, fwd);
            at("path_tangent_cam")[0] = tc[0], at("path_tangent_cam")[1] = tc[2];
            V3 prr{pr.tangent[2], 0.0, -pr.tangent[0]};
            V3 vp = to_frame(vel, prr, pr.tangent);
            for (int k = 0; k < 3; ++k) at("vel_path")[k] = vp[k] / c.vel_scale;
            if (c.waypoints) {
                double *w = at("waypoints");
                for (int j = 0; j < c.waypoints; ++j) {
                    V3 q = path->point_at(pr.s + c.waypoint_spacing * (j + 1));
                    V3 d{q[0] - pos[0], q[1] - pos[1], q[2] - pos[2]};
                    V3 f = to_frame(d, right, fwd);
                    for (int k = 0; k < 3; ++k) w[3 * j + k] = f[k] / c.pos_scale;
                }
            }
            if (!c.far_waypoints.empty()) {
                double *w = at("far_waypoints");
                for (size_t j = 0; j < c.far_waypoints.size(); ++j) {
                    V3 q = path->point_at(pr.s + c.far_waypoints[j]);
                    V3 d{q[0] - pos[0], q[1] - pos[1], q[2] - pos[2]};
                    V3 f = to_frame(d, right, fwd);
                    for (int k = 0; k < 3; ++k) w[3 * j + k] = f[k] / c.pos_scale;
                }
            }
            if (!c.ref_speed_at.empty()) {
                double *w = at("ref_speed");
                for (size_t j = 0; j < c.ref_speed_at.size(); ++j) {
                    auto v = path->speed_at(pr.s + c.ref_speed_at[j]);
                    w[j] = (v ? *v : 0.0) / c.vel_scale;
                }
            }
        }
        if (c.bodies) bodies(t, F, pos, vel);
    }
    if (c.counters) {
        *at("lives") = t.lives() / 10.0;
        *at("points") = t.points() / 1000.0;
        auto sr = t.sr_ms();
        *at("sr_s") = (sr ? *sr : 0.0) / 100000.0;
        int npath = path ? path->n_sectors : 1;
        *at("sector") = double(t.sector()) / std::max(npath, 1);
        *at("transforming") = t.transforming() ? 1.0 : 0.0;
        *at("ball_nav") = t.flags.ball_nav ? 1.0 : 0.0;
        if (have_ball) *at("depth_margin") = depth_margin(t, pos) / c.pos_scale;
    }
    if (c.hs_block) {
        int n = t.followers();
        *at("followers") = n / 6.0;
        FollowerState st = t.follower_state();
        std::vector<int> per;
        for (auto &kv : st.per_source) per.push_back(kv.second);
        std::sort(per.begin(), per.end(), std::greater<int>());
        for (size_t i = 0; i < per.size() && int(i) < c.follower_sources; ++i) at("follower_sources")[i] = per[i] / 6.0;
        if (have_ball && !st.positions.empty()) {
            auto d = t.nearest_follower(pos.data());
            *at("follower_near") = std::min(*d, c.pos_scale) / c.pos_scale;
        }
        *at("paused") = t.flags.paused ? 1.0 : 0.0;
        *at("since_checkpoint") = t.frames_since_checkpoint() / c.checkpoint_frames_scale;
    }
    std::vector<float> vec(F.size());
    float cl = float(c.clip);
    for (size_t i = 0; i < F.size(); ++i) {
        float v = float(F[i]);
        if (std::isnan(v)) v = 0.0f;  // np.clip keeps NaN, nan_to_num then zeroes it
        else v = std::min(std::max(v, -cl), cl);
        vec[i] = v;
    }
    if (info_out) *info_out = std::move(info);
    return vec;
}

void FeatureExtractor::contacts(const GameTracker &t, std::vector<double> &F, FeatureInfo &info, const V3 &pos,
                                const V3 &vel, double radius, std::optional<V3> &contact, std::optional<V3> &support) {
    const Observation &obs = *t.obs;
    const LevelData &lvl = t.level;
    auto at = [&](const char *block) -> double * { return &F[size_t(off_.at(block))]; };
    const auto &mind = obs.mind;
    const auto &fric = obs.fric;
    *at("n_mind") = mind.size() / 10.0;
    *at("n_fric") = fric.size() / 4.0;
    *at("fric_contact") = fric.empty() ? 0.0 : 1.0;
    std::optional<uint32_t> partner;
    if (!mind.empty()) {
        std::vector<V3> N = mind_normals(t, pos);
        size_t k = 0;
        for (size_t j = 1; j < mind.size(); ++j)
            if (mind[j].gap < mind[k].gap) k = j;
        double gap = mind[k].gap;
        const V3 &normal = N[k];
        *at("min_gap") = std::min(gap, NO_CONTACT_GAP);
        *at("geom_contact") = gap <= GEOM_EPS ? 1.0 : 0.0;
        V3 nc = to_frame(normal, right_, fwd_);
        for (int q = 0; q < 3; ++q) at("normal_cam")[q] = nc[q];
        partner = mind[k].partner;
        info.gap = gap;
        contact = V3{pos[0] - normal[0] * (radius + gap), pos[1] - normal[1] * (radius + gap), pos[2] - normal[2] * (radius + gap)};
        std::optional<size_t> jb;
        for (size_t j = 0; j < mind.size(); ++j)
            if (N[j][1] >= UP_WALKABLE && (!jb || mind[j].gap < mind[*jb].gap)) jb = j;
        if (jb) {
            double gj = mind[*jb].gap;
            if (gj <= SUPPORT_GAP)
                support = V3{pos[0] - N[*jb][0] * (radius + gj), pos[1] - N[*jb][1] * (radius + gj),
                             pos[2] - N[*jb][2] * (radius + gj)};
        }
    } else {
        *at("min_gap") = NO_CONTACT_GAP;
    }
    if (!fric.empty()) {
        size_t k = 0;
        for (size_t j = 1; j < fric.size(); ++j)
            if (fric[j].gap < fric[k].gap) k = j;
        partner = fric[k].partner;
    }
    int slot = 0;
    if (partner) {
        int cid = lvl.contact_id(*partner);
        slot = cid == 1 ? 1 : cid == 2 ? 2 : cid == 3 ? 3 : 4;
        bool moving = !lvl.is_fixed(*partner);
        *at("partner_moving") = moving ? 1.0 : 0.0;
        *at("partner_rail") = lvl.in_group(*partner, G_RAILS) ? 1.0 : 0.0;
        if (moving)
            for (auto &b : obs.body)
                if (b.ckid == *partner) {
                    V3 rel{vel[0] - double(b.speed[0]), vel[1] - double(b.speed[1]), vel[2] - double(b.speed[2])};
                    V3 f = to_frame(rel, right_, fwd_);
                    for (int q = 0; q < 3; ++q) at("rel_vel_partner")[q] = f[q] / cfg_.vel_scale;
                    break;
                }
    }
    at("surface")[slot] = 1;
    info.surface = SURFACE_SLOTS[slot];
}

void FeatureExtractor::terrain(const GameTracker &t, std::vector<double> &F, FeatureInfo &info, const V3 &pos,
                               double radius, const std::optional<V3> &contact, const std::optional<V3> &support) {
    const FeatureConfig &c = cfg_;
    auto at = [&](const char *block) -> double * { return &F[size_t(off_.at(block))]; };
    LevelGeometry &geo = geometry(t);
    const V3 &right = right_, &fwd = fwd_;
    double bottom = pos[1] - radius;
    std::vector<V3> Pg, Pf;
    std::vector<double> steps;
    if (c.height_grid) {
        int g = c.height_grid;
        for (int i = 0; i < g; ++i)
            for (int j = 0; j < g; ++j) {
                double gx = (i - (g - 1) / 2.0) * c.height_spacing, gz = (j - (g - 1) / 2.0) * c.height_spacing;
                V3 p;
                for (int k = 0; k < 3; ++k) p[k] = pos[k] + (gx * right[k] + gz * fwd[k]);
                p[1] = pos[1] + c.height_up;
                Pg.push_back(p);
            }
    }
    if (c.fan_dirs) {
        int ns = int(std::ceil((c.fan_max + 1e-9 - c.fan_step) / c.fan_step));
        for (int s = 0; s < ns; ++s) steps.push_back(c.fan_step + s * c.fan_step);
        V3 base = support ? *support : pos;
        for (int i = 0; i < c.fan_dirs; ++i) {
            double ang = i * (2 * PI / c.fan_dirs);
            double ca = std::cos(ang), sa = std::sin(ang);
            V3 dir;
            for (int k = 0; k < 3; ++k) dir[k] = ca * fwd[k] + sa * right[k];
            for (double st : steps) {
                V3 p;
                for (int k = 0; k < 3; ++k) p[k] = base[k] + st * dir[k];
                p[1] = pos[1] + c.fan_up;
                Pf.push_back(p);
            }
        }
    }
    std::optional<TriangleSet> dyn;
    std::vector<std::vector<V3>> keep;
    if (c.dynamic_geometry) {
        double x0 = pos[0], x1 = pos[0], z0 = pos[2], z1 = pos[2];
        for (auto *v : {&Pg, &Pf})
            for (auto &p : *v) x0 = std::min(x0, p[0]), x1 = std::max(x1, p[0]), z0 = std::min(z0, p[2]), z1 = std::max(z1, p[2]);
        dyn = dynamic_triangles(t, pos, c.body_radius, x0, z0, x1, z1, keep);
    }
    auto ray = [&](const V3 &P) {
        double y = geo.tris.ray_down(P);
        if (dyn && dyn->size()) y = merge_ray(y, dyn->ray_down(P));
        return y;
    };
    if (!Pg.empty()) {
        double *h = at("heights"), *hh = at("height_hit");
        for (size_t i = 0; i < Pg.size(); ++i) {
            double y = ray(Pg[i]);
            bool hit = !std::isnan(y);
            h[i] = (hit ? clipd(y - bottom, -c.height_clip, c.height_clip) : -c.height_clip) / c.height_clip;
            hh[i] = hit ? 1.0 : 0.0;
        }
    }
    if (!Pf.empty()) {
        double h_ref;
        if (support) h_ref = (*support)[1];
        else h_ref = ray(V3{pos[0], pos[1] + c.fan_up, pos[2]});  // NaN = no reference
        double *fe = at("edge_fan");
        size_t ns = steps.size();
        double fmin = 0;
        for (int i = 0; i < c.fan_dirs; ++i) {
            double first = c.fan_max;
            double prev = h_ref;
            for (size_t s = 0; s < ns; ++s) {
                double y = ray(Pf[size_t(i) * ns + s]);
                bool drop;
                if (c.fan_rule == "contact") drop = std::isnan(y) || y < h_ref - c.fan_drop;
                else drop = std::isnan(y) || (prev - y > c.fan_step_drop);
                if (drop) {
                    first = steps[s];
                    break;
                }
                prev = y;
            }
            fe[i] = first / c.fan_max;
            fmin = i == 0 ? first : std::min(fmin, first);
        }
        info.edge_fan_min = fmin;
    }
    if (c.mesh_edge) {
        V3 q = contact ? *contact : V3{pos[0] - 0.0, pos[1] - radius, pos[2] - 0.0};
        double d = geo.edges.distance(q, 12.0);
        *at("edge_mesh") = std::min(d, 12.0) / 12.0;
    }
    std::optional<Closest> rc;
    if (geo.rails.size()) rc = geo.rails.closest(pos, c.rail_radius);
    if (rc) {
        *at("rail_dist") = (rc->d - radius) / c.rail_radius;
        V3 dv = to_frame(V3{rc->q[0] - pos[0], rc->q[1] - pos[1], rc->q[2] - pos[2]}, right, fwd);
        double h = std::hypot(dv[0], dv[2]);
        if (h > 1e-9) at("rail_dir")[0] = dv[0] / h, at("rail_dir")[1] = dv[2] / h;
    } else {
        *at("rail_dist") = 1.0;
    }
}

void FeatureExtractor::bodies(const GameTracker &t, std::vector<double> &F, const V3 &pos, const V3 &vel) {
    const FeatureConfig &c = cfg_;
    struct Row { double dist; V3 d, rv; };
    std::vector<Row> rows;
    for (auto &b : t.obs->body) {
        if (t.level.in_group(b.ckid, G_STOPPER)) continue;
        V3 d{b.pos[0] - pos[0], b.pos[1] - pos[1], b.pos[2] - pos[2]};
        double dist = norm3(d.data());
        if (dist <= c.body_radius)
            rows.push_back({dist, d, V3{double(b.speed[0]) - vel[0], double(b.speed[1]) - vel[1], double(b.speed[2]) - vel[2]}});
    }
    std::stable_sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) { return a.dist < b.dist; });
    double *out = &F[size_t(off_.at("bodies"))];
    for (size_t i = 0; i < rows.size() && int(i) < c.bodies; ++i) {
        out[7 * i] = 1;
        V3 a = to_frame(rows[i].d, right_, fwd_), b = to_frame(rows[i].rv, right_, fwd_);
        for (int k = 0; k < 3; ++k) out[7 * i + 1 + k] = a[k] / c.pos_scale, out[7 * i + 4 + k] = b[k] / c.vel_scale;
    }
}

double FeatureExtractor::depth_margin(const GameTracker &t, const V3 &pos) const {
    std::optional<double> best;
    for (auto &b : t.level.depth)
        if (b.lo[0] <= pos[0] && pos[0] <= b.hi[0] && b.lo[2] <= pos[2] && pos[2] <= b.hi[2]) {
            double m = pos[1] - double(b.hi[1]);
            best = best ? std::min(*best, m) : m;
        }
    return best ? *best : 20 * cfg_.pos_scale;
}

}  // namespace ballrain
