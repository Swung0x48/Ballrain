#include "ballrain/branch.h"

#include <algorithm>
#include <set>
#include <cmath>
#include <cstdio>

#include "ballrain/sha256.h"
#include "ballrain/util.h"

namespace ballrain {

static int64_t ss_left(const std::vector<double> &a, double x) {
    return int64_t(std::lower_bound(a.begin(), a.end(), x) - a.begin());
}
static int64_t ss_right(const std::vector<double> &a, double x) {
    return int64_t(std::upper_bound(a.begin(), a.end(), x) - a.begin());
}

BranchRules BranchRules::from_desc(const Json &desc) {
    BranchRules r;
    const Json *raw = desc.find("rules");
    if (!raw) return r;
    if (!raw->is_object()) throw RouteError("branch rules must be an object");
    auto bad = [](const std::string &k) { return RouteError("invalid branch rule " + k); };
    auto number = [&](const std::string &k, const Json &v) {  // float fields: int or float, 0 <= v (finite)
        if (!v.is_number()) throw bad(k);
        double x = v.as_double();
        if (!(x >= 0) || !std::isfinite(x)) throw bad(k);
        return x;
    };
    auto frames = [&](const std::string &k, const Json &v) {  // int fields: an integer in [1, 2147483647]
        if (!v.is_int()) throw bad(k);
        int64_t x = v.as_int();
        if (x < 1 || x > 2147483647) throw bad(k);
        return x;
    };
    auto choice = [&](const std::string &k, const Json &v, const char *a, const char *b) {
        if (!v.is_string() || (v.as_string() != a && v.as_string() != b)) throw bad(k);
        return v.as_string();
    };
    for (auto &kv : raw->members()) {
        const std::string &k = kv.first;
        const Json &v = kv.second;
        if (k == "stall_pause_speed") r.stall_pause_speed = number(k, v);
        else if (k == "stall_speed_source") r.stall_speed_source = choice(k, v, "speed", "speed+speed_change");
        else if (k == "stall_odometer") r.stall_odometer = choice(k, v, "local_s", "s");
        else if (k == "rest_stop_speed") {
            if (v.is_null()) r.rest_stop_speed.reset();
            else if (double x = number(k, v); x == 0) throw bad(k);
            else r.rest_stop_speed = x;
        } else if (k == "rest_max_drive_frames") {
            if (v.is_null()) r.rest_max_drive_frames.reset();
            else r.rest_max_drive_frames = frames(k, v);
        } else if (k == "rest_max_brake_frames") {
            if (v.is_null()) r.rest_max_brake_frames.reset();
            else r.rest_max_brake_frames = frames(k, v);
        } else if (k == "rest_require_support") {
            if (!v.is_bool()) throw bad(k);
            r.rest_require_support = v.as_bool();
        } else if (k == "rest_position_tolerance") r.rest_position_tolerance = number(k, v);
        else if (k == "rest_fallback") r.rest_fallback = choice(k, v, "same_tas_frame", "error");
        else if (k == "drive_arrival_radius") r.drive_arrival_radius = number(k, v);
        else if (k == "drive_gain") r.drive_gain = number(k, v);
        else if (k == "drive_speed_cap") r.drive_speed_cap = number(k, v);
        else if (k == "drive_deadband") r.drive_deadband = number(k, v);
        else if (k == "rest_control_repeat") r.rest_control_repeat = frames(k, v);
        else throw RouteError("unknown branch rule: " + k);
    }
    return r;
}

bool BranchRules::is_default() const {
    BranchRules d;
    return stall_pause_speed == d.stall_pause_speed && stall_speed_source == d.stall_speed_source &&
           stall_odometer == d.stall_odometer && rest_stop_speed == d.rest_stop_speed &&
           rest_max_drive_frames == d.rest_max_drive_frames && rest_max_brake_frames == d.rest_max_brake_frames &&
           rest_require_support == d.rest_require_support && rest_position_tolerance == d.rest_position_tolerance &&
           rest_fallback == d.rest_fallback && drive_arrival_radius == d.drive_arrival_radius &&
           drive_gain == d.drive_gain && drive_speed_cap == d.drive_speed_cap && drive_deadband == d.drive_deadband &&
           rest_control_repeat == d.rest_control_repeat;
}

static Json read_desc(const std::string &dir, int level) {
    char n[16];
    std::snprintf(n, sizeof n, "/L%02d.json", level);
    Bytes jb;
    if (!read_file(dir + n, jb)) throw RouteError("branch description " + dir + n + " missing");
    if (jb.compare(0, 3, "\xEF\xBB\xBF") == 0) jb = jb.substr(3);  // utf-8-sig
    return Json::parse(jb, 16);
}

BranchRules load_branch_rules(const std::string &dir, int level) { return BranchRules::from_desc(read_desc(dir, level)); }

std::string branch_bundle_sha256(const Bytes &line, const Bytes &desc, const Bytes &drive) {
    Sha256 h;
    for (const Bytes *b : {&line, &desc, &drive}) h.update(Sha256::hash(b->data(), b->size()));
    return h.hexdigest();
}

BranchFollower::BranchFollower(std::shared_ptr<const ReferencePath> m, std::shared_ptr<const ReferencePath> a,
                               const Json &meta, double back, double ahead, double margin, double reacquire)
    : main(std::move(m)), alt(std::move(a)),
      routes_{PathFollower(main, back, ahead, margin, reacquire), PathFollower(alt, back, ahead, margin, reacquire)} {
    rules = BranchRules::from_desc(meta);
    ball_type = int(meta.at("ball_type").as_int());
    sector = int(meta.at("sector").as_int());
    lo = meta.at("branch_s").at(size_t(0)).as_double();
    hi = meta.at("branch_s").at(size_t(1)).as_double();
    // np.asarray([(0, 0), (main_s[0], lo), *stations, (main_s[1], hi), (main.length, alt.length)]).T
    const Json &mse = meta.at("main_s");
    ms = {0.0, mse.at(size_t(0)).as_double()};
    bs = {0.0, lo};
    for (auto &p : meta.at("stations").items()) ms.push_back(p.at(size_t(0)).as_double()), bs.push_back(p.at(size_t(1)).as_double());
    ms.push_back(mse.at(size_t(1)).as_double()), bs.push_back(hi);
    ms.push_back(main->length), bs.push_back(alt->length);
    for (size_t i = 1; i < ms.size(); ++i)
        if (!(ms[i] - ms[i - 1] > 0) || !(bs[i] - bs[i - 1] > 0)) throw RouteError("branch stations must increase on both lines");
    if (!alt->t_ms) throw RouteError("branch timing must cover every point and increase strictly");
    clocks[1] = *alt->t_ms;
    if (main->t_ms) {
        clocks[0] = *main->t_ms;
    } else {
        // older TAS assets have no t_ms: the generator supplies the replaced samples; before and after the splice both
        // clocks differ only by a constant (np.r_[alt.t_ms[:k], span, alt.t_ms[l + 1:] - alt.t_ms[l] + span[-1]])
        int64_t k = ss_left(alt->cum, lo), l = ss_left(alt->cum, hi);
        std::vector<double> span;
        for (auto &v : meta.at("main_span_t_ms").items()) span.push_back(v.as_double());
        if (span.empty() || l >= int64_t(clocks[1].size())) throw RouteError("branch: main_span_t_ms / branch_s");
        std::vector<double> c(clocks[1].begin(), clocks[1].begin() + k);
        c.insert(c.end(), span.begin(), span.end());
        for (size_t i = size_t(l) + 1; i < clocks[1].size(); ++i) c.push_back(clocks[1][i] - clocks[1][size_t(l)] + span.back());
        clocks[0] = std::move(c);
    }
    const ReferencePath *paths[2] = {main.get(), alt.get()};
    for (int r = 0; r < 2; ++r) {
        const auto &c = clocks[r];
        bool ok = c.size() == paths[r]->P.size();
        for (size_t i = 1; ok && i < c.size(); ++i) ok = c[i] - c[i - 1] > 0;
        if (!ok) throw RouteError("branch timing must cover every point and increase strictly");
    }
    reset();
}

void BranchFollower::reset() {
    active_.reset();
    peak_ = credit_ = 0;
    clock_ms_ = 0.0;
    anchor_.reset();
    timing = BranchTiming();
    routes_[0].reset(), routes_[1].reset();
}

BranchStep BranchFollower::update(const V3 &pos, int sec, int bt) {
    bool use = sec == sector && bt == ball_type;
    Projection br;
    if (use) {
        br = routes_[1].update(pos, sec);
        use = lo <= br.s && br.s <= hi;
    } else {
        routes_[1].reset();
    }
    if (use) routes_[0].reset();
    Projection pr = use ? br : routes_[0].update(pos, sec);
    int u = use ? 1 : 0;
    const ReferencePath *path = routes_[u].path.get();
    double local = pr.s;
    double station = use ? np_interp(local, bs, ms) : local;
    if (!active_) {
        credit_ = peak_ = station;
    } else if (*active_ != use) {
        // the projection discontinuity is consumed without pay; only never-reached stations earn credit afterwards
        peak_ = std::max(peak_, station);
        double end = ms[size_t(std::min<int64_t>(ss_right(ms, peak_), int64_t(ms.size()) - 1))];
        anchor_ = std::array<double, 3>{peak_, credit_, end};
    } else if (station > peak_) {
        double before = peak_;
        peak_ = station;
        if (anchor_ && station < (*anchor_)[2]) {
            double x = (*anchor_)[0], y = (*anchor_)[1], end = (*anchor_)[2];
            credit_ = y + (station - x) * (end - y) / (end - x);
        } else {
            anchor_.reset();
            credit_ = station;
        }
        double start = use ? np_interp(before, ms, bs) : before;
        clock_ms_ += np_interp(local, path->cum, clocks[u]) - np_interp(start, path->cum, clocks[u]);
    }
    active_ = use;
    double frontier = use ? np_interp(peak_, ms, bs) : peak_;
    double remaining = clocks[u].back() - np_interp(frontier, path->cum, clocks[u]);
    timing.set = true, timing.branch = use, timing.ref_elapsed_ms = clock_ms_, timing.ref_remaining_ms = remaining;
    BranchStep out;
    out.path = path;
    out.pr = pr;
    out.pr.s = credit_;
    out.local_s = local;
    out.branch = use;
    out.ref_elapsed_ms = clock_ms_;
    out.ref_remaining_ms = remaining;
    return out;
}

static std::unique_ptr<BranchFollower> load_line(std::shared_ptr<const ReferencePath> main, int level,
                                                 const std::string &dir, const std::string &main_sha256, double back,
                                                 double ahead, double margin, double reacquire) {
    char n[16];
    std::snprintf(n, sizeof n, "/L%02d", level);
    std::string base = dir + n;
    Bytes jb;
    if (!read_file(base + ".json", jb)) throw RouteError("branch description " + base + ".json missing");
    if (jb.compare(0, 3, "\xEF\xBB\xBF") == 0) jb = jb.substr(3);  // utf-8-sig
    Json desc = Json::parse(jb, 16);
    if (desc.at("level").as_int() != level || desc.at("main_sha256").as_string() != main_sha256)
        throw RouteError("branch is not for this main reference asset");
    std::string npz = base + ".npz";
    if (!file_exists(npz)) throw RouteError("branch line " + npz + " missing");
    std::string sha = file_sha256(npz);
    std::string want = desc.at("branch_sha256").as_string();
    if (sha != want)
        throw RouteError("reference line " + npz + " has sha256 " + sha.substr(0, 12) + "..., the configuration expects " +
                         want.substr(0, 12) + "...");
    std::shared_ptr<const ReferencePath> alt = ReferencePath::load(npz);
    return std::make_unique<BranchFollower>(std::move(main), std::move(alt), desc, back, ahead, margin, reacquire);
}

std::unique_ptr<RouteFollower> load_branch(std::shared_ptr<const ReferencePath> main, int level,
                                           const std::string &dir, const std::string &main_sha256, double back,
                                           double ahead, double margin, double reacquire) {
    Bytes raw;
    if (read_file(dir + "/branches.json", raw)) {  // wire v2: the members in manifest order, each a v1 child directory
        std::vector<BranchEntry> entries = parse_branch_manifest(raw);
        std::vector<std::unique_ptr<BranchFollower>> members;
        for (const BranchEntry &e : entries)
            members.push_back(load_line(main, level, dir + "/" + e.id, main_sha256, back, ahead, margin, reacquire));
        return std::make_unique<MultiFollower>(std::move(main), std::move(entries), std::move(members));
    }
    return load_line(std::move(main), level, dir, main_sha256, back, ahead, margin, reacquire);
}

// ---------------------------------------------------------------------------------------------- multi-branch (v2)

bool BranchGate::matches(const V3 &pos, int sec, int bt, double arc) const {
    if (sector && *sector != sec) return false;
    if (ball && *ball != bt) return false;
    if (s && !((*s)[0] <= arc && arc <= (*s)[1])) return false;
    if (box)
        for (int k = 0; k < 3; ++k)
            if (!((*box)[size_t(k)] <= pos[size_t(k)] && pos[size_t(k)] <= (*box)[size_t(3 + k)])) return false;
    return true;
}

static bool numeric(const Json &v) { return v.is_number(); }

// multi_branch.span: numbers, the shape, finite, lo <= hi (element-wise)
static bool span_pair(const Json &v, std::array<double, 2> &out) {
    if (!v.is_array() || v.items().size() != 2) return false;
    for (int i = 0; i < 2; ++i) {
        const Json &x = v.items()[size_t(i)];
        if (!numeric(x) || !std::isfinite(x.as_double())) return false;
        out[size_t(i)] = x.as_double();
    }
    return out[0] <= out[1];
}
static bool span_box(const Json &v, std::array<double, 6> &out) {
    if (!v.is_array() || v.items().size() != 2) return false;
    for (int r = 0; r < 2; ++r) {
        const Json &row = v.items()[size_t(r)];
        if (!row.is_array() || row.items().size() != 3) return false;
        for (int k = 0; k < 3; ++k) {
            const Json &x = row.items()[size_t(k)];
            if (!numeric(x) || !std::isfinite(x.as_double())) return false;
            out[size_t(3 * r + k)] = x.as_double();
        }
    }
    for (int k = 0; k < 3; ++k)
        if (!(out[size_t(k)] <= out[size_t(3 + k)])) return false;
    return true;
}

static BranchGate parse_gate(const Json &g) {
    if (!g.is_object()) throw RouteError("invalid phase gate");
    BranchGate out;
    bool region = false;
    for (auto &kv : g.members()) {
        const std::string &k = kv.first;
        const Json &v = kv.second;
        if (k == "sector") {
            if (!v.is_int()) throw RouteError("invalid phase gate");
            out.sector = int(v.as_int()), region = true;
        } else if (k == "ball") {
            if (!v.is_int() || v.as_int() < 0 || v.as_int() > 2) throw RouteError("invalid phase gate");
            out.ball = int(v.as_int());
        } else if (k == "s") {
            std::array<double, 2> a;
            if (!span_pair(v, a)) throw RouteError("invalid phase gate");
            out.s = a, region = true;
        } else if (k == "box") {
            std::array<double, 6> b;
            if (!span_box(v, b)) throw RouteError("invalid phase gate");
            out.box = b, region = true;
        } else {
            throw RouteError("invalid phase gate");
        }
    }
    if (!region) throw RouteError("invalid phase gate");
    return out;
}

static bool valid_id(const std::string &id) {
    if (id.empty() || !(id[0] >= 'a' && id[0] <= 'z')) return false;
    for (char c : id)
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-')) return false;
    if (id == "con" || id == "prn" || id == "aux" || id == "nul") return false;
    if (id.size() == 4 && (id.compare(0, 3, "com") == 0 || id.compare(0, 3, "lpt") == 0) && id[3] >= '1' && id[3] <= '9')
        return false;
    return true;
}

std::vector<BranchEntry> parse_branch_manifest(const Bytes &raw) {
    Json doc;
    try {
        doc = Json::parse(raw, 32);
    } catch (const Error &e) {
        throw RouteError(std::string("invalid branch manifest: ") + e.what());
    }
    const Json *ver = doc.is_object() ? doc.find("version") : nullptr;
    const Json *items = doc.is_object() ? doc.find("branches") : nullptr;
    if (!ver || !ver->is_number() || ver->as_double() != 2 || !items || !items->is_array() || items->items().empty())
        throw RouteError("expected version 2 branch manifest");
    std::vector<BranchEntry> out;
    std::set<std::string> seen;
    for (const Json &item : items->items()) {
        if (!item.is_object()) throw RouteError("invalid branch manifest");
        const Json *idj = item.find("id");
        if (!idj || !idj->is_string() || !valid_id(idj->as_string()) || seen.count(idj->as_string()))
            throw RouteError("invalid/duplicate branch id");
        BranchEntry e;
        e.id = idj->as_string();
        seen.insert(e.id);
        std::set<std::string> keys;
        for (auto &kv : item.members()) keys.insert(kv.first);
        const Json *leg = item.find("legacy");
        if (leg && leg->is_bool() && leg->as_bool() && keys == std::set<std::string>{"id", "legacy"}) {
            e.legacy = true;
            out.push_back(std::move(e));
            continue;
        }
        const Json *phases = item.find("phases");
        if (!phases || !phases->is_array() || phases->items().empty() ||
            keys != std::set<std::string>{"id", "phases", "merge"})
            throw RouteError("invalid branch phases");
        std::optional<double> prev;
        int idx = 0;
        for (const Json &ph : phases->items()) {
            if (!ph.is_object()) throw RouteError("invalid phase");
            BranchPhase p;
            if (const Json *src = ph.find("from")) {
                if (!src->is_array() || src->items().empty()) throw RouteError("invalid phase predecessor list");
                std::set<int> uniq;
                for (const Json &x : src->items()) {
                    if (!x.is_int() || x.as_int() < 0 || x.as_int() >= idx) throw RouteError("invalid phase predecessor list");
                    p.from.push_back(int(x.as_int()));
                    uniq.insert(int(x.as_int()));
                }
                if (!uniq.count(idx - 1) || uniq.size() != p.from.size()) throw RouteError("invalid phase predecessor list");
            }
            for (auto &kv : ph.members())
                if (kv.first != "when" && kv.first != "local_s" && kv.first != "from") throw RouteError("invalid phase");
            const Json *when = ph.find("when");
            if (!when || !when->is_object() || !when->find("ball")) throw RouteError("invalid phase");
            const Json *ls = ph.find("local_s");
            std::array<double, 2> b;
            if (!ls || !span_pair(*ls, b) || b[0] >= b[1] || (prev && *prev != b[0]))
                throw RouteError("phase spans must partition the diversion");
            p.lo = b[0], p.hi = b[1];
            prev = b[1];
            p.when = parse_gate(*when);
            e.phases.push_back(std::move(p));
            ++idx;
        }
        e.merge = parse_gate(item.at("merge"));
        out.push_back(std::move(e));
    }
    return out;
}

MultiFollower::MultiFollower(std::shared_ptr<const ReferencePath> m, std::vector<BranchEntry> es,
                             std::vector<std::unique_ptr<BranchFollower>> ms)
    : main(std::move(m)), entries(std::move(es)), members(std::move(ms)) {
    if (members.empty() || members.size() != entries.size()) throw RouteError("manifest/member mismatch");
    for (size_t i = 0; i < entries.size(); ++i) {
        const BranchEntry &e = entries[i];
        if (e.legacy) continue;
        const BranchFollower &f = *members[i];
        std::vector<double> cuts;
        for (auto &p : e.phases) cuts.push_back(p.lo);
        cuts.push_back(e.phases.back().hi);
        bool ok = cuts.front() == f.lo && cuts.back() == f.hi;
        for (double x : cuts) ok = ok && std::find(f.alt->cum.begin(), f.alt->cum.end(), x) != f.alt->cum.end();
        if (!ok) throw RouteError("phase spans must cover branch_s and end at line vertices");
    }
    reset();
}

void MultiFollower::reset() {
    key.reset();
    phase = 0;
    done.clear();
    peak_.reset();
    credit_ = 0, clock_ms_ = 0.0;
    anchor_.reset();
    timing = BranchTiming();
    primary().reset();
    for (auto &f : members) f->reset();
}

int MultiFollower::index_of(const std::string &id) const {
    for (size_t i = 0; i < entries.size(); ++i)
        if (entries[i].id == id) return int(i);
    return -1;
}

int MultiFollower::rest_member(const std::string &id) const {
    int found = -1, n = 0;
    for (size_t i = 0; i < entries.size(); ++i)
        if (id.empty() ? entries[i].legacy : entries[i].id == id) found = int(i), ++n;
    if (n != 1) throw RouteError("unknown or ambiguous rest branch '" + id + "'; specify branch:<id>:<s>");
    return found;
}

void MultiFollower::restore(const std::string &id, int ph, double local) {
    int i = index_of(id);
    if (i < 0) throw RouteError("unknown branch " + id);
    key = i, phase = ph;
    members[size_t(i)]->route(1).set_s_prev(local);
}

Projection MultiFollower::project(int i, const V3 &pos, int sector) {
    const BranchEntry &e = entries[size_t(i)];
    PathFollower &route = members[size_t(i)]->route(1);
    if (e.legacy) return route.update(pos, sector);
    double lo = e.phases[size_t(phase)].lo, hi = e.phases[size_t(phase)].hi;
    const ReferencePath &path = *route.path;
    auto near = [&](double a, double b) {  // the nearest segment within the arc range (no reacquire across legs)
        int64_t i0, i1;
        path.segments(a, b, i0, i1);
        return path.nearest(pos, i0, i1);
    };
    Projection pr = near(lo, hi);
    bool jump = true;
    if (route.s_prev()) {
        double a = std::max(lo, *route.s_prev() - route.back()), b = std::min(hi, *route.s_prev() + route.ahead());
        if (a < b) {
            Projection local = near(a, b);
            if (local.dist <= pr.dist + route.reacquire()) pr = local, jump = false;
        }
    }
    route.set_s_prev(pr.s);
    pr.jump = jump;
    pr.mode = jump ? "reacquire" : "window";
    return pr;
}

void MultiFollower::account(double local, const ReferencePath &path, const std::vector<double> &clock,
                            const std::vector<double> &ms, const std::vector<double> &bs, bool switched) {
    double station = np_interp(local, bs, ms);
    if (!peak_) {
        peak_ = station, credit_ = station;
    } else if (switched) {
        peak_ = std::max(*peak_, station);
        double end = ms[size_t(std::min<int64_t>(ss_right(ms, *peak_), int64_t(ms.size()) - 1))];
        anchor_ = std::array<double, 3>{*peak_, credit_, end};
    } else if (station > *peak_) {
        double start = np_interp(*peak_, ms, bs);
        clock_ms_ += np_interp(local, path.cum, clock) - np_interp(start, path.cum, clock);
        peak_ = station;
        if (anchor_ && station < (*anchor_)[2]) {
            double x = (*anchor_)[0], y = (*anchor_)[1], end = (*anchor_)[2];
            credit_ = y + (station - x) * (end - y) / (end - x);
        } else {
            credit_ = station;
            anchor_.reset();
        }
    }
}

BranchStep MultiFollower::update(const V3 &pos, int sector, int bt) {
    std::optional<int> old = key;
    bool changed = false;
    Projection pr;
    if (!old) {
        pr = primary().update(pos, sector);
        for (size_t i = 0; i < entries.size(); ++i) {
            if (done.count(int(i))) continue;
            const BranchEntry &e = entries[i];
            const BranchFollower &f = *members[i];
            Projection alt;
            if (e.legacy) {
                if (sector != f.sector || bt != f.ball_type) continue;
                alt = project(int(i), pos, sector);
                if (!(f.lo <= alt.s && alt.s <= f.hi)) continue;
            } else {
                if (!e.phases[0].when.matches(pos, sector, bt, pr.s)) continue;
                phase = 0;
                alt = project(int(i), pos, sector);
            }
            phase = 0;
            key = int(i);
            pr = alt;
            break;
        }
    } else {
        const BranchEntry &e = entries[size_t(*old)];
        BranchFollower &f = *members[size_t(*old)];
        pr = project(*old, pos, sector);
        bool leave;
        if (e.legacy) {
            leave = sector != f.sector || bt != f.ball_type || !(f.lo <= pr.s && pr.s <= f.hi);
        } else {
            const auto &phases = e.phases;
            const BranchGate &w = phases[size_t(phase)].when;
            bool left_sector = w.sector && *w.sector != sector;
            leave = phase == int(phases.size()) - 1 && e.merge.matches(pos, sector, bt, pr.s);
            if (leave && !left_sector) account(pr.s, *f.alt, f.clocks[1], f.ms, f.bs, false);
            std::optional<int> target;
            for (int i = phase + 1; i < int(phases.size()); ++i) {
                const auto &from = phases[size_t(i)].from;
                bool eligible = from.empty() ? phase == i - 1 : std::find(from.begin(), from.end(), phase) != from.end();
                if (eligible && phases[size_t(i)].when.matches(pos, sector, bt, pr.s)) {
                    target = i;
                    break;
                }
            }
            if (!leave && target) {
                account(pr.s, *f.alt, f.clocks[1], f.ms, f.bs, false);
                phase = *target;
                pr = project(*old, pos, sector);
                changed = true;
            } else if (left_sector) {
                leave = true;
            }
        }
        if (leave) {
            if (!e.legacy) done.insert(*old);
            key.reset();
            primary().reset();
            f.route(1).reset();
            pr = primary().update(pos, sector);
        }
    }
    double local = pr.s;
    const ReferencePath *path;
    const std::vector<double> *clock, *ms, *bs;
    static thread_local std::vector<double> ident;
    if (key) {
        BranchFollower &f = *members[size_t(*key)];
        path = f.alt.get(), clock = &f.clocks[1], ms = &f.ms, bs = &f.bs;
    } else {
        path = main.get(), clock = &members[0]->clocks[0];
        if (old) {
            ms = bs = &members[size_t(*old)]->ms;
        } else {
            ident = {0.0, main->length};
            ms = bs = &ident;
        }
    }
    account(local, *path, *clock, *ms, *bs, key != old || changed);
    double frontier = key ? np_interp(*peak_, *ms, *bs) : *peak_;
    double remaining = clock->back() - np_interp(frontier, path->cum, *clock);
    timing.set = true, timing.branch = bool(key), timing.multi = true;
    timing.branch_id = key ? entries[size_t(*key)].id : std::string();
    timing.phase = key ? phase : -1;
    timing.ref_elapsed_ms = clock_ms_, timing.ref_remaining_ms = remaining;
    BranchStep out;
    out.path = path;
    out.pr = pr;
    out.pr.s = credit_;
    out.local_s = local;
    out.branch = bool(key);
    out.multi = true;
    out.branch_id = timing.branch_id;
    out.phase = timing.phase;
    out.ref_elapsed_ms = clock_ms_;
    out.ref_remaining_ms = remaining;
    return out;
}

}  // namespace ballrain
