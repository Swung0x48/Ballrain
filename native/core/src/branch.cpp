#include "ballrain/branch.h"

#include <algorithm>
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

std::unique_ptr<BranchFollower> load_branch(std::shared_ptr<const ReferencePath> main, int level,
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

}  // namespace ballrain
