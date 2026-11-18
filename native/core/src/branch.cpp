#include "ballrain/branch.h"

#include <algorithm>

#include "ballrain/sha256.h"
#include "ballrain/util.h"

namespace ballrain {

static int64_t ss_left(const std::vector<double> &a, double x) {
    return int64_t(std::lower_bound(a.begin(), a.end(), x) - a.begin());
}
static int64_t ss_right(const std::vector<double> &a, double x) {
    return int64_t(std::upper_bound(a.begin(), a.end(), x) - a.begin());
}

BranchFollower::BranchFollower(std::shared_ptr<const ReferencePath> m, std::shared_ptr<const ReferencePath> a,
                               const Json &meta, double back, double ahead, double margin, double reacquire)
    : main(std::move(m)), alt(std::move(a)),
      routes_{PathFollower(main, back, ahead, margin, reacquire), PathFollower(alt, back, ahead, margin, reacquire)} {
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
