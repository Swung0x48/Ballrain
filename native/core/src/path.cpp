#include "ballrain/path.h"

#include <algorithm>
#include <limits>

#include "ballrain/npz.h"
#include "ballrain/sha256.h"
#include "ballrain/state.h"

namespace ballrain {

static double norm3(const double *d) { return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]); }

// numpy.searchsorted on a sorted array: side "right" = upper bound, "left" = lower bound
static int64_t ss_right(const std::vector<double> &a, double x) {
    return int64_t(std::upper_bound(a.begin(), a.end(), x) - a.begin());
}
static int64_t ss_left(const std::vector<double> &a, double x) {
    return int64_t(std::lower_bound(a.begin(), a.end(), x) - a.begin());
}
template <typename T>
static T clip(T v, T lo, T hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// numpy.interp for one x (left / right defaults: the end values)
static double interp(double x, const std::vector<double> &xp, const std::vector<double> &fp) {
    size_t n = xp.size();
    if (std::isnan(x)) return x;
    if (x < xp[0]) return fp[0];
    if (x > xp[n - 1]) return fp[n - 1];
    size_t j = size_t(std::upper_bound(xp.begin(), xp.end(), x) - xp.begin()) - 1;  // xp[j] <= x < xp[j+1]
    if (j >= n - 1) return fp[n - 1];
    if (xp[j] == x) return fp[j];
    double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    double r = slope * (x - xp[j]) + fp[j];
    if (std::isnan(r)) {
        r = slope * (x - xp[j + 1]) + fp[j + 1];
        if (std::isnan(r) && fp[j] == fp[j + 1]) r = fp[j];
    }
    return r;
}

double np_interp(double x, const std::vector<double> &xp, const std::vector<double> &fp) { return interp(x, xp, fp); }

ReferencePath::ReferencePath(std::vector<V3> points, std::vector<int64_t> psec, std::string src, Json m,
                             std::optional<std::vector<double>> t)
    : P(std::move(points)), point_sector(std::move(psec)), t_ms(std::move(t)), source(std::move(src)), meta(std::move(m)) {
    if (P.size() < 2) throw Error("a reference path needs at least two points");
    if (point_sector.empty()) {
        point_sector.push_back(1);
        for (size_t i = 1; i < P.size(); ++i) point_sector.push_back(int64_t(i));
    }
    size_t ns = P.size() - 1;
    seg.resize(ns), seg_len.resize(ns), L2.resize(ns);
    cum.assign(1, 0.0);
    double acc = 0.0;
    for (size_t i = 0; i < ns; ++i) {
        for (int k = 0; k < 3; ++k) seg[i][k] = P[i + 1][k] - P[i][k];
        seg_len[i] = norm3(seg[i].data());
        L2[i] = seg_len[i] == 0 ? 1.0 : seg_len[i] * seg_len[i];
        acc += seg_len[i];
        cum.push_back(acc);
    }
    length = cum.back();
    seg_sector.assign(point_sector.begin() + 1, point_sector.end());
    n_sectors = int(*std::max_element(point_sector.begin(), point_sector.end()));
    for (size_t i = 0; i < ns; ++i) {
        int s = int(seg_sector[i]);
        auto it = sector_range.find(s);
        if (it == sector_range.end()) sector_range[s] = {cum[i], cum[i + 1]};
        else it->second.second = cum[i + 1];
    }
    double span = meta.get_double("tangent_span", 0.0);
    std::vector<V3> d(ns);
    if (span > 0) {
        for (size_t i = 0; i < ns; ++i) {
            double mid = 0.5 * (cum[i] + cum[i + 1]);
            V3 a = point_at(clip(mid - span, 0.0, length)), b = point_at(clip(mid + span, 0.0, length));
            for (int k = 0; k < 3; ++k) d[i][k] = b[k] - a[k];
        }
    } else {
        d = seg;
    }
    tan_.resize(ns);
    V3 last{0.0, 0.0, 1.0};
    for (size_t i = 0; i < ns; ++i) {
        d[i][1] = 0;
        double ln = norm3(d[i].data());
        if (ln > 1e-6)
            for (int k = 0; k < 3; ++k) last[k] = d[i][k] / ln;
        tan_[i] = last;
    }
}

std::shared_ptr<ReferencePath> ReferencePath::load(const std::string &npz_path) {
    auto z = load_npz(npz_path);
    auto pts = z.at("points").f64();
    auto sec = z.at("sector").ints();
    Json meta = Json::parse(z.at("meta").unicode());
    std::optional<std::vector<double>> t;
    if (z.count("t_ms")) t = z.at("t_ms").f64();
    std::vector<V3> P(pts.size() / 3);
    for (size_t i = 0; i < P.size(); ++i) P[i] = {pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]};
    return std::make_shared<ReferencePath>(std::move(P), std::move(sec), "tas", std::move(meta), std::move(t));
}

std::pair<double, double> ReferencePath::arc_window(std::optional<int> sector, double margin) const {
    if (!sector || sector_range.empty()) return {0.0, length};
    int s = clip(*sector, sector_range.begin()->first, sector_range.rbegin()->first);
    auto it = sector_range.find(s);
    if (it == sector_range.end()) return {0.0, length};
    return {std::max(0.0, it->second.first - margin), std::min(length, it->second.second + margin)};
}

void ReferencePath::segments(double s_lo, double s_hi, int64_t &i0, int64_t &i1) const {
    int64_t ns = int64_t(seg.size());
    i0 = clip<int64_t>(ss_right(cum, s_lo) - 1, 0, ns - 1);
    i1 = clip<int64_t>(ss_left(cum, s_hi), i0 + 1, ns);
}

Projection ReferencePath::nearest(const V3 &p, int64_t i0, int64_t i1) const {
    Projection best;
    bool any = false;
    for (int64_t i = i0; i < i1; ++i) {
        const V3 &A = P[size_t(i)], &d = seg[size_t(i)];
        double t = ((p[0] - A[0]) * d[0] + (p[1] - A[1]) * d[1] + (p[2] - A[2]) * d[2]) / L2[size_t(i)];
        t = clip(t, 0.0, 1.0);
        V3 Q;
        for (int k = 0; k < 3; ++k) Q[k] = A[k] + d[k] * t;
        double e[3] = {p[0] - Q[0], p[1] - Q[1], p[2] - Q[2]};
        double dist = norm3(e);
        if (!any || dist < best.dist) {
            any = true;
            best.s = cum[size_t(i)] + t * seg_len[size_t(i)];
            best.seg = i;
            best.t = t;
            best.point = Q;
            best.tangent = tan_[size_t(i)];
            best.dist = dist;
            best.sector = int(seg_sector[size_t(i)]);
        }
    }
    return best;
}

Projection ReferencePath::project(const V3 &p, std::optional<int> sector, std::optional<double> s_prev, double back,
                                  double ahead, double margin, double reacquire) const {
    auto [lo, hi] = arc_window(sector, margin);
    int64_t i0, i1;
    segments(lo, hi, i0, i1);
    Projection glob = nearest(p, i0, i1);
    if (!s_prev) {
        glob.jump = true, glob.mode = "init";
        return glob;
    }
    double wlo = std::max(lo, *s_prev - back), whi = std::min(hi, *s_prev + ahead);
    if (whi <= wlo) {
        glob.jump = true, glob.mode = "window_empty";
        return glob;
    }
    segments(wlo, whi, i0, i1);
    Projection win = nearest(p, i0, i1);
    if (win.dist <= glob.dist + reacquire) {
        win.jump = false, win.mode = "window";
        return win;
    }
    glob.jump = true, glob.mode = "reacquire";
    return glob;
}

V3 ReferencePath::point_at(double s) const {
    s = clip(s, 0.0, length);
    int64_t ns = int64_t(seg.size());
    size_t i = size_t(clip<int64_t>(ss_right(cum, s) - 1, 0, ns - 1));
    double t = seg_len[i] > 0 ? (s - cum[i]) / seg_len[i] : 0.0;
    V3 out;
    for (int k = 0; k < 3; ++k) out[k] = P[i][k] + seg[i][k] * t;
    return out;
}

std::optional<double> ReferencePath::speed_at(double s, double half) const {
    if (!t_ms) return std::nullopt;
    double a = std::max(0.0, s - half), b = std::min(length, s + half);
    double ta = interp(a, cum, *t_ms), tb = interp(b, cum, *t_ms);
    return (b - a) / std::max(tb - ta, 1e-6) * 1000.0;
}

// ---------------------------------------------------------------------------------------------- anchors

static std::optional<V3> matrix_pos(const std::optional<CellValue> &v) {
    if (!v || v->kind != CellValue::Matrix) return std::nullopt;
    return V3{double(v->m[12]), double(v->m[13]), double(v->m[14])};
}

Anchors level_anchors(const GameTracker &t) {
    Anchors a;
    a.start = matrix_pos(t.arrays.get("ResetPoints", 0, 0));
    if (!a.start) {
        double p[3];
        if (t.level.entity_position("PS_FourFlames_MF", p)) a.start = V3{p[0], p[1], p[2]};
    }
    int rows = t.arrays.rows("Checkpoints");
    for (int r = 0; r < rows; ++r) {
        std::optional<V3> p;
        for (int c = 0; c < 2; ++c) {
            p = matrix_pos(t.arrays.get("Checkpoints", r, c));
            if (p) break;
        }
        if (p) a.checkpoints.push_back(*p);
    }
    for (const char *n : {"PE_Balloon_Platform", "PE_Balloon_MF"}) {
        double p[3];
        if (t.level.physical_position(n, p)) {
            a.airship = V3{p[0], p[1], p[2]};
            break;
        }
    }
    return a;
}

static bool json_v3(const Json &j, V3 &out) {
    if (!j.is_array() || j.size() != 3) return false;
    for (size_t k = 0; k < 3; ++k) out[k] = j.at(k).as_double();
    return true;
}

bool anchors_match(const Json &meta, const Anchors &a, double tol) {
    const Json *ms = meta.find("start");
    V3 s;
    if (!a.start || !ms || ms->is_null() || !json_v3(*ms, s)) return false;
    double d[3] = {s[0] - (*a.start)[0], s[1] - (*a.start)[1], s[2] - (*a.start)[2]};
    if (norm3(d) > tol) return false;
    const Json *cps = meta.find("checkpoints");
    size_t n = cps && cps->is_array() ? cps->size() : 0;
    if (n != a.checkpoints.size()) return false;
    for (size_t i = 0; i < n; ++i) {
        V3 c;
        if (!json_v3(cps->at(i), c)) return false;
        double e[3] = {c[0] - a.checkpoints[i][0], c[1] - a.checkpoints[i][1], c[2] - a.checkpoints[i][2]};
        if (norm3(e) > tol) return false;
    }
    return true;
}

bool anchors_ready(const Json &meta, const Anchors &a) {
    const Json *cps = meta.find("checkpoints");
    size_t n = cps && cps->is_array() ? cps->size() : 0;
    return a.start && a.checkpoints.size() >= n;
}

std::shared_ptr<ReferencePath> ReferencePath::from_tracker(const GameTracker &t) {
    Anchors a = level_anchors(t);
    if (!a.start) return nullptr;
    std::vector<V3> pts{*a.start};
    for (auto &c : a.checkpoints) pts.push_back(c);
    if (a.airship) pts.push_back(*a.airship);
    if (pts.size() < 2) return nullptr;
    std::vector<int64_t> sec{1};
    for (size_t i = 1; i < pts.size(); ++i) sec.push_back(int64_t(i));
    return std::make_shared<ReferencePath>(std::move(pts), std::move(sec), "checkpoints", Json::object(), std::nullopt);
}

std::shared_ptr<PathAsset> load_path_asset(const std::string &npz_path) {
    if (!file_exists(npz_path)) return nullptr;
    auto a = std::make_shared<PathAsset>();
    a->path = npz_path;
    a->sha256 = file_sha256(npz_path);
    a->line = ReferencePath::load(npz_path);
    return a;
}

}  // namespace ballrain
