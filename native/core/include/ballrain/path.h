// Reference paths and the continuity projection (a port of ballrain/path.py).
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ballrain/geometry.h"
#include "ballrain/json.h"

namespace ballrain {

class GameTracker;

struct Projection {
    double s = 0;
    int64_t seg = 0;
    double t = 0;
    V3 point{};
    V3 tangent{};
    double dist = 0;
    int sector = 0;
    bool jump = false;
    const char *mode = "";
};

class ReferencePath {
public:
    ReferencePath(std::vector<V3> points, std::vector<int64_t> point_sector, std::string source, Json meta,
                  std::optional<std::vector<double>> t_ms);
    static std::shared_ptr<ReferencePath> load(const std::string &npz_path);
    static std::shared_ptr<ReferencePath> from_tracker(const GameTracker &t);  // the checkpoint polyline

    std::vector<V3> P, seg;
    std::vector<double> seg_len, L2, cum;
    std::vector<int64_t> point_sector, seg_sector;
    std::optional<std::vector<double>> t_ms;
    double length = 0;
    int n_sectors = 1;
    std::string source;
    Json meta;
    std::map<int, std::pair<double, double>> sector_range;

    std::pair<double, double> arc_window(std::optional<int> sector, double margin) const;
    Projection nearest(const V3 &p, int64_t i0, int64_t i1) const;  // segments [i0, i1)
    Projection project(const V3 &p, std::optional<int> sector, std::optional<double> s_prev, double back, double ahead,
                       double margin, double reacquire) const;
    V3 point_at(double s) const;
    std::optional<double> speed_at(double s, double half = 2.5) const;
    void segments(double s_lo, double s_hi, int64_t &i0, int64_t &i1) const;  // ReferencePath._segments: [i0, i1)

private:
    std::vector<V3> tan_;
};

class PathFollower {
public:
    PathFollower(std::shared_ptr<const ReferencePath> path, double back, double ahead, double margin, double reacquire)
        : path(std::move(path)), back_(back), ahead_(ahead), margin_(margin), reacquire_(reacquire) {}
    void reset() { s_prev_.reset(); }
    Projection update(const V3 &p, std::optional<int> sector) {
        Projection r = path->project(p, sector, s_prev_, back_, ahead_, margin_, reacquire_);
        s_prev_ = r.s;
        return r;
    }
    std::shared_ptr<const ReferencePath> path;
    // the follower's continuity state and window (multi_branch.MultiFollower.project reads them, restore sets s_prev)
    const std::optional<double> &s_prev() const { return s_prev_; }
    void set_s_prev(double s) { s_prev_ = s; }
    double back() const { return back_; }
    double ahead() const { return ahead_; }
    double reacquire() const { return reacquire_; }

private:
    double back_, ahead_, margin_, reacquire_;
    std::optional<double> s_prev_;
};

struct Anchors {
    std::optional<V3> start;
    std::vector<V3> checkpoints;
    std::optional<V3> airship;
};
Anchors level_anchors(const GameTracker &t);
bool anchors_match(const Json &meta, const Anchors &a, double tol = 1.0);
bool anchors_ready(const Json &meta, const Anchors &a);

// A reference-line asset (L<nn>.npz) loaded once, with its file sha256.
// numpy.interp for one x (the end values outside xp), the same arithmetic as numpy (path.cpp)
double np_interp(double x, const std::vector<double> &xp, const std::vector<double> &fp);

struct PathAsset {
    std::string path;
    std::string sha256;
    std::shared_ptr<ReferencePath> line;
};
std::shared_ptr<PathAsset> load_path_asset(const std::string &npz_path);  // nullptr when the file is missing

}  // namespace ballrain
