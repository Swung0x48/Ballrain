// Observation features (a port of ballrain/features.py, FEATURE_VERSION f6), computed from the raw game state.
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ballrain/branch.h"
#include "ballrain/geometry.h"
#include "ballrain/json.h"
#include "ballrain/path.h"
#include "ballrain/state.h"

namespace ballrain {

struct FeatureConfig {
    std::string version = "f6";
    std::string camera = "orient_ref";
    double pos_scale = 50, vel_scale = 20, angvel_scale = 20;
    bool absolute_position = false;
    int height_grid = 7;
    double height_spacing = 1, height_clip = 5, height_up = 3;
    int fan_dirs = 8;
    double fan_step = 0.25, fan_max = 8;
    std::string fan_rule = "step";
    double fan_step_drop = 0.75, fan_drop = 1, fan_up = 1;
    bool mesh_edge = true;
    double rail_radius = 6;
    std::string path_source = "tas";
    std::string path_dir;  // directory of the L<nn>.npz asset ("" = none configured)
    std::string path_sha256;
    std::string branch_dir;  // the optional ball-type diversion (L<nn>.npz + L<nn>.json; "" = off: layout and values unchanged)
    double path_lateral_scale = 10, path_back = 10, path_ahead = 30, path_margin = 20, path_reacquire = 5;
    int waypoints = 3;
    double waypoint_spacing = 5;
    std::vector<double> far_waypoints, ref_speed_at;
    int bodies = 4;
    double body_radius = 60;
    bool dynamic_geometry = true, counters = true, hs_block = true;
    int follower_sources = 4;
    double checkpoint_frames_scale = 10000, clip = 10;

    static FeatureConfig from_json(const Json &j);  // the manifest's "features" (unknown keys: Error)
};

// What compute() learned besides the vector (the parts of the Python info dict the env / rewards read)
struct FeatureInfo {
    bool have_ball = false;
    V3 pos{}, vel{};
    std::optional<Projection> path;  // with a branch: path->s = the earned main-line progress
    std::string path_source;
    // the branch follower's view of this step (branch.BranchFollower: pr local_s / branch / ref_elapsed_ms / ...)
    bool has_branch = false, branch = false;
    double local_s = 0, ref_elapsed_ms = 0, ref_remaining_ms = 0;
    bool supported = false;
    std::optional<double> gap;
    std::string surface = "none";
    std::optional<double> edge_fan_min;
};

class LevelGeometry {
public:
    explicit LevelGeometry(const GameTracker &t);
    TriangleSet tris, rails;
    BoundaryEdges edges;
    uint64_t key;
};

class FeatureExtractor {
public:
    explicit FeatureExtractor(FeatureConfig cfg);
    const FeatureConfig &config() const { return cfg_; }
    const std::vector<std::pair<std::string, int>> &layout() const { return layout_; }
    std::vector<std::string> names() const;
    int size() const { return size_; }
    int offset(const std::string &block) const;

    std::vector<float> compute(const GameTracker &t, FeatureInfo *info = nullptr);
    void reset_episode() {
        if (follower_) follower_->reset();
        if (branch_) branch_->reset();
    }
    const BranchFollower *branch() const { return branch_.get(); }

private:
    struct Slices;
    const ReferencePath *reference_path(const GameTracker &t);
    bool needs_asset(const GameTracker &t) const;
    void check_route(const GameTracker &t) const;
    LevelGeometry &geometry(const GameTracker &t);
    void contacts(const GameTracker &t, std::vector<double> &F, FeatureInfo &info, const V3 &pos, const V3 &vel,
                  double radius, std::optional<V3> &contact, std::optional<V3> &support);
    void terrain(const GameTracker &t, std::vector<double> &F, FeatureInfo &info, const V3 &pos, double radius,
                 const std::optional<V3> &contact, const std::optional<V3> &support);
    void bodies(const GameTracker &t, std::vector<double> &F, const V3 &pos, const V3 &vel);
    double depth_margin(const GameTracker &t, const V3 &pos) const;
    std::vector<V3> mind_normals(const GameTracker &t, const V3 &pos);

    FeatureConfig cfg_;
    std::vector<std::pair<std::string, int>> layout_;
    std::map<std::string, int> off_;
    int size_ = 0;
    V3 right_{1, 0, 0}, fwd_{0, 0, 1};
    // reference path state
    std::shared_ptr<PathAsset> asset_;
    bool asset_loaded_ = false;
    std::shared_ptr<const ReferencePath> path_;
    std::unique_ptr<PathFollower> follower_;
    std::unique_ptr<BranchFollower> branch_;
    bool have_key_ = false;
    uint64_t key_tracker_ = 0;
    int key_version_ = -1, key_rows_ = -1;
    // geometry cache (one level geometry; rebuilt when the static content changes)
    std::unique_ptr<LevelGeometry> geo_;
    uint64_t geo_tracker_ = 0;
    int geo_version_ = -1;
};

}  // namespace ballrain
