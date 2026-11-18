// The f7 "reach" observation block (ballrain/reach_features.py ReachBlock + the FastReach.perceive wrapper;
// docs/f7-native-port.md): the fields of the tracker's current static geometry (chosen by FastReach.fields_key over
// the canonical triangles), the unknown zones of the level data, the moving bodies near the ball, the fastreach kernel
// (core/src/fastreach.cpp), unknown-clears-doomed, then the scaling to [-1, 1].
//
// Fields are never built here: they come from a directory of FastReach.fields_for_tracker files
// (L<nn>-<sha16 of the key>.npz, whose info.key is the key), e.g. the task's distributed reach assets. A geometry whose
// fields are missing is an error (the task fails), never a zero block.
#pragma once

#include <array>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ballrain/fastreach.h"
#include "ballrain/geometry.h"
#include "ballrain/state.h"

namespace ballrain {

constexpr int REACH_BLOCK_VERSION = 1;  // reach_features.BLOCK_VERSION
constexpr int N_REACH = fastreach::N_FEATURES;
std::string reach_version();  // reach_features.reach_version(): "fastreach<BUILD>.<QUERY>-block<BLOCK>"

// FastReach.canonical_geometry: the static triangles in an order independent of CK ids
struct CanonicalGeometry {
    std::vector<double> A, B, C;  // (n, 3) row-major
    std::vector<int64_t> rank;     // the owner entity's rank
    std::vector<uint8_t> rail;     // the owner is a rail entity
};
class LevelGeometry;
CanonicalGeometry canonical_geometry(const GameTracker &t, const LevelGeometry &geo);
// FastReach.fields_key (static triangles, the death boxes the build keeps, the fan push boxes, the constants, BUILD_VERSION)
std::string reach_fields_key(const GameTracker &t, const CanonicalGeometry &g);
// the file name of a key's fields: L<nn>-<sha16 of the key>.npz
std::string reach_fields_name(int level, const std::string &key);
// fastreach.fan_obbs(fan_boxes(level)): (k, 18) row-major
std::vector<double> reach_fan_obbs(const LevelData &lvl);
// reach_features._zone_boxes: the unknown zones (k, 6) (lo - 4, hi + 4) of the physicalized non-fixed entities
std::vector<double> reach_zone_boxes(const LevelData &lvl);
// reach_features.scale
void reach_scale(const double raw[N_REACH], double out[N_REACH]);

class ReachBlock {
public:
    // fields_dir: where the fields files are; expect_version: FeatureConfig.reach_version (another one is refused)
    ReachBlock(std::string fields_dir, double body_radius, const std::string &expect_version);
    // ReachBlock.compute: the scaled block (zeros for a non-finite state), the raw vector, the fields key
    void compute(const GameTracker &t, const LevelGeometry &geo, const V3 &pos, const V3 &vel, int quarter,
                 int ball_type, double out[N_REACH], double raw[N_REACH], std::string *key = nullptr);

private:
    struct LevelState {
        std::string key;
        std::vector<double> zones;
    };
    const LevelState &level_state(const GameTracker &t, const LevelGeometry &geo);
    std::shared_ptr<const fastreach::Fields> fields(int level, const std::string &key);

    std::string dir_;
    double body_radius_;
    uint64_t state_tracker_ = 0;
    int state_version_ = -1;
    LevelState state_;
    std::list<std::pair<std::string, std::shared_ptr<const fastreach::Fields>>> cache_;  // LRU, 2 geometries
    fastreach::Workspace ws_;
};

}  // namespace ballrain
