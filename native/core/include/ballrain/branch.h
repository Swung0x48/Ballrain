// The optional ball-type diversion of a level's reference line (a port of ballrain/branch.py; docs/l2-stone-branch.md):
// while the ball has the branch's type in the branch's sector and projects inside the branch's arc range, the path
// features follow the alternative line; progress stays in main-line metres (the JSON station pairs), is monotone
// within a life and pays nothing for the jump at a switch; the reference clock integrates the active line's t_ms over
// newly covered ground.
#pragma once

#include <array>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "ballrain/json.h"
#include "ballrain/path.h"

namespace ballrain {

// ballrain/branch_rules.py BranchRules: the tunable rules of a diversion, from its description's "rules" object
// (docs/l2-stone-branch.md, v1). Closed schema: an unknown name or an invalid value refuses the task
struct BranchRules {
    double stall_pause_speed = 1.0;                      // the branch's stall pause threshold: min(this, stall_speed)
    std::string stall_speed_source = "speed+speed_change";  // | "speed"
    std::string stall_odometer = "local_s";              // | "s" (the main-line credit)
    std::optional<double> rest_stop_speed;               // none: the env's rest_speed
    std::optional<int64_t> rest_max_drive_frames, rest_max_brake_frames;  // none: the env's rest_max_frames
    bool rest_require_support = true;
    double rest_position_tolerance = 2.0;
    std::string rest_fallback = "same_tas_frame";        // | "error"
    double drive_arrival_radius = 1.0, drive_gain = 0.8, drive_speed_cap = 6.0, drive_deadband = 0.3;
    int64_t rest_control_repeat = 2;

    static BranchRules from_desc(const Json &desc);
    bool is_default() const;
};
// the rules of <dir>/L<nn>.json (BranchRules::from_desc)
BranchRules load_branch_rules(const std::string &dir, int level);
// ballrain.branch_assets.validate's digest: sha256 over the three files' raw 32-byte digests (line, description,
// drive): features.branch_sha256
std::string branch_bundle_sha256(const Bytes &line, const Bytes &desc, const Bytes &drive);

struct BranchStep {
    const ReferencePath *path = nullptr;  // the active line (local quantities: lookahead, tangent, remaining)
    Projection pr;                        // pr.s = the earned main-line progress (credit)
    double local_s = 0;                   // arc length on the active line
    bool branch = false;
    double ref_elapsed_ms = 0, ref_remaining_ms = 0;
    // multi-branch (wire v2) only: the active member id ("" on the main line) and its phase (-1 on the main line)
    bool multi = false;
    std::string branch_id;
    int phase = -1;
};

struct BranchTiming {  // branch.timing (env info when no ball is visible: transforms, finish)
    bool set = false, branch = false;
    double ref_elapsed_ms = 0, ref_remaining_ms = 0;
    bool multi = false;
    std::string branch_id;
    int phase = -1;
};

// A diversion of the reference line: one v1 branch (BranchFollower) or the wire-v2 multi-branch routes (MultiFollower)
class RouteFollower {
public:
    virtual ~RouteFollower() = default;
    virtual void reset() = 0;
    virtual BranchStep update(const V3 &pos, int sector, int ball_type) = 0;
    BranchTiming timing;
};

class BranchFollower : public RouteFollower {
public:
    BranchFollower(std::shared_ptr<const ReferencePath> main, std::shared_ptr<const ReferencePath> alt, const Json &meta,
                   double back, double ahead, double margin, double reacquire);
    void reset() override;
    BranchStep update(const V3 &pos, int sector, int ball_type) override;
    PathFollower &route(int i) { return routes_[i]; }  // 0 main, 1 the branch line (MultiFollower)

    std::shared_ptr<const ReferencePath> main, alt;
    int ball_type = 0, sector = 0;
    double lo = 0, hi = 0;
    std::vector<double> ms, bs;  // the station pairs: main-line and branch-line metres
    std::vector<double> clocks[2];
    BranchRules rules;

private:
    PathFollower routes_[2];
    std::optional<bool> active_;
    double peak_ = 0, credit_ = 0, clock_ms_ = 0;
    std::optional<std::array<double, 3>> anchor_;
};

// multi_branch: a gate (when / merge): all supplied predicates ANDed, bounds inclusive
struct BranchGate {
    std::optional<int> sector, ball;
    std::optional<std::array<double, 2>> s;
    std::optional<std::array<double, 6>> box;  // lo x y z, hi x y z
    bool matches(const V3 &pos, int sector, int ball, double arc) const;
};
struct BranchPhase {
    BranchGate when;
    double lo = 0, hi = 0;   // local_s
    std::vector<int> from;   // eligible predecessors (empty: the immediately previous phase)
};
struct BranchEntry {  // one member of branches.json
    std::string id;
    bool legacy = false;
    std::vector<BranchPhase> phases;
    BranchGate merge;
};
// multi_branch.manifest: the closed schema of branches.json (version 2, ordered members); RouteError when invalid
std::vector<BranchEntry> parse_branch_manifest(const Bytes &raw);

// multi_branch.MultiFollower: the wire-v2 routes over one main line; one ledger and clock per life across all lines
class MultiFollower : public RouteFollower {
public:
    MultiFollower(std::shared_ptr<const ReferencePath> main, std::vector<BranchEntry> entries,
                  std::vector<std::unique_ptr<BranchFollower>> members);
    void reset() override;
    BranchStep update(const V3 &pos, int sector, int ball_type) override;
    // the verified rest start's route cursor (MultiFollower.restore)
    void restore(const std::string &key, int phase, double local);
    int index_of(const std::string &id) const;  // -1: unknown
    // rest_member: the member a rest start addresses (id, or "" = the unique legacy one); RouteError otherwise
    int rest_member(const std::string &id) const;

    std::shared_ptr<const ReferencePath> main;
    std::vector<BranchEntry> entries;
    std::vector<std::unique_ptr<BranchFollower>> members;
    std::optional<int> key;  // the active member
    int phase = 0;
    std::set<int> done;

private:
    Projection project(int i, const V3 &pos, int sector);
    void account(double local, const ReferencePath &path, const std::vector<double> &clock, const std::vector<double> &ms,
                 const std::vector<double> &bs, bool switched);
    PathFollower &primary() { return members[0]->route(0); }
    std::optional<double> peak_;
    double credit_ = 0, clock_ms_ = 0;
    std::optional<std::array<double, 3>> anchor_;
};

// branch.load_branch: <branch_dir>/L<nn>.npz + L<nn>.json, checked against the level and the main line's sha256;
// a directory with branches.json is the wire-v2 multi-branch routes (each member a v1 child directory)
std::unique_ptr<RouteFollower> load_branch(std::shared_ptr<const ReferencePath> main, int level,
                                            const std::string &branch_dir, const std::string &main_sha256, double back,
                                            double ahead, double margin, double reacquire);

}  // namespace ballrain
