// The optional ball-type diversion of a level's reference line (a port of ballrain/branch.py; docs/l2-stone-branch.md):
// while the ball has the branch's type in the branch's sector and projects inside the branch's arc range, the path
// features follow the alternative line; progress stays in main-line metres (the JSON station pairs), is monotone
// within a life and pays nothing for the jump at a switch; the reference clock integrates the active line's t_ms over
// newly covered ground.
#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ballrain/json.h"
#include "ballrain/path.h"

namespace ballrain {

struct BranchStep {
    const ReferencePath *path = nullptr;  // the active line (local quantities: lookahead, tangent, remaining)
    Projection pr;                        // pr.s = the earned main-line progress (credit)
    double local_s = 0;                   // arc length on the active line
    bool branch = false;
    double ref_elapsed_ms = 0, ref_remaining_ms = 0;
};

struct BranchTiming {  // branch.timing (env info when no ball is visible: transforms, finish)
    bool set = false, branch = false;
    double ref_elapsed_ms = 0, ref_remaining_ms = 0;
};

class BranchFollower {
public:
    BranchFollower(std::shared_ptr<const ReferencePath> main, std::shared_ptr<const ReferencePath> alt, const Json &meta,
                   double back, double ahead, double margin, double reacquire);
    void reset();
    BranchStep update(const V3 &pos, int sector, int ball_type);

    std::shared_ptr<const ReferencePath> main, alt;
    int ball_type = 0, sector = 0;
    double lo = 0, hi = 0;
    std::vector<double> ms, bs;  // the station pairs: main-line and branch-line metres
    std::vector<double> clocks[2];
    BranchTiming timing;

private:
    PathFollower routes_[2];
    std::optional<bool> active_;
    double peak_ = 0, credit_ = 0, clock_ms_ = 0;
    std::optional<std::array<double, 3>> anchor_;
};

// branch.load_branch: <branch_dir>/L<nn>.npz + L<nn>.json, checked against the level and the main line's sha256
std::unique_ptr<BranchFollower> load_branch(std::shared_ptr<const ReferencePath> main, int level,
                                            const std::string &branch_dir, const std::string &main_sha256, double back,
                                            double ahead, double margin, double reacquire);

}  // namespace ballrain
