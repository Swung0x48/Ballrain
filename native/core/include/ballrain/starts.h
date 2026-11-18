// TAS start states of a pool task (ballrain/curriculum.py TasStarts, read from the pool's starts pack:
// ballrain/pool/task.py export_starts_pack / install_starts_pack).
#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "ballrain/json.h"
#include "ballrain/wire.h"

namespace ballrain {

struct StartState {  // TasStarts.expected(k)
    int64_t frame, sector, checkpoints, points, lives;
    double sr_double;
    std::array<double, 3> pos;
};

class TasStarts {
public:
    // Checks the pack as install_starts_pack does (file sha256 = expect_sha, array names / dtypes / dimensions,
    // finiteness, meta schema, version, lengths, increasing start frames); Error on any problem.
    static TasStarts load_pack(const std::string &path, const std::string &expect_sha);
    // the env's TasStarts(record, camera_tol, camera_grace_ms) then exclude(exclude_starts)
    void apply_camera_filter(double tol, double grace_ms);
    void exclude(const std::vector<std::pair<int64_t, int64_t>> &ranges);

    std::optional<size_t> index_of(int64_t k) const;
    std::optional<StartState> expected(int64_t k) const;
    std::vector<Frame> prefix(int64_t k) const;

    std::string record, record_sha256;
    int level = 1;
    std::vector<Frame> frames;  // the record's frames
    // the kept start frames (after the camera filter and exclude)
    std::vector<int64_t> valid, sector, points, lives, checkpoints;
    std::vector<double> sr_double;
    std::vector<std::array<double, 3>> pos;
    std::optional<int64_t> levelfinish_frame;
    double sr_final = 0;  // NaN when unknown
    std::vector<int8_t> camera_quarter;
    std::vector<float> camera_dev;

private:
    struct Arrays {
        std::vector<int64_t> valid, sector, points, lives, checkpoints;
        std::vector<double> sr_double;
        std::vector<std::array<double, 3>> pos;
    };
    void restrict(const std::vector<bool> &mask);
    std::optional<Arrays> all_;  // before exclude (expected() knows the excluded frames' state)
};

}  // namespace ballrain
