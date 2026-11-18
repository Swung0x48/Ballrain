// The fastreach query kernel (a port of ballrain/fastreach.py: _perceive and everything it calls; docs/f7-native-port.md).
// Bit-identical to the Python kernel (numba and plain Python agree bit for bit): IEEE double, the same operation order,
// no FMA contraction, the libm-dependent tables embedded from Python.
//
// Contract with the reach block (core/src/reach_block.cpp, the FastReach.perceive wrapper): the kernel computes the raw
// vector of one state from the static fields, the unknown zones and the ball; the wrapper adds the moving bodies
// (unknown_zone) and the unknown-clears-doomed rule, then scales.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ballrain {
namespace fastreach {

constexpr int BUILD_VERSION = 5;  // fastreach.BUILD_VERSION (the fields)
constexpr int QUERY_VERSION = 2;  // fastreach.QUERY_VERSION (perceive)
constexpr int N_FEATURES = 59;    // fastreach.N_FEATURES (FEATURE_NAMES order)
constexpr int N_PSI = 396;

// The static fields of one geometry (FastReach.load of a fields_for_tracker file L<nn>-<sha16>.npz). The arrays are
// kept as the file has them (float32 / int8 / int16 / ...) and converted to double where the kernel reads them.
class Fields {
public:
    static std::shared_ptr<const Fields> load(const std::string &npz_path);
    std::string key;  // info["key"] (FastReach.fields_key of the geometry it was built from)
    struct Impl;
    std::shared_ptr<const Impl> impl;
};

// Per-caller scratch (the FastReach instance's work arrays: yy, xf, zf, D, fins, yl, ya, stack, stamp, ...). One per
// thread / env; reusable across states and fields.
class Workspace {
public:
    Workspace();
    ~Workspace();
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// fastreach._perceive for one state, as FastReach.perceive calls it (before its bodies / unknown-clears-doomed steps):
// pos, vel (BALL pos, speed + speed_change; finite, checked by the caller), ball type 0 wood / 1 paper / 2 stone,
// camera quarter 0..3 (the key-direction table), uzones (k x 6: lo, hi of the unknown zones, already inflated).
// out: N_FEATURES raw values; dbg (N_DBG) and pdd (8 x N_PDD), when not null, the kernel's debug arrays (validation).
void perceive(const Fields &fields, Workspace &ws, const std::vector<double> &uzones, int ball_type, int quarter,
              const double pos[3], const double vel[3], double *out, double *dbg = nullptr, double *pdd = nullptr);

int n_dbg();  // len(DBG_NAMES)
int n_pdd();  // len(PER_DIR_DBG)

// Validation (tools/native/reach_diff.py, nc_reach_diff): the workspace's stamp_id after the last perceive (the
// kernel's return value, which FastReach.perceive stores; one sequence per workspace as per FastReach instance), the
// deepest pyramid stack used (Python's is fixed at 512 rows; the C++ one grows).
int64_t stamp_id(const Workspace &ws);
int64_t stack_high_water(const Workspace &ws);
// The kernel's live log / pow (brake stop, rolling displacement): std::log / std::pow by default (nullptr restores
// them). Python uses the C runtime's; on Windows the system ucrtbase.dll's pow differs from the static CRT's by 1 ulp
// on a few arguments (fastreach.cpp), so the validation harness installs the DLL's. Not thread-safe: call before any
// perceive.
void set_libm(double (*log_fn)(double), double (*pow_fn)(double, double));
// One kernel helper on raw arguments (names and argument lists in fastreach.cpp: h_top, sdf, rail_dist, r_ground,
// r_hold, h_top_disc, march, fan_dist, brake_axis, v_reach, v_need, mount_speed, air_scan, fan_touch, log, pow).
// false: unknown name.
bool leaf(const Fields &fields, Workspace &ws, const std::string &name, const double *args, size_t n_args,
          std::vector<double> &out);

}  // namespace fastreach
}  // namespace ballrain
