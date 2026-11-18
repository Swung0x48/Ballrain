// Pool policy weights (ballrain/pool/weights.py: plain-array npz + JSON manifest) and the numpy-equivalent forward
// pass (NumpyPolicy), plus the deterministic action sampling of ballrain/pool/sampling.py.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ballrain/json.h"

namespace ballrain {

constexpr int SEED_BITS = 53;
constexpr double ACTION_EPS = 1e-4;
constexpr double LOGP_TOL = 1e-3;
constexpr double VALUE_TOL = 1e-3;

uint64_t splitmix64(uint64_t x);
double sample_u(uint64_t seed, uint64_t t);           // the uniform draw of task step t, in [0, 1)
std::vector<double> cdf(const std::vector<float> &log_probs);
int sample_action(const std::vector<float> &log_probs, double u);
bool action_consistent(const std::vector<float> &log_probs, double u, int a, double eps = ACTION_EPS);
double numpy_sum(const double *a, size_t n);          // np.add.reduce order (pairwise, 8 accumulators)

struct Layer {
    int n_in = 0, n_out = 0;
    std::vector<float> w;  // (n_out, n_in) row-major, as torch / the npz store it
    std::vector<float> b;
};

class Policy {
public:
    // Checks the npz against the manifest (size, sha256, names, dtypes, shapes, finiteness) and recomputes the
    // policy_sha (and compares it with expect_sha when given). Error on any mismatch.
    static Policy load(const std::string &npz_path, const Json &manifest, const std::string &expect_sha = "");
    static Json read_manifest(const std::string &path);
    static std::string compute_policy_sha(const Json &manifest, const std::string &npz_path);

    int n_obs = 0, n_act = 0;
    std::string sha;
    Json manifest;
    std::vector<uint32_t> action_keys;  // manifest actions.keys

    std::vector<float> normalize(const std::vector<float> &raw) const;  // float64 statistics, clip, float32
    std::vector<float> logits(const std::vector<float> &x) const;
    float value(const std::vector<float> &x) const;
    std::vector<float> log_probs(const std::vector<float> &x) const;
    static std::vector<float> log_softmax(const std::vector<float> &logits);

private:
    std::vector<float> mlp(const std::vector<Layer> &layers, const std::vector<float> &x) const;
    std::vector<Layer> pi_, v_;
    bool tanh_ = true;
    std::vector<double> mean_, std_;
    double clip_ = 10.0;
};

}  // namespace ballrain
