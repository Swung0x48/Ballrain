#include "ballrain/policy.h"

#include <algorithm>
#include <cmath>
#include <set>

#include "ballrain/npz.h"
#include "ballrain/sha256.h"

namespace ballrain {

static const char *FORMAT = "ballrain-pool-weights/1";
static const size_t MANIFEST_MAX_BYTES = 256 * 1024;
static const size_t NPZ_MAX_BYTES = 64 * 1024 * 1024;

// ---------------------------------------------------------------------------------------------- sampling

uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

double sample_u(uint64_t seed, uint64_t t) {
    uint64_t z = splitmix64(seed ^ splitmix64(t));
    return double(z >> 11) * (1.0 / double(uint64_t(1) << 53));
}

double numpy_sum(const double *a, size_t n) {
    if (n < 8) {
        double r = 0.0;  // numpy: -0.0 identity for floats is irrelevant here (sums of exp)
        for (size_t i = 0; i < n; ++i) r += a[i];
        return r;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + size_t(j)];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    size_t n2 = (n / 2) - ((n / 2) % 8);
    return numpy_sum(a, n2) + numpy_sum(a + n2, n - n2);
}

std::vector<double> cdf(const std::vector<float> &lp) {
    std::vector<double> p(lp.size());
    for (size_t i = 0; i < lp.size(); ++i) p[i] = std::exp(double(lp[i]));
    double s = numpy_sum(p.data(), p.size());
    std::vector<double> c(p.size());
    double acc = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        double q = p[i] / s;
        acc = i == 0 ? q : acc + q;
        c[i] = acc;
    }
    return c;
}

int sample_action(const std::vector<float> &lp, double u) {
    std::vector<double> c = cdf(lp);
    int64_t i = int64_t(std::upper_bound(c.begin(), c.end(), u) - c.begin());  // searchsorted side="right"
    return int(std::min<int64_t>(i, int64_t(c.size()) - 1));
}

bool action_consistent(const std::vector<float> &lp, double u, int a, double eps) {
    std::vector<double> c = cdf(lp);
    int n = int(c.size());
    if (a < 0 || a >= n) return false;
    double lo = a > 0 ? c[size_t(a) - 1] : 0.0;
    double hi = a < n - 1 ? c[size_t(a)] : 1.0;
    return lo - eps <= u && u < hi + eps;
}

// ---------------------------------------------------------------------------------------------- manifest / weights

Json Policy::read_manifest(const std::string &path) {
    Bytes b;
    if (!read_file(path, b)) throw Error("cannot read manifest " + path);
    if (b.size() > MANIFEST_MAX_BYTES) throw Error("manifest too large");
    Json m = Json::parse(b, 13);
    if (!m.is_object() || m.get_string("format", "") != FORMAT) throw Error(std::string("not a ") + FORMAT + " manifest");
    return m;
}

static const char *IDENTITY_FIELDS[] = {"format", "source", "arch", "obs_norm", "actions", "env", "engine_options",
                                        "features", "reward", "rollout", "arrays"};

static std::string policy_sha_of(const Json &m, const std::map<std::string, NpyArray> &arrays) {
    Sha256 h;
    h.update("ballrain-pool/policy/v1\0", 24);
    Json id = Json::object();
    for (auto *k : IDENTITY_FIELDS) id.set(k, m.at(k));
    h.update(id.dump(true));
    for (auto &kv : arrays) {  // std::map: sorted by name
        const NpyArray &a = kv.second;
        Json shape = Json::array();
        for (auto s : a.shape) shape.push(Json::integer(s));
        std::string head = std::string("\0", 1) + kv.first + std::string("\0", 1) + a.descr + std::string("\0", 1) +
                           shape.dump(true) + std::string("\0", 1);
        h.update(head);
        h.update(a.data);
    }
    return h.hexdigest();
}

std::string Policy::compute_policy_sha(const Json &manifest, const std::string &npz_path) {
    return policy_sha_of(manifest, load_npz(npz_path));
}

static bool all_finite(const NpyArray &a) {
    if (a.descr == "<f4") {
        for (float v : a.f32())
            if (!std::isfinite(v)) return false;
        return true;
    }
    for (double v : a.f64())
        if (!std::isfinite(v)) return false;
    return true;
}

Policy Policy::load(const std::string &npz_path, const Json &m, const std::string &expect_sha) {
    Bytes file;
    if (!read_file(npz_path, file)) throw Error("cannot read weights " + npz_path);
    if (file.size() > NPZ_MAX_BYTES || int64_t(file.size()) != m.at("npz_bytes").as_int())
        throw Error("weights file size differs from the manifest");
    if (Sha256::hex(file) != m.at("npz_sha256").as_string()) throw Error("weights file sha256 differs from the manifest");
    const Json &arch = m.at("arch");
    auto arrays = load_npz_bytes(file);
    const Json &spec = m.at("arrays");
    if (arrays.size() != spec.size()) throw Error("weights file arrays differ from the manifest");
    for (auto &kv : arrays) {
        const Json *s = spec.find(kv.first);
        if (!s) throw Error("weights file arrays differ from the manifest");
        if (kv.second.descr != s->at("dtype").as_string()) throw Error(kv.first + ": dtype differs from the manifest");
        const Json &shape = s->at("shape");
        if (shape.size() != kv.second.shape.size()) throw Error(kv.first + ": shape differs from the manifest");
        for (size_t i = 0; i < shape.size(); ++i)
            if (shape.at(i).as_int() != kv.second.shape[i]) throw Error(kv.first + ": shape differs from the manifest");
        if (kv.second.descr != "<f4" && kv.second.descr != "<f8") throw Error(kv.first + ": dtype not allowed");
        if (!all_finite(kv.second)) throw Error(kv.first + ": non-finite values");
    }
    Policy p;
    p.n_obs = int(arch.at("n_obs").as_int());
    p.n_act = int(arch.at("n_act").as_int());
    // policy_graph_v1: the manifest's graph (a layout this client has no code for), else the arch's (format-3 kinds)
    // the graph is not an identity field: bound by arch (its own graph for a format-3 kind, graph_sha256 for "graph")
    const Json *g = m.find("graph");
    Json graph;
    if (arch.get_string("kind", "") == "graph") {
        if (!g || Sha256::hex(g->dump(true)) != arch.get_string("graph_sha256", ""))
            throw Error("unsupported architecture: the graph does not match arch.graph_sha256");
        graph = *g;
    } else {
        graph = PolicyGraph::from_arch(arch);  // throws "unsupported architecture" for another kind / activation
        if (g && g->dump(true) != graph.dump(true)) throw Error("unsupported architecture: the graph is not the arch's");
    }
    p.graph_ = PolicyGraph::parse(graph, arrays, p.n_obs, p.n_act);
    if (arrays.size() != p.graph_.used_arrays().size() + 2 || !arrays.count("rms_mean") || !arrays.count("rms_var"))
        throw Error("array names unexpected / missing");
    auto mean = arrays.at("rms_mean").f64(), var = arrays.at("rms_var").f64();
    if (int(mean.size()) != p.n_obs || int(var.size()) != p.n_obs) throw Error("observation statistics shape");
    const Json &on = m.at("obs_norm");
    double eps = on.at("eps").as_double();
    p.clip_ = on.at("clip").as_double();
    p.mean_ = mean;
    p.std_.resize(var.size());
    for (size_t i = 0; i < var.size(); ++i) {
        if (var[i] < 0) throw Error("negative observation variance");
        p.std_[i] = std::sqrt(var[i] + eps);
    }
    std::string sha = policy_sha_of(m, arrays);
    if (sha != m.get_string("policy_sha", "") || (!expect_sha.empty() && sha != expect_sha))
        throw Error("policy_sha does not match the weights");
    p.sha = sha;
    p.manifest = m;
    for (auto &k : m.at("actions").at("keys").items()) p.action_keys.push_back(uint32_t(k.as_int()));
    if (int(p.action_keys.size()) != p.n_act) throw Error("action set size differs from the model");
    return p;
}

// ---------------------------------------------------------------------------------------------- forward

std::vector<float> Policy::normalize(const std::vector<float> &raw) const {
    if (int(raw.size()) != n_obs) throw Error("observation size");
    std::vector<float> x(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        double z = (double(raw[i]) - mean_[i]) / std_[i];
        z = std::min(std::max(z, -clip_), clip_);
        x[i] = float(z);
    }
    return x;
}

std::vector<float> Policy::logits(const std::vector<float> &x) const { return graph_.logits(x); }
float Policy::value(const std::vector<float> &x) const { return graph_.value(x); }

std::vector<float> Policy::log_softmax(const std::vector<float> &lg) {
    std::vector<double> z(lg.size());
    double mx = -INFINITY;
    for (size_t i = 0; i < lg.size(); ++i) z[i] = lg[i], mx = std::max(mx, z[i]);
    std::vector<double> e(z.size());
    for (size_t i = 0; i < z.size(); ++i) z[i] -= mx, e[i] = std::exp(z[i]);
    double lse = std::log(numpy_sum(e.data(), e.size()));
    std::vector<float> out(z.size());
    for (size_t i = 0; i < z.size(); ++i) out[i] = float(z[i] - lse);
    return out;
}

std::vector<float> Policy::log_probs(const std::vector<float> &x) const { return log_softmax(logits(x)); }

}  // namespace ballrain
