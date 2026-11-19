// The policy network as data (policy_graph_v1): the manifest's "graph" names the operations and the npz arrays they
// use, so a new network layout needs no client update. The executor runs in native code (the network is ~12-16% of a
// step: too hot for the wasm module). Python: ballrain/pool/weights.py build_graph / GraphPolicy.
//
//   {"version": 1, "input": "x", "nodes": [...], "logits": "<value name>", "value": "<value name>"}
//   nodes, executed in order, each writing "out":
//     {"op": "linear", "in": v, "w": array (out, in) f4, "b": array (out,) f4}   h @ w.T + b (float32 result)
//     {"op": "tanh" | "relu", "in": v}
//     {"op": "ctx_argmax", "in": v, "lo": i, "hi": j}   an index: argmax of v[lo:hi] when its max > 0, else 0
//     {"op": "select_linear", "in": v, "index": idx, "heads": [{"w": .., "b": ..}, ...]}   linear with heads[idx]
//   "input" is the normalised observation (obs_norm stays the manifest's own field).
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ballrain/json.h"
#include "ballrain/npz.h"

namespace ballrain {

constexpr int GRAPH_VERSION = 1;

class PolicyGraph {
public:
    // checks every node (op known, inputs defined before use, arrays present with float32 shapes that chain), throws
    // Error otherwise; n_obs / n_act: the manifest's sizes (the input's and the logits' length)
    static PolicyGraph parse(const Json &graph, const std::map<std::string, NpyArray> &arrays, int n_obs, int n_act);
    // the graph of a manifest without one (format-3 arch kinds): mlp_actor_critic and mlp_actor_critic_ball_type
    static Json from_arch(const Json &arch);

    std::vector<float> logits(const std::vector<float> &x) const;
    float value(const std::vector<float> &x) const;
    // the arrays the graph uses (all of them: the policy's npz may hold nothing else besides rms_mean / rms_var)
    const std::vector<std::string> &used_arrays() const { return used_; }

private:
    struct Lin {
        int n_in = 0, n_out = 0;
        std::shared_ptr<const std::vector<float>> w, b;  // shared: a copied graph stays valid
    };
    struct Node {
        int op = 0;  // 1 linear, 2 tanh, 3 relu, 4 ctx_argmax, 5 select_linear
        int in = -1, index = -1, out = -1;
        int lo = 0, hi = 0;
        std::vector<Lin> lin;  // linear: one; select_linear: the heads
    };
    std::vector<float> run(const std::vector<float> &x, int want) const;
    std::vector<Node> nodes_;
    int input_ = 0, logits_ = -1, value_ = -1, n_values_ = 0;
    std::vector<bool> is_index_;
    std::vector<std::string> used_;
};

}  // namespace ballrain
