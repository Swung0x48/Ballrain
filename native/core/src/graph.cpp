#include "ballrain/graph.h"

#include <algorithm>
#include <cmath>

#include "ballrain/util.h"

namespace ballrain {

static const size_t MAX_NODES = 256, MAX_VALUES = 256, MAX_HEADS = 16;

Json PolicyGraph::from_arch(const Json &arch) {
    std::string kind = arch.get_string("kind", ""), act = arch.get_string("activation", "");
    if ((kind != "mlp_actor_critic" && kind != "mlp_actor_critic_ball_type") || (act != "tanh" && act != "relu"))
        throw Error("unsupported architecture");
    size_t nl = arch.at("hidden").size() + 1;
    Json nodes = Json::array();
    auto node = [&](const char *op, const std::string &in, const std::string &out) {
        Json n = Json::object();
        n.set("op", Json::string(op)), n.set("in", Json::string(in)), n.set("out", Json::string(out));
        return n;
    };
    for (const char *hn : {"pi", "v"}) {
        std::string h = "x";
        for (size_t i = 0; i < nl; ++i) {
            std::string out = std::string(hn) + std::to_string(i), pre = std::string(hn) + "." + std::to_string(2 * i);
            bool last = i + 1 == nl;
            if (last && kind == "mlp_actor_critic_ball_type" && std::string(hn) == "pi") {
                Json c = node("ctx_argmax", "x", "ctx");
                c.set("lo", Json::integer(1)), c.set("hi", Json::integer(4));
                nodes.push(c);
                Json s = node("select_linear", h, "logits");
                s.set("index", Json::string("ctx"));
                Json heads = Json::array();
                for (std::string p : {pre, std::string("extra_heads.0"), std::string("extra_heads.1")}) {
                    Json hd = Json::object();
                    hd.set("w", Json::string(p + ".weight")), hd.set("b", Json::string(p + ".bias"));
                    heads.push(hd);
                }
                s.set("heads", heads);
                nodes.push(s);
                h = "logits";
                break;
            }
            Json l = node("linear", h, last && std::string(hn) == "pi" ? "logits" : last ? "value" : out);
            l.set("w", Json::string(pre + ".weight")), l.set("b", Json::string(pre + ".bias"));
            nodes.push(l);
            h = l.at("out").as_string();
            if (!last) nodes.push(node(act.c_str(), h, h));
        }
    }
    Json g = Json::object();
    g.set("version", Json::integer(GRAPH_VERSION)), g.set("input", Json::string("x")), g.set("nodes", nodes);
    g.set("logits", Json::string("logits")), g.set("value", Json::string("value"));
    return g;
}

PolicyGraph PolicyGraph::parse(const Json &g, const std::map<std::string, NpyArray> &arrays, int n_obs, int n_act) {
    if (!g.is_object() || g.get_int("version", 0) != GRAPH_VERSION) throw Error("policy graph: unsupported version");
    PolicyGraph p;
    std::map<std::string, int> ids;     // value name -> slot
    std::vector<int> width;             // vector length per slot (-1: an index)
    auto def = [&](const std::string &name, int w) {
        if (name.empty() || name.size() > 64) throw Error("policy graph: bad value name");
        auto it = ids.find(name);
        if (it != ids.end()) {  // in-place ops overwrite their input: same shape only
            if (width[size_t(it->second)] != w) throw Error("policy graph: " + name + " redefined with another shape");
            return it->second;
        }
        if (ids.size() >= MAX_VALUES) throw Error("policy graph: too many values");
        int id = int(ids.size());
        ids[name] = id, width.push_back(w);
        return id;
    };
    auto use = [&](const Json &n, const char *key) {
        auto it = ids.find(n.at(key).as_string());
        if (it == ids.end()) throw Error(std::string("policy graph: ") + n.at(key).as_string() + " used before defined");
        return it->second;
    };
    std::map<std::string, std::shared_ptr<const std::vector<float>>> loaded;
    auto weight = [&](const std::string &wn, const std::string &bn, int n_in) {
        auto wi = arrays.find(wn), bi = arrays.find(bn);
        if (wi == arrays.end() || bi == arrays.end()) throw Error("policy graph: array " + wn + " / " + bn + " missing");
        const NpyArray &w = wi->second, &b = bi->second;
        if (w.descr != "<f4" || b.descr != "<f4" || w.shape.size() != 2 || b.shape.size() != 1 || w.shape[1] != n_in ||
            b.shape[0] != w.shape[0] || w.shape[0] < 1 || w.shape[0] > 65536)
            throw Error("policy graph: " + wn + " has the wrong dtype / shape for its input");
        auto get = [&](const std::string &k) {
            auto it = loaded.find(k);
            if (it != loaded.end()) return it->second;
            auto v = std::make_shared<const std::vector<float>>(arrays.at(k).f32());
            loaded[k] = v, p.used_.push_back(k);
            return v;
        };
        Lin l;
        l.n_in = n_in, l.n_out = int(w.shape[0]);
        l.w = get(wn), l.b = get(bn);
        return l;
    };
    p.input_ = def(g.at("input").as_string(), n_obs);
    const Json &nodes = g.at("nodes");
    if (!nodes.is_array() || nodes.size() > MAX_NODES) throw Error("policy graph: nodes");
    for (const Json &n : nodes.items()) {
        std::string op = n.at("op").as_string();
        Node d;
        d.in = use(n, "in");
        int win = width[size_t(d.in)];
        if (op == "linear") {
            if (win < 0) throw Error("policy graph: linear of an index");
            d.op = 1;
            d.lin.push_back(weight(n.at("w").as_string(), n.at("b").as_string(), win));
            d.out = def(n.at("out").as_string(), d.lin[0].n_out);
        } else if (op == "tanh" || op == "relu") {
            if (win < 0) throw Error("policy graph: activation of an index");
            d.op = op == "tanh" ? 2 : 3;
            d.out = def(n.at("out").as_string(), win);
        } else if (op == "ctx_argmax") {
            d.op = 4;
            d.lo = int(n.at("lo").as_int()), d.hi = int(n.at("hi").as_int());
            if (win < 0 || d.lo < 0 || d.hi <= d.lo || d.hi > win || d.hi - d.lo > int(MAX_HEADS))
                throw Error("policy graph: ctx_argmax range");
            d.out = def(n.at("out").as_string(), -1);
        } else if (op == "select_linear") {
            d.op = 5;
            if (win < 0) throw Error("policy graph: select_linear of an index");
            d.index = use(n, "index");
            if (width[size_t(d.index)] != -1) throw Error("policy graph: select_linear index is not an index");
            const Json &heads = n.at("heads");
            if (!heads.is_array() || heads.size() < 1 || heads.size() > MAX_HEADS) throw Error("policy graph: heads");
            for (const Json &h : heads.items()) d.lin.push_back(weight(h.at("w").as_string(), h.at("b").as_string(), win));
            for (auto &l : d.lin)
                if (l.n_out != d.lin[0].n_out) throw Error("policy graph: heads of different sizes");
            d.out = def(n.at("out").as_string(), d.lin[0].n_out);
        } else {
            throw Error("policy graph: unknown op " + op);
        }
        p.nodes_.push_back(std::move(d));
    }
    p.n_values_ = int(ids.size());
    for (int w : width) p.is_index_.push_back(w < 0);
    auto out = [&](const char *key, int want) {
        auto it = ids.find(g.at(key).as_string());
        if (it == ids.end() || width[size_t(it->second)] != want)
            throw Error(std::string("policy graph: ") + key + " output missing or of the wrong size");
        return it->second;
    };
    p.logits_ = out("logits", n_act);
    p.value_ = out("value", 1);
    return p;
}

// the same arithmetic as the 0.5 Policy::mlp: a double accumulator per row, rounded to float32, + float32 bias
static void linear(const std::vector<float> &h, int n_in, int n_out, const float *w, const float *b, std::vector<float> &o) {
    o.assign(size_t(n_out), 0.0f);
    // 16 float32 partial sums per row (vectorised, as BLAS accumulates in float32; the order differs from numpy's
    // anyway: the verifier's log-prob tolerance covers it)
    const float *x = h.data();
    for (int r = 0; r < n_out; ++r) {
        const float *wr = w + size_t(r) * size_t(n_in);
        float a[16] = {0};
        int c = 0;
        for (; c + 16 <= n_in; c += 16)
            for (int j = 0; j < 16; ++j) a[j] += x[c + j] * wr[c + j];
        double acc = 0.0;
        for (int j = 0; j < 16; ++j) acc += double(a[j]);
        for (; c < n_in; ++c) acc += double(x[c]) * double(wr[c]);
        o[size_t(r)] = float(acc) + b[r];
    }
}

std::vector<float> PolicyGraph::run(const std::vector<float> &x, int want) const {
    std::vector<std::vector<float>> v(static_cast<size_t>(n_values_));
    std::vector<int> idx(static_cast<size_t>(n_values_), 0);
    v[size_t(input_)] = x;
    for (const Node &d : nodes_) {
        const std::vector<float> &h = v[size_t(d.in)];
        std::vector<float> o;
        switch (d.op) {
        case 1: linear(h, d.lin[0].n_in, d.lin[0].n_out, d.lin[0].w->data(), d.lin[0].b->data(), o); break;
        case 2:
            o = h;
            for (float &e : o) e = std::tanh(e);
            break;
        case 3:
            o = h;
            for (float &e : o) e = std::max(e, 0.0f);
            break;
        case 4: {  // numpy: where(bits.max() > 0, bits.argmax(), 0) (argmax: the first maximum)
            int best = d.lo;
            for (int i = d.lo + 1; i < d.hi; ++i)
                if (h[size_t(i)] > h[size_t(best)]) best = i;
            idx[size_t(d.out)] = h[size_t(best)] > 0 ? best - d.lo : 0;
            continue;
        }
        case 5: {
            int i = idx[size_t(d.index)];
            if (i < 0 || i >= int(d.lin.size())) throw Error("policy graph: head index out of range");
            const Lin &l = d.lin[size_t(i)];
            linear(h, l.n_in, l.n_out, l.w->data(), l.b->data(), o);
            break;
        }
        }
        v[size_t(d.out)] = std::move(o);
    }
    return v[size_t(want)];
}

std::vector<float> PolicyGraph::logits(const std::vector<float> &x) const { return run(x, logits_); }
float PolicyGraph::value(const std::vector<float> &x) const { return run(x, value_)[0]; }

}  // namespace ballrain
