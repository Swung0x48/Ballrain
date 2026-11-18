// Offline golden test: feed a recorded stream (tools/native/golden.py, or this client's own runner) through the native
// tracker / features / policy and compare with the recorded feature vectors and policy outputs.
//
//   nc_golden --manifest <policy>.json --stream <dir>/stream.bin --path-dir <dir of L01.npz> [--tol 1e-4] [--json out]
//
// A feature value matches when |cpp - py| <= tol or |cpp - py| <= tol * |py|. Log-probs and values are compared with
// the pool's LOGP_TOL / VALUE_TOL; the sampled action (policy streams) must be the recorded one unless u lies within
// ACTION_EPS of a cdf boundary.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "ballrain/features.h"
#include "ballrain/json.h"
#include "ballrain/policy.h"
#include "ballrain/session.h"
#include "ballrain/stream.h"

using namespace ballrain;

namespace {

struct Cursor {
    const std::vector<Record> &recs;
    size_t i = 0;
    const Record &next(uint32_t kind) {
        if (i >= recs.size()) throw Error("stream ended, expected record kind " + std::to_string(kind));
        const Record &r = recs[i++];
        if (r.kind != kind)
            throw Error("record " + std::to_string(i - 1) + " is kind " + std::to_string(r.kind) + ", expected " +
                        std::to_string(kind));
        return r;
    }
};

class ReplayLevelSource : public LevelSource {
public:
    explicit ReplayLevelSource(Cursor &c) : c_(c) {}
    std::optional<Bytes> load_full(bool, bool) override { return pop(K_LVL_FULL); }
    std::optional<Bytes> load_inc(uint64_t) override { return pop(K_LVL_INC); }

private:
    std::optional<Bytes> pop(uint32_t k) {
        const Record &r = c_.next(k);
        if (r.data.empty() || r.data[0] == 0) return std::nullopt;
        return r.data.substr(1);
    }
    Cursor &c_;
};

std::string arg(int argc, char **argv, const char *name, const char *dflt = nullptr) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    if (!dflt) throw Error(std::string("missing ") + name);
    return dflt;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        std::string mpath = arg(argc, argv, "--manifest");
        std::string spath = arg(argc, argv, "--stream");
        std::string pdir = arg(argc, argv, "--path-dir");
        double tol = std::atof(arg(argc, argv, "--tol", "1e-4").c_str());
        std::string out_json = arg(argc, argv, "--json", "");
        Json manifest = Policy::read_manifest(mpath);
        std::string npz = mpath.substr(0, mpath.size() - 5) + ".npz";
        Policy pol = Policy::load(npz, manifest);
        FeatureConfig fc = FeatureConfig::from_json(manifest.at("features"));
        fc.path_dir = pdir;
        auto recs = read_stream(spath);
        Cursor cur{recs};
        std::unique_ptr<FeatureExtractor> fx;
        std::unique_ptr<ReplayLevelSource> src;
        std::unique_ptr<Ingest> ing;
        std::vector<std::string> names;
        int n = 0;
        // per column: rows over tolerance, max abs diff and where
        std::vector<int64_t> bad;
        std::vector<double> maxd;
        std::vector<int64_t> maxrow;
        int64_t rows = 0, bad_rows = 0;
        std::vector<float> last_py;
        double max_lp = 0, max_v = 0;
        int64_t n_pol = 0, act_same = 0, act_boundary = 0, act_diff = 0, first_act_diff = -1;
        int64_t first_bad_row = -1;
        while (cur.i < recs.size()) {
            const Record &r = recs[cur.i++];
            switch (r.kind) {
                case K_HELLO:
                    fx = std::make_unique<FeatureExtractor>(fc);
                    if (!n) {
                        n = fx->size();
                        names = fx->names();
                        bad.assign(size_t(n), 0), maxd.assign(size_t(n), 0.0), maxrow.assign(size_t(n), -1);
                        if (n != pol.n_obs) throw Error("feature size " + std::to_string(n) + " != policy n_obs");
                    }
                    src = std::make_unique<ReplayLevelSource>(cur);
                    ing = std::make_unique<Ingest>(1, src.get(), nullptr);
                    break;
                case K_OBS: {
                    if (!ing) throw Error("OBS before HELLO");
                    Reader h(r.data);
                    ChanHeader ch;
                    ch.mod_state = h.u32(), ch.end_reason = h.u32(), ch.frames_done = h.u32(), ch.last_error = h.u32();
                    ch.mod_seq = h.u64();
                    ing->ingest(parse_observation(r.data.data() + 24, r.data.size() - 24), ch);
                    break;
                }
                case K_RESET: fx->reset_episode(); break;
                case K_FEAT: {
                    std::vector<float> py(r.data.size() / 4);
                    std::memcpy(py.data(), r.data.data(), r.data.size());
                    std::vector<float> cc = fx->compute(ing->tracker);
                    if (int(py.size()) != n) throw Error("recorded feature size differs");
                    bool any = false;
                    for (int c = 0; c < n; ++c) {
                        double d = std::fabs(double(cc[size_t(c)]) - double(py[size_t(c)]));
                        if (d > tol && d > tol * std::fabs(double(py[size_t(c)]))) ++bad[size_t(c)], any = true;
                        if (d > maxd[size_t(c)]) maxd[size_t(c)] = d, maxrow[size_t(c)] = rows;
                    }
                    if (any) {
                        ++bad_rows;
                        if (first_bad_row < 0) first_bad_row = rows;
                    }
                    ++rows;
                    last_py = py;
                    break;
                }
                case K_POLICY: {
                    Reader p(r.data);
                    uint32_t step = p.u32();
                    double u = p.f64();
                    uint32_t a = p.u32();
                    float v = p.f32();
                    std::vector<float> lp_py(size_t(pol.n_act));
                    p.arr(lp_py.data(), lp_py.size());
                    std::vector<float> x = pol.normalize(last_py);
                    std::vector<float> lp = pol.log_probs(x);
                    float vc = pol.value(x);
                    for (int k = 0; k < pol.n_act; ++k)
                        max_lp = std::max(max_lp, std::fabs(double(lp[size_t(k)]) - double(lp_py[size_t(k)])));
                    max_v = std::max(max_v, std::fabs(double(vc) - double(v)) / std::max(1.0, std::fabs(double(v))));
                    ++n_pol;
                    int ac = sample_action(lp, u);
                    if (ac == int(a)) ++act_same;
                    else if (action_consistent(lp, u, int(a))) ++act_boundary;
                    else {
                        ++act_diff;
                        if (first_act_diff < 0) first_act_diff = step;
                    }
                    break;
                }
                default: break;  // ACT, NOTE (LVL records are consumed by the ingest)
            }
        }
        std::printf("feature rows %lld, rows with a value over tol %g: %lld (first %lld)\n", (long long)rows, tol,
                    (long long)bad_rows, (long long)first_bad_row);
        double worst = 0;
        for (int c = 0; c < n; ++c) {
            worst = std::max(worst, maxd[size_t(c)]);
            if (bad[size_t(c)])
                std::printf("  %-22s %6lld rows over tol, max |diff| %.3g at row %lld\n", names[size_t(c)].c_str(),
                            (long long)bad[size_t(c)], maxd[size_t(c)], (long long)maxrow[size_t(c)]);
        }
        std::printf("max |diff| over all features %.3g\n", worst);
        if (n_pol)
            std::printf("policy rows %lld: max |logp diff| %.3g (tol %g), max value diff %.3g (tol %g x max(1,|v|)); "
                        "recorded action = cpp draw %lld, within eps of a boundary %lld, different %lld (first step %lld)\n",
                        (long long)n_pol, max_lp, LOGP_TOL, max_v, VALUE_TOL, (long long)act_same,
                        (long long)act_boundary, (long long)act_diff, (long long)first_act_diff);
        if (!out_json.empty()) {
            Json j = Json::object();
            j.set("rows", Json::integer(rows));
            j.set("bad_rows", Json::integer(bad_rows));
            j.set("first_bad_row", Json::integer(first_bad_row));
            j.set("tol", Json::number(tol));
            j.set("max_abs_diff", Json::number(worst));
            Json cols = Json::object();
            for (int c = 0; c < n; ++c)
                if (bad[size_t(c)] || maxd[size_t(c)] > 0) {
                    Json e = Json::object();
                    e.set("rows_over_tol", Json::integer(bad[size_t(c)]));
                    e.set("max_abs_diff", Json::number(maxd[size_t(c)]));
                    e.set("at_row", Json::integer(maxrow[size_t(c)]));
                    cols.set(names[size_t(c)], e);
                }
            j.set("columns", cols);
            j.set("policy_rows", Json::integer(n_pol));
            j.set("max_logp_diff", Json::number(max_lp));
            j.set("max_value_diff", Json::number(max_v));
            j.set("actions_same", Json::integer(act_same));
            j.set("actions_boundary", Json::integer(act_boundary));
            j.set("actions_different", Json::integer(act_diff));
            write_file(out_json, j.dump(true) + "\n");
        }
        bool ok = bad_rows == 0 && max_lp <= LOGP_TOL && max_v <= VALUE_TOL;
        std::printf("%s\n", ok ? "OK" : "MISMATCH");
        return ok ? 0 : 1;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "nc_golden: %s\n", e.what());
        return 2;
    }
}
