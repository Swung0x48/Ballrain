// The feature module (wasm32-wasip1, built by tools/native/build_module.py with wasi-sdk): the per-step logic that the
// trainer changes often, loaded per task from a signed .wasm (docs/native-client.md §12). It holds its own tracker and
// feature extractor (core/src/state.cpp, features.cpp, path.cpp, branch.cpp, reach_block.cpp, ...: the same sources
// as the native build) and is fed the session's record stream (core/include/ballrain/stream.h: HELLO, OBS, LVL_FULL,
// LVL_INC, RESET), exactly as nc_golden replays a recorded stream. The reach kernel and its fields stay native
// (fastreach_import.cpp: the host's br_host_reach_* imports), so the fields are loaded once per process.
//
// Files: the host preopens its cache directory read-only as "/cache"; the config's paths are under it.
//
// ABI (version 1), every export returns >= 0 on success, -1 on an error (br_error: the message):
//   br_abi() -> 1
//   br_alloc(n) -> ptr, br_free(ptr)
//   br_init(json, len): {"level": n, "features": {FeatureConfig}} (paths in the module's namespace)
//   br_feed(kind, ptr, len): one record
//   br_compute() -> n: the features of the current state (br_out: n float32), the info (br_info / br_info_len)
//   br_error() -> ptr to a NUL-terminated message
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "ballrain/features.h"
#include "ballrain/session.h"
#include "ballrain/stream.h"
#include "ballrain/util.h"
#include "ballrain/wire.h"

#define BR_EXPORT(name) extern "C" __attribute__((export_name(#name)))

using namespace ballrain;

namespace {

struct QueueSource : LevelSource {
    std::deque<Record> *q = nullptr;
    std::optional<Bytes> pop(uint32_t kind) {
        if (q->empty() || q->front().kind != kind)
            throw Error("record stream: expected level record " + std::to_string(kind));
        Record r = std::move(q->front());
        q->pop_front();
        if (r.data.empty() || r.data[0] == 0) return std::nullopt;
        return r.data.substr(1);
    }
    std::optional<Bytes> load_full(bool, bool) override { return pop(K_LVL_FULL); }
    std::optional<Bytes> load_inc(uint64_t) override { return pop(K_LVL_INC); }
};

struct State {
    int level = 1;
    FeatureConfig fc;
    bool configured = false;
    std::unique_ptr<FeatureExtractor> fx;
    std::unique_ptr<Ingest> ing;
    QueueSource src;
    std::deque<Record> lvl;
    bool have_obs = false;
    Bytes obs;  // the pending OBS record (its level records follow it in the stream)
    std::vector<float> out;
    Bytes info;
    std::string error;
};
State *S = nullptr;

// the pending observation once its level records are queued (the ingest pops them as the recorder wrote them)
void flush() {
    if (!S->have_obs) return;
    S->have_obs = false;
    if (!S->ing) throw Error("record stream: OBS before HELLO");
    Reader h(S->obs);
    ChanHeader ch;
    ch.mod_state = h.u32(), ch.end_reason = h.u32(), ch.frames_done = h.u32(), ch.last_error = h.u32();
    ch.mod_seq = h.u64();
    S->ing->ingest(parse_observation(S->obs.data() + 24, S->obs.size() - 24), ch);
    if (!S->lvl.empty()) throw Error("record stream: level records left over");
}

void put_str(Bytes &b, const std::string &s) {
    put<uint32_t>(b, uint32_t(s.size()));
    b += s;
}

// FeatureInfo, packed (native/core/src/feature_backend.cpp unpacks it): the fields the env and the task runner read
Bytes pack_info(const FeatureInfo &fi) {
    Bytes b;
    uint32_t flags = (fi.have_ball ? 1u : 0u) | (fi.path ? 2u : 0u) | (fi.has_branch ? 4u : 0u) | (fi.branch ? 8u : 0u) |
                     (fi.branch_multi ? 16u : 0u) | (fi.supported ? 32u : 0u) | (fi.gap ? 64u : 0u) |
                     (fi.edge_fan_min ? 128u : 0u) | (fi.path && fi.path->jump ? 256u : 0u);
    put<uint32_t>(b, flags);
    for (double v : fi.pos) put<double>(b, v);
    for (double v : fi.vel) put<double>(b, v);
    const Projection p = fi.path ? *fi.path : Projection();
    put<double>(b, p.s), put<int64_t>(b, p.seg), put<double>(b, p.t);
    for (double v : p.point) put<double>(b, v);
    for (double v : p.tangent) put<double>(b, v);
    put<double>(b, p.dist), put<int32_t>(b, p.sector);
    put_str(b, p.mode ? p.mode : "");
    put_str(b, fi.path_source);
    put<double>(b, fi.local_s), put<double>(b, fi.ref_elapsed_ms), put<double>(b, fi.ref_remaining_ms);
    put_str(b, fi.branch_id);
    put<int32_t>(b, fi.branch_phase);
    put<double>(b, fi.gap ? *fi.gap : 0.0);
    put_str(b, fi.surface);
    put<double>(b, fi.edge_fan_min ? *fi.edge_fan_min : 0.0);
    put_str(b, fi.reach_key);
    return b;
}

template <class F>
int guarded(F f) {
    try {
        return f();
    } catch (const std::exception &e) {
        S->error = e.what();
        if (S->error.empty()) S->error = "error";
        return -1;
    }
}

}  // namespace

BR_EXPORT(br_abi) int br_abi() { return 1; }
BR_EXPORT(br_alloc) void *br_alloc(uint32_t n) { return std::malloc(n ? n : 1); }
BR_EXPORT(br_free) void br_free(void *p) { std::free(p); }
BR_EXPORT(br_error) const char *br_error() { return S ? S->error.c_str() : "not initialised"; }

BR_EXPORT(br_init) int br_init(const char *json, uint32_t len) {
    if (!S) S = new State();
    return guarded([&] {
        Json c = Json::parse(Bytes(json, len), 13);
        S->level = int(c.get_int("level", 1));
        S->fc = FeatureConfig::from_json(c.at("features"));
        S->src.q = &S->lvl;
        S->configured = true;
        S->fx.reset(), S->ing.reset();
        return 0;
    });
}

BR_EXPORT(br_feed) int br_feed(uint32_t kind, const char *p, uint32_t len) {
    return guarded([&] {
        if (!S || !S->configured) throw Error("br_init first");
        switch (kind) {
            case K_HELLO:
                flush();
                S->fx = std::make_unique<FeatureExtractor>(S->fc);
                S->ing = std::make_unique<Ingest>(S->level, &S->src, nullptr);
                S->lvl.clear();
                break;
            case K_OBS:
                flush();
                S->obs.assign(p, len), S->have_obs = true;
                break;
            case K_LVL_FULL:
            case K_LVL_INC: S->lvl.push_back({kind, Bytes(p, len)}); break;
            case K_RESET:
                flush();
                if (!S->fx) throw Error("record stream: RESET before HELLO");
                S->fx->reset_episode();
                break;
            default: break;  // ACT, FEAT, POLICY, NOTE: not the module's
        }
        return 0;
    });
}

BR_EXPORT(br_compute) int br_compute() {
    return guarded([&] {
        if (!S || !S->fx) throw Error("record stream: no HELLO yet");
        flush();
        FeatureInfo fi;
        S->out = S->fx->compute(S->ing->tracker, &fi);
        S->info = pack_info(fi);
        return int(S->out.size());
    });
}

BR_EXPORT(br_out) const float *br_out() { return S ? S->out.data() : nullptr; }
BR_EXPORT(br_info) const char *br_info() { return S ? S->info.data() : nullptr; }
BR_EXPORT(br_info_len) uint32_t br_info_len() { return S ? uint32_t(S->info.size()) : 0; }
BR_EXPORT(br_size) int br_size() {
    return guarded([&] { return S && S->configured ? FeatureExtractor(S->fc).size() : throw Error("br_init first"); });
}
