// The recording format shared with tools/native/golden.py: b"BRGS", u32 version, then records (u32 kind, u32 length,
// payload). The live runner writes it, the offline replay reads it, tools/native/stream_diff.py compares two.
#pragma once

#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "ballrain/util.h"

namespace ballrain {

enum RecKind : uint32_t {
    K_HELLO = 1,     // the engine's hello string
    K_OBS = 2,       // socket OBS payload: u32 mod_state, end_reason, frames_done, last_error, u64 mod_seq, OBS1
    K_ACT = 3,       // ACT1 sent
    K_LVL_FULL = 4,  // u8 status (0 = not read), then level.msg bytes
    K_LVL_INC = 5,   // u8 status (0 = not read), then the unread tail of level.msg.inc
    K_RESET = 6,     // FeatureExtractor.reset_episode()
    K_FEAT = 7,      // float32[n_obs]
    K_POLICY = 8,    // u32 step, f64 u, u32 action, f32 value, f32 log_probs[n_act]
    K_NOTE = 9,      // JSON
    K_ROUTE = 10,    // JSON {"id", "phase", "s"}: a verified rest start's route cursor (MultiFollower::restore), after RESET
};

struct Record {
    uint32_t kind;
    Bytes data;
};

class StreamWriter {
public:
    explicit StreamWriter(const std::string &path);
    // records to a callback instead of a file (the feature module's feed; both: tee = a file writer to copy them to)
    using Sink = std::function<void(uint32_t kind, const void *p, size_t n)>;
    explicit StreamWriter(Sink sink, StreamWriter *tee = nullptr) : sink_(std::move(sink)), tee_(tee) {}
    ~StreamWriter();
    void put(uint32_t kind, const Bytes &payload);
    void put(uint32_t kind, const void *p, size_t n);
    void flush();

private:
    std::FILE *f_ = nullptr;
    Sink sink_;
    StreamWriter *tee_ = nullptr;
};

std::vector<Record> read_stream(const std::string &path);

}  // namespace ballrain
