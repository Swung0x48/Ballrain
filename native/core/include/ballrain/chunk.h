// Pool chunks (ballrain/pool/protocol.py encode_chunk, CHUNK_FORMAT 2): b"BRPC", u16 format, u16 flags, u32 hlen,
// the canonical JSON header, the array sections (raw or zstd), sha256 over everything before it.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ballrain/json.h"
#include "ballrain/util.h"

namespace ballrain {

constexpr uint16_t CHUNK_FORMAT = 2;
constexpr int PROTOCOL_VERSION = 1;
constexpr double MIN_LOGP = -60.0;
constexpr uint8_t FLAG_TERMINATED = 1, FLAG_TRUNCATED = 2;

uint16_t f32_to_f16(float v);  // IEEE half, round to nearest even (numpy astype("<f2"))
float f16_to_f32(uint16_t h);

struct ChunkData {
    std::string task_id, policy_sha;
    int64_t seq = 0, step0 = 0;
    int n_obs = 0, n_act = 0;
    Bytes chain_prev;               // 32 bytes
    bool final = false;
    std::string end;                // "" unless final
    // stats (optional)
    bool has_stats = false;
    int64_t stat_frames = 0, stat_episodes = 0;
    double stat_wall_s = 0;
    // arrays (T = actions.size())
    std::vector<uint8_t> actions, flags, sector;
    std::vector<float> rewards, logp, values, env_rewards, arc;  // values: T + 1
    std::vector<uint16_t> frames;
    std::vector<double> sr_ms;
    std::vector<float> obs;          // (T + 1) x n_obs float32 (converted to float16)
    std::vector<Bytes> obs_hash;     // T + 1 chain values
};

Bytes encode_chunk(const ChunkData &c, int zstd_level = 3);

}  // namespace ballrain
