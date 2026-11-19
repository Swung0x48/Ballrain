#include "ballrain/chunk.h"

#include <cmath>
#include <cstring>

#include "ballrain/sha256.h"
#include "zstd.h"

namespace ballrain {

uint16_t f32_to_f16(float v) {
    uint32_t x;
    std::memcpy(&x, &v, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t e = (x >> 23) & 0xFFu;
    uint32_t m = x & 0x7FFFFFu;
    if (e == 0xFF) return uint16_t(sign | 0x7C00u | (m ? 0x200u | (m >> 13) : 0u));  // inf / NaN
    int32_t exp = int32_t(e) - 127 + 15;
    if (exp >= 0x1F) return uint16_t(sign | 0x7C00u);  // overflow -> inf
    if (exp <= 0) {                                   // subnormal half (or zero)
        if (exp < -10) return uint16_t(sign);
        m |= 0x800000u;
        uint32_t shift = uint32_t(14 - exp);
        uint32_t half = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1u), mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1u))) ++half;
        return uint16_t(sign | half);
    }
    uint32_t half = (uint32_t(exp) << 10) | (m >> 13);
    uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;  // may carry into the exponent (-> inf): correct
    return uint16_t(sign | half);
}

float f16_to_f32(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    uint32_t x;
    if (e == 0) {
        if (!m) x = sign;
        else {
            int k = -1;
            do ++k, m <<= 1;
            while (!(m & 0x400u));
            x = sign | (uint32_t(127 - 15 - k) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (e == 0x1F) {
        x = sign | 0x7F800000u | (m << 13);
    } else {
        x = sign | ((e + 127 - 15) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

namespace {

struct Section {
    const char *name;
    Bytes raw;
    bool zstd;
};

template <typename T>
Bytes raw_of(const std::vector<T> &v) {
    return Bytes(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(T));
}

}  // namespace

Bytes encode_chunk(const ChunkData &c, int zstd_level) {
    size_t T = c.actions.size();
    if (T == 0) throw Error("empty chunk");
    auto need = [&](size_t got, size_t want, const char *n) {
        if (got != want) throw Error(std::string("chunk array ") + n + ": wrong length");
    };
    need(c.rewards.size(), T, "rewards"), need(c.logp.size(), T, "logp"), need(c.flags.size(), T, "flags");
    need(c.values.size(), T + 1, "values"), need(c.obs.size(), (T + 1) * size_t(c.n_obs), "obs");
    need(c.obs_hash.size(), T + 1, "obs_hash");
    std::vector<Section> secs;
    secs.push_back({"actions", raw_of(c.actions), false});
    secs.push_back({"rewards", raw_of(c.rewards), false});
    std::vector<float> lp(c.logp);
    for (auto &x : lp) x = float(std::min(std::max(double(x), MIN_LOGP), 0.0));
    secs.push_back({"logp", raw_of(lp), false});
    secs.push_back({"values", raw_of(c.values), false});
    secs.push_back({"flags", raw_of(c.flags), false});
    std::vector<uint16_t> o16(c.obs.size());
    for (size_t i = 0; i < c.obs.size(); ++i) o16[i] = f32_to_f16(c.obs[i]);
    secs.push_back({"obs", raw_of(o16), true});
    Bytes hashes;
    for (auto &h : c.obs_hash) {
        if (h.size() != 32) throw Error("chain value is not 32 bytes");
        hashes += h;
    }
    secs.push_back({"obs_hash", hashes, false});
    if (!c.env_rewards.empty()) need(c.env_rewards.size(), T, "env_rewards"), secs.push_back({"env_rewards", raw_of(c.env_rewards), false});
    if (!c.sector.empty()) need(c.sector.size(), T, "sector"), secs.push_back({"sector", raw_of(c.sector), false});
    if (!c.frames.empty()) need(c.frames.size(), T, "frames"), secs.push_back({"frames", raw_of(c.frames), false});
    if (!c.sr_ms.empty()) need(c.sr_ms.size(), T, "sr_ms"), secs.push_back({"sr_ms", raw_of(c.sr_ms), false});
    if (!c.arc.empty()) need(c.arc.size(), T, "arc"), secs.push_back({"arc", raw_of(c.arc), false});
    if (!c.rin.empty()) need(c.rin.size(), T * size_t(c.rin_cols), "rin"), secs.push_back({"rin", raw_of(c.rin), true});

    Bytes body;
    Json arrays = Json::array();
    for (auto &s : secs) {
        Bytes stored = s.raw;
        if (s.zstd) {
            stored.resize(ZSTD_compressBound(s.raw.size()));
            size_t n = ZSTD_compress(&stored[0], stored.size(), s.raw.data(), s.raw.size(), zstd_level);
            if (ZSTD_isError(n)) throw Error(std::string("zstd: ") + ZSTD_getErrorName(n));
            stored.resize(n);
        }
        Json e = Json::object();
        e.set("name", Json::string(s.name));
        e.set("codec", Json::string(s.zstd ? "zstd" : "raw"));
        e.set("nbytes", Json::integer(int64_t(stored.size())));
        e.set("sha256", Json::string(Sha256::hex(s.raw)));
        arrays.push(e);
        body += stored;
    }
    Json h = Json::object();
    h.set("protocol", Json::integer(PROTOCOL_VERSION));
    h.set("type", Json::string("chunk"));
    h.set("task_id", Json::string(c.task_id));
    h.set("seq", Json::integer(c.seq));
    h.set("step0", Json::integer(c.step0));
    h.set("steps", Json::integer(int64_t(T)));
    h.set("n_obs", Json::integer(c.n_obs));
    h.set("n_act", Json::integer(c.n_act));
    h.set("policy_sha", Json::string(c.policy_sha));
    h.set("chain_prev", Json::string(hex(c.chain_prev)));
    h.set("final", Json::boolean(c.final));
    h.set("end", c.final ? Json::string(c.end) : Json());
    if (c.has_stats) {
        Json st = Json::object();
        st.set("frames", Json::integer(c.stat_frames));
        st.set("wall_s", Json::number(std::round(c.stat_wall_s * 1000.0) / 1000.0));
        st.set("episodes", Json::integer(c.stat_episodes));
        h.set("stats", st);
    }
    if (c.episodes.is_array() && c.episodes.size()) h.set("episodes", c.episodes);
    h.set("arrays", arrays);
    std::string hj = h.dump(true);
    Bytes out("BRPC", 4);
    put<uint16_t>(out, CHUNK_FORMAT);
    put<uint16_t>(out, 0);
    put<uint32_t>(out, uint32_t(hj.size()));
    out += hj;
    out += body;
    out += Sha256::hash(out.data(), out.size());
    return out;
}

}  // namespace ballrain

namespace ballrain {

Bytes zstd_decompress_capped(const Bytes &in, size_t max_out) {
    ZSTD_DCtx *d = ZSTD_createDCtx();
    if (!d) throw Error("zstd: out of memory");
    Bytes out;
    std::string buf(1 << 20, '\0');
    ZSTD_inBuffer ib{in.data(), in.size(), 0};
    size_t ret = 1;
    try {
        while (ib.pos < ib.size || ret != 0) {
            ZSTD_outBuffer ob{&buf[0], buf.size(), 0};
            size_t before = ib.pos;
            ret = ZSTD_decompressStream(d, &ob, &ib);
            if (ZSTD_isError(ret)) throw Error(std::string("zstd: ") + ZSTD_getErrorName(ret));
            if (out.size() + ob.pos > max_out) throw Error("zstd: the data is larger than announced");
            out.append(buf.data(), ob.pos);
            if (ib.pos == ib.size && ob.pos == 0 && ib.pos == before) {
                if (ret != 0) throw Error("zstd: truncated data");
                break;
            }
        }
    } catch (...) {
        ZSTD_freeDCtx(d);
        throw;
    }
    ZSTD_freeDCtx(d);
    return out;
}

Bytes zstd_compress(const Bytes &in, int level) {
    Bytes out(ZSTD_compressBound(in.size()), '\0');
    size_t n = ZSTD_compress(&out[0], out.size(), in.data(), in.size(), level);
    if (ZSTD_isError(n)) throw Error(std::string("zstd: ") + ZSTD_getErrorName(n));
    out.resize(n);
    return out;
}

}  // namespace ballrain
