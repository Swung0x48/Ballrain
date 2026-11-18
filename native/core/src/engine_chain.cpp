#include "ballrain/engine_chain.h"

#include <cstring>

namespace ballrain {

static const char ROW_TAG[] = "ballrain-pool/engine-row/v2";      // + NUL
static const char SEED_TAG[] = "ballrain-pool/engine-chain/v2";  // + NUL

void EngineDigest::reset() {
    h_.reset();
    h_.update(ROW_TAG, sizeof ROW_TAG);  // includes the terminating NUL, as the Python tag
    records = 0;
}

// canonical bytes of one message at p (OBS1 / LVL1 layout), sections whose tag is in skip left out; returns byte_len
static size_t canonical(const char *p, size_t avail, const char *skip1, const char *skip2, Bytes &out) {
    Reader r(p, avail);
    if (avail < 24) throw ProtocolError("message shorter than its header");
    uint32_t magic = r.u32();
    uint16_t version = r.u16(), flags = r.u16();
    uint32_t byte_len = r.u32(), frames_done = r.u32(), count = r.u32();
    r.u32();
    if (byte_len < 24 || byte_len > avail) throw ProtocolError("byte_len outside buffer");
    struct Sec { uint32_t tag; uint16_t v, f; uint32_t n; const char *p; };
    std::vector<Sec> kept;
    size_t pos = 24;
    for (uint32_t i = 0; i < count; ++i) {
        if (pos + 12 > byte_len) throw ProtocolError("section header beyond message end");
        Sec s;
        std::memcpy(&s.tag, p + pos, 4);
        std::memcpy(&s.v, p + pos + 4, 2);
        std::memcpy(&s.f, p + pos + 6, 2);
        std::memcpy(&s.n, p + pos + 8, 4);
        pos += 12;
        if (pos + s.n > byte_len) throw ProtocolError("section payload beyond message end");
        s.p = p + pos;
        pos += align4(s.n);
        bool skip = (skip1 && std::memcmp(&s.tag, skip1, 4) == 0) || (skip2 && std::memcmp(&s.tag, skip2, 4) == 0);
        if (!skip) kept.push_back(s);
    }
    put<uint32_t>(out, magic);
    put<uint16_t>(out, version);
    put<uint16_t>(out, flags);
    put<uint32_t>(out, frames_done);
    put<uint32_t>(out, uint32_t(kept.size()));
    for (auto &s : kept) {
        put<uint32_t>(out, s.tag);
        put<uint16_t>(out, s.v);
        put<uint16_t>(out, s.f);
        put<uint32_t>(out, s.n);
        out.append(s.p, s.n);
    }
    return byte_len;
}

void EngineDigest::obs(const char *msg, size_t n) {
    Bytes c("O", 1);
    canonical(msg, n, "SCOR", "SREN", c);
    h_.update(c);
    ++records;
}

void EngineDigest::level(char kind, const char *data, size_t n) {
    Bytes parts;
    uint32_t count = 0;
    size_t pos = 0;
    while (n - pos >= 24) {
        uint32_t bl;
        std::memcpy(&bl, data + pos + 8, 4);
        if (pos + bl > n) break;
        pos += canonical(data + pos, n - pos, "IVPO", nullptr, parts);
        ++count;
        if (kind == 'F') break;
    }
    Bytes rec(1, kind);
    put<uint32_t>(rec, count);
    h_.update(rec);
    h_.update(parts);
    ++records;
}

Bytes EngineDigest::take() {
    Bytes d = h_.digest();
    reset();
    return d;
}

Bytes chain_seed(const std::string &task_id) {
    Sha256 h;
    h.update(SEED_TAG, sizeof SEED_TAG);
    h.update(task_id);
    return h.digest();
}

Bytes chain_next(const Bytes &prev, const Bytes &row_digest) {
    Sha256 h;
    h.update(prev);
    h.update(row_digest);
    return h.digest();
}

}  // namespace ballrain
