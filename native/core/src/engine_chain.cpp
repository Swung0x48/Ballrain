#include "ballrain/engine_chain.h"

#include <cstring>
#include <vector>

#include "ballrain/wire.h"

namespace ballrain {

// chain v3 (ballrain/pool/engine_chain.py is the reference: the records and what is left out are documented there)
static const char ROW_TAG[] = "ballrain-pool/engine-row/v3";      // + NUL
static const char SEED_TAG[] = "ballrain-pool/engine-chain/v3";  // + NUL

void EngineDigest::reset() {
    h_.reset();
    h_.update(ROW_TAG, sizeof ROW_TAG);  // includes the terminating NUL, as the Python tag
    records = 0;
}

// OBS1 ARRS: u16 n; u32 array, u16 row, u16 col, u8 type, u8 length, data -- string cells written with length 0
static Bytes obs_arrs(const char *p, size_t n) {
    Reader r(p, n);
    uint16_t cnt = r.u16();
    Bytes out(p, 2);
    for (uint16_t i = 0; i < cnt; ++i) {
        size_t at = r.pos();
        r.skip(8);
        uint8_t type = r.u8(), ln = r.u8();
        if (ln > r.left()) throw ProtocolError("ARRS cell beyond payload");
        if (type == ARR_STRING) {
            out.append(p + at, 9);
            out.push_back('\0');
        } else {
            out.append(p + at, 10 + size_t(ln));
        }
        r.skip(ln);
    }
    return out;
}

// LVL1 ARRS: u32 n; per array u32 ckid, u32 name, u16 rows, u16 cols, cols x (u32 name, u8 type, u32 param type),
// rows x cols cells (u8 length, data) -- string cells written with length 0
static Bytes lvl_arrs(const char *p, size_t n) {
    Reader r(p, n);
    uint32_t cnt = r.u32();
    Bytes out(p, 4);
    for (uint32_t a = 0; a < cnt; ++a) {
        size_t start = r.pos();
        r.skip(8);
        uint16_t rows = r.u16(), cols = r.u16();
        std::vector<uint8_t> types(cols);
        for (uint16_t c = 0; c < cols; ++c) {
            r.skip(4);
            types[c] = r.u8();
            r.skip(4);
        }
        out.append(p + start, r.pos() - start);
        for (uint32_t row = 0; row < rows; ++row)
            for (uint16_t c = 0; c < cols; ++c) {
                size_t at = r.pos();
                uint8_t ln = r.u8();
                r.skip(ln);
                if (types[c] == ARR_STRING) out.push_back('\0');
                else out.append(p + at, 1 + size_t(ln));
            }
    }
    return out;
}

// canonical bytes of one message at p (OBS1 / LVL1 layout), sections whose tag is in skip left out; returns byte_len
static size_t canonical(const char *p, size_t avail, bool obs, const char *const *skip, Bytes &out) {
    Reader r(p, avail);
    if (avail < 24) throw ProtocolError("message shorter than its header");
    uint32_t magic = r.u32();
    uint16_t version = r.u16(), flags = r.u16();
    uint32_t byte_len = r.u32(), frames_done = r.u32(), count = r.u32();
    r.u32();
    if (byte_len < 24 || byte_len > avail) throw ProtocolError("byte_len outside buffer");
    struct Sec { uint32_t tag; uint16_t v, f; Bytes payload; };
    std::vector<Sec> kept;
    size_t pos = 24;
    for (uint32_t i = 0; i < count; ++i) {
        if (pos + 12 > byte_len) throw ProtocolError("section header beyond message end");
        Sec s;
        uint32_t len;
        std::memcpy(&s.tag, p + pos, 4);
        std::memcpy(&s.v, p + pos + 4, 2);
        std::memcpy(&s.f, p + pos + 6, 2);
        std::memcpy(&len, p + pos + 8, 4);
        pos += 12;
        if (pos + len > byte_len) throw ProtocolError("section payload beyond message end");
        const char *pl = p + pos;
        pos += align4(len);
        bool drop = false;
        for (const char *const *k = skip; *k; ++k) drop = drop || std::memcmp(&s.tag, *k, 4) == 0;
        if (drop) continue;
        if (std::memcmp(&s.tag, "ARRS", 4) == 0) s.payload = obs ? obs_arrs(pl, len) : lvl_arrs(pl, len);
        else s.payload.assign(pl, len);
        kept.push_back(std::move(s));
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
        put<uint32_t>(out, uint32_t(s.payload.size()));
        out += s.payload;
    }
    return byte_len;
}

static const char *const SKIP_OBS[] = {"SCOR", "SREN", "WTCH", nullptr};
static const char *const SKIP_LVL[] = {"IVPO", "GRPS", nullptr};

void EngineDigest::obs(const char *msg, size_t n) {
    Bytes c("O", 1);
    canonical(msg, n, true, SKIP_OBS, c);
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
        pos += canonical(data + pos, n - pos, false, SKIP_LVL, parts);
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
