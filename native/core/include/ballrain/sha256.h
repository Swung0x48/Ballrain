// SHA-256 (FIPS 180-4), incremental.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ballrain {

class Sha256 {
public:
    Sha256() { reset(); }
    void reset();
    void update(const void *data, size_t n);
    void update(const std::string &s) { update(s.data(), s.size()); }
    void final(uint8_t out[32]);
    std::string digest();     // 32 raw bytes
    std::string hexdigest();  // 64 hex chars

    static std::string hash(const void *data, size_t n);  // raw
    static std::string hex(const void *data, size_t n);
    static std::string hex(const std::string &s) { return hex(s.data(), s.size()); }

private:
    void block(const uint8_t *p);
    uint32_t h_[8];
    uint8_t buf_[64];
    size_t nbuf_ = 0;
    uint64_t total_ = 0;
};

std::string file_sha256(const std::string &path);  // hex; Error when unreadable

}  // namespace ballrain
