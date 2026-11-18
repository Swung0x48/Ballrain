// Small portable helpers of the Ballrain native core: little-endian reading, errors, files.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace ballrain {

using Bytes = std::string;  // raw byte buffers (std::string: cheap moves, comparisons, substrings)

struct Error : std::runtime_error {
    explicit Error(const std::string &m) : std::runtime_error(m) {}
};
struct ProtocolError : Error {
    explicit ProtocolError(const std::string &m) : Error(m) {}
};
struct RouteError : Error {  // ballrain.path.RouteError
    explicit RouteError(const std::string &m) : Error(m) {}
};

// Little-endian cursor over a byte range (the wire formats are packed LE: docs/agent-protocol.md, protocol.py A1).
class Reader {
public:
    Reader(const char *data, size_t n) : p_(data), n_(n) {}
    explicit Reader(const Bytes &b) : p_(b.data()), n_(b.size()) {}
    size_t pos() const { return i_; }
    size_t size() const { return n_; }
    size_t left() const { return n_ - i_; }
    void seek(size_t at) {
        if (at > n_) throw ProtocolError("seek beyond buffer");
        i_ = at;
    }
    void skip(size_t k) { need(k), i_ += k; }
    const char *ptr() const { return p_ + i_; }
    template <typename T>
    T get() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p_ + i_, sizeof(T));
        i_ += sizeof(T);
        return v;
    }
    uint8_t u8() { return get<uint8_t>(); }
    uint16_t u16() { return get<uint16_t>(); }
    uint32_t u32() { return get<uint32_t>(); }
    uint64_t u64() { return get<uint64_t>(); }
    int8_t i8() { return get<int8_t>(); }
    int32_t i32() { return get<int32_t>(); }
    float f32() { return get<float>(); }
    double f64() { return get<double>(); }
    template <typename T>
    void arr(T *out, size_t k) {
        need(sizeof(T) * k);
        std::memcpy(out, p_ + i_, sizeof(T) * k);
        i_ += sizeof(T) * k;
    }
    Bytes raw(size_t k) {
        need(k);
        Bytes b(p_ + i_, k);
        i_ += k;
        return b;
    }

private:
    void need(size_t k) const {
        if (k > n_ - i_) throw ProtocolError("short read");
    }
    const char *p_;
    size_t n_;
    size_t i_ = 0;
};

inline void put_bytes(Bytes &out, const void *p, size_t n) { out.append(static_cast<const char *>(p), n); }
template <typename T>
inline void put(Bytes &out, T v) {
    put_bytes(out, &v, sizeof(T));
}

uint32_t fourcc(const char s[4]);
std::string tag_str(uint32_t tag);
inline size_t align4(size_t n) { return (n + 3) & ~size_t(3); }

// Whole-file read (nullopt-like: returns false when the file does not exist / cannot be opened).
bool read_file(const std::string &path, Bytes &out);
bool read_file_from(const std::string &path, uint64_t offset, Bytes &out, uint64_t *file_size);
bool file_exists(const std::string &path);
std::filesystem::path u8_path(const std::string &utf8);  // a UTF-8 path for the standard library's file streams
void write_file(const std::string &path, const Bytes &data);

std::string hex(const void *p, size_t n);
inline std::string hex(const Bytes &b) { return hex(b.data(), b.size()); }

}  // namespace ballrain
