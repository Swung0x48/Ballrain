#include "ballrain/util.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

namespace ballrain {

// paths are UTF-8 everywhere in the client; on Windows a narrow std::fstream path would be read in the ANSI code page
// (a Chinese user name in %LOCALAPPDATA%, a Ballance folder on the 桌面)
std::filesystem::path u8_path(const std::string &p) { return std::filesystem::u8path(p); }

uint32_t fourcc(const char s[4]) {
    uint32_t v;
    std::memcpy(&v, s, 4);
    return v;
}

std::string tag_str(uint32_t tag) {
    char s[5] = {};
    std::memcpy(s, &tag, 4);
    return std::string(s, 4);
}

bool read_file(const std::string &path, Bytes &out) {
    std::ifstream f(u8_path(path), std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    std::streamoff n = f.tellg();
    if (n < 0) return false;
    f.seekg(0, std::ios::beg);
    out.resize(size_t(n));
    if (n && !f.read(&out[0], n)) return false;
    return true;
}

bool read_file_from(const std::string &path, uint64_t offset, Bytes &out, uint64_t *file_size) {
    std::ifstream f(u8_path(path), std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    std::streamoff n = f.tellg();
    if (n < 0) return false;
    if (file_size) *file_size = uint64_t(n);
    out.clear();
    if (uint64_t(n) <= offset) return true;
    f.seekg(std::streamoff(offset), std::ios::beg);
    out.resize(size_t(uint64_t(n) - offset));
    if (!f.read(&out[0], std::streamsize(out.size()))) return false;
    return true;
}

bool file_exists(const std::string &path) {
    std::ifstream f(u8_path(path), std::ios::binary);
    return bool(f);
}

void write_file(const std::string &path, const Bytes &data) {
    std::ofstream f(u8_path(path), std::ios::binary | std::ios::trunc);
    if (!f) throw Error("cannot write " + path);
    f.write(data.data(), std::streamsize(data.size()));
    if (!f) throw Error("write failed: " + path);
}

std::string hex(const void *p, size_t n) {
    static const char *d = "0123456789abcdef";
    std::string s(n * 2, '0');
    auto b = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < n; ++i) {
        s[2 * i] = d[b[i] >> 4];
        s[2 * i + 1] = d[b[i] & 15];
    }
    return s;
}

}  // namespace ballrain
