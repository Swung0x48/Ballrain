#include "ballrain/stream.h"

#include <cstring>

namespace ballrain {

StreamWriter::StreamWriter(const std::string &path) {
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_) throw Error("cannot write " + path);
    uint32_t v = 1;
    std::fwrite("BRGS", 1, 4, f_);
    std::fwrite(&v, 4, 1, f_);
}

StreamWriter::~StreamWriter() {
    if (f_) std::fclose(f_);
}

void StreamWriter::put(uint32_t kind, const void *p, size_t n) {
    if (sink_) {
        sink_(kind, p, n);
        if (tee_) tee_->put(kind, p, n);
        return;
    }
    uint32_t h[2] = {kind, uint32_t(n)};
    std::fwrite(h, 4, 2, f_);
    if (n) std::fwrite(p, 1, n, f_);
}

void StreamWriter::put(uint32_t kind, const Bytes &b) { put(kind, b.data(), b.size()); }

void StreamWriter::flush() {
    if (f_) std::fflush(f_);
    if (tee_) tee_->flush();
}

std::vector<Record> read_stream(const std::string &path) {
    Bytes b;
    if (!read_file(path, b)) throw Error("cannot read " + path);
    if (b.size() < 8 || std::memcmp(b.data(), "BRGS", 4) != 0) throw Error(path + ": not a stream");
    std::vector<Record> out;
    size_t p = 8;
    while (p + 8 <= b.size()) {
        uint32_t k, n;
        std::memcpy(&k, b.data() + p, 4);
        std::memcpy(&n, b.data() + p + 4, 4);
        if (p + 8 + n > b.size()) throw Error(path + ": truncated record");
        out.push_back({k, b.substr(p + 8, n)});
        p += 8 + n;
    }
    return out;
}

}  // namespace ballrain
