#include "ballrain/npz.h"

#include <cstdlib>
#include <cstring>

#include "miniz.h"

namespace ballrain {

int64_t NpyArray::count() const {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
}

size_t NpyArray::itemsize() const {
    if (descr.size() >= 3 && (descr[0] == '<' || descr[0] == '|' || descr[0] == '>' || descr[0] == '=')) {
        size_t n = size_t(std::atoi(descr.c_str() + 2));
        return descr[1] == 'U' ? 4 * n : n;
    }
    int64_t c = count();
    return c ? data.size() / size_t(c) : 0;
}

std::vector<float> NpyArray::f32() const {
    if (descr != "<f4") throw Error("npy: expected <f4, got " + descr);
    std::vector<float> v(static_cast<size_t>(count()));
    if (!v.empty()) std::memcpy(v.data(), data.data(), v.size() * 4);
    return v;
}

std::vector<double> NpyArray::f64() const {
    std::vector<double> v(static_cast<size_t>(count()));
    if (descr == "<f8") {
        if (!v.empty()) std::memcpy(v.data(), data.data(), v.size() * 8);
    } else if (descr == "<f4") {
        for (size_t i = 0; i < v.size(); ++i) {
            float f;
            std::memcpy(&f, data.data() + 4 * i, 4);
            v[i] = f;
        }
    } else {
        throw Error("npy: expected a float dtype, got " + descr);
    }
    return v;
}

std::vector<int64_t> NpyArray::ints() const {
    size_t n = size_t(count());
    std::vector<int64_t> v(n);
    const char *p = data.data();
    auto rd = [&](size_t i, size_t w, bool sgn) -> int64_t {
        uint64_t u = 0;
        std::memcpy(&u, p + i * w, w);
        if (sgn && w < 8 && (u >> (8 * w - 1)) & 1) u |= ~uint64_t(0) << (8 * w);
        return int64_t(u);
    };
    size_t w;
    bool sgn;
    if (descr == "|u1") w = 1, sgn = false;
    else if (descr == "|i1") w = 1, sgn = true;
    else if (descr == "<u2") w = 2, sgn = false;
    else if (descr == "<i2") w = 2, sgn = true;
    else if (descr == "<u4") w = 4, sgn = false;
    else if (descr == "<i4") w = 4, sgn = true;
    else if (descr == "<i8") w = 8, sgn = true;
    else if (descr == "<u8") w = 8, sgn = false;
    else if (descr == "|b1") w = 1, sgn = false;
    else throw Error("npy: expected an integer dtype, got " + descr);
    if (data.size() < n * w) throw Error("npy: short data");
    for (size_t i = 0; i < n; ++i) v[i] = rd(i, w, sgn);
    return v;
}

std::string NpyArray::unicode() const {
    if (descr.size() < 3 || descr[1] != 'U' || descr[0] != '<') throw Error("npy: expected <U, got " + descr);
    std::string out;
    for (size_t i = 0; i + 4 <= data.size(); i += 4) {
        uint32_t cp;
        std::memcpy(&cp, data.data() + i, 4);
        if (cp == 0) continue;
        if (cp < 0x80) out += char(cp);
        else if (cp < 0x800) out += char(0xC0 | (cp >> 6)), out += char(0x80 | (cp & 0x3F));
        else if (cp < 0x10000)
            out += char(0xE0 | (cp >> 12)), out += char(0x80 | ((cp >> 6) & 0x3F)), out += char(0x80 | (cp & 0x3F));
        else
            out += char(0xF0 | (cp >> 18)), out += char(0x80 | ((cp >> 12) & 0x3F)),
                out += char(0x80 | ((cp >> 6) & 0x3F)), out += char(0x80 | (cp & 0x3F));
    }
    return out;
}

// The header is a Python dict literal: {'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }
static std::string dict_value(const std::string &h, const std::string &key) {
    size_t k = h.find("'" + key + "'");
    if (k == std::string::npos) throw Error("npy: header without " + key);
    size_t c = h.find(':', k);
    if (c == std::string::npos) throw Error("npy: bad header");
    size_t i = c + 1;
    while (i < h.size() && h[i] == ' ') ++i;
    size_t j = i;
    int depth = 0;
    bool in_str = false;
    char q = 0;
    for (; j < h.size(); ++j) {
        char ch = h[j];
        if (in_str) {
            if (ch == q) in_str = false;
            continue;
        }
        if (ch == '\'' || ch == '"') in_str = true, q = ch;
        else if (ch == '(' || ch == '[' || ch == '{') ++depth;
        else if (ch == ')' || ch == ']' || ch == '}') {
            if (depth == 0) break;
            --depth;
        } else if (ch == ',' && depth == 0) break;
    }
    std::string v = h.substr(i, j - i);
    while (!v.empty() && v.back() == ' ') v.pop_back();
    return v;
}

NpyArray parse_npy(const Bytes &f) {
    if (f.size() < 10 || std::memcmp(f.data(), "\x93NUMPY", 6) != 0) throw Error("npy: bad magic");
    uint8_t major = uint8_t(f[6]);
    size_t hl, off;
    if (major == 1) {
        uint16_t n;
        std::memcpy(&n, f.data() + 8, 2);
        hl = n, off = 10;
    } else {
        if (f.size() < 12) throw Error("npy: short header");
        uint32_t n;
        std::memcpy(&n, f.data() + 8, 4);
        hl = n, off = 12;
    }
    if (off + hl > f.size()) throw Error("npy: header beyond file");
    std::string h = f.substr(off, hl);
    NpyArray a;
    std::string d = dict_value(h, "descr");
    if (!d.empty() && (d[0] == '\'' || d[0] == '"')) d = d.substr(1, d.size() - 2);
    a.descr = d;
    a.fortran = dict_value(h, "fortran_order") == "True";
    std::string s = dict_value(h, "shape");
    for (size_t i = 0; i < s.size();) {
        while (i < s.size() && !(s[i] >= '0' && s[i] <= '9')) ++i;
        if (i >= s.size()) break;
        size_t j = i;
        while (j < s.size() && s[j] >= '0' && s[j] <= '9') ++j;
        a.shape.push_back(std::atoll(s.substr(i, j - i).c_str()));
        i = j;
    }
    if (a.fortran && a.shape.size() > 1) throw Error("npy: fortran order not supported");
    a.data = f.substr(off + hl);
    size_t want = size_t(a.count()) * a.itemsize();
    if (a.descr.size() >= 3 && a.descr[0] != '[' && a.data.size() < want) throw Error("npy: data shorter than shape");
    if (a.descr.size() >= 3 && a.descr[0] != '[') a.data.resize(want);
    return a;
}

std::map<std::string, NpyArray> load_npz_bytes(const Bytes &data) {
    mz_zip_archive z;
    std::memset(&z, 0, sizeof z);
    if (!mz_zip_reader_init_mem(&z, data.data(), data.size(), 0)) throw Error("npz: not a zip file");
    std::map<std::string, NpyArray> out;
    try {
        mz_uint n = mz_zip_reader_get_num_files(&z);
        for (mz_uint i = 0; i < n; ++i) {
            mz_zip_archive_file_stat st;
            if (!mz_zip_reader_file_stat(&z, i, &st)) throw Error("npz: bad entry");
            if (st.m_is_directory) continue;
            std::string name = st.m_filename;
            size_t sz = 0;
            void *p = mz_zip_reader_extract_to_heap(&z, i, &sz, 0);
            if (!p) throw Error("npz: cannot extract " + name);
            Bytes b(static_cast<const char *>(p), sz);
            mz_free(p);
            if (name.size() > 4 && name.compare(name.size() - 4, 4, ".npy") == 0) name.resize(name.size() - 4);
            out[name] = parse_npy(b);
        }
    } catch (...) {
        mz_zip_reader_end(&z);
        throw;
    }
    mz_zip_reader_end(&z);
    return out;
}

std::map<std::string, NpyArray> load_npz(const std::string &path) {
    Bytes b;
    if (!read_file(path, b)) throw Error("npz: cannot read " + path);
    return load_npz_bytes(b);
}

}  // namespace ballrain
