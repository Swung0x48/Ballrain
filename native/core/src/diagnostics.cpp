#include "ballrain/diagnostics.h"

#include <algorithm>
#include <cctype>
#include <cstring>

#include "ballrain/util.h"
#include "miniz.h"

namespace ballrain {

static char fold(char c) {
    if (c == '\\') return '/';
    return char(std::tolower(static_cast<unsigned char>(c)));
}

static std::string replace_all(const std::string &s, const std::string &what, const std::string &with) {
    if (what.empty()) return s;
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        bool hit = i + what.size() <= s.size();
        for (size_t k = 0; hit && k < what.size(); ++k) hit = fold(s[i + k]) == fold(what[k]);
        if (hit) {
            out += with;
            i += what.size();
        } else {
            out += s[i++];
        }
    }
    return out;
}

static std::string mask_ipv4(const std::string &s) {
    std::string out;
    size_t i = 0;
    auto digits = [&](size_t p, size_t &n) {
        n = 0;
        while (p + n < s.size() && std::isdigit(static_cast<unsigned char>(s[p + n])) && n < 4) ++n;
        return n >= 1 && n <= 3;
    };
    while (i < s.size()) {
        bool start = std::isdigit(static_cast<unsigned char>(s[i])) &&
                     (i == 0 || !(std::isalnum(static_cast<unsigned char>(s[i - 1])) || s[i - 1] == '.'));
        if (start) {
            size_t p = i, n;
            int parts = 0;
            while (parts < 4 && digits(p, n)) {
                p += n, ++parts;
                if (parts < 4) {
                    if (p < s.size() && s[p] == '.') ++p;
                    else break;
                }
            }
            bool end_ok = p >= s.size() || !(std::isalnum(static_cast<unsigned char>(s[p])) || s[p] == '.');
            if (parts == 4 && end_ok) {
                out += "<ip>";
                i = p;
                continue;
            }
        }
        out += s[i++];
    }
    return out;
}

std::string scrub_text(const std::string &text, std::vector<std::pair<std::string, std::string>> replace) {
    std::sort(replace.begin(), replace.end(), [](const auto &a, const auto &b) { return a.first.size() > b.first.size(); });
    std::string s = text;
    for (auto &r : replace)
        if (r.first.size() >= 2) s = replace_all(s, r.first, r.second);
    return mask_ipv4(s);
}

void write_zip(const DiagFiles &files, const std::string &path) {
    mz_zip_archive zip;
    std::memset(&zip, 0, sizeof zip);
    if (!mz_zip_writer_init_heap(&zip, 0, 0)) throw Error("zip: init");
    for (auto &f : files)
        if (!mz_zip_writer_add_mem(&zip, f.first.c_str(), f.second.data(), f.second.size(), MZ_DEFAULT_COMPRESSION)) {
            mz_zip_writer_end(&zip);
            throw Error("zip: cannot add " + f.first);
        }
    void *buf = nullptr;
    size_t n = 0;
    if (!mz_zip_writer_finalize_heap_archive(&zip, &buf, &n)) {
        mz_zip_writer_end(&zip);
        throw Error("zip: finalize");
    }
    Bytes data(static_cast<const char *>(buf), n);
    zip.m_pFree(zip.m_pAlloc_opaque, buf);
    mz_zip_writer_end(&zip);
    write_file(path, data);
}

}  // namespace ballrain
