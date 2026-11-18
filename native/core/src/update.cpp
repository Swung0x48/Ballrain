#include "ballrain/update.h"

#include <cctype>
#include <cstdlib>
#include <vector>

#include "ballrain/json.h"
#include "ballrain/util.h"
#include "ed25519.h"
#include "platform.h"

namespace ballrain {

static std::vector<long> version_parts(const std::string &v, std::string *suffix) {
    std::string core = v.substr(0, v.find('-'));
    if (suffix) *suffix = v.size() > core.size() ? v.substr(core.size() + 1) : "";
    std::vector<long> out;
    size_t i = 0;
    while (i <= core.size()) {
        size_t j = core.find('.', i);
        if (j == std::string::npos) j = core.size();
        out.push_back(std::strtol(core.substr(i, j - i).c_str(), nullptr, 10));
        i = j + 1;
    }
    return out;
}

int compare_versions(const std::string &a, const std::string &b) {
    std::string sa, sb;
    auto pa = version_parts(a, &sa), pb = version_parts(b, &sb);
    for (size_t i = 0; i < std::max(pa.size(), pb.size()); ++i) {
        long x = i < pa.size() ? pa[i] : 0, y = i < pb.size() ? pb[i] : 0;
        if (x != y) return x < y ? -1 : 1;
    }
    if (sa == sb) return 0;
    if (sa.empty()) return 1;  // 0.4.0 > 0.4.0-beta1
    if (sb.empty()) return -1;
    return sa < sb ? -1 : 1;
}

static bool unhex(const std::string &h, std::vector<unsigned char> &out, size_t n) {
    std::string s;
    for (char c : h)
        if (!std::isspace(static_cast<unsigned char>(c))) s += c;
    if (s.size() != 2 * n) return false;
    out.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        int v = 0;
        for (int k = 0; k < 2; ++k) {
            char c = char(std::tolower(static_cast<unsigned char>(s[2 * i + size_t(k)])));
            int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
            if (d < 0) return false;
            v = v * 16 + d;
        }
        out[i] = static_cast<unsigned char>(v);
    }
    return true;
}

bool verify_release_signature(const std::string &manifest, const std::string &sig_hex, const std::string &pubkey_hex) {
    std::vector<unsigned char> sig, pub;
    if (!unhex(sig_hex, sig, 64) || !unhex(pubkey_hex, pub, 32)) return false;
    return ed25519_verify(sig.data(), reinterpret_cast<const unsigned char *>(manifest.data()), manifest.size(),
                          pub.data()) == 1;
}

std::string release_platform_key() {
#if defined(_WIN32)
    return sizeof(void *) == 8 ? "windows-x64" : "windows-x86";
#elif defined(__APPLE__)
#if defined(__aarch64__)
    return "macos-arm64";
#else
    return "macos-x64";
#endif
#elif defined(__ANDROID__)
    return "android-arm64";
#else
#if defined(__aarch64__)
    return "linux-arm64";
#else
    return "linux-x64";
#endif
#endif
}

std::optional<ReleaseInfo> check_for_update(const std::string &url, const std::string &pubkey_hex,
                                            const std::string &current, const std::string &key, std::string *error) {
    auto fail = [&](const std::string &e) -> std::optional<ReleaseInfo> {
        if (error) *error = e;
        return std::nullopt;
    };
    if (url.empty() || pubkey_hex.empty()) return fail("这个版本没有配置更新检查");
    if (url.rfind("https://", 0) != 0) return fail("更新地址必须是 https");
    std::vector<std::pair<std::string, std::string>> h{{"User-Agent", "ballrain-pool-update-check"}};
    platform::HttpResponse m, s;
    try {
        m = platform::http_request("GET", url, h, "", 20.0, 64 * 1024);
        s = platform::http_request("GET", url + ".sig", h, "", 20.0, 1024);
    } catch (const Error &e) {
        return fail(std::string("无法检查更新：") + e.what());
    }
    if (m.status != 200 || s.status != 200) return fail("无法检查更新（HTTP " + std::to_string(m.status) + "）");
    if (!verify_release_signature(m.body, s.body, pubkey_hex)) return fail("发布清单的签名不对，已忽略");
    Json j;
    try {
        j = Json::parse(m.body, 8);
    } catch (const Error &) {
        return fail("发布清单格式不对");
    }
    if (!j.is_object() || j.get_string("product", "") != "ballrain-pool") return fail("发布清单格式不对");
    ReleaseInfo r;
    r.version = j.get_string("version", "");
    r.notes = j.get_string("notes", "");
    if (r.version.empty()) return fail("发布清单没有版本号");
    if (error) error->clear();
    if (compare_versions(r.version, current) <= 0) return std::nullopt;
    const Json *assets = j.find("assets");
    const Json *a = assets ? assets->find(key) : nullptr;
    if (!a || !a->is_object()) return fail("新版本 " + r.version + " 没有这个平台（" + key + "）的下载");
    r.url = a->get_string("url", "");
    r.sha256 = a->get_string("sha256", "");
    r.size = a->get_int("size", 0);
    if (r.url.rfind("https://", 0) != 0) return fail("发布清单里的下载地址不是 https");
    return r;
}

}  // namespace ballrain
