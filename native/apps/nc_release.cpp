// Release signing for the volunteer client's update check (docs/native-client.md section 8; native/core/src/update.cpp).
//
//   nc_release keygen --out-dir DIR
//       a new ed25519 key: DIR/release-signing.key (the 32-byte seed, hex; owner-only on POSIX) and
//       DIR/release-signing.pub (the public key, hex: give it to the build as BALLRAIN_RELEASE_PUBKEY). Refuses to
//       overwrite. Keep DIR outside the repository (the default of tools/release/keygen.cmd is
//       %USERPROFILE%\.ballrain-release); never commit or print the .key file.
//   nc_release manifest --version V --out release.json [--notes TEXT] --asset KEY=URL=FILE [--asset ...]
//       release.json with each asset's sha256 and size (KEY: windows-x64, ...)
//   nc_release sign --key FILE --manifest release.json      -> release.json.sig (hex signature of the exact bytes)
//   nc_release verify --pub HEX|FILE --manifest release.json [--sig release.json.sig]
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "ballrain/json.h"
#include "ballrain/sha256.h"
#include "ballrain/update.h"
#include "ballrain/util.h"
#include "ed25519.h"

using namespace ballrain;

static std::string arg(int argc, char **argv, const char *name, const char *dflt = nullptr) {
    for (int i = 2; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    if (!dflt) throw Error(std::string("missing ") + name);
    return dflt;
}

static std::vector<std::string> args_all(int argc, char **argv, const char *name) {
    std::vector<std::string> out;
    for (int i = 2; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) out.push_back(argv[i + 1]);
    return out;
}

static std::string to_hex(const unsigned char *p, size_t n) { return hex(p, n); }

static std::vector<unsigned char> from_hex(std::string s, size_t n) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    if (s.size() != 2 * n) throw Error("expected " + std::to_string(2 * n) + " hex characters");
    std::vector<unsigned char> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<unsigned char>(std::stoi(s.substr(2 * i, 2), nullptr, 16));
    return out;
}

static std::string read_text(const std::string &p) {
    Bytes b;
    if (!read_file(p, b)) throw Error("cannot read " + p);
    return b;
}

static int keygen(int argc, char **argv) {
    std::string d = arg(argc, argv, "--out-dir");
    std::filesystem::create_directories(u8_path(d));
    std::string kf = d + "/release-signing.key", pf = d + "/release-signing.pub";
    if (file_exists(kf) || file_exists(pf)) throw Error("a release key exists already in " + d + " (not overwritten)");
    unsigned char seed[32], pub[32], priv[64];
    if (ed25519_create_seed(seed) != 0) throw Error("no random seed from the OS");
    ed25519_create_keypair(pub, priv, seed);
    write_file(kf, to_hex(seed, 32) + "\n");
    std::error_code ec;
    std::filesystem::permissions(u8_path(kf), std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::replace, ec);
    write_file(pf, to_hex(pub, 32) + "\n");
    std::memset(seed, 0, sizeof seed), std::memset(priv, 0, sizeof priv);
    std::printf("public key written to %s (the private key file next to it stays private)\n", pf.c_str());
    return 0;
}

static int manifest(int argc, char **argv) {
    Json m = Json::object();
    m.set("product", Json::string("ballrain-pool"));
    m.set("version", Json::string(arg(argc, argv, "--version")));
    m.set("notes", Json::string(arg(argc, argv, "--notes", "")));
    Json assets = Json::object();
    for (auto &a : args_all(argc, argv, "--asset")) {
        size_t e1 = a.find('='), e2 = a.rfind('=');
        if (e1 == std::string::npos || e2 == e1) throw Error("--asset KEY=URL=FILE");
        std::string key = a.substr(0, e1), url = a.substr(e1 + 1, e2 - e1 - 1), file = a.substr(e2 + 1);
        if (url.rfind("https://", 0) != 0) throw Error("asset URL must be https: " + url);
        Json j = Json::object();
        j.set("url", Json::string(url));
        j.set("sha256", Json::string(file_sha256(file)));
        j.set("size", Json::integer(int64_t(std::filesystem::file_size(u8_path(file)))));
        assets.set(key, j);
    }
    m.set("assets", assets);
    write_file(arg(argc, argv, "--out"), m.dump(true) + "\n");
    return 0;
}

static int sign(int argc, char **argv) {
    auto seed = from_hex(read_text(arg(argc, argv, "--key")), 32);
    unsigned char pub[32], priv[64], sig[64];
    ed25519_create_keypair(pub, priv, seed.data());
    std::string mf = arg(argc, argv, "--manifest");
    std::string msg = read_text(mf);
    ed25519_sign(sig, reinterpret_cast<const unsigned char *>(msg.data()), msg.size(), pub, priv);
    std::memset(priv, 0, sizeof priv), std::fill(seed.begin(), seed.end(), 0);
    write_file(mf + ".sig", to_hex(sig, 64) + "\n");
    std::printf("signed %s with the key of public key %s\n", mf.c_str(), to_hex(pub, 32).c_str());
    return 0;
}

static int verify(int argc, char **argv) {
    std::string pub = arg(argc, argv, "--pub");
    if (pub.size() != 64) pub = read_text(pub);
    std::string mf = arg(argc, argv, "--manifest");
    bool ok = verify_release_signature(read_text(mf), read_text(arg(argc, argv, "--sig", (mf + ".sig").c_str())), pub);
    std::printf("%s\n", ok ? "signature OK" : "signature BAD");
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: nc_release keygen|manifest|sign|verify ... (see the source header)\n");
        return 2;
    }
    try {
        std::string c = argv[1];
        if (c == "keygen") return keygen(argc, argv);
        if (c == "manifest") return manifest(argc, argv);
        if (c == "sign") return sign(argc, argv);
        if (c == "verify") return verify(argc, argv);
        std::fprintf(stderr, "unknown command %s\n", c.c_str());
        return 2;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "nc_release: %s\n", e.what());
        return 1;
    }
}
