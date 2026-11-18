// nc_reach_diff: the fastreach kernel (core/src/fastreach.cpp) on a state file, for tools/native/reach_diff.py, which
// compares it bit for bit with ballrain/fastreach.py _perceive.
//   nc_reach_diff perceive --fields <a.npz>[,<b.npz>...] --in <states.npz> --out <out.bin> [--repeat <k>]
//                          [--libm system|crt] [--shared]
//     states.npz: pos (n, 3) f8, vel (n, 3) f8, ball (n,) i8, quarter (n,) i8, field (n,) i8 (index into --fields),
//     zoff (n + 1,) i8 and zones (m, 6) f8 (the unknown zones of state i: zones[zoff[i]:zoff[i + 1]]).
//     One workspace per field (the stamp_id sequence of one FastReach instance per field file), states in order.
//     out.bin: per state f8 out[59], dbg[33], pdd[8 x 13], stamp_id, best time of one perceive in us (of k runs).
//     --libm system (default): the kernel's log / pow from the C runtime Python uses (Windows: ucrtbase.dll);
//     crt: this executable's (the kernel's own default). --shared: the reach block's call pattern, one workspace for
//     every field and no dbg / pdd (those records hold NaN).
//   nc_reach_diff leaf --fields <a.npz> --name <helper> --in <args.npz> --out <out.bin> [--libm system|crt]
//     args.npz: args (n, k) f8; out.bin: i64 n, i64 m, then n x m f8 (fastreach::leaf of each row).
#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <memory>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ballrain/fastreach.h"
#include "ballrain/npz.h"
#include "ballrain/util.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

using namespace ballrain;
namespace fr = ballrain::fastreach;

static std::string arg(int argc, char **argv, const char *name, const char *dflt = nullptr) {
    for (int i = 2; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], name)) return argv[i + 1];
    if (dflt) return dflt;
    std::fprintf(stderr, "missing %s\n", name);
    std::exit(2);
}

static std::vector<std::string> split(const std::string &s) {
    std::vector<std::string> out;
    size_t a = 0;
    while (a <= s.size()) {
        size_t b = s.find(',', a);
        if (b == std::string::npos) b = s.size();
        if (b > a) out.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return out;
}

static const NpyArray &get(const std::map<std::string, NpyArray> &z, const char *k) {
    auto it = z.find(k);
    if (it == z.end()) throw Error(std::string("missing array ") + k);
    return it->second;
}

// --libm: system (default) = the C runtime Python uses (Windows: ucrtbase.dll), crt = this executable's.
static bool install_libm(const std::string &lm) {
    if (lm != "system" && lm != "crt") throw Error("--libm system|crt");
    bool sys_libm = false;
#ifdef _WIN32
    if (lm == "system") {
        if (HMODULE h = LoadLibraryExW(L"ucrtbase.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32)) {
            auto lg = reinterpret_cast<double (*)(double)>(reinterpret_cast<void *>(GetProcAddress(h, "log")));
            auto pw = reinterpret_cast<double (*)(double, double)>(reinterpret_cast<void *>(GetProcAddress(h, "pow")));
            if (lg && pw) fr::set_libm(lg, pw), sys_libm = true;
        }
        if (!sys_libm) throw Error("ucrtbase.dll log / pow not found");
    }
#else
    sys_libm = lm == "system";  // the platform libm is the one Python uses
#endif
    return sys_libm;
}

static int run_perceive(int argc, char **argv) {
    bool sys_libm = install_libm(arg(argc, argv, "--libm", "system"));
    bool shared = false;
    for (int i = 2; i < argc; ++i)
        if (!std::strcmp(argv[i], "--shared")) shared = true;
    std::vector<std::shared_ptr<const fr::Fields>> fields;
    for (auto &p : split(arg(argc, argv, "--fields"))) fields.push_back(fr::Fields::load(p));
    auto z = load_npz(arg(argc, argv, "--in"));
    std::vector<double> pos = get(z, "pos").f64(), vel = get(z, "vel").f64(), zones = get(z, "zones").f64();
    std::vector<int64_t> ball = get(z, "ball").ints(), quarter = get(z, "quarter").ints(), fi = get(z, "field").ints(),
                         zoff = get(z, "zoff").ints();
    const size_t n = ball.size();
    if (pos.size() != 3 * n || vel.size() != 3 * n || quarter.size() != n || fi.size() != n || zoff.size() != n + 1)
        throw Error("state arrays disagree");
    int repeat = std::stoi(arg(argc, argv, "--repeat", "1"));
    if (repeat < 1) repeat = 1;
    std::vector<fr::Workspace> ws(fields.size());
    const int nd = fr::n_dbg(), np = fr::n_pdd();
    const size_t rec = size_t(fr::N_FEATURES + nd + 8 * np + 2);
    std::vector<double> outv(n * rec);
    std::vector<double> scratch(rec);
    for (int rep = 0; rep < repeat; ++rep) {
        // a repeat runs on fresh workspaces so that every repeat has the same stamp_id sequence
        std::vector<fr::Workspace> wr(fields.size());
        std::vector<fr::Workspace> &W = rep == 0 ? ws : wr;
        for (size_t i = 0; i < n; ++i) {
            if (fi[i] < 0 || size_t(fi[i]) >= fields.size()) throw Error("field index out of range");
            std::vector<double> uz(zones.begin() + zoff[i] * 6, zones.begin() + zoff[i + 1] * 6);
            double *o = rep == 0 ? &outv[i * rec] : scratch.data();
            auto t0 = std::chrono::steady_clock::now();
            fr::Workspace &wk = W[shared ? 0 : fi[i]];
            if (shared) std::fill(o + fr::N_FEATURES, o + rec - 2, std::numeric_limits<double>::quiet_NaN());
            fr::perceive(*fields[fi[i]], wk, uz, int(ball[i]), int(quarter[i]), &pos[3 * i], &vel[3 * i], o,
                         shared ? nullptr : o + fr::N_FEATURES, shared ? nullptr : o + fr::N_FEATURES + nd);
            auto t1 = std::chrono::steady_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            double *dst = &outv[i * rec];
            if (rep == 0) {
                dst[rec - 2] = double(fr::stamp_id(wk));
                dst[rec - 1] = us;
            } else {
                if (std::memcmp(scratch.data(), dst, (rec - 2) * sizeof(double)) != 0)
                    throw Error("a repeat gave another result at state " + std::to_string(i));
                if (us < dst[rec - 1]) dst[rec - 1] = us;
            }
        }
    }
    Bytes b(reinterpret_cast<const char *>(outv.data()), outv.size() * sizeof(double));
    write_file(arg(argc, argv, "--out"), b);
    int64_t hw = 0;
    for (auto &w : ws) hw = std::max(hw, fr::stack_high_water(w));
    std::printf("%zu states, %zu fields, stack high water %lld\n", n, fields.size(), (long long)hw);
    return 0;
}

static int run_leaf(int argc, char **argv) {
    install_libm(arg(argc, argv, "--libm", "system"));
    auto f = fr::Fields::load(arg(argc, argv, "--fields"));
    fr::Workspace ws;
    std::string name = arg(argc, argv, "--name");
    auto z = load_npz(arg(argc, argv, "--in"));
    const NpyArray &A = get(z, "args");
    if (A.shape.size() != 2) throw Error("args must be (n, k)");
    std::vector<double> a = A.f64();
    int64_t n = A.shape[0], k = A.shape[1];
    std::vector<double> o, all;
    int64_t m = -1;
    for (int64_t i = 0; i < n; ++i) {
        if (!fr::leaf(*f, ws, name, &a[size_t(i * k)], size_t(k), o)) throw Error("unknown leaf " + name);
        if (m < 0) m = int64_t(o.size());
        if (int64_t(o.size()) != m) throw Error("leaf output size changed");
        all.insert(all.end(), o.begin(), o.end());
    }
    if (m < 0) m = 0;
    Bytes b;
    put_bytes(b, &n, 8);
    put_bytes(b, &m, 8);
    put_bytes(b, all.data(), all.size() * sizeof(double));
    write_file(arg(argc, argv, "--out"), b);
    return 0;
}

int main(int argc, char **argv) {
    try {
        if (argc < 2) {
            std::fprintf(stderr, "usage: nc_reach_diff perceive|leaf ...\n");
            return 2;
        }
        std::string mode = argv[1];
        if (mode == "perceive") return run_perceive(argc, argv);
        if (mode == "leaf") return run_leaf(argc, argv);
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
