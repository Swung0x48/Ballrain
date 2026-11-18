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
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <immintrin.h>
#include <cmath>
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
#include "ballrain/reach_block.h"
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

// check: the client's call pattern (one workspace, no dbg / pdd, this executable's libm unless --libm system) on a
// state file, against expected raw outputs (expect.npz "out" (n, 59): Python's on the reference machine); prints a
// short report (cross-machine check: run the same files on two machines, compare the reports)
static int run_check(int argc, char **argv) {
    std::string lm = arg(argc, argv, "--libm", "crt");
    install_libm(lm);
    std::vector<std::shared_ptr<const fr::Fields>> fields;
    for (auto &p : split(arg(argc, argv, "--fields"))) fields.push_back(fr::Fields::load(p));
    auto z = load_npz(arg(argc, argv, "--in"));
    auto e = load_npz(arg(argc, argv, "--expect"));
    std::vector<double> pos = get(z, "pos").f64(), vel = get(z, "vel").f64(), zones = get(z, "zones").f64();
    std::vector<int64_t> ball = get(z, "ball").ints(), quarter = get(z, "quarter").ints(), fi = get(z, "field").ints(),
                         zoff = get(z, "zoff").ints();
    std::vector<double> want = get(e, "out").f64();
    const size_t n = ball.size(), nf = size_t(fr::N_FEATURES);
    if (want.size() != n * nf) throw Error("expect.npz: out is not (n, 59)");
    for (auto &f : fields)
        std::printf("field %s\n", f->key.substr(f->key.size() - 20).c_str());
#if defined(_M_X64) || defined(__x86_64__)
    {  // the machine: CPU brand and features, the floating-point control (MXCSR: rounding, FTZ / DAZ)
        int r[4];
        char brand[49] = {};
        for (int k = 0; k < 3; ++k) {
            __cpuid(r, int(0x80000002u + unsigned(k)));
            std::memcpy(brand + 16 * k, r, 16);
        }
        __cpuid(r, 1);
        int fma = (r[2] >> 12) & 1, avx = (r[2] >> 28) & 1;
        __cpuidex(r, 7, 0);
        int avx2 = (r[1] >> 5) & 1, avx512f = (r[1] >> 16) & 1;
        std::printf("cpu %s | fma %d avx %d avx2 %d avx512f %d | mxcsr 0x%04x\n", brand, fma, avx, avx2, avx512f,
                    unsigned(_mm_getcsr()));
    }
#endif
    fr::Workspace ws;
    const int flags[6] = {0, 1, 2, 3, 11, 12};  // grounded doomed unknown_zone air_any air_near_can_stop air_near_clear
    size_t differ = 0, flipped = 0, jumped = 0;
    double max_raw = 0, max_scaled = 0;
    std::vector<size_t> first;
    std::vector<double> o(nf), so(nf), sw(nf);
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> uz(zones.begin() + zoff[i] * 6, zones.begin() + zoff[i + 1] * 6);
        fr::perceive(*fields[size_t(fi[i])], ws, uz, int(ball[i]), int(quarter[i]), &pos[3 * i], &vel[3 * i], o.data());
        const double *w = &want[i * nf];
        if (std::memcmp(o.data(), w, nf * sizeof(double)) == 0) continue;
        bool bitdiff = false;
        for (size_t k = 0; k < nf; ++k)
            if (!(o[k] == w[k] || (o[k] != o[k] && w[k] != w[k]))) bitdiff = true;
            else if (std::memcmp(&o[k], &w[k], 8) != 0) bitdiff = true;  // -0 / +0
        if (!bitdiff) continue;
        ++differ;
        bool flip = false;
        for (int f : flags) flip = flip || (o[size_t(f)] != w[size_t(f)]);
        flipped += flip;
        reach_scale(o.data(), so.data());
        reach_scale(w, sw.data());
        bool jump = false;
        for (size_t k = 0; k < nf; ++k) {
            double d = std::abs(so[k] - sw[k]);
            max_scaled = std::max(max_scaled, d);
            jump = jump || d > 1e-4 * std::max(1.0, std::abs(sw[k]));
            if (std::isfinite(o[k]) && std::isfinite(w[k])) max_raw = std::max(max_raw, std::abs(o[k] - w[k]));
        }
        jumped += jump;
        if (first.size() < 20) first.push_back(i);
    }
    std::printf("libm %s: %zu states, differing in any bit %zu, flag flips %zu, scaled jumps > 1e-4 %zu, "
                "max raw |d| %.3g, max scaled |d| %.3g\n", lm.c_str(), n, differ, flipped, jumped, max_raw, max_scaled);
    for (size_t i : first) {
        fr::perceive(*fields[size_t(fi[i])], ws, std::vector<double>(zones.begin() + zoff[i] * 6, zones.begin() + zoff[i + 1] * 6),
                     int(ball[i]), int(quarter[i]), &pos[3 * i], &vel[3 * i], o.data());
        std::printf("  state %zu:", i);
        for (size_t k = 0; k < nf; ++k)
            if (std::memcmp(&o[k], &want[i * nf + k], 8) != 0) std::printf(" [%zu] %.17g vs %.17g", k, o[k], want[i * nf + k]);
        std::printf("\n");
    }
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
        if (mode == "check") return run_check(argc, argv);
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
