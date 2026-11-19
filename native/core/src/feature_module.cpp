#include "ballrain/feature_module.h"

#include <cstring>
#include <map>
#include <mutex>

#include "ballrain/fastreach.h"
#include "ballrain/sha256.h"
#include "ballrain/update.h"

#ifdef BALLRAIN_HAS_WASMTIME
#include <wasi.h>
#include <wasmtime.h>
#ifdef _WIN32
#include <windows.h>
#endif
#endif

#ifndef BALLRAIN_MODULE_PUBKEY
#define BALLRAIN_MODULE_PUBKEY ""
#endif

namespace ballrain {

static std::mutex key_mu;
static std::string module_pubkey = BALLRAIN_MODULE_PUBKEY;

void FeatureModule::set_module_pubkey(const std::string &hex) {
    std::lock_guard<std::mutex> g(key_mu);
    module_pubkey = hex;
}

bool FeatureModule::has_pubkey() {
    std::lock_guard<std::mutex> g(key_mu);
    return module_pubkey.size() == 64;
}

bool FeatureModule::signature_ok(const Bytes &wasm, const std::string &sig_hex) {
    std::string key;
    {
        std::lock_guard<std::mutex> g(key_mu);
        key = module_pubkey;
    }
    return !key.empty() && verify_release_signature(wasm, sig_hex, key);
}

// ---------------------------------------------------------------------------------------------- info

FeatureInfo unpack_feature_info(const Bytes &b) {
    static const char *MODES[] = {"", "init", "window", "window_empty", "reacquire"};
    Reader r(b);
    auto str = [&] {
        uint32_t n = r.u32();
        return r.raw(n);
    };
    FeatureInfo fi;
    uint32_t f = r.u32();
    for (double &v : fi.pos) v = r.f64();
    for (double &v : fi.vel) v = r.f64();
    Projection p;
    p.s = r.f64(), p.seg = int64_t(r.u64()), p.t = r.f64();
    for (double &v : p.point) v = r.f64();
    for (double &v : p.tangent) v = r.f64();
    p.dist = r.f64(), p.sector = int32_t(r.u32());
    std::string mode = str();
    p.mode = "?";
    for (const char *m : MODES)
        if (mode == m) p.mode = m;
    p.jump = (f & 256u) != 0;
    fi.have_ball = (f & 1u) != 0;
    if (f & 2u) fi.path = p;
    fi.path_source = str();
    fi.has_branch = (f & 4u) != 0, fi.branch = (f & 8u) != 0, fi.branch_multi = (f & 16u) != 0;
    fi.supported = (f & 32u) != 0;
    fi.local_s = r.f64(), fi.ref_elapsed_ms = r.f64(), fi.ref_remaining_ms = r.f64();
    fi.branch_id = str();
    fi.branch_phase = int32_t(r.u32());
    double gap = r.f64();
    if (f & 64u) fi.gap = gap;
    fi.surface = str();
    double efm = r.f64();
    if (f & 128u) fi.edge_fan_min = efm;
    fi.reach_key = str();
    fi.timing_set = (f & 512u) != 0, fi.timing_branch = (f & 1024u) != 0;
    fi.timing_elapsed_ms = r.f64(), fi.timing_remaining_ms = r.f64();
    return fi;
}

#ifndef BALLRAIN_HAS_WASMTIME

bool FeatureModule::available(std::string *why) {
    if (why) *why = "built without wasmtime";
    return false;
}
struct FeatureModule::Impl {};
FeatureModule::FeatureModule(const Bytes &, const std::string &, const Json &) {
    throw Error("this build has no feature module support (wasmtime)");
}
FeatureModule::~FeatureModule() = default;
void FeatureModule::feed(uint32_t, const void *, size_t) {}
std::vector<float> FeatureModule::compute(FeatureInfo *) { return {}; }
StreamWriter::Sink FeatureModule::sink() { return nullptr; }
std::string FeatureModule::guest_path(const std::string &) const { return ""; }

#else

// ---------------------------------------------------------------------------------------------- engine

static std::string err_text(wasmtime_error_t *e, wasm_trap_t *t) {
    wasm_byte_vec_t m;
    if (e) wasmtime_error_message(e, &m), wasmtime_error_delete(e);
    else if (t) wasm_trap_message(t, &m), wasm_trap_delete(t);
    else return "unknown";
    std::string s(m.data, m.size);
    wasm_byte_vec_delete(&m);
    return s;
}

bool FeatureModule::available(std::string *why) {
#ifdef _WIN32
    static int ok = -1;
    if (ok < 0) {
        // delay-loaded: present next to the program (the DLL search starts there)
        HMODULE h = LoadLibraryExW(L"wasmtime.dll", nullptr, LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        ok = h != nullptr;
    }
    if (!ok && why) *why = "wasmtime.dll was not found next to the program";
    return ok == 1;
#else
    (void)why;
    return true;
#endif
}

static std::mutex eng_mu;
static wasm_engine_t *engine() {
    static wasm_engine_t *e = nullptr;
    std::lock_guard<std::mutex> g(eng_mu);
    if (!e) {
        wasm_config_t *c = wasm_config_new();
        wasmtime_config_cranelift_nan_canonicalization_set(c, true);  // the same NaN bits on every machine
        wasmtime_config_wasm_exceptions_set(c, true);                  // the module's C++ exceptions (exnref)
        wasmtime_config_cranelift_opt_level_set(c, WASMTIME_OPT_LEVEL_SPEED);
        e = wasm_engine_new_with_config(c);
    }
    return e;
}

// compiled modules by sha256, shared by every instance and task of the process
static std::map<std::string, wasmtime_module_t *> modules;

// the reach fields the modules loaded (handles), shared by every module of the process
static std::mutex reach_mu;
static std::vector<std::shared_ptr<const fastreach::Fields>> reach_fields;
static std::map<std::string, int32_t> reach_by_path;

struct FeatureModule::Impl {
    wasmtime_store_t *store = nullptr;
    wasmtime_context_t *ctx = nullptr;
    wasmtime_instance_t inst{};
    wasmtime_memory_t mem{};
    wasmtime_func_t f_alloc{}, f_free{}, f_feed{}, f_compute{}, f_out{}, f_info{}, f_info_len{}, f_error{};
    std::string cache_dir;
    std::map<int32_t, std::unique_ptr<fastreach::Workspace>> ws;  // the module's workspaces (by its handle)

    uint8_t *data() { return wasmtime_memory_data(ctx, &mem); }
    size_t size() { return wasmtime_memory_data_size(ctx, &mem); }
    int64_t call(wasmtime_func_t &f, std::initializer_list<int32_t> args, bool result = true) {
        wasmtime_val_t a[4], r;
        size_t n = 0;
        for (int32_t v : args) a[n].kind = WASMTIME_I32, a[n].of.i32 = v, ++n;
        wasm_trap_t *trap = nullptr;
        wasmtime_error_t *e = wasmtime_func_call(ctx, &f, a, n, result ? &r : nullptr, result ? 1 : 0, &trap);
        if (e || trap) throw Error("feature module: " + err_text(e, trap));
        return result ? (r.kind == WASMTIME_I32 ? r.of.i32 : int64_t(r.of.i64)) : 0;
    }
    std::string error() {
        int64_t p = call(f_error, {});
        std::string s;
        for (size_t i = size_t(p); i < size() && data()[i] && s.size() < 1000; ++i) s += char(data()[i]);
        return s;
    }
    // bounds-checked view of [p, p + n) in the module's memory
    uint8_t *at(uint64_t p, uint64_t n) {
        if (p > size() || n > size() - p) throw Error("feature module: pointer out of bounds");
        return data() + p;
    }
};

static wasm_trap_t *trap_of(const std::string &m) { return wasmtime_trap_new(m.data(), m.size()); }

// br_host_reach_load(path, path_len, key_out, key_cap, key_len_out) -> handle
static wasm_trap_t *host_reach_load(void *env, wasmtime_caller_t *, const wasmtime_val_t *a, size_t, wasmtime_val_t *r,
                                    size_t) {
    auto *I = static_cast<FeatureModule::Impl *>(env);
    try {
        std::string guest(reinterpret_cast<char *>(I->at(uint32_t(a[0].of.i32), uint32_t(a[1].of.i32))), size_t(uint32_t(a[1].of.i32)));
        // only fields under the module's /cache (no "..")
        if (guest.rfind("/cache/", 0) != 0 || guest.find("..") != std::string::npos) throw Error("reach fields path");
        std::string host = I->cache_dir + guest.substr(6);
        int32_t h;
        std::string key;
        {
            std::lock_guard<std::mutex> g(reach_mu);
            auto it = reach_by_path.find(host);
            if (it == reach_by_path.end()) {
                reach_fields.push_back(fastreach::Fields::load(host));
                it = reach_by_path.emplace(host, int32_t(reach_fields.size() - 1)).first;
            }
            h = it->second;
            key = reach_fields[size_t(h)]->key;
        }
        uint32_t cap = uint32_t(a[3].of.i32);
        if (key.size() > cap) throw Error("reach fields key too long");
        std::memcpy(I->at(uint32_t(a[2].of.i32), key.size()), key.data(), key.size());
        uint32_t n = uint32_t(key.size());
        std::memcpy(I->at(uint32_t(a[4].of.i32), 4), &n, 4);
        r[0].kind = WASMTIME_I32, r[0].of.i32 = h;
    } catch (const std::exception &) {
        r[0].kind = WASMTIME_I32, r[0].of.i32 = -1;
    }
    return nullptr;
}

// br_host_reach_perceive(fields, workspace, uzones, n_uzones, ball_type, quarter, pos, vel, out) -> 0 / -1
static wasm_trap_t *host_reach_perceive(void *env, wasmtime_caller_t *, const wasmtime_val_t *a, size_t,
                                        wasmtime_val_t *r, size_t) {
    auto *I = static_cast<FeatureModule::Impl *>(env);
    try {
        std::shared_ptr<const fastreach::Fields> f;
        {
            std::lock_guard<std::mutex> g(reach_mu);
            int32_t h = a[0].of.i32;
            if (h < 0 || size_t(h) >= reach_fields.size()) throw Error("fields handle");
            f = reach_fields[size_t(h)];
        }
        auto &w = I->ws[a[1].of.i32];
        if (!w) w = std::make_unique<fastreach::Workspace>();
        uint32_t nz = uint32_t(a[3].of.i32);
        if (nz > 6u * 4096u) throw Error("uzones");
        std::vector<double> uz(nz);
        if (nz) std::memcpy(uz.data(), I->at(uint32_t(a[2].of.i32), uint64_t(nz) * 8), size_t(nz) * 8);
        double pos[3], vel[3], out[fastreach::N_FEATURES];
        std::memcpy(pos, I->at(uint32_t(a[6].of.i32), 24), 24);
        std::memcpy(vel, I->at(uint32_t(a[7].of.i32), 24), 24);
        fastreach::perceive(*f, *w, uz, a[4].of.i32, a[5].of.i32, pos, vel, out);
        std::memcpy(I->at(uint32_t(a[8].of.i32), sizeof out), out, sizeof out);
        r[0].kind = WASMTIME_I32, r[0].of.i32 = 0;
    } catch (const std::exception &) {
        r[0].kind = WASMTIME_I32, r[0].of.i32 = -1;
    }
    return nullptr;
}

static wasm_functype_t *ftype(size_t n_params) {
    wasm_valtype_vec_t p, r;
    wasm_valtype_vec_new_uninitialized(&p, n_params);
    for (size_t i = 0; i < n_params; ++i) p.data[i] = wasm_valtype_new(WASM_I32);
    wasm_valtype_vec_new_uninitialized(&r, 1);
    r.data[0] = wasm_valtype_new(WASM_I32);
    return wasm_functype_new(&p, &r);
}

FeatureModule::FeatureModule(const Bytes &wasm, const std::string &cache_dir, const Json &init) : impl_(new Impl()) {
    std::string why;
    if (!available(&why)) throw Error("feature module: " + why);
    Impl &I = *impl_;
    I.cache_dir = cache_dir;
    wasm_engine_t *eng = engine();
    std::string sha = Sha256::hex(wasm);
    wasmtime_module_t *mod = nullptr;
    {
        std::lock_guard<std::mutex> g(eng_mu);
        auto it = modules.find(sha);
        if (it != modules.end()) mod = it->second;
    }
    if (!mod) {
        wasmtime_error_t *e =
            wasmtime_module_new(eng, reinterpret_cast<const uint8_t *>(wasm.data()), wasm.size(), &mod);
        if (e) throw Error("feature module: compile: " + err_text(e, nullptr));
        std::lock_guard<std::mutex> g(eng_mu);
        auto it = modules.find(sha);
        if (it != modules.end()) wasmtime_module_delete(mod), mod = it->second;
        else modules[sha] = mod;
    }
    I.store = wasmtime_store_new(eng, nullptr, nullptr);
    I.ctx = wasmtime_store_context(I.store);
    wasi_config_t *wc = wasi_config_new();
    if (!wasi_config_preopen_dir(wc, cache_dir.c_str(), "/cache", false)) {
        wasi_config_delete(wc);
        throw Error("feature module: cannot open the cache directory for it");
    }
    wasmtime_error_t *e = wasmtime_context_set_wasi(I.ctx, wc);
    if (e) throw Error("feature module: wasi: " + err_text(e, nullptr));
    wasmtime_linker_t *lk = wasmtime_linker_new(eng);
    e = wasmtime_linker_define_wasi(lk);
    wasm_functype_t *t5 = ftype(5), *t9 = ftype(9);
    if (!e) e = wasmtime_linker_define_func(lk, "ballrain", 8, "br_host_reach_load", 18, t5, host_reach_load, &I, nullptr);
    if (!e)
        e = wasmtime_linker_define_func(lk, "ballrain", 8, "br_host_reach_perceive", 22, t9, host_reach_perceive, &I,
                                        nullptr);
    wasm_functype_delete(t5), wasm_functype_delete(t9);
    wasm_trap_t *trap = nullptr;
    if (!e) e = wasmtime_linker_instantiate(lk, I.ctx, mod, &I.inst, &trap);
    wasmtime_linker_delete(lk);
    if (e || trap) throw Error("feature module: instantiate: " + err_text(e, trap));
    auto get = [&](const char *name, wasmtime_func_t &f) {
        wasmtime_extern_t x;
        if (!wasmtime_instance_export_get(I.ctx, &I.inst, name, std::strlen(name), &x) || x.kind != WASMTIME_EXTERN_FUNC)
            throw Error(std::string("feature module: no export ") + name);
        f = x.of.func;
    };
    wasmtime_extern_t m;
    if (!wasmtime_instance_export_get(I.ctx, &I.inst, "memory", 6, &m) || m.kind != WASMTIME_EXTERN_MEMORY)
        throw Error("feature module: no memory");
    I.mem = m.of.memory;
    wasmtime_func_t f_init, f_abi, f_start;
    get("_initialize", f_start);
    I.call(f_start, {}, false);
    get("br_abi", f_abi);
    if (I.call(f_abi, {}) != MODULE_ABI) throw Error("feature module: another ABI version");
    get("br_alloc", I.f_alloc), get("br_free", I.f_free), get("br_feed", I.f_feed), get("br_compute", I.f_compute);
    get("br_out", I.f_out), get("br_info", I.f_info), get("br_info_len", I.f_info_len), get("br_error", I.f_error);
    get("br_init", f_init);
    std::string js = init.dump(true);
    int32_t p = int32_t(I.call(I.f_alloc, {int32_t(js.size())}));
    std::memcpy(I.at(uint32_t(p), js.size()), js.data(), js.size());
    int64_t rc = I.call(f_init, {p, int32_t(js.size())});
    I.call(I.f_free, {p}, false);
    if (rc < 0) throw Error("feature module: init: " + I.error());
}

FeatureModule::~FeatureModule() {
    if (impl_ && impl_->store) wasmtime_store_delete(impl_->store);
}

void FeatureModule::feed(uint32_t kind, const void *src, size_t n) {
    Impl &I = *impl_;
    if (n > (64u << 20)) throw Error("feature module: record too large");
    int32_t p = int32_t(I.call(I.f_alloc, {int32_t(n)}));
    if (n) std::memcpy(I.at(uint32_t(p), n), src, n);
    int64_t rc = I.call(I.f_feed, {int32_t(kind), p, int32_t(n)});
    I.call(I.f_free, {p}, false);
    if (rc < 0) throw Error("feature module: " + I.error());
}

std::vector<float> FeatureModule::compute(FeatureInfo *info) {
    Impl &I = *impl_;
    int64_t n = I.call(I.f_compute, {});
    if (n < 0) throw Error("feature module: " + I.error());
    std::vector<float> v(static_cast<size_t>(n));
    std::memcpy(v.data(), I.at(uint32_t(I.call(I.f_out, {})), uint64_t(n) * 4), size_t(n) * 4);
    if (info) {
        int64_t ln = I.call(I.f_info_len, {});
        Bytes b(reinterpret_cast<char *>(I.at(uint32_t(I.call(I.f_info, {})), uint64_t(ln))), size_t(ln));
        *info = unpack_feature_info(b);
    }
    return v;
}

StreamWriter::Sink FeatureModule::sink() {
    return [this](uint32_t kind, const void *p, size_t n) { feed(kind, p, n); };
}

std::string FeatureModule::guest_path(const std::string &host) const {
    const std::string &c = impl_->cache_dir;
    if (host.size() < c.size() || host.compare(0, c.size(), c) != 0) return "";
    std::string rest = host.substr(c.size());
    for (char &ch : rest)
        if (ch == '\\') ch = '/';
    if (!rest.empty() && rest[0] != '/') return "";
    return "/cache" + rest;
}

#endif

}  // namespace ballrain
