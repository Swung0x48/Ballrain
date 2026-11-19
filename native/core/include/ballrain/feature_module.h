// The feature module host (feature_module_v1; native/wasm/feature_module.cpp is the module): a signed .wasm per
// policy, run with wasmtime (Cranelift, NaN canonicalisation, wasm exceptions), fed the session's record stream, asked
// for the features and their info each step. The reach kernel stays native (the module's br_host_reach_* imports).
//
// wasmtime.dll is delay-loaded (Windows): a client without it runs everything else and does not declare the
// capability (available()).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ballrain/features.h"
#include "ballrain/json.h"
#include "ballrain/stream.h"
#include "ballrain/util.h"

namespace ballrain {

constexpr int MODULE_ABI = 1;

class FeatureModule {
public:
    // wasmtime can be loaded (and was built in); why: the reason when not
    static bool available(std::string *why = nullptr);
    // the module's signature (ed25519 over its exact bytes, hex) against the build's module key
    // (BALLRAIN_MODULE_PUBKEY, or set_module_pubkey for tests); false: refuse to load it
    static bool signature_ok(const Bytes &wasm, const std::string &sig_hex);
    static void set_module_pubkey(const std::string &hex);  // tests / dev gateways (the build's key otherwise)

    // wasm: the module bytes (its sha256 already checked by the caller); cache_dir: the host directory the module sees
    // read-only as "/cache"; init: br_init's JSON ({"level", "features"} with paths under "/cache")
    FeatureModule(const Bytes &wasm, const std::string &cache_dir, const Json &init);
    ~FeatureModule();
    FeatureModule(const FeatureModule &) = delete;
    FeatureModule &operator=(const FeatureModule &) = delete;

    void feed(uint32_t kind, const void *p, size_t n);
    std::vector<float> compute(FeatureInfo *info);
    // a record sink for the session (StreamWriter(Sink)): every record goes to feed()
    StreamWriter::Sink sink();
    // the path the module sees for a host path under cache_dir ("" when it is not under it)
    std::string guest_path(const std::string &host_path) const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// FeatureInfo from the module's packed form (feature_module.cpp pack_info)
FeatureInfo unpack_feature_info(const Bytes &b);

}  // namespace ballrain
