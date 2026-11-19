// Engine-free unit checks of the native core against values computed by the Python side (ballrain.pool.sampling,
// ballrain.pool.protocol.dumps, hashlib). Run: nc_selftest (exit code 0 = all passed).
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "ballrain/branch.h"
#include "ballrain/chunk.h"
#include "ballrain/diagnostics.h"
#include "ballrain/features.h"
#include "ballrain/geometry.h"
#include "ballrain/json.h"
#include "ballrain/policy.h"
#include "ballrain/sha256.h"
#include "ballrain/update.h"
#include "ballrain/wire.h"

using namespace ballrain;

static int failures = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);              \
            ++failures;                                                             \
        }                                                                           \
    } while (0)

int main() {
    // sampling.sample_u / splitmix64 (exact)
    CHECK(py_float_repr(sample_u(1, 0)) == "0.03401170130434639");
    CHECK(py_float_repr(sample_u(1, 1)) == "0.9140224628703029");
    CHECK(py_float_repr(sample_u(123456789, 4095)) == "0.11257283599099865");
    CHECK(py_float_repr(sample_u((uint64_t(1) << 53) - 1, 7)) == "0.02300620177497892");
    CHECK(splitmix64(0) == 16294208416658607535ull);
    CHECK(splitmix64(~uint64_t(0)) == 16490336266968443936ull);
    // sample_action / action_consistent
    std::vector<float> lp;
    for (double p : {0.1, 0.2, 0.3, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05}) lp.push_back(float(std::log(p)));
    int want[] = {0, 1, 1, 2, 8};
    double us[] = {0.0, 0.1, 0.29999, 0.3, 0.99999};
    for (int i = 0; i < 5; ++i) CHECK(sample_action(lp, us[i]) == want[i]);
    CHECK(action_consistent(lp, 0.30000001, 1));
    {  // feature_module_v1: the host's view of a feature set it does not know (the module's layout)
        Json f = Json::parse(R"({"version": "f8", "contacts_k": 4, "pos_scale": 40, "reach": false})");
        bool strict = false;
        try {
            FeatureConfig::from_json(f);
        } catch (const Error &) {
            strict = true;
        }
        CHECK(strict);
        FeatureConfig h = FeatureConfig::host_view(f);
        CHECK(h.pos_scale == 40 && h.version == "f6");
    }
    CHECK(!action_consistent(lp, 0.5, 1));
    // protocol.dumps (canonical JSON)
    Json j = Json::parse(R"({"b":[1,2.5,1e-05,1e16,123456789012345.6,0.1,-0.0,3.0],"a":"xé\n","c":null,"d":true})");
    CHECK(j.dump(true) == "{\"a\":\"x\xc3\xa9\\n\",\"b\":[1,2.5,1e-05,1e+16,123456789012345.6,0.1,-0.0,3.0],\"c\":null,\"d\":true}");
    CHECK(py_float_repr(7.5757575035095215) == "7.5757575035095215");
    CHECK(py_float_repr(1e22) == "1e+22" && py_float_repr(0.0001) == "0.0001" && py_float_repr(123.0) == "123.0");
    bool threw = false;
    try {
        Json::parse("{\"a\":1,\"a\":2}");
    } catch (const Error &) {
        threw = true;
    }
    CHECK(threw);
    // feature version gate (FeatureConfig.__post_init__): reach makes it f7, pinned by reach_version; an f7 label
    // without the block is f6; another reach_version is refused
    CHECK(FeatureConfig::from_json(Json::parse(R"({"version":"f6","reach":false})")).version == "f6");
    CHECK(FeatureConfig::from_json(Json::parse(R"({"version":"f7"})")).version == "f6");
    {
        FeatureConfig f7 = FeatureConfig::from_json(Json::parse(R"({"version":"f6","reach":true})"));
        CHECK(f7.version == "f7" && f7.reach && f7.reach_version == reach_version());
        bool refused = false;
        try {
            FeatureConfig::from_json(Json::parse(R"({"version":"f7","reach":true,"reach_version":"fastreach4.1-block1"})"));
        } catch (const Error &e) {
            refused = std::string(e.what()).find("reach_version") != std::string::npos;
        }
        CHECK(refused);
    }
    // branch rules (ballrain/branch_rules.py, docs/l2-stone-branch.md v1): defaults, overrides, the closed schema
    {
        BranchRules d = BranchRules::from_desc(Json::parse(R"({"level":2})"));
        CHECK(d.is_default() && d.drive_gain == 0.8 && d.rest_control_repeat == 2 && !d.rest_stop_speed);
        BranchRules r = BranchRules::from_desc(Json::parse(
            R"({"rules":{"stall_pause_speed":2,"stall_odometer":"s","rest_max_brake_frames":300,)"
            R"("rest_require_support":false,"rest_fallback":"error","rest_stop_speed":null}})"));
        CHECK(!r.is_default() && r.stall_pause_speed == 2.0 && r.stall_odometer == "s" && !r.rest_stop_speed &&
              r.rest_max_brake_frames && *r.rest_max_brake_frames == 300 && !r.rest_require_support &&
              r.rest_fallback == "error");
    }
    for (const char *bad : {R"({"rules":{"stall_pasue_speed":0}})", R"({"rules":{"rest_require_support":1}})",
                            R"({"rules":{"stall_speed_source":"velocity"}})", R"({"rules":{"rest_control_repeat":0}})",
                            R"({"rules":{"rest_control_repeat":2.0}})", R"({"rules":{"rest_stop_speed":0}})",
                            R"({"rules":{"drive_gain":-1}})", R"({"rules":{"drive_gain":true}})",
                            R"({"rules":{"stall_pause_speed":null}})", R"({"rules":[]})"}) {
        bool refused = false;
        try {
            BranchRules::from_desc(Json::parse(bad));
        } catch (const Error &) {
            refused = true;
        }
        CHECK(refused);
    }
    // zstd with an output cap (the reach fields download: a bomb guard, truncated data refused)
    {
        Bytes raw;
        for (int i = 0; i < 3000000; ++i) raw.push_back(char((i * 7) % 251));
        Bytes z = zstd_compress(raw, 9);
        CHECK(zstd_decompress_capped(z, raw.size()) == raw);
        for (int k = 0; k < 2; ++k) {
            bool refused = false;
            try {
                if (k == 0) zstd_decompress_capped(z, raw.size() - 1);       // larger than announced
                else zstd_decompress_capped(z.substr(0, z.size() / 2), raw.size());  // truncated
            } catch (const Error &) {
                refused = true;
            }
            CHECK(refused);
        }
    }
    // sha256
    CHECK(Sha256::hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    std::string mil(1000000, 'a');
    CHECK(Sha256::hex(mil) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    // numpy add.reduce order for 9 doubles (pairwise with 8 accumulators)
    double a[9] = {1e16, 1.0, -1e16, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0};
    double r = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
    CHECK(numpy_sum(a, 9) == r + a[8]);
    // geometry: a vertical ray onto a unit triangle, closest point on its face
    V3 A{0, 0, 0}, B{1, 0, 0}, C{0, 0, 1};
    CHECK(ray_down_pair(V3{0.2, 5, 0.2}, A, B, C) == 0.0);
    CHECK(std::isnan(ray_down_pair(V3{0.8, 5, 0.8}, A, B, C)));
    int reg;
    V3 q = closest_point_triangle(V3{0.2, 3, 0.2}, A, B, C, &reg);
    CHECK(reg == 2 && py_float_repr(q[0]) == "0.19999999999999998" && q[1] == 0.0 && q[2] == q[0]);  // numpy: same value
    // ACT1 encoding (protocol.encode_action layout: 36-byte header + frames)
    Action act;
    act.frames = {{7.5757575f, KEY_UP}};
    act.repeat = 2;
    act.stop_mask = event_mask({Ev::BallOff, Ev::LevelFinish, Ev::GameOver});
    Bytes b = encode_action(act, 5);
    CHECK(b.size() == 44 && b.substr(0, 4) == "ACT1");
    // update check: versions, ed25519 (RFC 8032 test 1: the empty message)
    CHECK(compare_versions("0.3.10", "0.3.9") == 1 && compare_versions("0.4.0-beta1", "0.4.0") == -1 &&
          compare_versions("0.3.0", "0.3") == 0 && compare_versions("1.0.0", "0.9.9") == 1);
    const char *pub = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";
    std::string sig = "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46b"
                      "d25bf5f0595bbe24655141438e7a100b";
    CHECK(verify_release_signature("", sig, pub));
    CHECK(!verify_release_signature("x", sig, pub));
    sig[0] = 'f';
    CHECK(!verify_release_signature("", sig, pub) && !verify_release_signature("", "zz", pub));
    // diagnostics scrub: replacements (case / slash insensitive, longest first), IPv4, versions kept
    std::string s = scrub_text("C:\\Users\\Wang\\AppData\\Local\\BallrainPool\\logs and c:/users/wang/x, token abcdefgh, "
                               "from 192.168.1.20:80, client 0.3.0, 1.2.3.4.5 stays",
                               {{"C:/Users/Wang", "~"}, {"C:/Users/Wang/AppData/Local/BallrainPool", "<数据目录>"},
                                {"abcdefgh", "<令牌>"}});
    CHECK(s == "<数据目录>\\logs and ~/x, token <令牌>, from <ip>:80, client 0.3.0, 1.2.3.4.5 stays");
    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
