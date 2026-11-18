// Engine-free unit checks of the native core against values computed by the Python side (ballrain.pool.sampling,
// ballrain.pool.protocol.dumps, hashlib). Run: nc_selftest (exit code 0 = all passed).
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "ballrain/features.h"
#include "ballrain/geometry.h"
#include "ballrain/json.h"
#include "ballrain/policy.h"
#include "ballrain/sha256.h"
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
    // feature version gate: f6 (reach false or absent) parses, f7 / reach true is refused (no fastreach port)
    FeatureConfig::from_json(Json::parse(R"({"version":"f6","reach":false})"));
    for (const char *f7 : {R"({"version":"f6","reach":true})", R"({"version":"f7"})"}) {
        bool refused = false;
        try {
            FeatureConfig::from_json(Json::parse(f7));
        } catch (const Error &e) {
            refused = std::string(e.what()).find("fastreach") != std::string::npos;
        }
        CHECK(refused);
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
    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
