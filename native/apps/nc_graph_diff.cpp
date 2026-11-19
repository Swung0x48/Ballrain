// Policy graph check (policy_graph_v1): loads a policy (npz + manifest, its graph or its arch's) and prints, for each
// row of an observation file (raw float32 .npy, (N, n_obs), already normalised), the logits and the value as text
// lines "logits... value" for tools/native/graph_diff.py to compare with ballrain/pool/weights.py.
//   nc_graph_diff <policy.npz> <manifest.json> <x.npy> [--timing]
#include <chrono>
#include <cstdio>
#include <cstring>

#include "ballrain/npz.h"
#include "ballrain/policy.h"
#include "platform.h"

using namespace ballrain;

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: nc_graph_diff <policy.npz> <manifest.json> <x.npy> [--timing]\n");
        return 2;
    }
    try {
        Policy p = Policy::load(argv[1], Policy::read_manifest(argv[2]));
        Bytes f;
        if (!read_file(argv[3], f)) throw Error("cannot read x");
        NpyArray x = parse_npy(f);
        if (x.descr != "<f4" || x.shape.size() != 2 || x.shape[1] != p.n_obs) throw Error("x: (N, n_obs) float32");
        std::vector<float> all = x.f32();
        bool timing = argc > 4 && std::strcmp(argv[4], "--timing") == 0;
        size_t n = size_t(x.shape[0]), k = size_t(p.n_obs);
        auto t0 = std::chrono::steady_clock::now();
        double sink = 0;
        for (size_t i = 0; i < n; ++i) {
            std::vector<float> row(all.begin() + std::ptrdiff_t(i * k), all.begin() + std::ptrdiff_t((i + 1) * k));
            std::vector<float> lg = p.logits(row);
            float v = p.value(row);
            if (timing) {
                sink += lg[0] + v;
                continue;
            }
            for (float e : lg) std::printf("%.9g ", e);
            std::printf("%.9g\n", v);
        }
        if (timing) {
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("%zu rows in %.3f s: %.1f us per row (logits + value) %g\n", n, s, 1e6 * s / double(n), sink);
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "nc_graph_diff: %s\n", e.what());
        return 1;
    }
    return 0;
}
