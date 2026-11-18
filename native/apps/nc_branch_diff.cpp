// nc_branch_diff: the BranchFollower on a recorded step sequence (tools/native/branch_diff.py compares it with
// ballrain/branch.py record by record).
//   nc_branch_diff --main <L<nn>.npz> --branch-dir <dir> --level <n> --in <steps.bin> --out <out.bin>
// steps.bin: records f64 x, y, z; i32 sector, ball_type, reset (1: reset() before the update)
// out.bin: records f64 s, local_s, ref_elapsed_ms, ref_remaining_ms, dist, point[3], tangent[3]; i32 branch, active
// (0 main, 1 alt); i64 seg
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ballrain/branch.h"
#include "ballrain/path.h"
#include "ballrain/sha256.h"

using namespace ballrain;

static std::string arg(int argc, char **argv, const char *name) {
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], name)) return argv[i + 1];
    std::fprintf(stderr, "missing %s\n", name);
    std::exit(2);
}

#pragma pack(push, 1)
struct In {
    double p[3];
    int32_t sector, bt, reset;
};
struct Out {
    double s, local_s, elapsed, remaining, dist, point[3], tangent[3];
    int32_t branch, active;
    int64_t seg;
};
#pragma pack(pop)

int main(int argc, char **argv) {
    try {
        std::string mp = arg(argc, argv, "--main");
        auto main_path = ReferencePath::load(mp);
        auto f = load_branch(main_path, std::stoi(arg(argc, argv, "--level")), arg(argc, argv, "--branch-dir"),
                             file_sha256(mp), 10, 30, 20, 5);
        FILE *fi = std::fopen(arg(argc, argv, "--in").c_str(), "rb");
        FILE *fo = std::fopen(arg(argc, argv, "--out").c_str(), "wb");
        if (!fi || !fo) throw std::runtime_error("cannot open in/out");
        In r;
        size_t n = 0;
        while (std::fread(&r, sizeof r, 1, fi) == 1) {
            if (r.reset) f->reset();
            BranchStep st = f->update(V3{r.p[0], r.p[1], r.p[2]}, r.sector, r.bt);
            Out o{};
            o.s = st.pr.s, o.local_s = st.local_s, o.elapsed = st.ref_elapsed_ms, o.remaining = st.ref_remaining_ms;
            o.dist = st.pr.dist;
            for (int k = 0; k < 3; ++k) o.point[k] = st.pr.point[k], o.tangent[k] = st.pr.tangent[k];
            o.branch = st.branch, o.active = st.path == f->alt.get(), o.seg = st.pr.seg;
            std::fwrite(&o, sizeof o, 1, fo);
            ++n;
        }
        std::fclose(fi), std::fclose(fo);
        std::printf("%zu steps\n", n);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
