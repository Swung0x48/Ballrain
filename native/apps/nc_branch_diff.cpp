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
#include "ballrain/pool_client.h"
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
    int32_t member, phase;  // wire v2: the active member's manifest index and phase (-1 on the main line / v1)
};
#pragma pack(pop)

int main(int argc, char **argv) {
    try {
        for (int i = 1; i + 1 < argc; ++i)  // --install-pack P --pin SHA --cache D: install_branch_pack + check_branch_dir
            if (std::strcmp(argv[i], "--install-pack") == 0) {
                FeatureConfig fc;
                fc.branch_sha256 = arg(argc, argv, "--pin");
                fc.path_sha256 = file_sha256(arg(argc, argv, "--main"));
                std::string pack = argv[i + 1];
                std::string d = install_branch_pack(pack, arg(argc, argv, "--cache"), std::stoi(arg(argc, argv, "--level")),
                                                    file_sha256(pack), fc);
                check_branch_dir(d, std::stoi(arg(argc, argv, "--level")), fc);
                std::printf("installed %s\n", d.c_str());
                return 0;
            }
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
            auto *multi = dynamic_cast<MultiFollower *>(f.get());
            auto *one = dynamic_cast<BranchFollower *>(f.get());
            o.branch = st.branch, o.active = one ? st.path == one->alt.get() : st.branch, o.seg = st.pr.seg;
            o.member = multi && !st.branch_id.empty() ? multi->index_of(st.branch_id) : -1;
            o.phase = st.phase;
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
