// The feature module's fastreach (instead of core/src/fastreach.cpp): the kernel and its fields stay in the host
// (native speed, the fields loaded once per process and shared by every instance), called through two imports of
// module "ballrain":
//   br_host_reach_load(path, path_len, key_out, key_cap) -> fields handle (>= 0; key_out: the fields' key, its length
//       at key_out[-4..]: see below), -1 on an error
//   br_host_reach_perceive(fields, workspace, uzones, n_uzones, ball_type, quarter, pos, vel, out) -> 0 / -1
// Every pointer is into the module's memory; the host checks the bounds. The workspace handle is the module's own
// (one per Workspace object, created by the host on first use).
#include <cstdint>
#include <cstring>
#include <string>

#include "ballrain/fastreach.h"
#include "ballrain/util.h"

#define BR_IMPORT(name) extern "C" __attribute__((import_module("ballrain"), import_name(#name)))

BR_IMPORT(br_host_reach_load) int32_t br_host_reach_load(const char *path, uint32_t path_len, char *key_out,
                                                         uint32_t key_cap, uint32_t *key_len);
BR_IMPORT(br_host_reach_perceive) int32_t br_host_reach_perceive(int32_t fields, int32_t workspace, const double *uzones,
                                                                 uint32_t n_uzones, int32_t ball_type, int32_t quarter,
                                                                 const double *pos, const double *vel, double *out);

namespace ballrain {
namespace fastreach {

struct Fields::Impl {
    int32_t handle = -1;
};
struct Workspace::Impl {
    int32_t handle = -1;
};

static int32_t next_workspace = 0;

std::shared_ptr<const Fields> Fields::load(const std::string &npz_path) {
    std::string key(4096, '\0');
    uint32_t n = 0;
    int32_t h = br_host_reach_load(npz_path.data(), uint32_t(npz_path.size()), &key[0], uint32_t(key.size()), &n);
    if (h < 0) throw Error("the host could not load the reach fields " + npz_path);
    auto f = std::make_shared<Fields>();
    key.resize(n);
    f->key = key;
    auto impl = std::make_shared<Impl>();
    impl->handle = h;
    f->impl = impl;
    return f;
}

Workspace::Workspace() : impl(new Impl()) { impl->handle = next_workspace++; }
Workspace::~Workspace() = default;

void perceive(const Fields &fields, Workspace &ws, const std::vector<double> &uzones, int ball_type, int quarter,
              const double pos[3], const double vel[3], double *out, double *dbg, double *pdd) {
    if (dbg || pdd) throw Error("the module's reach kernel has no debug arrays");
    if (br_host_reach_perceive(fields.impl->handle, ws.impl->handle, uzones.data(), uint32_t(uzones.size()), ball_type,
                               quarter, pos, vel, out) != 0)
        throw Error("the host's reach kernel failed");
}

int n_dbg() { return 0; }
int n_pdd() { return 0; }
int64_t stamp_id(const Workspace &) { return 0; }
int64_t stack_high_water(const Workspace &) { return 0; }
void set_libm(double (*)(double), double (*)(double, double)) {}
bool leaf(const Fields &, Workspace &, const std::string &, const double *, size_t, std::vector<double> &) { return false; }

}  // namespace fastreach
}  // namespace ballrain
