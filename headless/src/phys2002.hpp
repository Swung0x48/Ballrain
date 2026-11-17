#pragma once
// Ballrain hl-win: the ORIGINAL 2002 physics_RT.dll (sha256 e72e4afc...) inside the 32-bit headless engine
// (Windows x86 only; patches/0003-hlwin-phys2002.diff builds it into BallanceMMOSim and defines BALLRAIN_PHYS2002_HOST).
//
// The DLL file is never modified: it is mapped with LoadLibraryEx(DONT_RESOLVE_DLL_REFERENCES) after its sha256 is
// checked, and its import table is filled by hand: CK2.dll / VxMath.dll names resolve to thiscall shims over the
// statically linked Ballanced CK2 / VxMath (the retail decorated names differ only in char* vs const char*, long vs
// int: the same x86 ABI), VxQuaternion::FromMatrix to the retail VxMath.dll's own export, MSVCRT / KERNEL32 to the
// system DLLs (as in Player.exe). Then its DllMain runs and its three plugin exports are registered as the
// "physics_RT" static plugin in place of Ballanced's source build.
//
// One change to the mapped image (not to the file), unless BALLRAIN_HL_PHYS2002_HOOK=0: the single ball-pair order
// compare (IVP_Mindist_Manager, RVA 0x164C1: `client_data0 < client_data1`, the CK3dEntity heap addresses) is
// redirected to a key function, so the contact order of two balls follows a key (default: the CK id, as the product
// build's IVP_Environment::bmmo_ball_order_key) instead of this process's heap layout. The original bytes are
// checked before the patch. With a key of the address itself the compare is the original one.
//
// Which DLL: select() before the engine is created (the agent mode: BALLRAIN_HL_PHYS2002 = path | 0, default
// <root>/BuildingBlocks/physics_RT.dll); otherwise the environment variable BALLRAIN_PHYS2002 (SimTool --tas).
// The retail VxMath.dll for FromMatrix: BALLRAIN_PHYS2002_VXMATH, default <dir of the DLL>/../Bin/VxMath.dll.

#include <string>

class CKPluginInfo;
class CKObjectDeclaration;
template <class T> class XArray;
typedef XArray<CKObjectDeclaration *> XObjectDeclarationArray;
class CKContext;

namespace ballrain::phys2002 {
    constexpr const char *kSha256 = "e72e4afcfa5c33a7d3d27776137f8c997b3c52d89d8a8a4745f1ca21e45893ec";

    struct plugin {
        int (*count)() = nullptr;
        CKPluginInfo *(*info)(int) = nullptr;
        void (*declarations)(XObjectDeclarationArray *) = nullptr;
    };

    // The DLL to load ("" = the source physics). select() overrides the environment; call it before the engine.
    void select(const std::string &path);
    const std::string &requested_path();
    inline bool requested() { return !requested_path().empty(); }

    // Load (once per process). Fails on a wrong hash, a missing import, unexpected code at the hook or DllMain.
    bool load(plugin &out, std::string &error);
    bool loaded();
    unsigned char *module_base();

    // The ball-pair order key (smaller key = first of the pair). nullptr = the CK id. Called from inside the DLL's
    // simulation with the CK3dEntity of each ball.
    using ball_key_fn = unsigned int (*)(void *client_data);
    void set_ball_key(ball_key_fn key);
    bool ball_hook_installed();
    unsigned int ball_key_ckid(void *client_data);

    // Raw state of the 2002 DLL (BallrainIO Ivp2002.h, TASSupport offsets).
    unsigned char *manager(CKContext *context);     // the DLL's CKIpionManager (TT_PHYSICS_MANAGER_GUID)
    unsigned char *environment(CKContext *context); // *(mgr + 0xC0)
    unsigned *rand_seed();                          // IVP_RAND_SEED, RVA 0x685B4

    // TASSupport OnStart's clock reset on the DLL's memory (BallrainIO ResetPhysicsTime) plus the IVP state
    // Player.exe has at that point: next_movement_check (IVP_Environment +0x140, i16; < 0 keeps it) and the seed.
    bool reset_clock(CKContext *context, float smoothed_delta, int movement_check, unsigned seed, std::string &error);
}
