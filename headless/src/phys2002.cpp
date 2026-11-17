// Ballrain hl-win: load the ORIGINAL 2002 physics_RT.dll into the 32-bit headless engine (phys2002.hpp).
#include "phys2002.hpp"

#include "CKAll.h"

#include <windows.h>
#include <bcrypt.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <new>
#include <string>
#include <vector>

#pragma comment(lib, "bcrypt.lib")

#if !defined(_M_IX86)
#error "the 2002 physics_RT.dll is a 32-bit x86 DLL: build the engine for x86"
#endif

namespace {
    const CKGUID kPhysicsManagerGuid(0x6BED328B, 0x141F5148);   // TT_PHYSICS_MANAGER_GUID

    template <class M> void *code_of(M m) {
        static_assert(sizeof(M) == sizeof(void *), "single-inheritance member pointer expected");
        void *p;
        std::memcpy(&p, &m, sizeof(p));
        return p;
    }

    // ---- thiscall shims with the retail (Virtools SDK 2.1) signatures over the Ballanced CK2 / VxMath ----
    struct S_ObjectDeclaration {
        CKObjectDeclaration *me() { return reinterpret_cast<CKObjectDeclaration *>(this); }
        void SetCreationFunction(long (*f)(CKBehaviorPrototype **)) { me()->SetCreationFunction(reinterpret_cast<CKDLL_CREATEPROTOFUNCTION>(f)); }
        void SetVersion(unsigned long v) { me()->SetVersion(v); }
        void SetAuthorName(char *n) { me()->SetAuthorName(n); }
        void SetAuthorGuid(CKGUID g) { me()->SetAuthorGuid(g); }
        void SetGuid(CKGUID g) { me()->SetGuid(g); }
        void SetType(int t) { me()->SetType(t); }
        void SetCategory(char *c) { me()->SetCategory(c); }
        void SetDescription(char *d) { me()->SetDescription(d); }
        void SetCompatibleClassId(long c) { me()->SetCompatibleClassId(c); }
    };
    struct S_BehaviorPrototype {
        CKBehaviorPrototype *me() { return reinterpret_cast<CKBehaviorPrototype *>(this); }
        void SetBehaviorCallbackFct(long (*f)(const CKBehaviorContext &), unsigned long mask, void *arg) {
            me()->SetBehaviorCallbackFct(reinterpret_cast<CKBEHAVIORCALLBACKFCT>(f), mask, arg);
        }
        void SetBehaviorFlags(CK_BEHAVIOR_FLAGS f) { me()->SetBehaviorFlags(f); }
        void SetFunction(int (*f)(const CKBehaviorContext &)) { me()->SetFunction(reinterpret_cast<CKBEHAVIORFCT>(f)); }
        void SetFlags(CK_BEHAVIORPROTOTYPE_FLAGS f) { me()->SetFlags(f); }
        int DeclareLocalParameter(char *n, CKGUID g, char *d) { return me()->DeclareLocalParameter(n, g, static_cast<CKSTRING>(d)); }
        int DeclareSetting(char *n, CKGUID g, char *d) { return me()->DeclareSetting(n, g, static_cast<CKSTRING>(d)); }
    };
    struct S_Behavior {
        CKBehavior *me() { return reinterpret_cast<CKBehavior *>(this); }
        void ActivateInput(int p, int a) { me()->ActivateInput(p, a); }
        CKBeObject *GetTarget() { return me()->GetTarget(); }
        long GetLocalParameterValue(int p, void *b) { return me()->GetLocalParameterValue(p, b); }
        int IsInputActive(int p) { return me()->IsInputActive(p); }
        int AddOutput(char *n) { return me()->AddOutput(n); }
        long DeleteOutput(int p) { return me()->DeleteOutput(p); }
        int GetOutputCount() { return me()->GetOutputCount(); }
        CKBeObject *GetOwner() { return me()->GetOwner(); }
        long SetLocalParameterValue(int p, const void *b, int s) { return me()->SetLocalParameterValue(p, b, s); }
        long GetInputParameterValue(int p, void *b) { return me()->GetInputParameterValue(p, b); }
        void ActivateOutput(int p, int a) { me()->ActivateOutput(p, a); }
        long SetOutputParameterValue(int p, const void *b, int s) { return me()->SetOutputParameterValue(p, b, s); }
        CKObject *GetInputParameterObject(int p) { return me()->GetInputParameterObject(p); }
        void *GetInputParameterReadDataPtr(int p) { return me()->GetInputParameterReadDataPtr(p); }
        CKParameterIn *GetInputParameter(int p) { return me()->GetInputParameter(p); }
        CKParameterIn *CreateInputParameter(char *n, CKGUID g) { return me()->CreateInputParameter(n, g); }
        CKParameterIn *RemoveInputParameter(int p) { return me()->RemoveInputParameter(p); }
        int GetInputParameterCount() { return me()->GetInputParameterCount(); }
        long SetOutputParameterObject(int p, CKObject *o) { return me()->SetOutputParameterObject(p, o); }
        int IsOutputActive(int p) { return me()->IsOutputActive(p); }
    };
    struct S_Context {
        CKContext *me() { return reinterpret_cast<CKContext *>(this); }
        CKBaseManager *GetManagerByGuid(CKGUID g) { return me()->GetManagerByGuid(g); }
        CKAttributeManager *GetAttributeManager() { return me()->GetAttributeManager(); }
        long OutputToConsole(char *s, int beep) { return me()->OutputToConsole(s, beep); }
        long RegisterNewManager(CKBaseManager *m) { return me()->RegisterNewManager(m); }
        int IsReseted() { return me()->IsReseted(); }
        long DestroyObject(CKObject *o, unsigned long f, CKDependencies *d) { return me()->DestroyObject(o, f, d); }
        CKTimeManager *GetTimeManager() { return me()->GetTimeManager(); }
        const XObjectPointerArray &GetObjectListByType(long c, int derived) { return me()->GetObjectListByType(c, derived); }
    };
    struct S_BeObject {
        CKBeObject *me() { return reinterpret_cast<CKBeObject *>(this); }
        CKParameterOut *GetAttributeParameter(int t) { return me()->GetAttributeParameter(t); }
        int GetAttributeType(int i) { return me()->GetAttributeType(i); }
        int GetAttributeCount() { return me()->GetAttributeCount(); }
    };
    struct S_AttributeManager {
        CKAttributeManager *me() { return reinterpret_cast<CKAttributeManager *>(this); }
        int GetAttributeTypeByName(char *n) { return me()->GetAttributeTypeByName(n); }
        char *GetAttributeNameByType(int t) { return const_cast<char *>(static_cast<const char *>(me()->GetAttributeNameByType(t))); }
    };
    struct S_Object {
        void SetName(char *n, int shared) { reinterpret_cast<CKObject *>(this)->SetName(n, shared); }
    };
    struct S_BaseManager {
        CKBaseManager *Construct(CKContext *c, CKGUID g, char *n) { return ::new (static_cast<void *>(this)) CKBaseManager(c, g, n); }
        void Destruct() { reinterpret_cast<CKBaseManager *>(this)->CKBaseManager::~CKBaseManager(); }
    };
    struct S_XString {
        XString &Assign(const char *s) { return *reinterpret_cast<XString *>(this) = s; }
        void Destruct() { reinterpret_cast<XString *>(this)->~XString(); }
    };
    void __cdecl s_CKStoreDeclaration(XObjectDeclarationArray *a, CKObjectDeclaration *d) { CKStoreDeclaration(a, d); }
    CKObjectDeclaration *__cdecl s_CreateCKObjectDeclaration(char *n) { return CreateCKObjectDeclaration(n); }
    CKBehaviorPrototype *__cdecl s_CreateCKBehaviorPrototypeRunTime(char *n) { return CreateCKBehaviorPrototypeRunTime(n); }
    void __cdecl s_CKDestroyObject(CKObject *o, unsigned long f, CKDependencies *d) { CKDestroyObject(o, f, d); }

    std::map<std::string, void *> ck_table() {
        std::map<std::string, void *> t;
        t["?CKStoreDeclaration@@YAXPAV?$XArray@PAVCKObjectDeclaration@@@@PAVCKObjectDeclaration@@@Z"] = reinterpret_cast<void *>(&s_CKStoreDeclaration);
        t["?SetCreationFunction@CKObjectDeclaration@@QAEXP6AJPAPAVCKBehaviorPrototype@@@Z@Z"] = code_of(&S_ObjectDeclaration::SetCreationFunction);
        t["?SetVersion@CKObjectDeclaration@@QAEXK@Z"] = code_of(&S_ObjectDeclaration::SetVersion);
        t["?SetAuthorName@CKObjectDeclaration@@QAEXPAD@Z"] = code_of(&S_ObjectDeclaration::SetAuthorName);
        t["?SetAuthorGuid@CKObjectDeclaration@@QAEXUCKGUID@@@Z"] = code_of(&S_ObjectDeclaration::SetAuthorGuid);
        t["?SetGuid@CKObjectDeclaration@@QAEXUCKGUID@@@Z"] = code_of(&S_ObjectDeclaration::SetGuid);
        t["?SetType@CKObjectDeclaration@@QAEXH@Z"] = code_of(&S_ObjectDeclaration::SetType);
        t["?SetCategory@CKObjectDeclaration@@QAEXPAD@Z"] = code_of(&S_ObjectDeclaration::SetCategory);
        t["?SetDescription@CKObjectDeclaration@@QAEXPAD@Z"] = code_of(&S_ObjectDeclaration::SetDescription);
        t["?CreateCKObjectDeclaration@@YAPAVCKObjectDeclaration@@PAD@Z"] = reinterpret_cast<void *>(&s_CreateCKObjectDeclaration);
        t["?SetBehaviorCallbackFct@CKBehaviorPrototype@@QAEXP6AJABUCKBehaviorContext@@@ZKPAX@Z"] = code_of(&S_BehaviorPrototype::SetBehaviorCallbackFct);
        t["?SetBehaviorFlags@CKBehaviorPrototype@@QAEXW4CK_BEHAVIOR_FLAGS@@@Z"] = code_of(&S_BehaviorPrototype::SetBehaviorFlags);
        t["?SetFunction@CKBehaviorPrototype@@QAEXP6AHABUCKBehaviorContext@@@Z@Z"] = code_of(&S_BehaviorPrototype::SetFunction);
        t["?SetFlags@CKBehaviorPrototype@@QAEXW4CK_BEHAVIORPROTOTYPE_FLAGS@@@Z"] = code_of(&S_BehaviorPrototype::SetFlags);
        t["?DeclareLocalParameter@CKBehaviorPrototype@@QAEHPADUCKGUID@@0@Z"] = code_of(&S_BehaviorPrototype::DeclareLocalParameter);
        t["?DeclareSetting@CKBehaviorPrototype@@QAEHPADUCKGUID@@0@Z"] = code_of(&S_BehaviorPrototype::DeclareSetting);
        t["?CreateCKBehaviorPrototypeRunTime@@YAPAVCKBehaviorPrototype@@PAD@Z"] = reinterpret_cast<void *>(&s_CreateCKBehaviorPrototypeRunTime);
        t["?ActivateInput@CKBehavior@@QAEXHH@Z"] = code_of(&S_Behavior::ActivateInput);
        t["?GetManagerByGuid@CKContext@@QAEPAVCKBaseManager@@UCKGUID@@@Z"] = code_of(&S_Context::GetManagerByGuid);
        t["?GetTarget@CKBehavior@@QAEPAVCKBeObject@@XZ"] = code_of(&S_Behavior::GetTarget);
        t["?GetLocalParameterValue@CKBehavior@@QAEJHPAX@Z"] = code_of(&S_Behavior::GetLocalParameterValue);
        t["?IsInputActive@CKBehavior@@QAEHH@Z"] = code_of(&S_Behavior::IsInputActive);
        t["?AddOutput@CKBehavior@@QAEHPAD@Z"] = code_of(&S_Behavior::AddOutput);
        t["?DeleteOutput@CKBehavior@@QAEJH@Z"] = code_of(&S_Behavior::DeleteOutput);
        t["?GetOutputCount@CKBehavior@@QAEHXZ"] = code_of(&S_Behavior::GetOutputCount);
        t["?GetOwner@CKBehavior@@QAEPAVCKBeObject@@XZ"] = code_of(&S_Behavior::GetOwner);
        t["?SetLocalParameterValue@CKBehavior@@QAEJHPBXH@Z"] = code_of(&S_Behavior::SetLocalParameterValue);
        t["?GetInputParameterValue@CKBehavior@@QAEJHPAX@Z"] = code_of(&S_Behavior::GetInputParameterValue);
        t["?ActivateOutput@CKBehavior@@QAEXHH@Z"] = code_of(&S_Behavior::ActivateOutput);
        t["?GetAttributeParameter@CKBeObject@@QAEPAVCKParameterOut@@H@Z"] = code_of(&S_BeObject::GetAttributeParameter);
        t["?GetAttributeTypeByName@CKAttributeManager@@QAEHPAD@Z"] = code_of(&S_AttributeManager::GetAttributeTypeByName);
        t["?GetAttributeManager@CKContext@@QAEPAVCKAttributeManager@@XZ"] = code_of(&S_Context::GetAttributeManager);
        t["?OutputToConsole@CKContext@@QAEJPADH@Z"] = code_of(&S_Context::OutputToConsole);
        t["?SetOutputParameterValue@CKBehavior@@QAEJHPBXH@Z"] = code_of(&S_Behavior::SetOutputParameterValue);
        t["?GetInputParameterObject@CKBehavior@@QAEPAVCKObject@@H@Z"] = code_of(&S_Behavior::GetInputParameterObject);
        t["?GetInputParameterReadDataPtr@CKBehavior@@QAEPAXH@Z"] = code_of(&S_Behavior::GetInputParameterReadDataPtr);
        t["?GetInputParameter@CKBehavior@@QAEPAVCKParameterIn@@H@Z"] = code_of(&S_Behavior::GetInputParameter);
        t["?CreateInputParameter@CKBehavior@@QAEPAVCKParameterIn@@PADUCKGUID@@@Z"] = code_of(&S_Behavior::CreateInputParameter);
        t["?CKDestroyObject@@YAXPAVCKObject@@KPAVCKDependencies@@@Z"] = reinterpret_cast<void *>(&s_CKDestroyObject);
        t["?RemoveInputParameter@CKBehavior@@QAEPAVCKParameterIn@@H@Z"] = code_of(&S_Behavior::RemoveInputParameter);
        t["?GetInputParameterCount@CKBehavior@@QAEHXZ"] = code_of(&S_Behavior::GetInputParameterCount);
        t["?SetName@CKObject@@QAEXPADH@Z"] = code_of(&S_Object::SetName);
        t["?SetOutputParameterObject@CKBehavior@@QAEJHPAVCKObject@@@Z"] = code_of(&S_Behavior::SetOutputParameterObject);
        t["?IsOutputActive@CKBehavior@@QAEHH@Z"] = code_of(&S_Behavior::IsOutputActive);
        t["??1CKBaseManager@@UAE@XZ"] = code_of(&S_BaseManager::Destruct);
        t["?RegisterNewManager@CKContext@@QAEJPAVCKBaseManager@@@Z"] = code_of(&S_Context::RegisterNewManager);
        t["??0CKBaseManager@@QAE@PAVCKContext@@UCKGUID@@PAD@Z"] = code_of(&S_BaseManager::Construct);
        t["?IsReseted@CKContext@@QAEHXZ"] = code_of(&S_Context::IsReseted);
        t["?DestroyObject@CKContext@@QAEJPAVCKObject@@KPAVCKDependencies@@@Z"] = code_of(&S_Context::DestroyObject);
        t["?GetTimeManager@CKContext@@QAEPAVCKTimeManager@@XZ"] = code_of(&S_Context::GetTimeManager);
        t["?GetAttributeNameByType@CKAttributeManager@@QAEPADH@Z"] = code_of(&S_AttributeManager::GetAttributeNameByType);
        t["?GetAttributeType@CKBeObject@@QAEHH@Z"] = code_of(&S_BeObject::GetAttributeType);
        t["?GetAttributeCount@CKBeObject@@QAEHXZ"] = code_of(&S_BeObject::GetAttributeCount);
        t["?GetObjectListByType@CKContext@@QAEABVXObjectPointerArray@@JH@Z"] = code_of(&S_Context::GetObjectListByType);
        t["?SetCompatibleClassId@CKObjectDeclaration@@QAEXJ@Z"] = code_of(&S_ObjectDeclaration::SetCompatibleClassId);
        return t;
    }


    HMODULE g_module = nullptr;
    HMODULE g_retail_vxmath = nullptr;
    bool g_selected = false;
    std::string g_path;

    std::string sha256_file(const std::string &path) {
        FILE *f = std::fopen(path.c_str(), "rb");
        if (!f) return {};
        std::vector<unsigned char> data;
        unsigned char buf[65536];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) data.insert(data.end(), buf, buf + n);
        std::fclose(f);
        unsigned char digest[32] = {};
        if (BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0, data.data(), static_cast<ULONG>(data.size()), digest, 32) != 0)
            return {};
        char hex[65];
        for (int i = 0; i < 32; ++i) std::snprintf(hex + 2 * i, 3, "%02x", digest[i]);
        return hex;
    }

    // ---- the ball-pair order compare (IVP_Mindist_Manager, the one `pop0->client_data < pop1->client_data`):
    //   100164C1  8B 8D B0 00 00 00   mov ecx, [ebp+0B0h]   ; pop0 (ebp) client_data
    //   100164C7  8B 90 B0 00 00 00   mov edx, [eax+0B0h]   ; pop1 (eax) client_data
    //   100164CD  3B CA               cmp ecx, edx
    //   100164CF  73 06               jae 100164D7          ; else the two synapses swap roles
    // The 14 bytes at 164C1 become `call thunk` + 9 nops; the thunk sets the flags of cmp key(pop0), key(pop1) and
    // keeps every register the original kept (ecx / edx are dead after the jae on both paths).
    constexpr uint32_t kHookRva = 0x164C1;
    const unsigned char kHookOriginal[16] = {0x8B, 0x8D, 0xB0, 0x00, 0x00, 0x00, 0x8B, 0x90,
                                             0xB0, 0x00, 0x00, 0x00, 0x3B, 0xCA, 0x73, 0x06};
    ballrain::phys2002::ball_key_fn g_key = nullptr;
    bool g_hooked = false;
}

extern "C" unsigned int __cdecl ballrain_phys2002_key(void *client_data) {
    return g_key ? g_key(client_data) : ballrain::phys2002::ball_key_ckid(client_data);
}

namespace {
    __declspec(naked) void ball_compare_thunk() {
        __asm {
            push eax
            push dword ptr [ebp + 0B0h]
            call ballrain_phys2002_key
            add esp, 4
            push eax
            mov eax, dword ptr [esp + 4]
            push dword ptr [eax + 0B0h]
            call ballrain_phys2002_key
            add esp, 4
            mov edx, eax
            pop ecx
            pop eax
            cmp ecx, edx
            ret
        }
    }

    bool install_ball_hook(unsigned char *base, std::string &error) {
        unsigned char *site = base + kHookRva;
        if (std::memcmp(site, kHookOriginal, sizeof(kHookOriginal)) != 0) {
            error = "phys2002: unexpected code at the ball-pair compare (RVA 0x164C1)";
            return false;
        }
        unsigned char code[14];
        code[0] = 0xE8;
        const int32_t rel = static_cast<int32_t>(reinterpret_cast<intptr_t>(&ball_compare_thunk) -
                                                 reinterpret_cast<intptr_t>(site + 5));
        std::memcpy(code + 1, &rel, 4);
        std::memset(code + 5, 0x90, sizeof(code) - 5);
        DWORD old = 0;
        if (!VirtualProtect(site, sizeof(code), PAGE_EXECUTE_READWRITE, &old)) {
            error = "phys2002: VirtualProtect failed at the ball-pair compare";
            return false;
        }
        std::memcpy(site, code, sizeof(code));
        VirtualProtect(site, sizeof(code), old, &old);
        FlushInstructionCache(GetCurrentProcess(), site, sizeof(code));
        g_hooked = true;
        return true;
    }
}

extern "C" int g_ballrain_phys2002_active = 0;   // CKIpionManager::GetManager returns NULL while set

namespace ballrain::phys2002 {
    void select(const std::string &path) {
        g_path = path;
        g_selected = true;
    }

    const std::string &requested_path() {
        if (!g_selected) {
            const char *v = std::getenv("BALLRAIN_PHYS2002");
            g_path = v ? v : "";
            g_selected = true;
        }
        return g_path;
    }

    bool loaded() { return g_module != nullptr; }
    unsigned char *module_base() { return reinterpret_cast<unsigned char *>(g_module); }
    void set_ball_key(ball_key_fn key) { g_key = key; }
    bool ball_hook_installed() { return g_hooked; }
    unsigned int ball_key_ckid(void *client_data) {
        return client_data ? static_cast<unsigned int>(static_cast<CKObject *>(client_data)->GetID()) : 0u;
    }

    static std::string vxmath_path() {
        if (const char *v = std::getenv("BALLRAIN_PHYS2002_VXMATH")) return v;
        const std::string p = requested_path();
        const size_t slash = p.find_last_of("/\\");
        const std::string dir = slash == std::string::npos ? std::string(".") : p.substr(0, slash);
        return dir + "\\..\\Bin\\VxMath.dll";
    }

    bool load(plugin &out, std::string &error) {
        if (g_module) {
            out.count = reinterpret_cast<int (*)()>(GetProcAddress(g_module, "CKGetPluginInfoCount"));
            out.info = reinterpret_cast<CKPluginInfo *(*)(int)>(GetProcAddress(g_module, "CKGetPluginInfo"));
            out.declarations = reinterpret_cast<void (*)(XObjectDeclarationArray *)>(GetProcAddress(g_module, "RegisterBehaviorDeclarations"));
            if (!out.count || !out.info || !out.declarations) { error = "phys2002: plugin exports missing"; return false; }
            return true;
        }
        const std::string path = requested_path();
        g_ballrain_phys2002_active = 1;
        const std::string hash = sha256_file(path);
        if (hash != kSha256) {
            error = "phys2002: " + path + " has sha256 " + (hash.empty() ? "<unreadable>" : hash) + ", want " + kSha256;
            return false;
        }
        const std::string vx = vxmath_path();
        g_retail_vxmath = LoadLibraryExA(vx.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        void *from_matrix = g_retail_vxmath
            ? reinterpret_cast<void *>(GetProcAddress(g_retail_vxmath, "?FromMatrix@VxQuaternion@@QAEXABVVxMatrix@@HH@Z")) : nullptr;
        if (!from_matrix) {
            error = "phys2002: no retail VxQuaternion::FromMatrix in " + vx + " (error " + std::to_string(GetLastError()) + ")";
            return false;
        }
        HMODULE module = LoadLibraryExA(path.c_str(), nullptr, DONT_RESOLVE_DLL_REFERENCES);
        if (!module) {
            error = "phys2002: LoadLibraryEx failed for " + path + " (error " + std::to_string(GetLastError()) + ")";
            return false;
        }
        auto *base = reinterpret_cast<unsigned char *>(module);
        auto *dos = reinterpret_cast<IMAGE_DOS_HEADER *>(base);
        auto *nt = reinterpret_cast<IMAGE_NT_HEADERS32 *>(base + dos->e_lfanew);
        const auto &dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
        const auto table = ck_table();
        int filled = 0;
        for (auto *imp = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR *>(base + dir.VirtualAddress); imp->Name; ++imp) {
            const char *dll = reinterpret_cast<const char *>(base + imp->Name);
            const bool ck2 = _stricmp(dll, "CK2.dll") == 0, vxmath = _stricmp(dll, "VxMath.dll") == 0;
            HMODULE system = nullptr;
            if (!ck2 && !vxmath) {
                system = LoadLibraryA(dll);
                if (!system) { error = std::string("phys2002: cannot load ") + dll; return false; }
            }
            auto *names = reinterpret_cast<IMAGE_THUNK_DATA32 *>(base + (imp->OriginalFirstThunk ? imp->OriginalFirstThunk : imp->FirstThunk));
            auto *slots = reinterpret_cast<IMAGE_THUNK_DATA32 *>(base + imp->FirstThunk);
            for (; names->u1.AddressOfData; ++names, ++slots) {
                if (IMAGE_SNAP_BY_ORDINAL32(names->u1.Ordinal)) { error = std::string("phys2002: ordinal import from ") + dll; return false; }
                const char *name = reinterpret_cast<const char *>(reinterpret_cast<IMAGE_IMPORT_BY_NAME *>(base + names->u1.AddressOfData)->Name);
                void *target = nullptr;
                if (ck2) {
                    auto it = table.find(name);
                    if (it != table.end()) target = it->second;
                } else if (vxmath) {
                    if (std::strcmp(name, "?FromMatrix@VxQuaternion@@QAEXABVVxMatrix@@HH@Z") == 0) target = from_matrix;
                    else if (std::strcmp(name, "??4XString@@QAEAAV0@PBD@Z") == 0) target = code_of(&S_XString::Assign);
                    else if (std::strcmp(name, "??1XString@@QAE@XZ") == 0) target = code_of(&S_XString::Destruct);
                } else {
                    target = reinterpret_cast<void *>(GetProcAddress(system, name));
                }
                if (!target) { error = std::string("phys2002: unresolved import ") + dll + "!" + name; return false; }
                DWORD old = 0;
                VirtualProtect(&slots->u1.Function, sizeof(DWORD), PAGE_READWRITE, &old);
                slots->u1.Function = static_cast<DWORD>(reinterpret_cast<uintptr_t>(target));
                VirtualProtect(&slots->u1.Function, sizeof(DWORD), old, &old);
                ++filled;
            }
        }
        const char *hook = std::getenv("BALLRAIN_HL_PHYS2002_HOOK");
        if (!(hook && std::strcmp(hook, "0") == 0) && !install_ball_hook(base, error))
            return false;
        FlushInstructionCache(GetCurrentProcess(), base, nt->OptionalHeader.SizeOfImage);
        using dll_main = BOOL(WINAPI *)(HINSTANCE, DWORD, LPVOID);
        auto entry = reinterpret_cast<dll_main>(base + nt->OptionalHeader.AddressOfEntryPoint);
        if (!entry(module, DLL_PROCESS_ATTACH, nullptr)) { error = "phys2002: DllMain failed"; return false; }
        g_module = module;
        std::fprintf(stderr, "phys2002: %s (sha256 %.16s) mapped at %p, %d imports filled, FromMatrix from %s, "
                     "ball-pair compare %s\n", path.c_str(), hash.c_str(), base, filled, vx.c_str(),
                     g_hooked ? "hooked (key: CK id unless the host sets one)" : "original (heap address order)");
        return load(out, error);
    }

    unsigned char *manager(CKContext *context) {
        return context ? reinterpret_cast<unsigned char *>(context->GetManagerByGuid(kPhysicsManagerGuid)) : nullptr;
    }
    unsigned *rand_seed() { return g_module ? reinterpret_cast<unsigned *>(module_base() + 0x685B4) : nullptr; }
    unsigned char *environment(CKContext *context) {
        unsigned char *m = manager(context);
        return m ? *reinterpret_cast<unsigned char **>(m + 0xC0) : nullptr;
    }

    bool reset_clock(CKContext *context, float smoothed_delta, int movement_check, unsigned seed, std::string &error) {
        unsigned char *mgr = manager(context);
        unsigned char *env = environment(context);
        unsigned *rs = rand_seed();
        if (!mgr || !env || !rs) {
            error = "phys2002: no manager / environment";
            return false;
        }
        *reinterpret_cast<double *>(*reinterpret_cast<unsigned char **>(env + 0x4) + 0x18) = 0;   // time_manager->base_time
        *reinterpret_cast<double *>(env + 0x120) = 0;                                           // current_time
        double &time_of_last_psi = *reinterpret_cast<double *>(env + 0x130);
        time_of_last_psi = 0;
        *reinterpret_cast<double *>(env + 0x128) = time_of_last_psi + 1.0 / 66;                 // time_of_next_psi
        if (movement_check >= 0) *reinterpret_cast<short *>(env + 0x140) = static_cast<short>(movement_check);
        *rs = seed;
        *reinterpret_cast<float *>(mgr + 0xC8) = smoothed_delta;                                // smoothed delta
        return true;
    }
}
