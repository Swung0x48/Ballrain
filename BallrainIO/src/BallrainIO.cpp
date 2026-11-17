#include "BallrainIO.h"

#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <float.h>
#include <algorithm>
#include <fstream>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <bcrypt.h>

#include "TASHook.h"

// Reference = reference/TASSupport/TASSupport.cpp (patched 0.3.12, LegacyMode 1). Line numbers cite that file.
//
// Contract (environment, read once at OnLoad):
//   BALLRAIN_MODE=replay       BALLRAIN_MODE=agent: agent mode, see docs/agent-protocol.md and Agent.cpp
//                              (same injection path; frames come from Python). Anything else: inert
//   BALLRAIN_FRAMES=<path>     headerless 8-byte frames (the uncompressed payload of a legacy .tas)
//   BALLRAIN_LEVEL=<1..13>     auto-load this level after the start menu
//   BALLRAIN_TRACE=<path>      per-frame ball trace, the reference's BMMO_TAS_TRACE line format (no core section)
//   BALLRAIN_FRAMELOG=<path>   the frames actually injected, same 8-byte format
//   BALLRAIN_EVENTS=<path>     "<frameIndex> <EventName>" lines and a final Summary line
//   BALLRAIN_SKIP_RENDER=1     skip rendering while a run plays
//   BALLRAIN_EXIT=1            exit the game when the (last) run ends
//   BALLRAIN_LOADS=<n>         load + replay n times in one process; outputs of load k>1 get ".k" appended.
//                              NOT trajectory-exact after a full level: use one process per episode for
//                              exact runs (see "Same-process reload" below; LOADS=1 never runs that code)
//   BALLRAIN_DEBUG=<path>      debug log (+ .dump/.dumpS/.act files); changes callback registration time
//   BALLRAIN_KEEP_FOCUS=1      do not hand the foreground back (any mode, also inert; see ReturnForeground)
//
// Same-process reload (LOADS > 1). After "Exit Level" the warm process differs from a cold one in ways
// that change the next load. Written back to the cold process's values, in this order:
//   - the menu waits until the behavior manager runs the same objects as at the cold autoload (the
//     previous level's Ball_ResetPieces_* fade keeps All_Balls in it for ~1.9 s), then puts them in the
//     cold order (equal-priority objects are qsorted, so their order depends on the list's history:
//     with All_Sound before All_Levelinit, Execute Script(Music_Init) ends one frame early and the
//     whole load, StartLevel included, is one frame short);
//   - scripts idle at the cold autoload are reset, the Gameplay_* ones also get their parameter values
//     back, and IngameParameter/Energy/LifeBalls their contents;
//   - at PreLoadLevel: the physics time factor (the level leaves 0.002, cold is 0.001: the physics clock
//     ran twice as fast during the load), IVP_RAND_SEED and Gameplay_Ingame's ActiveBall (cold: none,
//     so no trace line is written before this load's ball is set).
// With this a reload after a 3000-frame SR06 run replays byte-identically, but after a finished SR06
// level the trace still drifts at frame 7323 (shared IVP environment; see the M0 report).

// SHA-256 of the 2002 physics_RT.dll (the reference environment's BuildingBlocks/physics_RT.dll).
// The raw offsets in ResetPhysicsTime are valid for this exact file only.
static const char kPhysicsSha256[] = "e72e4afcfa5c33a7d3d27776137f8c997b3c52d89d8a8a4745f1ca21e45893ec";

MOD_EXPORT IMod *BMLEntry(IBML *bml) { return new BallrainIO(bml); }
MOD_EXPORT void BMLExit(IMod *mod) { delete mod; }

static unsigned FpuCw() {
    unsigned short cw = 0;
    __asm fnstcw cw;
    return cw;
}

void BallrainIO::Dbg(const char *fmt, ...) {
    if (!m_Dbg)
        return;
    std::fprintf(m_Dbg, "t%u ", m_TimeManager ? (unsigned) m_TimeManager->GetMainTickCount() : 0u);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(m_Dbg, fmt, ap);
    va_end(ap);
    std::fputc('\n', m_Dbg);
    std::fflush(m_Dbg);
}

static std::string GetEnv(const char *name) {
    const char *v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

static bool Sha256File(const wchar_t *path, std::string &hex) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        return false;
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    bool ok = BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) == 0 &&
              BCryptCreateHash(alg, &hash, nullptr, 0, nullptr, 0, 0) == 0;
    std::vector<char> buf(1 << 16);
    while (ok && file) {
        file.read(buf.data(), (std::streamsize) buf.size());
        const auto n = (ULONG) file.gcount();
        if (n > 0)
            ok = BCryptHashData(hash, reinterpret_cast<PUCHAR>(buf.data()), n, 0) == 0;
    }
    unsigned char digest[32] = {};
    ok = ok && file.eof() && BCryptFinishHash(hash, digest, sizeof(digest), 0) == 0;
    if (hash)
        BCryptDestroyHash(hash);
    if (alg)
        BCryptCloseAlgorithmProvider(alg, 0);
    if (!ok)
        return false;
    static const char digits[] = "0123456789abcdef";
    hex.clear();
    for (unsigned char b : digest) {
        hex += digits[b >> 4];
        hex += digits[b & 15];
    }
    return true;
}

// ---------------------------------------------------------------- foreground

// Player.exe activates its (hidden) main window at start-up, before any mod is loaded, and Windows lets
// the new process take the foreground, so the keyboard goes to an invisible window. The process holding
// the foreground may always give it away: when this process owns the foreground window and that window
// is hidden (instances.launch_direct starts Player.exe with SW_HIDE), hand it to the
// first window in z-order (top first) that the user could have been using: visible, not minimised, not
// cloaked, not a tool / no-activate / click-through window, a top-level window not owned by another, of
// another process. Window management only: no game object, input or timing state is read or written.
static bool IsCloaked(HWND hwnd) {
    using DwmGetWindowAttributeFn = HRESULT(WINAPI *)(HWND, DWORD, PVOID, DWORD);
    static DwmGetWindowAttributeFn fn = [] {
        HMODULE dwm = ::LoadLibraryW(L"dwmapi.dll");
        return dwm ? (DwmGetWindowAttributeFn) ::GetProcAddress(dwm, "DwmGetWindowAttribute") : nullptr;
    }();
    DWORD cloaked = 0;
    return fn && fn(hwnd, 14 /* DWMWA_CLOAKED */, &cloaked, sizeof(cloaked)) == S_OK && cloaked != 0;
}

static HWND PickForegroundTarget() {
    const DWORD self = ::GetCurrentProcessId();
    int n = 0;
    for (HWND h = ::GetTopWindow(nullptr); h && n < 4096; h = ::GetWindow(h, GW_HWNDNEXT), ++n) {
        if (!::IsWindowVisible(h) || ::IsIconic(h))
            continue;
        DWORD pid = 0;
        ::GetWindowThreadProcessId(h, &pid);
        if (pid == self || pid == 0)
            continue;
        const LONG ex = ::GetWindowLongW(h, GWL_EXSTYLE);
        if (ex & (WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_TRANSPARENT))
            continue;
        if (::GetWindow(h, GW_OWNER) && !(ex & WS_EX_APPWINDOW))
            continue;
        RECT r = {};
        if (!::GetWindowRect(h, &r) || r.right <= r.left || r.bottom <= r.top || IsCloaked(h))
            continue;
        return h;
    }
    return nullptr;
}

static std::string ProcessImage(DWORD pid) {
    char buf[MAX_PATH] = {};
    DWORD size = MAX_PATH;
    HANDLE h = ::OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!h)
        return "?";
    const BOOL ok = ::QueryFullProcessImageNameA(h, 0, buf, &size);
    ::CloseHandle(h);
    return ok ? std::string(buf, size) : std::string("?");
}

void BallrainIO::ReturnForeground(const char *where) {
    if (m_KeepFocus)
        return;
    HWND fg = ::GetForegroundWindow();
    DWORD fgPid = 0;
    if (!fg || !::GetWindowThreadProcessId(fg, &fgPid) || fgPid != ::GetCurrentProcessId())
        return;
    if (::IsWindowVisible(fg))
        return;   // a visible game window (started by hand, BALLRAIN_VISIBLE=1): the user may have clicked into it
    HWND target = PickForegroundTarget();
    if (!target)
        target = ::GetShellWindow();
    DWORD targetPid = 0;
    if (target)
        ::GetWindowThreadProcessId(target, &targetPid);
    const BOOL ok = target && ::SetForegroundWindow(target);
    HWND now = ::GetForegroundWindow();
    DWORD nowPid = 0;
    if (now)
        ::GetWindowThreadProcessId(now, &nowPid);
    GetLogger()->Info("focus (%s): foreground handed to pid %u %s: %s, foreground now pid %u", where,
                      (unsigned) targetPid, ProcessImage(targetPid).c_str(), ok ? "ok" : "failed", (unsigned) nowPid);
}

// ---------------------------------------------------------------- loading

bool BallrainIO::ReadEnvironment() {
    if (GetEnv("BALLRAIN_MODE") == "agent") {
        m_Mode = Mode::Agent;
        if (AgentReadEnvironment())
            return true;
        m_Mode = Mode::Inactive;
        return false;
    }
    if (GetEnv("BALLRAIN_MODE") != "replay") {
        GetLogger()->Info("BALLRAIN_MODE is not 'replay' or 'agent': inactive");
        return false;
    }
    m_Mode = Mode::Replay;
    m_FramesPath = GetEnv("BALLRAIN_FRAMES");
    m_TracePath = GetEnv("BALLRAIN_TRACE");
    m_FrameLogPath = GetEnv("BALLRAIN_FRAMELOG");
    m_EventsPath = GetEnv("BALLRAIN_EVENTS");
    m_Level = std::atoi(GetEnv("BALLRAIN_LEVEL").c_str());
    m_SkipRender = GetEnv("BALLRAIN_SKIP_RENDER") == "1";
    m_ExitWhenDone = GetEnv("BALLRAIN_EXIT") == "1";
    const std::string loads = GetEnv("BALLRAIN_LOADS");
    m_Loads = loads.empty() ? 1 : std::atoi(loads.c_str());
    if (m_Loads < 1) {
        GetLogger()->Error("BALLRAIN_LOADS=%s is invalid", loads.c_str());
        return false;
    }
    if (m_Level != 0 && (m_Level < 1 || m_Level > 13)) {
        GetLogger()->Error("BALLRAIN_LEVEL=%d is out of 1..13", m_Level);
        return false;
    }

    if (m_FramesPath.empty()) {
        GetLogger()->Error("BALLRAIN_FRAMES is not set");
        return false;
    }
    std::ifstream file(m_FramesPath, std::ios::binary | std::ios::ate);
    if (!file) {
        GetLogger()->Error("cannot open BALLRAIN_FRAMES %s", m_FramesPath.c_str());
        return false;
    }
    const auto size = (size_t) file.tellg();
    file.seekg(0);
    // As the reference's legacy Load: frameCount = length / 8, trailing bytes ignored (TASRecord.cpp:476).
    // A zero-frame input is accepted: the run stops at its first Time callback, as the reference's
    // OnPreProcessTime does when the record is not playing (TASSupport.cpp:506-513).
    m_Frames.resize(size / sizeof(Frame));
    if (!m_Frames.empty() && !file.read(reinterpret_cast<char *>(m_Frames.data()),
                                        (std::streamsize) (m_Frames.size() * sizeof(Frame)))) {
        GetLogger()->Error("cannot read BALLRAIN_FRAMES %s", m_FramesPath.c_str());
        m_Frames.clear();
        return false;
    }
    if (size % sizeof(Frame))
        GetLogger()->Warn("BALLRAIN_FRAMES: %u trailing bytes ignored", (unsigned) (size % sizeof(Frame)));
    GetLogger()->Info("replay: %u frames from %s, level %d, loads %d", (unsigned) m_Frames.size(),
                      m_FramesPath.c_str(), m_Level, m_Loads);
    return true;
}

void BallrainIO::OnLoad() {
    m_KeepFocus = GetEnv("BALLRAIN_KEEP_FOCUS") == "1";
    ReturnForeground("OnLoad");
    m_Active = ReadEnvironment();
    if (!m_Active)
        return;
    VerifyPhysicsDll(false);   // refuses (m_Active = false) on a wrong DLL; retried at PostStartMenu / start
    if (!m_Active)
        return;

    // Reference OnLoad (:91-100) + InitHooks (:667-679) with legacy mode: only the two PreProcess hooks.
    m_IpionManager = m_BML->GetCKContext()->GetManagerByGuid(CKGUID(0x6bed328b, 0x141f5148));
    m_TimeManager = m_BML->GetTimeManager();
    m_InputHook = m_BML->GetInputManager();

    const MH_STATUS status = MH_Initialize();
    if (status != MH_OK && status != MH_ERROR_ALREADY_INITIALIZED) {
        GetLogger()->Error("Failed to initialize MinHook: %s", MH_StatusToString(status));
        m_Active = false;
        return;
    }
    auto *inputManager = (CKInputManager *) m_BML->GetCKContext()->GetManagerByGuid(INPUT_MANAGER_GUID);
    if (!m_IpionManager || !m_TimeManager || !m_InputHook || !inputManager ||
        !CKTimeManagerHook::Enable(m_TimeManager) || !CKInputManagerHook::Enable(inputManager)) {
        GetLogger()->Error("managers or PreProcess hooks unavailable: inactive");
        m_Active = false;
    }
    if (m_Mode == Mode::Agent) {
        if (!m_Active)
            AgentFail(3, "agent: managers or PreProcess hooks unavailable");
        else if (m_PhysicsOk > 0)
            AgentHello();
    }
    const std::string dbg = GetEnv("BALLRAIN_DEBUG");
    if (m_Active && !dbg.empty()) {
        m_Dbg = std::fopen(dbg.c_str(), "w");
        CKTimeManagerHook::AddPostCallback([this](CKBaseManager *) { OnPreProcessTime(); });
        CKInputManagerHook::AddPostCallback([this](CKBaseManager *) { OnPreProcessInput(); });
        m_CallbacksRegistered = true;
    }
}

void BallrainIO::OnUnload() {
    if (m_Playing)
        CloseOutputs(0);
    FreeColdState();
    CKTimeManagerHook::Disable();
    CKInputManagerHook::Disable();
    MH_Uninitialize();
    if (m_LevelInc)
        std::fclose(m_LevelInc);
    m_LevelInc = nullptr;
    for (void *h : {m_EvObs, m_EvAct, m_ParentProc})
        if (h)
            ::CloseHandle(h);
    m_EvObs = m_EvAct = m_ParentProc = nullptr;
    if (m_Shm)
        ::UnmapViewOfFile(m_Shm);
    if (m_ShmMap)
        ::CloseHandle(m_ShmMap);
    m_Shm = nullptr;
    m_ShmMap = nullptr;
}

bool BallrainIO::VerifyPhysicsDll(bool mustBeLoaded) {
    if (m_PhysicsOk != 0)
        return m_PhysicsOk > 0;
    HMODULE module = ::GetModuleHandleA("physics_RT.dll");
    if (!module && !mustBeLoaded)
        return false;   // not loaded yet: checked again before any run
    m_PhysicsOk = -1;
    wchar_t path[MAX_PATH] = {};
    std::string hex;
    if (!module || !::GetModuleFileNameW(module, path, MAX_PATH) || !Sha256File(path, hex)) {
        GetLogger()->Error("physics_RT.dll is not loaded or cannot be hashed: refusing to run");
        m_Active = false;
        if (m_Mode == Mode::Agent)
            AgentFail(3, "agent: physics_RT.dll is not loaded or cannot be hashed");
        return false;
    }
    if (hex != kPhysicsSha256) {
        GetLogger()->Error("physics_RT.dll sha256 %s is not the 2002 DLL %s: refusing to run", hex.c_str(),
                           kPhysicsSha256);
        m_Active = false;
        if (m_Mode == Mode::Agent)
            AgentFail(3, "agent: physics_RT.dll is not the 2002 DLL");
        return false;
    }
    GetLogger()->Info("physics_RT.dll is the 2002 DLL (sha256 %.8s...)", hex.c_str());
    m_PhysicsHex = hex;
    m_PhysicsOk = 1;
    return true;
}

void BallrainIO::OnLoadObject(const char *filename, CKBOOL isMap, const char *masterName, CK_CLASSID filterClass,
                              CKBOOL addToScene, CKBOOL reuseMeshes, CKBOOL reuseMaterials, CKBOOL dynamic,
                              XObjectArray *objArray, CKObject *masterObj) {
    if (!m_Active)
        return;
    // Reference :125-136.
    Dbg("LoadObject %s isMap=%d", filename, (int) isMap);
    if (!strcmp(filename, "3D Entities\\Gameplay.nmo"))
        m_CurLevel = m_BML->GetArrayByName("CurrentLevel");
    if (!strcmp(filename, "3D Entities\\Menu.nmo")) {
        m_Keyboard = m_BML->GetArrayByName("Keyboard");
        m_ExitMain = ScriptHelper::FindFirstBB(m_BML->GetScriptByName("Menu_Main"), "Exit", false, 1, 0);
    }
}

void BallrainIO::OnLoadScript(const char *filename, CKBehavior *script) {
    if (!m_Active)
        return;
    // Reference :145-165, legacy branch.
    const char *name = script->GetName();
    if (!strcmp(name, "Ball_Explosion_Wood") || !strcmp(name, "Ball_Explosion_Paper") ||
        !strcmp(name, "Ball_Explosion_Stone")) {
        CKBehavior *beh = ScriptHelper::FindFirstBB(script, "Set Position");
        if (beh)
            ScriptHelper::DeleteBB(script, beh);
        else
            GetLogger()->Error("%s: no Set Position building block", name);
    }
    if (!strcmp(name, "Gameplay_Ingame")) {
        for (int i = 0; i < script->GetLocalParameterCount(); ++i) {
            CKParameter *param = script->GetLocalParameter(i);
            if (!strcmp(param->GetName(), "ActiveBall")) {
                m_ActiveBall = param;
                break;
            }
        }
    }
}

// ---------------------------------------------------------------- auto-load

void BallrainIO::OnPreStartMenu() { Dbg("PreStartMenu cw=%04x", FpuCw()); }
void BallrainIO::OnPostExitLevel() { Dbg("PostExitLevel cw=%04x", FpuCw()); }

void BallrainIO::OnPostStartMenu() {
    Dbg("PostStartMenu cw=%04x", FpuCw());
    if (!m_Active || !VerifyPhysicsDll())
        return;
    if (m_Mode == Mode::Agent && m_Phase == Phase::Idle) {
        AgentHello();
        AgentSetState(2);   // menu
    }
    if (m_AutoLoadArmed && m_LoadIndex < m_Loads && m_Level >= 1) {
        m_AutoLoadArmed = false;
        if (m_LoadIndex == 0)
            ScheduleAutoLoad();
        else
            m_WaitQuiet = true;   // reload: OnProcess schedules the autoload once the menu is quiet
    }
}

static bool SameNames(std::vector<std::string> a, std::vector<std::string> b) {
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return a == b;
}

std::vector<std::string> BallrainIO::BehaviorObjectNames() const {
    std::vector<std::string> names;
    CKBehaviorManager *bm = m_BML->GetCKContext()->GetBehaviorManager();
    for (int k = 0; bm && k < bm->GetObjectsCount(); ++k) {
        CKBeObject *o = bm->GetObject(k);
        names.emplace_back(o && o->GetName() ? o->GetName() : "-");
    }
    return names;
}

void BallrainIO::ScheduleAutoLoad() {
    // Reference OnPostStartMenu :294-309 (the 2-tick timer and the same messages).
    const int level = m_Level;
    m_BML->AddTimer(2ul, [this, level]() {
        Dbg("autoload timer fires cw=%04x", FpuCw());
        if (m_Loads > 1 && m_LoadIndex == 0) {
            m_ColdMenuObjects = BehaviorObjectNames();
            CaptureScripts();
        } else if (m_LoadIndex > 0) {
            RestoreScripts();
            ReorderBehaviorObjects();
        }
        if (m_Dbg) {
            std::string list;
            for (const std::string &n : BehaviorObjectNames())
                list += " " + n;
            Dbg("autoload behavior objects:%s", list.c_str());
        }
        if (!m_CurLevel || !m_ExitMain) {
            GetLogger()->Error("auto-load: CurrentLevel array or Menu_Main Exit not found");
            return;
        }
        m_CurLevel->SetElementValue(0, 0, (void *) &level);

        CKContext *ctx = m_BML->GetCKContext();
        CKMessageManager *mm = ctx->GetMessageManager();
        CKMessageType loadLevel = mm->AddMessageType((CKSTRING) "Load Level");
        CKMessageType loadMenu = mm->AddMessageType((CKSTRING) "Menu_Load");

        mm->SendMessageSingle(loadLevel, ctx->GetCurrentLevel());
        mm->SendMessageSingle(loadMenu, m_BML->GetGroupByName("All_Sound"));
        m_BML->Get2dEntityByName("M_BlackScreen")->Show(CKHIDE);
        m_ExitMain->ActivateInput(0);
        m_ExitMain->Activate();
    });
}

// ---------------------------------------------------------------- start / stop

static void DumpState(CKContext *ctx, CKBaseManager *ipion, const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "w");
    if (!f)
        return;
    std::vector<char> buf(1 << 16);
    const CK_CLASSID classes[] = {CKCID_DATAARRAY, CKCID_BEHAVIOR, CKCID_3DENTITY, CKCID_2DENTITY, CKCID_GROUP,
                                  CKCID_PARAMETER, CKCID_MESH, CKCID_SCENE, CKCID_LEVEL, CKCID_WAVESOUND};
    for (CK_CLASSID cid : classes)
        std::fprintf(f, "count class %d %d\n", (int) cid, ctx->GetObjectsCountByClassID(cid));
    auto *mgr = reinterpret_cast<unsigned char *>(ipion);
    for (int off = 0x20; off < 0xD4; off += 4)
        std::fprintf(f, "ipion +%02x %08x\n", off, *reinterpret_cast<unsigned *>(mgr + off));
    {
        const int n = ctx->GetObjectsCountByClassID(CKCID_DATAARRAY);
        CK_ID *ids = ctx->GetObjectsListByClassID(CKCID_DATAARRAY);
        for (int k = 0; k < n; ++k) {
            auto *a = (CKDataArray *) ctx->GetObject(ids[k]);
            if (!a)
                continue;
            std::fprintf(f, "array %s %dx%d\n", a->GetName() ? a->GetName() : "-", a->GetRowCount(), a->GetColumnCount());
            for (int r = 0; r < a->GetRowCount(); ++r)
                for (int c = 0; c < a->GetColumnCount(); ++c) {
                    buf[0] = 0;
                    if (a->GetElementStringValue(r, c, nullptr) < (int) buf.size())
                        a->GetElementStringValue(r, c, buf.data());
                    std::fprintf(f, "  [%d,%d] %s\n", r, c, buf.data());
                }
        }
    }
    {
        const int n = ctx->GetObjectsCountByClassID(CKCID_BEHAVIOR);
        CK_ID *ids = ctx->GetObjectsListByClassID(CKCID_BEHAVIOR);
        for (int k = 0; k < n; ++k) {
            auto *b = (CKBehavior *) ctx->GetObject(ids[k]);
            if (!b)
                continue;
            CKBehavior *owner = b->GetOwnerScript();
            std::fprintf(f, "beh %s/%s active=%d flags=%08x\n", owner && owner->GetName() ? owner->GetName() : "-",
                         b->GetName() ? b->GetName() : "-", (int) b->IsActive(), (unsigned) b->GetFlags());
            for (int p = 0; p < b->GetLocalParameterCount(); ++p) {
                CKParameter *param = b->GetLocalParameter(p);
                if (!param)
                    continue;
                buf[0] = 0;
                if (param->GetStringValue(nullptr) < (int) buf.size())
                    param->GetStringValue(buf.data());
                std::fprintf(f, "  local %s = %s\n", param->GetName() ? param->GetName() : "-", buf.data());
            }
        }
    }
    {
        const int n = ctx->GetObjectsCountByClassID(CKCID_3DENTITY);
        CK_ID *ids = ctx->GetObjectsListByClassID(CKCID_3DENTITY);
        for (int k = 0; k < n; ++k) {
            auto *e = (CK3dEntity *) ctx->GetObject(ids[k]);
            if (!e)
                continue;
            const VxMatrix &m = e->GetWorldMatrix();
            std::fprintf(f, "ent %s vis=%d pos=%.9g %.9g %.9g\n", e->GetName() ? e->GetName() : "-", (int) e->IsVisible(),
                         m[3][0], m[3][1], m[3][2]);
        }
    }
    std::fclose(f);
}

void BallrainIO::DbgGate(const char *where) {
    if (!m_Dbg)
        return;
    auto *mgr = reinterpret_cast<CKBYTE *>(m_IpionManager);
    auto *env = *reinterpret_cast<CKBYTE **>(mgr + 0xC0);
    auto *base = reinterpret_cast<CKBYTE *>(::GetModuleHandleA("physics_RT.dll"));
    Dbg("GATE %s idx=%u env120=%a env128=%a env130=%a c8=%08x cc=%08x d0=%08x rand=%08x colldist=%08x", where,
        (unsigned) m_FrameIndex, env ? *reinterpret_cast<double *>(env + 0x120) : -1.0,
        env ? *reinterpret_cast<double *>(env + 0x128) : -1.0, env ? *reinterpret_cast<double *>(env + 0x130) : -1.0,
        *reinterpret_cast<unsigned *>(mgr + 0xC8), *reinterpret_cast<unsigned *>(mgr + 0xCC),
        *reinterpret_cast<unsigned *>(mgr + 0xD0), base ? *reinterpret_cast<unsigned *>(base + 0x685b4) : 0u,
        base ? *reinterpret_cast<unsigned *>(base + 0x75edc) : 0u);
}

// CKBehaviorManager keeps its active objects in an XObjectPointerArray sorted by priority with qsort,
// so objects of equal priority run in an order that depends on the array's history. A reload puts the
// same objects in the order the cold process had at its autoload. The array is found by its content.
void BallrainIO::ReorderBehaviorObjects() {
    CKBehaviorManager *bm = m_BML->GetCKContext()->GetBehaviorManager();
    const int n = bm ? bm->GetObjectsCount() : 0;
    if (n == 0 || BehaviorObjectNames().size() != m_ColdMenuObjects.size()) {
        GetLogger()->Error("reload: behavior objects differ from the cold autoload (%d)", n);
        return;
    }
    auto *raw = reinterpret_cast<CKBYTE *>(bm);
    CKBeObject **arr = nullptr;
    for (int off = 0; off < 0x200 && !arr; off += 4) {
        auto **begin = *reinterpret_cast<CKBeObject ***>(raw + off);
        auto **end = *reinterpret_cast<CKBeObject ***>(raw + off + 4);
        if (!begin || end != begin + n || ::IsBadReadPtr(begin, n * sizeof(void *)))
            continue;
        bool match = true;
        for (int k = 0; k < n && match; ++k)
            match = begin[k] == bm->GetObject(k);
        if (match)
            arr = begin;
    }
    if (!arr) {
        GetLogger()->Error("reload: behavior manager object array not found");
        return;
    }
    std::vector<CKBeObject *> cur(arr, arr + n), sorted;
    for (const std::string &name : m_ColdMenuObjects)
        for (CKBeObject *&o : cur)
            if (o && o->GetName() && name == o->GetName()) {
                sorted.push_back(o);
                o = nullptr;
                break;
            }
    if ((int) sorted.size() != n) {
        GetLogger()->Error("reload: behavior objects differ from the cold autoload (names)");
        return;
    }
    std::copy(sorted.begin(), sorted.end(), arr);
    if (BehaviorObjectNames() != m_ColdMenuObjects)
        GetLogger()->Error("reload: behavior object order not restored");
}

// Process-global state that a level load starts from and that the previous run leaves changed.
// Captured at the first (cold) PreLoadLevel, written back at every later one (LOADS > 1 only).
static unsigned *PhysicsRandSeed() {   // IVP_RAND_SEED of the 2002 physics_RT.dll (RVA 0x685b4)
    auto *base = reinterpret_cast<CKBYTE *>(::GetModuleHandleA("physics_RT.dll"));
    return base ? reinterpret_cast<unsigned *>(base + 0x685b4) : nullptr;
}

// Scripts that are not running at the cold autoload. A reload resets each of them that is not running
// at its own autoload either (the engine's script reset: sub-behaviors off, IOs and link delays reset),
// so sub-behaviors the previous level left flagged active do not run on when the script is started
// again without a reset. The gameplay scripts also get their parameter values (locals, outputs) back
// as the cold process had them: e.g. Gameplay_Ingame otherwise enables Ball Navigation at the first
// key press of the camera intro. Also the gameplay arrays a finished level leaves filled.
static void CollectParameters(CKBehavior *b, std::vector<CKObject *> &out) {
    if (!b)
        return;
    for (int i = 0; i < b->GetOutputParameterCount(); ++i)
        out.push_back(b->GetOutputParameter(i));
    for (int i = 0; i < b->GetLocalParameterCount(); ++i)
        out.push_back(b->GetLocalParameter(i));
    for (int i = 0; i < b->GetParameterOperationCount(); ++i)
        if (CKParameterOperation *op = b->GetParameterOperation(i))
            out.push_back(op->GetOutParameter());
    for (int i = 0; i < b->GetSubBehaviorCount(); ++i)
        CollectParameters(b->GetSubBehavior(i), out);
}

void BallrainIO::CaptureScripts() {
    CKContext *ctx = m_BML->GetCKContext();
    const int n = ctx->GetObjectsCountByClassID(CKCID_BEHAVIOR);
    CK_ID *ids = ctx->GetObjectsListByClassID(CKCID_BEHAVIOR);
    size_t params = 0;
    for (int k = 0; k < n; ++k) {
        auto *b = (CKBehavior *) ctx->GetObject(ids[k]);
        if (!b || b->GetType() != CKBEHAVIORTYPE_SCRIPT || b->GetParent() || b->IsActive())
            continue;
        ScriptState st;
        st.script = b->GetID();
        if (b->GetName() && !strncmp(b->GetName(), "Gameplay_", 9)) {
            std::vector<CKObject *> objs;
            CollectParameters(b, objs);
            for (CKObject *o : objs)
                if (o)
                    st.chunks.emplace_back(o->GetID(), CKSaveObjectState(o));
            params += st.chunks.size();
        }
        m_ColdScripts.push_back(std::move(st));
    }
    static const char *const kArrays[] = {"IngameParameter", "Energy", "LifeBalls"};
    for (const char *name : kArrays)
        if (CKDataArray *a = m_BML->GetArrayByName(name))
            m_ColdArrays.emplace_back(a->GetID(), CKSaveObjectState(a));
    Dbg("captured %u idle scripts, %u parameters, %u arrays", (unsigned) m_ColdScripts.size(), (unsigned) params,
        (unsigned) m_ColdArrays.size());
}

void BallrainIO::RestoreScripts() {
    CKContext *ctx = m_BML->GetCKContext();
    unsigned scripts = 0, params = 0, skipped = 0, arrays = 0;
    for (const ScriptState &st : m_ColdScripts) {
        auto *b = (CKBehavior *) ctx->GetObject(st.script);
        if (!b || b->IsActive()) {
            ++skipped;
            Dbg("reload: script %s is running, not reset", b && b->GetName() ? b->GetName() : "-");
            continue;
        }
        ++scripts;
        b->Activate(FALSE, TRUE);
        for (const auto &c : st.chunks)
            if (CKObject *o = ctx->GetObject(c.first)) {
                CKReadObjectState(o, c.second);
                ++params;
            }
    }
    for (const auto &c : m_ColdArrays)
        if (CKObject *a = ctx->GetObject(c.first)) {
            CKReadObjectState(a, c.second);
            ++arrays;
        }
    Dbg("reload: reset %u scripts (%u parameters), %u running, %u arrays", scripts, params, skipped, arrays);
}

void BallrainIO::FreeColdState() {
    for (ScriptState &st : m_ColdScripts)
        for (auto &c : st.chunks)
            DeleteCKStateChunk(c.second);
    m_ColdScripts.clear();
    for (auto &c : m_ColdArrays)
        DeleteCKStateChunk(c.second);
    m_ColdArrays.clear();
}

void BallrainIO::CaptureColdState() {
    auto *mgr = reinterpret_cast<CKBYTE *>(m_IpionManager);
    m_ColdTimeFactor = *reinterpret_cast<unsigned *>(mgr + 0xD0);
    if (unsigned *seed = PhysicsRandSeed())
        m_ColdRandSeed = *seed;
    m_ColdActiveBall = 0;
    if (m_ActiveBall)
        m_ActiveBall->GetValue(&m_ColdActiveBall);
}

void BallrainIO::RestoreColdState() {
    auto *mgr = reinterpret_cast<CKBYTE *>(m_IpionManager);
    *reinterpret_cast<unsigned *>(mgr + 0xD0) = m_ColdTimeFactor;   // physics time factor
    if (unsigned *seed = PhysicsRandSeed())
        *seed = m_ColdRandSeed;
    if (m_ActiveBall)
        m_ActiveBall->SetValue(&m_ColdActiveBall);   // no ball until this load's level start sets it
    DbgGate("Restored");
}

void BallrainIO::OnPreLoadLevel() {
    DbgGate("PreLoadLevel");
    Dbg("PreLoadLevel cw=%04x playing=%d ball=%p", FpuCw(), (int) m_Playing, (void *) GetActiveBall());
    if (m_Dbg)
        DumpState(m_BML->GetCKContext(), m_IpionManager,
                  GetEnv("BALLRAIN_DEBUG") + ".dump" + std::to_string(m_LoadIndex + 1));
    Start();
}

void BallrainIO::Start() {
    // Reference OnStart :406-436.
    if (!m_Active || !VerifyPhysicsDll() || m_LoadIndex >= m_Loads)
        return;
    if (m_Playing) {   // a level load while a run plays: the run goes on (no second start)
        GetLogger()->Warn("level load during run %d at frame %u: ignored", m_LoadIndex, (unsigned) m_FrameIndex);
        return;
    }
    if (m_Loads > 1) {   // LOADS=1 never touches any of this
        if (m_LoadIndex == 0)
            CaptureColdState();
        else
            RestoreColdState();
    }
    ++m_LoadIndex;
    m_ReloadWaiting = false;

    m_BML->AddTimer(1ul, [this]() { ResetPhysicsTime(); });

    // The reference adds both post-callbacks here and clears them in OnStop. Here they are added once,
    // at the first start, and are inert while no run plays: the same per-frame effect, without clearing
    // a callback list from inside one of its own callbacks (OnStop runs inside them at record end).
    if (!m_CallbacksRegistered) {
        CKTimeManagerHook::AddPostCallback([this](CKBaseManager *) { OnPreProcessTime(); });
        CKInputManagerHook::AddPostCallback([this](CKBaseManager *) { OnPreProcessInput(); });
        m_CallbacksRegistered = true;
    }

    AcquireKeyBindings();

    m_FrameIndex = 0;
    m_TraceLines = 0;
    m_GameOverPending = false;
    m_DeltaFrame = SIZE_MAX;
    m_FrameLog.clear();
    m_FrameLog.reserve(m_Frames.size());
    OpenOutputs();
    m_StartTime = std::chrono::steady_clock::now();
    m_Playing = true;
    Event("PreLoadLevel");
    GetLogger()->Info("run %d/%d started", m_LoadIndex, m_Loads);
    if (m_Mode == Mode::Agent)
        AgentStart();
}

void BallrainIO::Stop(StopReason reason) {
    // Reference OnStop :438-474 (playing branch).
    if (!m_Playing)
        return;
    if (reason == StopReason::RecordEnd)
        Event("RecordEnd");
    m_Playing = false;
    m_GameOverPending = false;
    if (m_InputHook)
        ResetKeyboardState(m_InputHook->GetKeyboardState());

    const auto wallMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - m_StartTime).count();
    CloseOutputs(wallMs, m_Mode == Mode::Agent);
    GetLogger()->Info("run %d/%d ended at frame %u", m_LoadIndex, m_Loads, (unsigned) m_FrameIndex);

    if (m_Mode == Mode::Agent) {   // SR tail / final observation / exit: Agent.cpp
        AgentAfterStop(reason);
        return;
    }
    if (m_LoadIndex < m_Loads) {
        // Back to the start menu through the game's own pause-menu message ("Exit Level" to Level and
        // All_Balls), then the next PostStartMenu runs the same auto-load. Not done when the level is
        // already being exited. Sent from OnProcess, not from inside a PreProcess callback.
        m_SendExitLevel = reason != StopReason::ExitLevel;
        m_AutoLoadArmed = true;
        m_ReloadWaiting = true;
        m_ReloadDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        return;
    }
    if (m_ExitWhenDone)
        m_BML->ExitGame();   // reference :458-461 (ExitOnFinish)
}

// Agent mode: every hook is also an observation event (protocol 2.3) with its events-file line;
// replay mode writes the same lines as before through Event() and ignores the agent-only hooks.
#define BALLRAIN_EVENT(code, name)               \
    do {                                         \
        if (m_Mode == Mode::Agent)               \
            AgentEvent((code), 0, (name));       \
        else                                     \
            Event(name);                         \
    } while (0)
#define BALLRAIN_AGENT_EVENT(code, name)         \
    do {                                         \
        if (m_Mode == Mode::Agent)               \
            AgentEvent((code), 0, (name));       \
    } while (0)

void BallrainIO::OnPreResetLevel() {   // reference :335
    BALLRAIN_AGENT_EVENT(28, "PreResetLevel");
    Stop(StopReason::ResetLevel);
}
void BallrainIO::OnPreExitLevel() {   // reference :337
    BALLRAIN_AGENT_EVENT(29, "PreExitLevel");
    Stop(StopReason::ExitLevel);
}

void BallrainIO::OnLevelFinish() {   // reference :339-347
    BALLRAIN_EVENT(21, "LevelFinish");
    Stop(StopReason::LevelFinish);
}

void BallrainIO::OnGameOver() {
    // Contract: a game over ends the run (the reference plays on). The stop is deferred to the next
    // Time callback, the point where the reference stops when its record runs out: this frame's trace
    // line is still written, so the frame log replayed as a legacy record by the real TASSupport gives
    // the identical trace, terminal frame included.
    BALLRAIN_EVENT(22, "GameOver");
    if (m_Playing)
        m_GameOverPending = true;
}

void BallrainIO::OnStartLevel() {
    Dbg("StartLevel idx=%u cw=%04x ball=%p", (unsigned) m_FrameIndex, FpuCw(), (void *) GetActiveBall());
    DbgGate("StartLevel");
    if (m_Dbg)
        DumpState(m_BML->GetCKContext(), m_IpionManager,
                  GetEnv("BALLRAIN_DEBUG") + ".dumpS" + std::to_string(m_LoadIndex));
    BALLRAIN_EVENT(3, "StartLevel");
}
void BallrainIO::OnCamNavActive() { BALLRAIN_EVENT(4, "CamNavActive"); }
void BallrainIO::OnBallNavActive() { BALLRAIN_EVENT(6, "BallNavActive"); }
void BallrainIO::OnBallNavInactive() { BALLRAIN_EVENT(7, "BallNavInactive"); }
void BallrainIO::OnBallOff() { BALLRAIN_EVENT(12, "BallOff"); }
void BallrainIO::OnPreCheckpointReached() { BALLRAIN_EVENT(16, "PreCheckpoint"); }
void BallrainIO::OnPostCheckpointReached() { BALLRAIN_EVENT(17, "PostCheckpoint"); }
void BallrainIO::OnPreLifeUp() { BALLRAIN_EVENT(19, "PreLifeUp"); }

void BallrainIO::OnPostLoadLevel() {
    Dbg("PostLoadLevel idx=%u cw=%04x", (unsigned) m_FrameIndex, FpuCw());
    BALLRAIN_AGENT_EVENT(2, "PostLoadLevel");
}
void BallrainIO::OnCamNavInactive() { BALLRAIN_AGENT_EVENT(5, "CamNavInactive"); }
void BallrainIO::OnCounterActive() { BALLRAIN_AGENT_EVENT(8, "CounterActive"); }
void BallrainIO::OnCounterInactive() { BALLRAIN_AGENT_EVENT(9, "CounterInactive"); }
void BallrainIO::OnPauseLevel() { BALLRAIN_AGENT_EVENT(10, "PauseLevel"); }
void BallrainIO::OnUnpauseLevel() { BALLRAIN_AGENT_EVENT(11, "UnpauseLevel"); }
void BallrainIO::OnDead() { BALLRAIN_AGENT_EVENT(13, "Dead"); }
void BallrainIO::OnPreSubLife() { BALLRAIN_AGENT_EVENT(14, "PreSubLife"); }
void BallrainIO::OnPostSubLife() { BALLRAIN_AGENT_EVENT(15, "PostSubLife"); }
void BallrainIO::OnExtraPoint() { BALLRAIN_AGENT_EVENT(18, "ExtraPoint"); }
void BallrainIO::OnPostLifeUp() { BALLRAIN_AGENT_EVENT(20, "PostLifeUp"); }
void BallrainIO::OnPreEndLevel() { BALLRAIN_AGENT_EVENT(23, "PreEndLevel"); }
void BallrainIO::OnPostEndLevel() { BALLRAIN_AGENT_EVENT(24, "PostEndLevel"); }
void BallrainIO::OnPostResetLevel() {}

// ---------------------------------------------------------------- per frame
// GetFrame: Agent.cpp (replay: the BALLRAIN_FRAMES frames).

void BallrainIO::OnPreProcessTime() {
    // Reference :506-513. The index is not advanced here.
    if (m_Dbg && (!m_Playing || m_FrameIndex < 140))
        Dbg("T idx=%u play=%d cw=%04x d=%.9g abs=%.9g", (unsigned) m_FrameIndex, (int) m_Playing, FpuCw(),
            m_TimeManager->GetLastDeltaTime(), m_TimeManager->GetAbsoluteTime());
    if (m_Mode == Mode::Agent && AgentPreTime())
        return;   // SR tail frame, or a stop deferred to this callback
    if (!m_Playing)
        return;
    if (m_GameOverPending) {
        Stop(StopReason::GameOver);
        return;
    }
    const Frame *frame = GetFrame(m_FrameIndex);
    if (!frame && m_Mode == Mode::Agent) {
        // Agent mode: the frame is Python's to choose. Blocks until the action arrives (the delta must
        // be known here, before the game uses it); false when the run ended while waiting.
        if (!AgentWaitAction())
            return;
        frame = GetFrame(m_FrameIndex);
    }
    if (!frame) {
        Stop(StopReason::RecordEnd);
        return;
    }
    m_TimeManager->SetLastDeltaTime(frame->deltaTime);
    m_InjectedDelta = frame->deltaTime;
    m_DeltaFrame = m_FrameIndex;
}

void BallrainIO::OnPreProcessInput() {
    // Reference :481-490: write the keys, then advance the frame.
    if (m_Dbg && (!m_Playing || m_FrameIndex < 140))
        Dbg("I idx=%u play=%d", (unsigned) m_FrameIndex, (int) m_Playing);
    if (m_Mode == Mode::Agent && AgentPreInput())
        return;   // SR tail frame (not exported)
    if (!m_Playing)
        return;
    if (m_GameOverPending) {
        Stop(StopReason::GameOver);
        return;
    }
    const Frame *frame = GetFrame(m_FrameIndex);
    if (!frame) {
        Stop(StopReason::RecordEnd);
        return;
    }
    SetKeyboardState(m_InputHook->GetKeyboardState(), frame->keys);
    float delta = m_InjectedDelta;
    if (m_DeltaFrame != m_FrameIndex) {   // the Time callback did not run first in this frame
        delta = m_TimeManager->GetLastDeltaTime();
        GetLogger()->Error("frame %u: input before time; logged the game's own delta", (unsigned) m_FrameIndex);
    }
    m_FrameLog.push_back(Frame{delta, frame->keys});
    ++m_FrameIndex;
    if (m_Mode == Mode::Agent) {
        ++m_Covered;
        m_LastDelta = delta;
    }
}

void BallrainIO::OnProcess() {
    const unsigned long long nowMs = ::GetTickCount64();
    if (nowMs - m_FocusCheckMs >= 100) {   // a later activation of this process is handed back within 0.1 s
        m_FocusCheckMs = nowMs;
        ReturnForeground("OnProcess");
    }
    if (!m_Active)
        return;

    if (m_SendExitLevel) {
        m_SendExitLevel = false;
        CKContext *ctx = m_BML->GetCKContext();
        CKMessageManager *mm = ctx->GetMessageManager();
        const CKMessageType exitLevel = mm->AddMessageType((CKSTRING) "Exit Level");
        int sent = 0;
        for (const char *target : {"Level", "All_Balls"}) {
            auto *obj = (CKBeObject *) ctx->GetObjectByNameAndParentClass((CKSTRING) target, CKCID_BEOBJECT, nullptr);
            if (obj && mm->SendMessageSingle(exitLevel, obj))
                ++sent;
        }
        if (sent != 2)
            GetLogger()->Error("reload: \"Exit Level\" reached %d of Level/All_Balls", sent);
    }
    if (m_WaitQuiet && m_Dbg) {
        static std::vector<std::string> last;
        std::vector<std::string> now = BehaviorObjectNames();
        if (now != last) {
            std::string list;
            for (const std::string &n : now)
                list += " " + n;
            Dbg("waiting, behavior objects:%s", list.c_str());
            last = now;
        }
    }
    if (m_WaitQuiet && SameNames(BehaviorObjectNames(), m_ColdMenuObjects)) {
        // The previous level's leftovers (e.g. Ball_ResetPieces_* on All_Balls) have finished: the
        // behavior manager runs the same objects as at the cold autoload, so the level-load scripts
        // execute in the same order (equal-priority objects are qsorted, order depends on the set).
        m_WaitQuiet = false;
        Dbg("reload: menu quiet");
        ScheduleAutoLoad();
    }
    if (m_ReloadWaiting && std::chrono::steady_clock::now() > m_ReloadDeadline) {
        GetLogger()->Error("reload: no level load within 30 s after run %d; giving up", m_LoadIndex);
        m_ReloadWaiting = false;
        m_Loads = m_LoadIndex;
        if (m_ExitWhenDone)
            m_BML->ExitGame();
        return;
    }

    if (m_Dbg && (!m_Playing || m_FrameIndex < 140)) {
        CK3dEntity *ball = GetActiveBall();
        double physTime = -1;
        if (auto *env = *reinterpret_cast<CKBYTE **>(reinterpret_cast<CKBYTE *>(m_IpionManager) + 0xC0))
            physTime = *reinterpret_cast<double *>(env + 0x120);
        int cl = -1;
        if (m_CurLevel)
            m_CurLevel->GetElementValue(0, 0, &cl);
        auto *mgr = reinterpret_cast<CKBYTE *>(m_IpionManager);
        Dbg("P idx=%u play=%d cw=%04x ball=%p:%s id=%d curlevel=%d phys=%.9g c4=%08x c8=%.9g cc=%.9g d0=%.9g ts=%.9g",
            (unsigned) m_FrameIndex, (int) m_Playing, FpuCw(), (void *) ball,
            ball && ball->GetName() ? ball->GetName() : "-", ball ? (int) ball->GetID() : 0, cl, physTime,
            *reinterpret_cast<unsigned *>(mgr + 0xC4), *reinterpret_cast<float *>(mgr + 0xC8),
            *reinterpret_cast<float *>(mgr + 0xCC), *reinterpret_cast<float *>(mgr + 0xD0),
            m_TimeManager->GetTimeScaleFactor());
    }
    if (m_Dbg && m_Playing && m_FrameIndex <= 120) {
        if (!m_Act || m_ActLoad != m_LoadIndex) {
            if (m_Act)
                std::fclose(m_Act);
            m_Act = std::fopen((GetEnv("BALLRAIN_DEBUG") + ".act" + std::to_string(m_LoadIndex)).c_str(), "w");
            m_ActLoad = m_LoadIndex;
        }
        if (m_Act && m_FrameIndex >= 3 && m_FrameIndex <= 10) {
            CKBehaviorManager *bm = m_BML->GetCKContext()->GetBehaviorManager();
            for (int k = 0; bm && k < bm->GetObjectsCount(); ++k) {
                CKBeObject *o = bm->GetObject(k);
                std::fprintf(m_Act, "BEO %u %d %s id=%d pri=%d scripts=%d\n", (unsigned) m_FrameIndex, k,
                             o && o->GetName() ? o->GetName() : "-", o ? (int) o->GetID() : 0,
                             o ? o->GetPriority() : 0, o ? o->GetScriptCount() : 0);
            }
        }
        if (m_Act) {
            CKContext *ctx = m_BML->GetCKContext();
            const int n = ctx->GetObjectsCountByClassID(CKCID_BEHAVIOR);
            CK_ID *ids = ctx->GetObjectsListByClassID(CKCID_BEHAVIOR);
            for (int k = 0; k < n; ++k) {
                auto *b = (CKBehavior *) ctx->GetObject(ids[k]);
                if (!b || !b->IsActive())
                    continue;
                CKBehavior *owner = b->GetOwnerScript();
                std::fprintf(m_Act, "%u %s/%s %08x\n", (unsigned) m_FrameIndex,
                             owner && owner->GetName() ? owner->GetName() : "-", b->GetName() ? b->GetName() : "-",
                             (unsigned) b->GetFlags());
            }
            std::fflush(m_Act);
        }
    }
    if (!m_Playing) {
        if (m_Mode == Mode::Agent)
            AgentProcess();   // SR tail, final observation after a mid-frame stop
        return;
    }

    // Reference :208-252: the trace line, frame index after the Input callback advanced it, flushed
    // after every line as the reference does.
    if (m_Trace) {
        if (CK3dEntity *ball = GetActiveBall()) {
            const VxMatrix &m = ball->GetWorldMatrix();
            std::fprintf(m_Trace, "%u %s", (unsigned) m_FrameIndex, ball->GetName() ? ball->GetName() : "-");
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 3; ++c)
                    std::fprintf(m_Trace, " %.9g", m[r][c]);
            std::fputc('\n', m_Trace);
            std::fflush(m_Trace);
            if (m_Dbg && m_TraceLines < 3)
                DbgGate("Trace");
            ++m_TraceLines;
        }
    }
    if (m_Mode == Mode::Agent)
        AgentProcess();   // events / SR / physicalize export / observation (after the trace line)
    // Reference :254-255 with SkipRenderUntil beyond the record.
    if (m_SkipRender)
        m_BML->SkipRenderForNextTick();
}

// ---------------------------------------------------------------- game state helpers

void BallrainIO::AcquireKeyBindings() {   // reference :694-703
    if (m_Keyboard) {
        m_Keyboard->GetElementValue(0, 0, &m_KeyUp);
        m_Keyboard->GetElementValue(0, 1, &m_KeyDown);
        m_Keyboard->GetElementValue(0, 2, &m_KeyLeft);
        m_Keyboard->GetElementValue(0, 3, &m_KeyRight);
        m_Keyboard->GetElementValue(0, 4, &m_KeyShift);
        m_Keyboard->GetElementValue(0, 5, &m_KeySpace);
    } else {
        GetLogger()->Error("Keyboard array not found: default key bindings");
    }
}

void BallrainIO::ResetPhysicsTime() {
    // Reference :705-706, 747-780: the retail (2002) DLL layout. VerifyPhysicsDll gates every run.
    if (m_PhysicsOk <= 0)
        return;
    GetLogger()->Info("x87 control word at the reset: %08x (precision bits %x)", _controlfp(0, 0),
                      _controlfp(0, 0) & _MCW_PC);
    Dbg("ResetPhysicsTime idx=%u cw=%04x", (unsigned) m_FrameIndex, FpuCw());
    auto *mgr = reinterpret_cast<CKBYTE *>(m_IpionManager);
    auto *env = *reinterpret_cast<CKBYTE **>(mgr + 0xC0);   // IVP_Environment
    if (!env) {
        GetLogger()->Error("ResetPhysicsTime: no IVP_Environment");
        return;
    }
    *reinterpret_cast<double *>(*reinterpret_cast<CKBYTE **>(env + 0x4) + 0x18) = 0;   // time_manager->base_time
    *reinterpret_cast<double *>(env + 0x120) = 0;                                       // current_time
    auto &time_of_last_psi = *reinterpret_cast<double *>(env + 0x130);
    time_of_last_psi = 0;
    *reinterpret_cast<double *>(env + 0x128) = time_of_last_psi + 1.0 / 66;             // time_of_next_psi
    *reinterpret_cast<float *>(mgr + 0xC8) = m_TimeManager->GetLastDeltaTime();         // smoothed delta
}

void BallrainIO::SetKeyboardState(unsigned char *dest, uint32_t keys) const {
    // Reference SetKeyboardState :868-878 (same order), legacy bits as TASRecord.cpp:486-488. Raw 0/1 only.
    if (!dest)
        return;
    dest[m_KeyUp] = keys & 1u;
    dest[m_KeyDown] = keys >> 1 & 1u;
    dest[m_KeyLeft] = keys >> 2 & 1u;
    dest[m_KeyRight] = keys >> 3 & 1u;
    dest[CKKEY_Q] = keys >> 6 & 1u;
    dest[m_KeyShift] = keys >> 4 & 1u;
    dest[m_KeySpace] = keys >> 5 & 1u;
    dest[CKKEY_ESCAPE] = keys >> 7 & 1u;
    dest[CKKEY_RETURN] = keys >> 8 & 1u;
}

void BallrainIO::ResetKeyboardState(unsigned char *dest) const {
    // Reference :880-891 (RETURN is not reset there either).
    if (!dest)
        return;
    dest[m_KeyUp] = KS_IDLE;
    dest[m_KeyDown] = KS_IDLE;
    dest[m_KeyLeft] = KS_IDLE;
    dest[m_KeyRight] = KS_IDLE;
    dest[CKKEY_Q] = KS_IDLE;
    dest[m_KeyShift] = KS_IDLE;
    dest[m_KeySpace] = KS_IDLE;
    dest[CKKEY_ESCAPE] = KS_IDLE;
}

CK3dEntity *BallrainIO::GetActiveBall() const {
    return m_ActiveBall ? (CK3dEntity *) m_ActiveBall->GetValueObject() : nullptr;
}

// ---------------------------------------------------------------- outputs

std::string BallrainIO::OutputPath(const std::string &base) const {
    if (base.empty() || m_LoadIndex <= 1)
        return base;
    return base + "." + std::to_string(m_LoadIndex);
}

void BallrainIO::OpenOutputs() {
    // Text mode like the reference's fopen(path, "w") (CRLF line ends, as in the reference traces).
    if (!m_TracePath.empty() && !(m_Trace = std::fopen(OutputPath(m_TracePath).c_str(), "w")))
        GetLogger()->Error("cannot open trace %s", OutputPath(m_TracePath).c_str());
    if (!m_EventsPath.empty() && !(m_Events = std::fopen(OutputPath(m_EventsPath).c_str(), "w")))
        GetLogger()->Error("cannot open events %s", OutputPath(m_EventsPath).c_str());
}

void BallrainIO::CloseOutputs(long long wallMs, bool keepEvents) {
    if (m_Events) {
        std::fprintf(m_Events, "%u Summary frames=%u wall_ms=%lld\n", (unsigned) m_FrameIndex, (unsigned) m_FrameIndex,
                     wallMs);
        if (keepEvents) {   // agent mode: SR tail / final lines follow (Agent.cpp closes the file)
            std::fflush(m_Events);
        } else {
            std::fclose(m_Events);
            m_Events = nullptr;
        }
    }
    if (m_Trace) {
        std::fclose(m_Trace);
        m_Trace = nullptr;
    }
    if (!m_FrameLogPath.empty()) {
        const std::string path = OutputPath(m_FrameLogPath);
        FILE *log = std::fopen(path.c_str(), "wb");
        if (!log || std::fwrite(m_FrameLog.data(), sizeof(Frame), m_FrameLog.size(), log) != m_FrameLog.size())
            GetLogger()->Error("cannot write frame log %s", path.c_str());
        if (log)
            std::fclose(log);
    }
}

void BallrainIO::Event(const char *name) {
    Dbg("Event %s idx=%u play=%d", name, (unsigned) m_FrameIndex, (int) m_Playing);
    if (m_Playing && m_Events)
        std::fprintf(m_Events, "%u %s\n", (unsigned) m_FrameIndex, name), std::fflush(m_Events);
}
