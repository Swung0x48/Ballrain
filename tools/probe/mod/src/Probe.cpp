#include "Probe.h"

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

// Replay part = BallrainIO 0.2.0 with LOADS=1 (the reload code paths are not present; LOADS=1 never ran them).

static const char kPhysicsSha256[] = "e72e4afcfa5c33a7d3d27776137f8c997b3c52d89d8a8a4745f1ca21e45893ec";

MOD_EXPORT IMod *BMLEntry(IBML *bml) { return new BallrainProbe(bml); }
MOD_EXPORT void BMLExit(IMod *mod) { delete mod; }

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

bool BallrainProbe::ReadEnvironment() {
    if (GetEnv("BALLRAIN_MODE") != "replay")
        return false;
    m_FramesPath = GetEnv("BALLRAIN_FRAMES");
    m_TracePath = GetEnv("BALLRAIN_TRACE");
    m_FrameLogPath = GetEnv("BALLRAIN_FRAMELOG");
    m_EventsPath = GetEnv("BALLRAIN_EVENTS");
    m_ProbeDir = GetEnv("BALLRAIN_PROBE");
    m_Level = std::atoi(GetEnv("BALLRAIN_LEVEL").c_str());
    m_SkipRender = GetEnv("BALLRAIN_SKIP_RENDER") == "1";
    m_ExitWhenDone = GetEnv("BALLRAIN_EXIT") == "1";
    const std::string loads = GetEnv("BALLRAIN_LOADS");
    if (!loads.empty() && loads != "1") {
        GetLogger()->Error("probe supports BALLRAIN_LOADS=1 only");
        return false;
    }
    if (m_Level < 1 || m_Level > 13 || m_FramesPath.empty())
        return false;
    std::ifstream file(m_FramesPath, std::ios::binary | std::ios::ate);
    if (!file)
        return false;
    const auto size = (size_t) file.tellg();
    file.seekg(0);
    m_Frames.resize(size / sizeof(Frame));
    if (!m_Frames.empty() && !file.read(reinterpret_cast<char *>(m_Frames.data()),
                                        (std::streamsize) (m_Frames.size() * sizeof(Frame)))) {
        m_Frames.clear();
        return false;
    }
    return true;
}

void BallrainProbe::OnLoad() {
    m_Active = ReadEnvironment();
    if (!m_Active)
        return;
    VerifyPhysicsDll(false);
    if (!m_Active)
        return;
    m_IpionManager = m_BML->GetCKContext()->GetManagerByGuid(CKGUID(0x6bed328b, 0x141f5148));
    m_TimeManager = m_BML->GetTimeManager();
    m_InputHook = m_BML->GetInputManager();
    const MH_STATUS status = MH_Initialize();
    if (status != MH_OK && status != MH_ERROR_ALREADY_INITIALIZED) {
        m_Active = false;
        return;
    }
    auto *inputManager = (CKInputManager *) m_BML->GetCKContext()->GetManagerByGuid(INPUT_MANAGER_GUID);
    if (!m_IpionManager || !m_TimeManager || !m_InputHook || !inputManager ||
        !CKTimeManagerHook::Enable(m_TimeManager) || !CKInputManagerHook::Enable(inputManager)) {
        m_Active = false;
        return;
    }
    if (!m_ProbeDir.empty()) {
        m_PF = std::fopen((m_ProbeDir + "\\frames.txt").c_str(), "w");
        m_PP = std::fopen((m_ProbeDir + "\\phys.txt").c_str(), "w");
        m_PM = std::fopen((m_ProbeDir + "\\meshes.txt").c_str(), "w");
    }
}

void BallrainProbe::OnUnload() {
    if (m_Playing)
        CloseOutputs(0);
    for (FILE *f : {m_PF, m_PP, m_PM})
        if (f)
            std::fclose(f);
    m_PF = m_PP = m_PM = nullptr;
    CKTimeManagerHook::Disable();
    CKInputManagerHook::Disable();
    MH_Uninitialize();
}

bool BallrainProbe::VerifyPhysicsDll(bool mustBeLoaded) {
    if (m_PhysicsOk != 0)
        return m_PhysicsOk > 0;
    HMODULE module = ::GetModuleHandleA("physics_RT.dll");
    if (!module && !mustBeLoaded)
        return false;
    m_PhysicsOk = -1;
    wchar_t path[MAX_PATH] = {};
    std::string hex;
    if (!module || !::GetModuleFileNameW(module, path, MAX_PATH) || !Sha256File(path, hex) || hex != kPhysicsSha256) {
        GetLogger()->Error("physics_RT.dll is not the 2002 DLL: refusing to run");
        m_Active = false;
        return false;
    }
    m_PhysicsOk = 1;
    return true;
}

void BallrainProbe::OnLoadObject(const char *filename, CKBOOL isMap, const char *masterName, CK_CLASSID filterClass,
                                 CKBOOL addToScene, CKBOOL reuseMeshes, CKBOOL reuseMaterials, CKBOOL dynamic,
                                 XObjectArray *objArray, CKObject *masterObj) {
    if (!m_Active)
        return;
    if (!strcmp(filename, "3D Entities\\Gameplay.nmo"))
        m_CurLevel = m_BML->GetArrayByName("CurrentLevel");
    if (!strcmp(filename, "3D Entities\\Menu.nmo")) {
        m_Keyboard = m_BML->GetArrayByName("Keyboard");
        m_ExitMain = ScriptHelper::FindFirstBB(m_BML->GetScriptByName("Menu_Main"), "Exit", false, 1, 0);
    }
    if (m_PP)
        std::fprintf(m_PP, "LOADOBJECT f=%u %s isMap=%d\n", (unsigned) m_FrameIndex, filename, (int) isMap);
}

void BallrainProbe::OnLoadScript(const char *filename, CKBehavior *script) {
    if (!m_Active)
        return;
    const char *name = script->GetName();
    if (!strcmp(name, "Ball_Explosion_Wood") || !strcmp(name, "Ball_Explosion_Paper") ||
        !strcmp(name, "Ball_Explosion_Stone")) {
        CKBehavior *beh = ScriptHelper::FindFirstBB(script, "Set Position");
        if (beh)
            ScriptHelper::DeleteBB(script, beh);
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

void BallrainProbe::OnPostStartMenu() {
    if (!m_Active || !VerifyPhysicsDll())
        return;
    if (m_AutoLoadArmed && m_LoadIndex < 1 && m_Level >= 1) {
        m_AutoLoadArmed = false;
        ScheduleAutoLoad();
    }
}

void BallrainProbe::ScheduleAutoLoad() {
    const int level = m_Level;
    m_BML->AddTimer(2ul, [this, level]() {
        if (!m_CurLevel || !m_ExitMain)
            return;
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

void BallrainProbe::OnPreLoadLevel() {
    ProbeEvent("PreLoadLevel");
    Start();
}

void BallrainProbe::Start() {
    if (!m_Active || !VerifyPhysicsDll() || m_LoadIndex >= 1)
        return;
    if (m_Playing)
        return;
    ++m_LoadIndex;
    m_BML->AddTimer(1ul, [this]() { ResetPhysicsTime(); });
    if (!m_CallbacksRegistered) {
        CKTimeManagerHook::AddPostCallback([this](CKBaseManager *) { OnPreProcessTime(); });
        CKInputManagerHook::AddPostCallback([this](CKBaseManager *) { OnPreProcessInput(); });
        m_CallbacksRegistered = true;
    }
    AcquireKeyBindings();
    m_FrameIndex = 0;
    m_GameOverPending = false;
    m_DeltaFrame = SIZE_MAX;
    m_FrameLog.clear();
    m_FrameLog.reserve(m_Frames.size());
    OpenOutputs();
    m_StartTime = std::chrono::steady_clock::now();
    m_Playing = true;
    Event("PreLoadLevel");
}

void BallrainProbe::Stop(StopReason reason) {
    if (!m_Playing)
        return;
    if (reason == StopReason::RecordEnd)
        Event("RecordEnd");
    if (m_StaticDumps < 2)
        DumpStatic("end");
    m_Playing = false;
    m_GameOverPending = false;
    if (m_InputHook)
        ResetKeyboardState(m_InputHook->GetKeyboardState());
    const auto wallMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - m_StartTime).count();
    CloseOutputs(wallMs);
    for (FILE *f : {m_PF, m_PP, m_PM})
        if (f)
            std::fflush(f);
    if (m_ExitWhenDone)
        m_BML->ExitGame();
}

void BallrainProbe::OnPreResetLevel() { ProbeEvent("PreResetLevel"); Stop(StopReason::ResetLevel); }
void BallrainProbe::OnPreExitLevel() { ProbeEvent("PreExitLevel"); Stop(StopReason::ExitLevel); }

void BallrainProbe::OnLevelFinish() {
    ProbeEvent("LevelFinish");
    Event("LevelFinish");
    Stop(StopReason::LevelFinish);
}

void BallrainProbe::OnGameOver() {
    ProbeEvent("GameOver");
    Event("GameOver");
    if (m_Playing)
        m_GameOverPending = true;
}

void BallrainProbe::OnStartLevel() {
    ProbeEvent("StartLevel");
    Event("StartLevel");
    if (m_Active && m_StaticDumps == 0)
        DumpStatic("start");
}
void BallrainProbe::OnPostLoadLevel() { ProbeEvent("PostLoadLevel"); }
void BallrainProbe::OnPostResetLevel() { ProbeEvent("PostResetLevel"); }
void BallrainProbe::OnPauseLevel() { ProbeEvent("PauseLevel"); }
void BallrainProbe::OnUnpauseLevel() { ProbeEvent("UnpauseLevel"); }
void BallrainProbe::OnDead() { ProbeEvent("Dead"); }
void BallrainProbe::OnPreEndLevel() { ProbeEvent("PreEndLevel"); }
void BallrainProbe::OnPostEndLevel() { ProbeEvent("PostEndLevel"); }
void BallrainProbe::OnCounterActive() { ProbeEvent("CounterActive"); }
void BallrainProbe::OnCounterInactive() { ProbeEvent("CounterInactive"); }
void BallrainProbe::OnCamNavActive() { ProbeEvent("CamNavActive"); Event("CamNavActive"); }
void BallrainProbe::OnCamNavInactive() { ProbeEvent("CamNavInactive"); }
void BallrainProbe::OnBallNavActive() { ProbeEvent("BallNavActive"); Event("BallNavActive"); }
void BallrainProbe::OnBallNavInactive() { ProbeEvent("BallNavInactive"); Event("BallNavInactive"); }
void BallrainProbe::OnBallOff() { ProbeEvent("BallOff"); Event("BallOff"); }
void BallrainProbe::OnPreCheckpointReached() { ProbeEvent("PreCheckpoint"); Event("PreCheckpoint"); }
void BallrainProbe::OnPostCheckpointReached() { ProbeEvent("PostCheckpoint"); Event("PostCheckpoint"); }
void BallrainProbe::OnExtraPoint() { ProbeEvent("ExtraPoint"); }
void BallrainProbe::OnPreSubLife() { ProbeEvent("PreSubLife"); }
void BallrainProbe::OnPostSubLife() { ProbeEvent("PostSubLife"); }
void BallrainProbe::OnPreLifeUp() { ProbeEvent("PreLifeUp"); Event("PreLifeUp"); }
void BallrainProbe::OnPostLifeUp() { ProbeEvent("PostLifeUp"); }

const Frame *BallrainProbe::GetFrame(size_t index) const {
    return index < m_Frames.size() ? &m_Frames[index] : nullptr;
}

void BallrainProbe::OnPreProcessTime() {
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
    m_TimeManager->SetLastDeltaTime(frame->deltaTime);
    m_InjectedDelta = frame->deltaTime;
    m_DeltaFrame = m_FrameIndex;
}

void BallrainProbe::OnPreProcessInput() {
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
    if (m_DeltaFrame != m_FrameIndex)
        delta = m_TimeManager->GetLastDeltaTime();
    m_FrameLog.push_back(Frame{delta, frame->keys});
    ++m_FrameIndex;
}

void BallrainProbe::OnProcess() {
    if (!m_Active)
        return;
    ExportPendingMeshes();
    if (!m_Playing)
        return;
    if (m_Trace) {
        if (CK3dEntity *ball = GetActiveBall()) {
            const VxMatrix &m = ball->GetWorldMatrix();
            std::fprintf(m_Trace, "%u %s", (unsigned) m_FrameIndex, ball->GetName() ? ball->GetName() : "-");
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 3; ++c)
                    std::fprintf(m_Trace, " %.9g", m[r][c]);
            std::fputc('\n', m_Trace);
            std::fflush(m_Trace);
        }
    }
    ProbeFrame();
    if (m_SkipRender)
        m_BML->SkipRenderForNextTick();
}

void BallrainProbe::AcquireKeyBindings() {
    if (m_Keyboard) {
        m_Keyboard->GetElementValue(0, 0, &m_KeyUp);
        m_Keyboard->GetElementValue(0, 1, &m_KeyDown);
        m_Keyboard->GetElementValue(0, 2, &m_KeyLeft);
        m_Keyboard->GetElementValue(0, 3, &m_KeyRight);
        m_Keyboard->GetElementValue(0, 4, &m_KeyShift);
        m_Keyboard->GetElementValue(0, 5, &m_KeySpace);
    }
}

void BallrainProbe::ResetPhysicsTime() {
    if (m_PhysicsOk <= 0)
        return;
    auto *mgr = reinterpret_cast<CKBYTE *>(m_IpionManager);
    auto *env = *reinterpret_cast<CKBYTE **>(mgr + 0xC0);
    if (!env)
        return;
    *reinterpret_cast<double *>(*reinterpret_cast<CKBYTE **>(env + 0x4) + 0x18) = 0;
    *reinterpret_cast<double *>(env + 0x120) = 0;
    auto &time_of_last_psi = *reinterpret_cast<double *>(env + 0x130);
    time_of_last_psi = 0;
    *reinterpret_cast<double *>(env + 0x128) = time_of_last_psi + 1.0 / 66;
    *reinterpret_cast<float *>(mgr + 0xC8) = m_TimeManager->GetLastDeltaTime();
}

void BallrainProbe::SetKeyboardState(unsigned char *dest, uint32_t keys) const {
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

void BallrainProbe::ResetKeyboardState(unsigned char *dest) const {
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

CK3dEntity *BallrainProbe::GetActiveBall() const {
    return m_ActiveBall ? (CK3dEntity *) m_ActiveBall->GetValueObject() : nullptr;
}

void BallrainProbe::OpenOutputs() {
    if (!m_TracePath.empty())
        m_Trace = std::fopen(m_TracePath.c_str(), "w");
    if (!m_EventsPath.empty())
        m_Events = std::fopen(m_EventsPath.c_str(), "w");
}

void BallrainProbe::CloseOutputs(long long wallMs) {
    if (m_Events) {
        std::fprintf(m_Events, "%u Summary frames=%u wall_ms=%lld\n", (unsigned) m_FrameIndex, (unsigned) m_FrameIndex,
                     wallMs);
        std::fclose(m_Events);
        m_Events = nullptr;
    }
    if (m_Trace) {
        std::fclose(m_Trace);
        m_Trace = nullptr;
    }
    if (!m_FrameLogPath.empty()) {
        FILE *log = std::fopen(m_FrameLogPath.c_str(), "wb");
        if (log) {
            std::fwrite(m_FrameLog.data(), sizeof(Frame), m_FrameLog.size(), log);
            std::fclose(log);
        }
    }
}

void BallrainProbe::Event(const char *name) {
    if (m_Playing && m_Events)
        std::fprintf(m_Events, "%u %s\n", (unsigned) m_FrameIndex, name), std::fflush(m_Events);
}
