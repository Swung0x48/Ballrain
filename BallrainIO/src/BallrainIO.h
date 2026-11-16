#pragma once

// BallrainIO, M0 "replay mode": a re-implementation of TASSupport's LEGACY-mode playback
// (reference/TASSupport, LegacyMode 1) as its own mod, driven by BALLRAIN_* environment
// variables instead of TASSupport.cfg. Interface contract: see BallrainIO.cpp (ReadEnvironment).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <BML/BMLAll.h>

// One input frame, exactly the uncompressed payload layout of a legacy .tas:
// f32 LE deltaTime (ms) + u32 LE key bits (bit0 up, 1 down, 2 left, 3 right, 4 shift, 5 space, 6 Q, 7 esc, 8 enter).
struct Frame {
    float deltaTime;
    uint32_t keys;
};
static_assert(sizeof(Frame) == 8, "legacy frame is 8 bytes");

class BallrainIO final : public IMod {
public:
    explicit BallrainIO(IBML *bml) : IMod(bml) {}

    const char *GetID() override { return "BallrainIO"; }
    const char *GetVersion() override { return "0.2.0"; }
    const char *GetName() override { return "BallrainIO"; }
    const char *GetAuthor() override { return "Swung0x48"; }
    const char *GetDescription() override { return "Ballrain episode I/O (M0: legacy TAS replay)."; }
    DECLARE_BML_VERSION;

    void OnLoad() override;
    void OnUnload() override;
    void OnLoadObject(const char *filename, CKBOOL isMap, const char *masterName, CK_CLASSID filterClass,
                      CKBOOL addToScene, CKBOOL reuseMeshes, CKBOOL reuseMaterials, CKBOOL dynamic,
                      XObjectArray *objArray, CKObject *masterObj) override;
    void OnLoadScript(const char *filename, CKBehavior *script) override;
    void OnProcess() override;

    void OnPreStartMenu() override;
    void OnPostStartMenu() override;
    void OnPreLoadLevel() override;
    void OnPostLoadLevel() override;
    void OnPostExitLevel() override;
    void OnStartLevel() override;
    void OnPreResetLevel() override;
    void OnPreExitLevel() override;
    void OnLevelFinish() override;
    void OnGameOver() override;
    void OnCamNavActive() override;
    void OnBallNavActive() override;
    void OnBallNavInactive() override;
    void OnBallOff() override;
    void OnPreCheckpointReached() override;
    void OnPostCheckpointReached() override;
    void OnPreLifeUp() override;

private:
    enum class StopReason { RecordEnd, ResetLevel, ExitLevel, LevelFinish, GameOver };

    bool ReadEnvironment();
    bool VerifyPhysicsDll(bool mustBeLoaded = true);
    void ScheduleAutoLoad();

    void Start();
    void Stop(StopReason reason);
    void OnPreProcessTime();
    void OnPreProcessInput();

    // The single source of input frames. Replay mode: the BALLRAIN_FRAMES file. A future agent mode
    // supplies the frame for `index` here (delta + key bits) instead; nothing else needs to change.
    const Frame *GetFrame(size_t index) const;

    void AcquireKeyBindings();
    void ResetPhysicsTime();
    void SetKeyboardState(unsigned char *dest, uint32_t keys) const;
    void ResetKeyboardState(unsigned char *dest) const;
    CK3dEntity *GetActiveBall() const;

    void OpenOutputs();
    void CloseOutputs(long long wallMs);
    void Event(const char *name);
    std::string OutputPath(const std::string &base) const;
    // Debug instrumentation (BALLRAIN_DEBUG=<path>; inert without it).
    void Dbg(const char *fmt, ...);
    void DbgGate(const char *where);
    FILE *m_Dbg = nullptr;
    FILE *m_Act = nullptr;
    int m_ActLoad = 0;
    unsigned m_TraceLines = 0;

    // Same-process reload (LOADS > 1 only): the state of the cold process at its first autoload and
    // PreLoadLevel, written back before every later load. See BallrainIO.cpp.
    struct ScriptState {
        CK_ID script = 0;                                       // a script not running at the cold autoload
        std::vector<std::pair<CK_ID, CKStateChunk *>> chunks;   // its parameter values (gameplay scripts)
    };
    std::vector<std::string> BehaviorObjectNames() const;
    void ReorderBehaviorObjects();
    void CaptureScripts();
    void RestoreScripts();
    void CaptureColdState();
    void RestoreColdState();
    void FreeColdState();
    bool m_WaitQuiet = false;                    // reload: autoload once the menu runs the cold objects
    std::vector<std::string> m_ColdMenuObjects;  // behavior manager objects at the cold autoload, in order
    std::vector<ScriptState> m_ColdScripts;
    std::vector<std::pair<CK_ID, CKStateChunk *>> m_ColdArrays;
    unsigned m_ColdTimeFactor = 0;               // CKIpionManager +0xD0 at the cold PreLoadLevel
    unsigned m_ColdRandSeed = 0;                 // IVP_RAND_SEED at the cold PreLoadLevel
    CK_ID m_ColdActiveBall = 0;                  // Gameplay_Ingame ActiveBall at the cold PreLoadLevel

    // Configuration (environment).
    bool m_Active = false;
    std::string m_FramesPath, m_TracePath, m_FrameLogPath, m_EventsPath;
    int m_Level = 0;
    bool m_SkipRender = false;
    bool m_ExitWhenDone = false;
    int m_Loads = 1;

    // Game objects / managers.
    CKBaseManager *m_IpionManager = nullptr;
    CKTimeManager *m_TimeManager = nullptr;
    InputHook *m_InputHook = nullptr;
    CKDataArray *m_CurLevel = nullptr;
    CKDataArray *m_Keyboard = nullptr;
    CKBehavior *m_ExitMain = nullptr;
    CKParameter *m_ActiveBall = nullptr;
    CKKEYBOARD m_KeyUp = CKKEY_UP;
    CKKEYBOARD m_KeyDown = CKKEY_DOWN;
    CKKEYBOARD m_KeyLeft = CKKEY_LEFT;
    CKKEYBOARD m_KeyRight = CKKEY_RIGHT;
    CKKEYBOARD m_KeyShift = CKKEY_LSHIFT;
    CKKEYBOARD m_KeySpace = CKKEY_SPACE;

    // Physics DLL identity: 0 = not checked yet, 1 = the 2002 physics_RT.dll, -1 = refused.
    int m_PhysicsOk = 0;
    bool m_CallbacksRegistered = false;

    // Run state.
    std::vector<Frame> m_Frames;
    bool m_Playing = false;
    bool m_GameOverPending = false;   // GameOver seen: stop at the next Time callback
    size_t m_FrameIndex = 0;
    size_t m_DeltaFrame = SIZE_MAX;   // frame index whose delta the Time callback injected
    float m_InjectedDelta = 0.0f;
    int m_LoadIndex = 0;              // number of level loads started (1-based index of the current run)
    bool m_AutoLoadArmed = true;      // the next PostStartMenu schedules the autoload
    bool m_SendExitLevel = false;     // OnProcess sends "Exit Level" (LOADS > 1)
    bool m_ReloadWaiting = false;
    std::chrono::steady_clock::time_point m_ReloadDeadline;
    std::chrono::steady_clock::time_point m_StartTime;

    FILE *m_Trace = nullptr;
    FILE *m_Events = nullptr;
    std::vector<Frame> m_FrameLog;
};
