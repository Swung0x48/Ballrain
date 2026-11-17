#pragma once

// BallrainIO.
// Replay mode (M0): a re-implementation of TASSupport's LEGACY-mode playback (reference/TASSupport,
// LegacyMode 1) as its own mod, driven by BALLRAIN_* environment variables instead of TASSupport.cfg.
// Agent mode: the same injection path, with the frames chosen step by step by a Python client over
// shared memory, plus a raw game-state export (docs/agent-protocol.md, docs/state-spec.md).
// Interface contract: see BallrainIO.cpp (ReadEnvironment) and Agent.cpp.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    const char *GetVersion() override { return "0.3.3"; }
    const char *GetName() override { return "BallrainIO"; }
    const char *GetAuthor() override { return "Swung0x48"; }
    const char *GetDescription() override { return "Ballrain episode I/O (legacy TAS replay, agent mode)."; }
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

    // Agent mode only (inert in replay mode).
    void OnPostResetLevel() override;
    void OnPauseLevel() override;
    void OnUnpauseLevel() override;
    void OnDead() override;
    void OnPreEndLevel() override;
    void OnPostEndLevel() override;
    void OnCounterActive() override;
    void OnCounterInactive() override;
    void OnCamNavInactive() override;
    void OnExtraPoint() override;
    void OnPreSubLife() override;
    void OnPostSubLife() override;
    void OnPostLifeUp() override;
    void OnPhysicalize(CK3dEntity *target, CKBOOL fixed, float friction, float elasticity, float mass,
                       const char *collGroup, CKBOOL startFrozen, CKBOOL enableColl, CKBOOL calcMassCenter,
                       float linearDamp, float rotDamp, const char *collSurface, VxVector massCenter, int convexCnt,
                       CKMesh **convexMesh, int ballCnt, VxVector *ballCenter, float *ballRadius, int concaveCnt,
                       CKMesh **concaveMesh) override;
    void OnUnphysicalize(CK3dEntity *target) override;

private:
    enum class StopReason { RecordEnd, ResetLevel, ExitLevel, LevelFinish, GameOver };

    enum class Mode { Inactive, Replay, Agent };

    bool ReadEnvironment();
    bool VerifyPhysicsDll(bool mustBeLoaded = true);
    void ReturnForeground(const char *where);   // hand the foreground back if this process holds it
    void ScheduleAutoLoad();

    void Start();
    void Stop(StopReason reason);
    void OnPreProcessTime();
    void OnPreProcessInput();

    // The single source of input frames. Replay mode: the BALLRAIN_FRAMES file. Agent mode: the
    // BALLRAIN_PREFIX frames, then the frames of the current action block (Agent.cpp).
    const Frame *GetFrame(size_t index) const;

    void AcquireKeyBindings();
    void ResetPhysicsTime();
    void SetKeyboardState(unsigned char *dest, uint32_t keys) const;
    void ResetKeyboardState(unsigned char *dest) const;
    CK3dEntity *GetActiveBall() const;

    void OpenOutputs();
    void CloseOutputs(long long wallMs, bool keepEvents = false);
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
    Mode m_Mode = Mode::Inactive;
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

    // Video capture (Capture.cpp; BALLRAIN_CAPTURE, inert without it). Output only: reads the back buffer
    // after the frame is drawn, never touches game objects, input or timing.
    void CaptureInit();
    bool CaptureRegister();
    bool CaptureTailStart();
    bool CaptureTailTime();
    bool CaptureTailInput();
    uint32_t m_CaptureTail = 0;              // BALLRAIN_CAPTURE_TAIL: idle frames after the run's final observation
    uint32_t m_CaptureTailLeft = 0;
    bool m_CaptureTailActive = false;
    void CaptureProcess();
    void CaptureClose();
    static void CaptureRenderCallback(CKRenderContext *rc, void *self);
    void CaptureFrame(CKRenderContext *rc);
    std::string m_CapturePath;
    void *m_CaptureFile = nullptr;           // HANDLE (file or named pipe)
    CKRenderContext *m_CaptureRc = nullptr;  // where the render callback is registered
    double m_CaptureFps = 60.0;
    double m_CaptureTime = 0.0;              // game time (ms) since the run's first frame (sum of the frame deltas)
    long long m_CaptureNext = 0;             // index of the next video frame (due at m_CaptureNext * 1000 / fps ms)
    uint32_t m_CaptureRepeat = 0;            // video frames the frame being rendered now stands for (0 = skip it)
    size_t m_CaptureLastIndex = 0;           // m_FrameIndex at the previous OnProcess
    uint32_t m_CaptureFrames = 0, m_CaptureWritten = 0, m_CaptureFailed = 0;
    bool m_CaptureStarted = false, m_CaptureDone = false;
    std::vector<uint8_t> m_CaptureBuf;

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
    bool m_KeepFocus = false;         // BALLRAIN_KEEP_FOCUS=1: never hand the foreground back
    unsigned long long m_FocusCheckMs = 0;   // GetTickCount64 of the last foreground check in OnProcess
    bool m_ReloadWaiting = false;
    std::chrono::steady_clock::time_point m_ReloadDeadline;
    std::chrono::steady_clock::time_point m_StartTime;

    FILE *m_Trace = nullptr;
    FILE *m_Events = nullptr;
    std::vector<Frame> m_FrameLog;
    std::string m_PhysicsHex;

public:
    // ------------------------------------------------------------ agent mode (Agent.cpp)
    enum class Phase { Idle, Running, Tail, Post, FinalPending, Ended };
    struct Ev {
        uint32_t frame;
        uint16_t code;
        uint16_t param;
    };
    struct PhysRec {
        uint32_t seq = 0, frame = 0;
        CK_ID ent = 0;
        uint8_t kind = 0;   // 0 physicalize, 1 unphysicalize
        uint8_t fixed = 0, frozen = 0, coll = 0, calcmc = 0;
        float friction = 0, elasticity = 0, mass = 0, linDamp = 0, rotDamp = 0;
        float massCenter[3] = {};
        std::string collGroup, collSurface;
        std::vector<CK_ID> convex, concave;
        std::vector<float> balls;   // x y z r per ball
    };
    struct HsSnap {
        uint8_t valid = 0;
        uint8_t types[6] = {};
        uint32_t energy[6] = {};
        uint8_t bonusType = 0;
        uint32_t levelBonus = 0;
        int32_t bmlHs = 0;
        int32_t hs = 0;
    };
    struct WatchLast {
        uint8_t vis;
        float pos[3];
    };
    struct ArrSnap {
        CK_ID id = 0;
        int rows = -1, cols = -1;
        std::vector<std::string> cells;
    };
    struct FollowerSlot {   // a watched P_Extra_Point_Ball<k>[<nnn>] entity (k 0 = the score ball itself)
        uint32_t watch;     // index into m_Watch
        uint16_t source;    // score-ball instance (name suffix)
        uint8_t k;
    };

private:
    bool AgentReadEnvironment();
    bool AgentOpenTransport();
    void AgentHello();
    void AgentSetState(uint32_t state);
    void AgentFail(uint32_t code, const char *fmt, ...);
    void AgentStart();
    void AgentEvent(uint16_t code, uint16_t param = 0, const char *name = nullptr);
    bool AgentPreTime();
    bool AgentPreInput();
    void AgentProcess();
    void AgentAfterStop(StopReason reason);
    void AgentFinish(uint32_t endReason);
    bool AgentWaitAction();
    void AgentPublish(bool final);
    void AgentBuildState(bool minimal);
    void AgentSnapHs(HsSnap &hs);
    uint16_t AgentCountFollowers();
    void AgentPostLog(const char *tag);
    void AgentEventsLine(const char *fmt, ...);
    void AgentLevelTables();
    void AgentWriteLevelMessage();
    void AgentExportPhysicalized();
    void AgentRebuildEntityMap();
    unsigned char *AgentFindBallReal(CK3dEntity *ball);
    unsigned char *AgentClusterStart();
    void AgentWalkCluster(unsigned char *anyReal);
    CK_ID AgentCkidOf(const void *client) const;

    // Transport.
    void *m_ShmMap = nullptr, *m_EvObs = nullptr, *m_EvAct = nullptr, *m_ParentProc = nullptr;
    uint8_t *m_Shm = nullptr;
    uint32_t m_ShmSize = 0, m_ObsOff = 0, m_ObsCap = 0, m_ActOff = 0, m_ActCap = 0;
    uint64_t m_ModSeq = 0, m_PySeq = 0;
    uint32_t m_ModState = 0;

    // Agent configuration.
    std::string m_ShmName, m_LevelMsgPath, m_PrefixPath;
    bool m_PrefixEvents = false;
    uint32_t m_TimeoutMs = 60000;
    uint32_t m_ParentPid = 0;
    uint32_t m_MaxFrames = 0;   // 0 = no limit
    float m_BodyRadius = 60.0f;
    uint32_t m_SrTail = 32;
    uint32_t m_PostFinish = 0;      // BALLRAIN_POST_FINISH: idle frames run after the SR tail (diagnostic)
    std::string m_PostLogPath;      // BALLRAIN_POST_LOG: per-frame text log from LevelFinish to the end

    // Frame source.
    std::vector<Frame> m_Prefix;
    bool m_InPrefix = false;
    std::vector<Frame> m_ActFrames;   // listed frames of the current block (each repeated m_Repeat times)
    uint32_t m_Repeat = 1;
    size_t m_BlockStart = 0, m_BlockEnd = 0;   // frame indices [start, end) the current source may supply
    uint64_t m_StopMask = 0;
    uint32_t m_ObsMask = 0;
    bool m_WantTrac = false;
    uint32_t m_Covered = 0;   // frames injected since the last observation

    // Run phase.
    Phase m_Phase = Phase::Idle;
    uint32_t m_EndReason = 0;       // end reason of the run (protocol 2: end_reason)
    uint32_t m_EndAtNextTime = 0;   // end reason applied at the next Time callback (GameOver-like stops)
    uint32_t m_TailFrames = 0;
    uint32_t m_PostFrames = 0;      // frames of the post-finish window (after the SR tail)
    FILE *m_PostLog = nullptr;
    std::vector<ArrSnap> m_PostArrs;
    std::vector<WatchLast> m_PostWatch;
    float m_LastDelta = 0.0f;
    bool m_PrefixObs = false;       // the built state is a prefix-frame observation (BALLRAIN_PREFIX_EVENTS)
    bool m_RewriteLevelMsg = false;

    // Events.
    std::vector<Ev> m_Evs;          // since the last observation
    uint64_t m_FrameEvMask = 0;     // event codes seen while m_FrameIndex == m_FrameEvFrame
    uint32_t m_FrameEvFrame = 0;
    uint32_t m_BallOffs = 0, m_LastBallOff = 0, m_LastBallNav = 0;
    CK3dEntity *m_LastBall = nullptr;

    // SR timer (BML+ 0.3.12 rule) and HS.
    double m_Sr = 0.0;
    float m_Sr32 = 0.0f;
    bool m_SrRunning = false;
    bool m_SrInactiveSeen = false;   // the timer was paused after LevelFinish
    bool m_LevelFinished = false;
    double m_SrAtFinish = 0.0;
    float m_Sr32AtFinish = 0.0f;
    HsSnap m_HsAtFinish;
    uint16_t m_FollowersAtFinish = 0;   // small score balls still following at the LevelFinish hook

    // Observation buffers.
    std::vector<uint8_t> m_StateBuf;
    size_t m_StateLen = 0;
    uint32_t m_StateSections = 0;
    bool m_StateDropped = false;
    bool m_StateMinimal = false;
    struct TracRec {
        uint32_t frame;
        float m[12];
    };
    std::vector<TracRec> m_Trac;
    bool m_TracOverflow = false;

    // Level tables (resolved at StartLevel).
    CK_ID m_CamIds[5] = {};
    std::vector<CK_ID> m_ContactBBs, m_ContactHandles;
    std::vector<CK_ID> m_Watch;
    std::vector<CK_ID> m_WatchGroup;   // group of each watched object (first match)
    std::vector<WatchLast> m_WatchLast;
    std::vector<FollowerSlot> m_FollowerSlots;
    std::vector<int8_t> m_SourceMainVis;   // scratch: visibility of each source's score ball
    std::vector<ArrSnap> m_Arrs;
    CK_ID m_EnergyId = 0, m_AllLevelId = 0;
    bool m_TablesReady = false;
    bool m_LevelMsgPending = false;
    FILE *m_LevelInc = nullptr;

    // Physicalize records.
    std::vector<PhysRec> m_Phys;
    size_t m_PhysExported = 0;           // records [0, m_PhysExported) are in the .inc file
    std::vector<uint32_t> m_PhysNew;     // record seqs appended to the .inc since the last observation
    std::vector<std::pair<CK_ID, CK_ID>> m_MeshesDone;   // (entity, mesh) pairs exported to the .inc
    std::unordered_set<uint64_t> m_IvpoDone;            // (entity << 32 | IVP object) pairs exported to the .inc

    // IVP discovery.
    std::unordered_map<const void *, CK_ID> m_EntOf;   // CK3dEntity* -> ckid
    std::vector<unsigned char *> m_Reals;             // IVP real objects found by the last cluster walk
    unsigned char *m_BallReal = nullptr;
    CK3dEntity *m_BallRealOf = nullptr;
};

