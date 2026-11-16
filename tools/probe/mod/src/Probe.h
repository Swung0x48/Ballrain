#pragma once

// Throwaway state probe: BallrainIO 0.2.0 replay (LOADS=1 path only) + read-only state dumps.
// Env: the BallrainIO replay contract + BALLRAIN_PROBE=<dir> (outputs).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <BML/BMLAll.h>

struct Frame {
    float deltaTime;
    uint32_t keys;
};
static_assert(sizeof(Frame) == 8, "legacy frame is 8 bytes");

class BallrainProbe final : public IMod {
public:
    explicit BallrainProbe(IBML *bml) : IMod(bml) {}

    const char *GetID() override { return "BallrainProbe"; }
    const char *GetVersion() override { return "0.0.1"; }
    const char *GetName() override { return "BallrainProbe"; }
    const char *GetAuthor() override { return "Swung0x48"; }
    const char *GetDescription() override { return "Ballrain read-only state probe."; }
    DECLARE_BML_VERSION;

    void OnLoad() override;
    void OnUnload() override;
    void OnLoadObject(const char *filename, CKBOOL isMap, const char *masterName, CK_CLASSID filterClass,
                      CKBOOL addToScene, CKBOOL reuseMeshes, CKBOOL reuseMaterials, CKBOOL dynamic,
                      XObjectArray *objArray, CKObject *masterObj) override;
    void OnLoadScript(const char *filename, CKBehavior *script) override;
    void OnProcess() override;

    void OnPhysicalize(CK3dEntity *target, CKBOOL fixed, float friction, float elasticity, float mass,
                       const char *collGroup, CKBOOL startFrozen, CKBOOL enableColl, CKBOOL calcMassCenter,
                       float linearDamp, float rotDamp, const char *collSurface, VxVector massCenter,
                       int convexCnt, CKMesh **convexMesh, int ballCnt, VxVector *ballCenter, float *ballRadius,
                       int concaveCnt, CKMesh **concaveMesh) override;
    void OnUnphysicalize(CK3dEntity *target) override;

    void OnPostStartMenu() override;
    void OnPreLoadLevel() override;
    void OnPostLoadLevel() override;
    void OnStartLevel() override;
    void OnPreResetLevel() override;
    void OnPostResetLevel() override;
    void OnPauseLevel() override;
    void OnUnpauseLevel() override;
    void OnPreExitLevel() override;
    void OnDead() override;
    void OnPreEndLevel() override;
    void OnPostEndLevel() override;
    void OnCounterActive() override;
    void OnCounterInactive() override;
    void OnLevelFinish() override;
    void OnGameOver() override;
    void OnCamNavActive() override;
    void OnCamNavInactive() override;
    void OnBallNavActive() override;
    void OnBallNavInactive() override;
    void OnBallOff() override;
    void OnPreCheckpointReached() override;
    void OnPostCheckpointReached() override;
    void OnExtraPoint() override;
    void OnPreSubLife() override;
    void OnPostSubLife() override;
    void OnPreLifeUp() override;
    void OnPostLifeUp() override;

private:
    enum class StopReason { RecordEnd, ResetLevel, ExitLevel, LevelFinish, GameOver };

    bool ReadEnvironment();
    bool VerifyPhysicsDll(bool mustBeLoaded = true);
    void ScheduleAutoLoad();
    void Start();
    void Stop(StopReason reason);
    void OnPreProcessTime();
    void OnPreProcessInput();
    const Frame *GetFrame(size_t index) const;
    void AcquireKeyBindings();
    void ResetPhysicsTime();
    void SetKeyboardState(unsigned char *dest, uint32_t keys) const;
    void ResetKeyboardState(unsigned char *dest) const;
    CK3dEntity *GetActiveBall() const;
    void OpenOutputs();
    void CloseOutputs(long long wallMs);
    void Event(const char *name);

    // ---- probe (ProbeDump.cpp)
    void ProbeEvent(const char *name);
    void DumpStatic(const char *tag);
    void ExportPendingMeshes();
    void ProbeFrame();
    void ProbeIvpBall(CK3dEntity *ball);
    void ProbeBodies(CK3dEntity *ball);
    void ProbeArrays();
    void ProbeWatch();
    void ProbeBehaviors();
    unsigned char *FindRealObject(CK3dEntity *ent);
    void RebuildEntitySet();
    void WalkCluster(unsigned char *anyReal);

    struct PendingPhys {
        CK_ID entity;
        std::vector<std::pair<CK_ID, int>> meshes;   // mesh id, kind 0 convex 1 concave
    };
    std::vector<PendingPhys> m_PendingPhys;
    std::set<std::pair<CK_ID, CK_ID>> m_ExportedMeshes;
    std::set<CK_ID> m_PhysEntities, m_FixedEntities;
    std::set<const void *> m_EntitySet;
    std::unordered_map<const void *, unsigned char *> m_RealOf;   // CK3dEntity* -> IVP_Real_Object*
    std::set<unsigned char *> m_PartnersSeen;
    std::map<std::string, std::vector<std::string>> m_ArrayLast;
    std::map<CK_ID, std::string> m_WatchLast;
    std::vector<CK_ID> m_Watch, m_Cams, m_ContactBBs, m_CollBBs;
    std::map<CK_ID, std::string> m_CollLast;
    std::string m_ProbeDir;
    FILE *m_PF = nullptr;   // frames.txt
    FILE *m_PP = nullptr;   // phys.txt
    FILE *m_PM = nullptr;   // meshes.txt
    int m_SynL = -1, m_FrL = -1;   // detected l_obj offsets in IVP_Synapse_Real / IVP_Synapse_Friction
    int m_StaticDumps = 0;
    unsigned m_ProbeFrames = 0;

    bool m_Active = false;
    std::string m_FramesPath, m_TracePath, m_FrameLogPath, m_EventsPath;
    int m_Level = 0;
    bool m_SkipRender = false;
    bool m_ExitWhenDone = false;

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

    int m_PhysicsOk = 0;
    bool m_CallbacksRegistered = false;

    std::vector<Frame> m_Frames;
    bool m_Playing = false;
    bool m_GameOverPending = false;
    size_t m_FrameIndex = 0;
    size_t m_DeltaFrame = SIZE_MAX;
    float m_InjectedDelta = 0.0f;
    int m_LoadIndex = 0;
    bool m_AutoLoadArmed = true;
    std::chrono::steady_clock::time_point m_StartTime;

    FILE *m_Trace = nullptr;
    FILE *m_Events = nullptr;
    std::vector<Frame> m_FrameLog;
};
