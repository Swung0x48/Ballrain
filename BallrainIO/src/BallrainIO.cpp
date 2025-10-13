#include "BallrainIO.h"
#include "TASHook.h"

void BallrainIO::OnLoad() {
    GetLogger()->Info("Hello from BallrainIO!");
    m_BML->SendIngameMessage("\x1b[32mHello BML+!\x1b[0m");
    m_BML->AddTimer(1000ul, [](){});

    m_inputSystem = std::make_unique<InputSystem>(m_BML->GetInputManager());
    m_timeSystem = std::make_unique<TimeSystem>(m_BML->GetTimeManager());

    MH_STATUS status = MH_Initialize();
    if (status != MH_OK && status != MH_ERROR_ALREADY_INITIALIZED) {
        GetLogger()->Error("MinHook failed to initialize: %s", MH_StatusToString(status));
    }

    CKContext* context = m_BML->GetCKContext();
    auto* timemgr = context->GetManagerByGuid(TIME_MANAGER_GUID);
    if (!CKTimeManagerHook::Enable(timemgr)) {
        GetLogger()->Error("Failed to enable TimeManager hook.");
    }
    auto* inputmgr = context->GetManagerByGuid(INPUT_MANAGER_GUID);
    if (!CKInputManagerHook::Enable(inputmgr)) {
        GetLogger()->Error("Failed to enable InputManager hook.");
    }
    CKTimeManagerHook::AddPostCallback([this](CKBaseManager* man) {
        auto* timeManager = static_cast<CKTimeManager*>(man);
        if (m_BML->IsPlaying()) {
            timeManager->SetLastDeltaTime(1000.0f / 132.0f / 2.0f);
        }
    });
}

void BallrainIO::OnUnload() {}

void BallrainIO::OnModifyConfig(const char* category, const char* key, IProperty* prop) {}

void BallrainIO::OnLoadObject(const char* filename, CKBOOL isMap, const char* masterName, CK_CLASSID filterClass,
    CKBOOL addToScene, CKBOOL reuseMeshes, CKBOOL reuseMaterials, CKBOOL dynamic,
    XObjectArray* objArray, CKObject* masterObj) {

}
void BallrainIO::OnLoadScript(const char* filename, CKBehavior* script) {}

void BallrainIO::OnProcess() {
    if (!m_BML->IsPlaying())
        return;

    m_timeSystem->Process();
    m_inputSystem->Process();
}

void BallrainIO::OnRender(CK_RENDER_FLAGS flags) {}

void BallrainIO::OnCheatEnabled(bool enable) {}

void BallrainIO::OnPhysicalize(CK3dEntity* target, CKBOOL fixed, float friction, float elasticity, float mass,
    const char* collGroup, CKBOOL startFrozen, CKBOOL enableColl, CKBOOL calcMassCenter,
    float linearDamp, float rotDamp, const char* collSurface, VxVector massCenter,
    int convexCnt, CKMesh** convexMesh, int ballCnt, VxVector* ballCenter, float* ballRadius,
    int concaveCnt, CKMesh** concaveMesh) {
}

void BallrainIO::OnUnphysicalize(CK3dEntity* target) {}

void BallrainIO::OnPreCommandExecute(ICommand* command, const std::vector<std::string>& args) {}

void BallrainIO::OnPostCommandExecute(ICommand* command, const std::vector<std::string>& args) {}

void BallrainIO::OnPreStartMenu() {}
void BallrainIO::OnPostStartMenu() {}

void BallrainIO::OnExitGame() {}

void BallrainIO::OnPreLoadLevel() {}
void BallrainIO::OnPostLoadLevel() {}

void BallrainIO::OnStartLevel() {}

void BallrainIO::OnPreResetLevel() {}
void BallrainIO::OnPostResetLevel() {}

void BallrainIO::OnPauseLevel() {}
void BallrainIO::OnUnpauseLevel() {}

void BallrainIO::OnPreExitLevel() {}
void BallrainIO::OnPostExitLevel() {}

void BallrainIO::OnPreNextLevel() {}
void BallrainIO::OnPostNextLevel() {}

void BallrainIO::OnDead() {}

void BallrainIO::OnPreEndLevel() {}
void BallrainIO::OnPostEndLevel() {}

void BallrainIO::OnCounterActive() {}
void BallrainIO::OnCounterInactive() {}

void BallrainIO::OnBallNavActive() {}
void BallrainIO::OnBallNavInactive() {}

void BallrainIO::OnCamNavActive() {}
void BallrainIO::OnCamNavInactive() {}

void BallrainIO::OnBallOff() {}

void BallrainIO::OnPreCheckpointReached() {}
void BallrainIO::OnPostCheckpointReached() {}

void BallrainIO::OnLevelFinish() {}

void BallrainIO::OnGameOver() {}

void BallrainIO::OnExtraPoint() {}

void BallrainIO::OnPreSubLife() {}
void BallrainIO::OnPostSubLife() {}

void BallrainIO::OnPreLifeUp() {}
void BallrainIO::OnPostLifeUp() {}