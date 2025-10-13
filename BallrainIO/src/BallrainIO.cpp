#include "BallrainIO.h"

void BallrainIO::OnLoad() {
    GetLogger()->Info("Hello from BallrainIO!");
    m_BML->SendIngameMessage("\x1b[32mHello BML+!\x1b[0m");
    m_BML->AddTimer(1000ul, [](){});
}

void BallrainIO::OnUnload() {}

void BallrainIO::OnModifyConfig(const char* category, const char* key, IProperty* prop) {}

void BallrainIO::OnLoadObject(const char* filename, CKBOOL isMap, const char* masterName, CK_CLASSID filterClass,
    CKBOOL addToScene, CKBOOL reuseMeshes, CKBOOL reuseMaterials, CKBOOL dynamic,
    XObjectArray* objArray, CKObject* masterObj) {

}
void BallrainIO::OnLoadScript(const char* filename, CKBehavior* script) {}

void BallrainIO::OnProcess() {}

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