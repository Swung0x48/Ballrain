#include <BML/IMod.h>
#include <BML/ExecuteBB.h>
#include <BML/ScriptHelper.h>
#include <BML/Guids/Visuals.h>
#include <imgui.h>
#include <VxMath.h>
#include <memory>
#include <vector>
#include <array>

#include <imgui.h>
#include <imgui_internal.h>

#include "TCPClient.h"
#include "InputSystem.h"
#include "TimeSystem.h"
#include "MessageTypes.h"

class BallrainIO final : public IMod {
public:
    explicit BallrainIO(IBML* bml) : IMod(bml) {}
    const char* GetID() override        { return "BallrainIO"; }
    const char* GetVersion() override   { return "1.0.0"; }
    const char* GetName() override      { return "BallrainIO Mod"; }
    const char* GetAuthor() override    { return "Swung0x48"; }
    const char* GetDescription() override { return "blah blah blah"; }
    DECLARE_BML_VERSION;

    void OnLoad() override;
    void OnUnload() override;
    void OnModifyConfig(const char* category, const char* key, IProperty* prop) override;
    void OnLoadObject(const char* filename, CKBOOL isMap, const char* masterName, CK_CLASSID filterClass,
        CKBOOL addToScene, CKBOOL reuseMeshes, CKBOOL reuseMaterials, CKBOOL dynamic,
        XObjectArray* objArray, CKObject* masterObj);
    void OnLoadScript(const char* filename, CKBehavior* script) override;

    void OnProcess() override;
    void OnRender(CK_RENDER_FLAGS flags) override;

    void OnCheatEnabled(bool enable) override;

    void OnPhysicalize(CK3dEntity* target, CKBOOL fixed, float friction, float elasticity, float mass,
        const char* collGroup, CKBOOL startFrozen, CKBOOL enableColl, CKBOOL calcMassCenter,
        float linearDamp, float rotDamp, const char* collSurface, VxVector massCenter,
        int convexCnt, CKMesh** convexMesh, int ballCnt, VxVector* ballCenter, float* ballRadius,
        int concaveCnt, CKMesh** concaveMesh) override;
    void OnUnphysicalize(CK3dEntity* target) override;

    void OnPreCommandExecute(ICommand* command, const std::vector<std::string>& args) override;
    void OnPostCommandExecute(ICommand* command, const std::vector<std::string>& args) override;

    virtual void OnPreStartMenu() override;
    virtual void OnPostStartMenu() override;

    virtual void OnExitGame() override;

    virtual void OnPreLoadLevel() override;
    virtual void OnPostLoadLevel() override;

    virtual void OnStartLevel() override;

    virtual void OnPreResetLevel() override;
    virtual void OnPostResetLevel() override;

    virtual void OnPauseLevel() override;
    virtual void OnUnpauseLevel() override;

    virtual void OnPreExitLevel() override;
    virtual void OnPostExitLevel() override;

    virtual void OnPreNextLevel() override;
    virtual void OnPostNextLevel() override;

    virtual void OnDead() override;

    virtual void OnPreEndLevel() override;
    virtual void OnPostEndLevel() override;

    virtual void OnCounterActive() override;
    virtual void OnCounterInactive() override;

    virtual void OnBallNavActive() override;
    virtual void OnBallNavInactive() override;

    virtual void OnCamNavActive() override;
    virtual void OnCamNavInactive() override;

    virtual void OnBallOff() override;

    virtual void OnPreCheckpointReached() override;
    virtual void OnPostCheckpointReached() override;

    virtual void OnLevelFinish() override;

    virtual void OnGameOver() override;

    virtual void OnExtraPoint() override;

    virtual void OnPreSubLife() override;
    virtual void OnPostSubLife() override;

    virtual void OnPreLifeUp() override;
    virtual void OnPostLifeUp() override;

private:
    void InitBallInfo();

    CK3dObject* GetCurrentBall();

    int GetBallID(CK3dObject* ball);

    int GetCurrentSector();

    VxVector GetLastSectorPosition(int sector);
    VxVector GetNextSectorPosition(int sector);

    void RestartLevel();
    static CKBehavior* CreateShowObjectInformation(
        CKBehavior* script,
        CK3dEntity* target,
        CKBOOL showBoundingBox);
    void ShowBoundingBox(CK3dEntity* target, CKBOOL show);

    static CKBOOL ShowBoundingBoxRenderCallBack(CKRenderContext* rc, CKRenderObject* obj, void* arg);

    std::unique_ptr<InputSystem> m_inputSystem;
    std::unique_ptr<TimeSystem> m_timeSystem;
    std::unique_ptr<TCPClient> m_tcpClient;

    CKBehavior* m_escBeh = nullptr;
    CKBehavior* m_restartBeh = nullptr;

    CKDataArray* m_currentLevelArray = nullptr;
    CKDataArray* m_inGameParameterArray = nullptr;

    std::vector<VxVector> m_sectorPositions;

    bool m_ballNavActive = false;

    //CKBehavior* m_bbShowObjectInfo = nullptr;

    //std::vector<CK3dObject*> m_balls;
    std::vector<std::string> m_ballNames;

    std::vector<VxBbox> m_floorBoxes;

    CKBehavior* m_showBoxBB = nullptr;
    std::vector<CKBehavior*> m_bbShowInformation;

    MsgGameState gameState;
    KeyState currentkeyState;
    MsgDepthImage depthImage;
};

extern "C" __declspec(dllexport) IMod* BMLEntry(IBML* bml) { return new BallrainIO(bml); }
extern "C" __declspec(dllexport) void BMLExit(IMod* mod) { delete mod; }