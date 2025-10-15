#include "BallrainIO.h"
#include "TASHook.h"
#include <format>

void BallrainIO::OnLoad() {
    GetLogger()->Info("Hello from BallrainIO!");
    m_BML->SendIngameMessage("\x1b[32mHello BML+!\x1b[0m");
    m_BML->AddTimer(1000ul, [](){});

    m_inputSystem = std::make_unique<InputSystem>(m_BML->GetInputManager());
    m_timeSystem = std::make_unique<TimeSystem>(m_BML->GetTimeManager());
    m_tcpClient = std::make_unique<TCPClient>();

    if (!m_tcpClient->Connect("127.0.0.1", 27787)) {
        m_BML->SendIngameMessage("\x1b[32mConnect failed!\x1b[0m");
        return;
    }
    std::string msg = "Pong";
    assert(m_tcpClient->Receive(msg.length(), msg.data()) == msg.length());
    if (msg == "Ping") {
        msg = "Pong";
        assert(m_tcpClient->Send(msg.c_str(), msg.length()) == msg.length());
    }

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
        if (!m_BML->IsPlaying())
            return;
        auto* timeManager = static_cast<CKTimeManager*>(man);
        timeManager->SetLastDeltaTime(1000.0f / 132.0f / 2.0f);    
    });
}

void BallrainIO::OnUnload() {}

void BallrainIO::OnModifyConfig(const char* category, const char* key, IProperty* prop) {}

void BallrainIO::OnLoadObject(const char* filename, CKBOOL isMap, const char* masterName, CK_CLASSID filterClass,
    CKBOOL addToScene, CKBOOL reuseMeshes, CKBOOL reuseMaterials, CKBOOL dynamic,
    XObjectArray* objArray, CKObject* masterObj) {
    if (strcmp(filename, "3D Entities\\Gameplay.nmo") == 0) {
        m_currentLevelArray = m_BML->GetArrayByName("CurrentLevel");
        m_inGameParameterArray = m_BML->GetArrayByName("IngameParameter");
    } else if (strcmp(filename, "3D Entities\\Balls.nmo") == 0) {
        InitBallInfo();
    } else if (isMap) {
        m_sectorObjects.clear();

        auto* fourFlamesObj = m_BML->Get3dObjectByName("PS_FourFlames_01");
        m_sectorObjects.emplace_back(fourFlamesObj);

        char name[256];
        for (int i = 1; i < 100; ++i) {
            snprintf(name, 256, "PC_TwoFlames_%02d", i);
            auto* twoFlamesObj = m_BML->Get3dObjectByName(name);
            if (twoFlamesObj == nullptr)
                break;

            m_sectorObjects.emplace_back(twoFlamesObj);
        }

        auto* balloonObj = m_BML->Get3dObjectByName("PE_Balloon_01");
        m_sectorObjects.emplace_back(balloonObj);
    }
}

void BallrainIO::OnLoadScript(const char* filename, CKBehavior* script) {}

void BallrainIO::OnProcess() {
    if (!m_ballNavActive)
        return;

    auto* ball = GetCurrentBall();
    MsgGameState gameState;
    gameState.ballType = GetBallID(ball);
    ball->GetPosition(&gameState.position);
    ball->GetQuaternion(&gameState.quaternion);
    gameState.currentSector = GetCurrentSector();
    GetNextSectorObject(gameState.currentSector)->GetPosition(&gameState.nextSectorPosition);
    auto sentsz = m_tcpClient->SendMsg(MessageType::BRM_GameState, &gameState);
    assert(sentsz == sizeof(MessageType) + sizeof(MsgGameState));

    sentsz = m_tcpClient->SendMsg(MessageType::BRM_Tick);
    assert(sentsz == sizeof(MessageType));

    MessageType msgType = MessageType::BRM_InvalidType;
    while (msgType != MessageType::BRM_KbdInput)
        msgType = m_tcpClient->ReceiveMsg();
    m_inputSystem->Process(m_tcpClient->GetMessageFromBuf<MsgKbdInput>()->keyState);
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

void BallrainIO::OnBallNavActive() {
    m_tcpClient->SendMsg(MessageType::BRM_BallNavActive);
    m_ballNavActive = true;
}

void BallrainIO::OnBallNavInactive() {
    m_tcpClient->SendMsg(MessageType::BRM_BallNavInactive);
    m_ballNavActive = false;
}

void BallrainIO::OnCamNavActive() {}
void BallrainIO::OnCamNavInactive() {}

void BallrainIO::OnBallOff() {}

void BallrainIO::OnPreCheckpointReached() {}
void BallrainIO::OnPostCheckpointReached() {
    
}

void BallrainIO::OnLevelFinish() {}

void BallrainIO::OnGameOver() {}

void BallrainIO::OnExtraPoint() {}

void BallrainIO::OnPreSubLife() {}
void BallrainIO::OnPostSubLife() {}

void BallrainIO::OnPreLifeUp() {}
void BallrainIO::OnPostLifeUp() {}

CK3dObject* BallrainIO::GetCurrentBall()
{
    if (!m_currentLevelArray)
        return nullptr;
    return static_cast<CK3dObject*>(m_currentLevelArray->GetElementObject(0, 1));
}

int BallrainIO::GetBallID(CK3dObject* ball)
{
    if (ball == nullptr)
        return -1;

    for (int i = 0; i < m_ballNames.size(); ++i) {
        if (m_ballNames[i] == ball->GetName()) {
            return i;
        }
    }
    return -1;
}

int BallrainIO::GetCurrentSector()
{
    // Get sector index (1-indexed)
    int sector = -1;
    if (!m_inGameParameterArray)
        return sector;

    m_inGameParameterArray->GetElementValue(0, 1, &sector);
    return sector;
}

CK3dObject* BallrainIO::GetNextSectorObject(int sector)
{
    return m_sectorObjects[sector];
}

void BallrainIO::InitBallInfo()
{
    auto* phBallArray = m_BML->GetArrayByName("Physicalize_GameBall");
    
    auto ballTypeCount = phBallArray->GetRowCount();
    m_ballNames.clear();
    m_ballNames.reserve(ballTypeCount);
    for (int i = 0; i < ballTypeCount; ++i) {
        std::string ballName;
        ballName.resize(phBallArray->GetElementStringValue(i, 0, nullptr), '\0');
        phBallArray->GetElementStringValue(i, 0, ballName.data());
        ballName.pop_back();
        m_ballNames.emplace_back(ballName);

        //CK3dObject* ball;
        //ball = m_BML->Get3dObjectByName(ballName.data());
        /*CKDependencies dep;
        dep.Resize(40); dep.Fill(0);
        dep.m_Flags = CK_DEPENDENCIES_CUSTOM;
        dep[CKCID_OBJECT] = CK_DEPENDENCIES_COPY_OBJECT_NAME | CK_DEPENDENCIES_COPY_OBJECT_UNIQUENAME;
        dep[CKCID_MESH] = CK_DEPENDENCIES_COPY_MESH_MATERIAL;
        dep[CKCID_3DENTITY] = CK_DEPENDENCIES_COPY_3DENTITY_MESH;
        ball = static_cast<CK3dObject*>(m_BML->GetCKContext()->CopyObject(ball, &dep, "_Peer_"));
        if (!ball) {
            m_BML->SendIngameMessage(std::format("Failed to init ball: {}", i).c_str());
            continue;
        }
        assert(ball->GetMeshCount() == 1);
        for (int j = 0; j < ball->GetMeshCount(); j++) {
            CKMesh* mesh = ball->GetMesh(j);
            assert(mesh->GetMaterialCount() >= 1);
            for (int k = 0; k < mesh->GetMaterialCount(); k++) {
                CKMaterial* mat = mesh->GetMaterial(k);
                mat->EnableAlphaBlend();
                mat->SetSourceBlend(VXBLEND_SRCALPHA);
                mat->SetDestBlend(VXBLEND_INVSRCALPHA);
                VxColor color = mat->GetDiffuse();
                mat->SetDiffuse(color);
                m_BML->SetIC(mat);
            }
        }
        m_balls.emplace_back(ball);*/
    }
}
