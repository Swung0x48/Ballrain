#include "BallrainIO.h"
#include "TASHook.h"
#include <format>
#include <fstream>

void BallrainIO::OnLoad() {
    GetLogger()->Info("Hello from BallrainIO!");
    m_BML->SendIngameMessage("\x1b[32mLoaded Ballrain!\x1b[0m");
    m_BML->AddTimer(1000ul, [](){});

    m_config = new BallrainConfig([this]() { return GetConfig(); });
    m_config->CreateProperty<bool>("General", "OnlyPhysicalized", true, "Only show physicalized objects (including hidden ones).");

    m_inputSystem = std::make_unique<InputSystem>(m_BML->GetInputManager());
    m_timeSystem = std::make_unique<TimeSystem>(m_BML->GetTimeManager());
    m_tcpClient = std::make_unique<TCPClient>();

    m_showBoxBB = (CKBehavior*)m_BML->GetCKContext()->CreateObject(CKCID_BEHAVIOR);
    m_showBoxBB->InitFromGuid(CKGUID(0x17cb4c57, 0xf525fb));
    GetLogger()->Info("bb: %s", m_showBoxBB->GetName());

    if (!m_tcpClient->Connect("127.0.0.1", 27787)) {
        m_BML->SendIngameMessage("\x1b[32mConnect failed!\x1b[0m");
        return;
    }
    std::string msg = "Pong";
    auto len = m_tcpClient->Receive(msg.length(), msg.data());
    assert(len == msg.length());
    if (msg == "Ping") {
        msg = "Pong";
        len = m_tcpClient->Send(msg.c_str(), msg.length());
        assert(len == msg.length());
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
        if (m_config->Get<bool>("OnlyPhysicalized")) {
            CKGroup* physGroups[] = { m_BML->GetGroupByName("Phys_Floors"), m_BML->GetGroupByName("Phys_FloorRails") };
            const auto IsPhysicalized = [&physGroups](CK3dObject* obj) -> bool {
                for (auto* group : physGroups) {
                    if (!group) continue;
                    if (obj->IsInGroup(group))
                        return true;
                }
                return false;
            };

            for (int i = 0; i < objArray->Size(); ++i) {
                auto* obj = CK3dObject::Cast(m_BML->GetCKContext()->GetObject((*objArray)[i]));
                if (!obj) continue;
                obj->Show(IsPhysicalized(obj) ? CKSHOW : CKHIDE);
            }
        }

        // Sector
        m_sectorPositions.clear();

        VxVector pos;
        auto* fourFlamesObj = m_BML->Get3dObjectByName("PS_FourFlames_01");
        fourFlamesObj->GetPosition(&pos);
        m_sectorPositions.emplace_back(pos);

        char name[64];
        int sectorCount = 1;
        for (; sectorCount <= 999; ++sectorCount) { // ExtraSector mod allows loading up to 999 sectors
            // checks if the next sector exists
            int sector = sectorCount + 1; // assume we have at least 1 sector (map fails to load otherwise)
            snprintf(name, sizeof(name), "Sector_%02d", sector);
            if (m_BML->GetGroupByName(sector == 9 ? "Sector_9" : name) == nullptr) // see pull request #1
                break;
        }

        for (int i = 1; i <= sectorCount - 1; ++i) {
            // we could write `i < sectorCount` here but it's better to be explicit
            snprintf(name, sizeof(name), "PC_TwoFlames_%02d", i);
            auto* twoFlamesObj = m_BML->Get3dObjectByName(name);
            if (twoFlamesObj == nullptr)
                break;

            twoFlamesObj->GetPosition(&pos);
            m_sectorPositions.emplace_back(pos);
        }
        // NOTE: if we break before finding all sector's corresponding checkpoints
        //     it means the map can't be completed although it will load fine
        //     commonly seen with unfinished custom maps
        // TODO: display some warning about unable to finish here, instead of crashing
        assert(m_sectorPositions.size() == sectorCount);

        auto* balloonObj = m_BML->Get3dObjectByName("PE_Balloon_01");
        balloonObj->GetPosition(&pos);
        m_sectorPositions.emplace_back(pos);

        // Floors
        auto* script = m_BML->GetScriptByName("Gameplay_Refresh");

        auto* floorGroup = m_BML->GetGroupByName("Phys_Floors");
        int floorCount = floorGroup->GetObjectCount();
        for (auto i = 0; i < floorCount; ++i) {
            auto* floor = reinterpret_cast<CK3dObject*>(floorGroup->GetObjectA(i));
            auto& box = floor->GetBoundingBox();
            GetLogger()->Info("%s, box: (%.2f, %.2f, %.2f) (%.2f, %.2f, %.2f)", 
                floor->GetName(),
                box.Min.x, box.Min.y, box.Min.z,
                box.Max.x, box.Max.y, box.Max.z
            );
            m_floorBoxes.emplace_back(box);
            //floor->AddPostRenderCallBack(BallrainIO::ShowBoundingBoxRenderCallBack, m_showBoxBB);
        }
    }
}

CKBOOL
BallrainIO::ShowBoundingBoxRenderCallBack(CKRenderContext* rc, CKRenderObject* obj, void* arg)
{
    CKBehavior* beh = (CKBehavior*)arg;
    CK3dEntity* ent = (CK3dEntity*)obj;
    if (!beh || !ent)
        return FALSE;

    // Projection Trick
    VxMatrix& projmat = (VxMatrix&)rc->GetProjectionTransformationMatrix();
    float oldproj = projmat[3].z;
    projmat[3].z = oldproj * 1.008f; // to disable Z buffer fighting
    rc->SetProjectionTransformationMatrix(projmat);

    CKBOOL DrawBBox;
    beh->GetInputParameterValue(0, &DrawBBox);

    CKBOOL Local;
    CKBOOL Hierachical;
    beh->GetInputParameterValue(2, &Local);
    beh->GetInputParameterValue(3, &Hierachical);

    // Bounding Box
    if (DrawBBox)
    {
        VxColor color;
        beh->GetInputParameterValue(1, &color);

        const VxBbox& Box = Hierachical ? ent->GetHierarchicalBox(Local) : ent->GetBoundingBox(Local);

        rc->SetTexture(NULL);

        if (Local)
            rc->SetWorldTransformationMatrix(ent->GetWorldMatrix());
        else
            rc->SetWorldTransformationMatrix(VxMatrix::Identity());

        // Data
        VxDrawPrimitiveData* data = rc->GetDrawPrimitiveStructure(CKRST_DP_TR_CL_VC, 10);

        // Colors
#if CKVERSION == 0x13022002 || CKVERSION == 0x05082002
        CKDWORD* colors = (CKDWORD*)data->ColorPtr;
        CKDWORD col = RGBAFTOCOLOR(&color);
        VxFillStructure(10, colors, data->ColorStride, 4, &col);

        // Positions
        VxVector Min = Box.Min, Max = Box.Max;
        ((VxVector4*)data->PositionPtr)[0] = Min;
        ((VxVector4*)data->PositionPtr)[1].Set(Max.x, Min.y, Min.z, 1.0f);
        ((VxVector4*)data->PositionPtr)[2].Set(Max.x, Max.y, Min.z, 1.0f);
        ((VxVector4*)data->PositionPtr)[3].Set(Min.x, Max.y, Min.z, 1.0f);

        ((VxVector4*)data->PositionPtr)[4].Set(Min.x, Min.y, Max.z, 1.0f);
        ((VxVector4*)data->PositionPtr)[5].Set(Max.x, Min.y, Max.z, 1.0f);
        ((VxVector4*)data->PositionPtr)[6] = Max;
        ((VxVector4*)data->PositionPtr)[7].Set(Min.x, Max.y, Max.z, 1.0f);
#else
        CKDWORD* colors = (CKDWORD*)data->Colors.Ptr;
        CKDWORD col = RGBAFTOCOLOR(&color);
        VxFillStructure(10, colors, data->Colors.Stride, 4, &col);

        // Positions
        VxVector Min = Box.Min, Max = Box.Max;
        ((VxVector4*)data->Positions.Ptr)[0] = Min;
        ((VxVector4*)data->Positions.Ptr)[1].Set(Max.x, Min.y, Min.z, 1.0f);
        ((VxVector4*)data->Positions.Ptr)[2].Set(Max.x, Max.y, Min.z, 1.0f);
        ((VxVector4*)data->Positions.Ptr)[3].Set(Min.x, Max.y, Min.z, 1.0f);

        ((VxVector4*)data->Positions.Ptr)[4].Set(Min.x, Min.y, Max.z, 1.0f);
        ((VxVector4*)data->Positions.Ptr)[5].Set(Max.x, Min.y, Max.z, 1.0f);
        ((VxVector4*)data->Positions.Ptr)[6] = Max;
        ((VxVector4*)data->Positions.Ptr)[7].Set(Min.x, Max.y, Max.z, 1.0f);
#endif
        // Indices
        CKWORD indices[24] = { 0, 1, 0, 3, 2, 1, 2, 3, 2, 6, 6, 7, 6, 5, 5, 1, 5, 4, 4, 7, 4, 0, 7, 3 };
        rc->DrawPrimitive(VX_LINELIST, indices, 24, data);
    }

    CKBOOL vNormal = FALSE;
    beh->GetInputParameterValue(4, &vNormal);

    CKMesh* mesh = ent->GetCurrentMesh();

    // Vertices Normals
    if (vNormal && mesh)
    {
        VxColor color;
        beh->GetInputParameterValue(5, &color);

        float size = 1.0f;
        beh->GetInputParameterValue(6, &size);

        CKMaterial* mat = (CKMaterial*)beh->GetLocalParameterObject(0);

        int vcount = mesh->GetVertexCount();

        rc->SetTexture(NULL);

        rc->SetWorldTransformationMatrix(ent->GetWorldMatrix());

        CKDWORD flags = CKRST_DP_TR_CL_VC;
        if (mat)
            flags = CKRST_DP_TR_CL_VNT & ~CKRST_DP_STAGES0;

        // Data
        VxDrawPrimitiveData* data = rc->GetDrawPrimitiveStructure((CKRST_DPFLAGS)flags, vcount * 2);

        CKDWORD pS;
        CKBYTE* positions = (CKBYTE*)mesh->GetPositionsPtr(&pS);
        CKDWORD nS;
        CKBYTE* normals = (CKBYTE*)mesh->GetNormalsPtr(&nS);

#if CKVERSION == 0x13022002 || CKVERSION == 0x05082002
        XPtrStrided<VxVector> dpp(data->PositionPtr, data->PositionStride);
        XPtrStrided<VxVector> dpn(data->NormalPtr, data->NormalStride);
#else
        XPtrStrided<VxVector> dpp(data->Positions.Ptr, data->Positions.Stride);
        XPtrStrided<VxVector> dpn(data->Normals.Ptr, data->Normals.Stride);
#endif
        for (int i = 0; i < vcount; ++i, positions += pS, normals += nS)
        {
            VxVector* v = (VxVector*)positions;
            VxVector* n = (VxVector*)normals;
            dpp[i * 2] = *v;
            dpp[i * 2 + 1] = *v + size * (*n);
            if (mat)
            {
                dpn[i * 2] = Normalize(*n);
                dpn[i * 2 + 1] = dpn[i * 2];
            }
        }

        if (!mat)
        {
            // Colors
#if CKVERSION == 0x13022002 || CKVERSION == 0x05082002
            CKDWORD* colors = (CKDWORD*)data->ColorPtr;
            CKDWORD col = RGBAFTOCOLOR(&color);
            VxFillStructure(vcount * 2, colors, data->ColorStride, 4, &col);
#else
            CKDWORD* colors = (CKDWORD*)data->Colors.Ptr;
            CKDWORD col = RGBAFTOCOLOR(&color);
            VxFillStructure(vcount * 2, colors, data->Colors.Stride, 4, &col);
#endif
        }

        if (vcount)
        {
            if (mat)
                mat->SetAsCurrent(rc);

            rc->DrawPrimitive(VX_LINELIST, NULL, vcount * 2, data);
        }
    }

    CKBOOL fNormal = FALSE;
    beh->GetInputParameterValue(7, &fNormal);

    // Faces Normals
    if (fNormal && mesh)
    {
        VxColor color;
        beh->GetInputParameterValue(8, &color);

        float size = 1.0f;
        beh->GetInputParameterValue(9, &size);

        CKMaterial* mat = (CKMaterial*)beh->GetLocalParameterObject(1);

        int fcount = mesh->GetFaceCount();
        int vcount = mesh->GetVertexCount();

        rc->SetTexture(NULL);

        rc->SetWorldTransformationMatrix(ent->GetWorldMatrix());

        CKDWORD flags = CKRST_DP_TR_CL_VC;
        if (mat)
            flags = CKRST_DP_TR_CL_VNT & ~CKRST_DP_STAGES0;

        // Data
        VxDrawPrimitiveData* data = rc->GetDrawPrimitiveStructure((CKRST_DPFLAGS)flags, fcount * 2);

        CKDWORD pS;
        CKBYTE* positions = (CKBYTE*)mesh->GetPositionsPtr(&pS);

#if CKVERSION == 0x13022002 || CKVERSION == 0x05082002
        XPtrStrided<VxVector> dpp(data->PositionPtr, data->PositionStride);
        XPtrStrided<VxVector> dpn(data->NormalPtr, data->NormalStride);
#else
        XPtrStrided<VxVector> dpp(data->Positions.Ptr, data->Positions.Stride);
        XPtrStrided<VxVector> dpn(data->Normals.Ptr, data->Normals.Stride);
#endif

        for (int i = 0; i < fcount; ++i)
        {
            VxVector n = mesh->GetFaceNormal(i);
            VxVector v = (mesh->GetFaceVertex(i, 0) + mesh->GetFaceVertex(i, 1) + mesh->GetFaceVertex(i, 2)) * 0.33333f;
            dpp[i * 2] = v;
            dpp[i * 2 + 1] = v + size * n;

            if (mat)
            {
                dpn[i * 2] = v;
                dpn[i * 2 + 1] = v;
            }
        }

        if (!mat)
        {
            // Colors
#if CKVERSION == 0x13022002 || CKVERSION == 0x05082002
            CKDWORD* colors = (CKDWORD*)data->ColorPtr;
            CKDWORD col = RGBAFTOCOLOR(&color);
            VxFillStructure(fcount * 2, colors, data->ColorStride, 4, &col);
#else
            CKDWORD* colors = (CKDWORD*)data->Colors.Ptr;
            CKDWORD col = RGBAFTOCOLOR(&color);
            VxFillStructure(fcount * 2, colors, data->Colors.Stride, 4, &col);
#endif
        }

        if (fcount)
        {
            if (mat)
                mat->SetAsCurrent(rc);

            rc->DrawPrimitive(VX_LINELIST, NULL, fcount * 2, data);
        }
    }

    CKBOOL showSphere = FALSE;
    beh->GetInputParameterValue(10, &showSphere);

    if (showSphere && mesh)
    {

        rc->SetTexture(NULL);

        rc->SetWorldTransformationMatrix(ent->GetWorldMatrix());

        VxColor color;
        beh->GetInputParameterValue(11, &color);

        int count = 32;

        // Data
        VxDrawPrimitiveData* data = rc->GetDrawPrimitiveStructure(CKRST_DP_TR_CL_VC, count);

        VxVector center;
        mesh->GetBaryCenter(&center);

        float radius = mesh->GetRadius();

        int axis0 = 0;
        int axis1 = 1;
        for (int i = 0; i < 3; ++i)
        {
#if CKVERSION == 0x13022002 || CKVERSION == 0x05082002
            XPtrStrided<VxVector> positions(data->PositionPtr, data->PositionStride);
#else
            XPtrStrided<VxVector> positions(data->Positions.Ptr, data->Positions.Stride);
#endif
            // Positions
            float delta = PI * 2.0f / (count - 1);
            float t = 0.0f;
            for (int j = 0; j < count; ++j, t += delta)
            {
                positions[j] = center;
                positions[j][axis0] += radius * cosf(t);
                positions[j][axis1] += radius * sinf(t);
            }

            // Colors
            CKDWORD col = RGBAFTOCOLOR(&color);
#if CKVERSION == 0x13022002 || CKVERSION == 0x05082002
            VxFillStructure(count, data->ColorPtr, data->ColorStride, 4, &col);
#else
            VxFillStructure(count, data->Colors.Ptr, data->Colors.Stride, 4, &col);
#endif

            // Drawing
            rc->DrawPrimitive(VX_LINESTRIP, NULL, count, data);

            ++axis0;
            ++axis1;
            if (axis1 == 3)
                axis1 = 0;
        }
    }

    projmat[3].z = oldproj; // restore projection matrix
    rc->SetProjectionTransformationMatrix(projmat);

    return TRUE;
}

void BallrainIO::OnLoadScript(const char* filename, CKBehavior* script) {
    if (strcmp(script->GetName(), "Gameplay_Events") == 0) {
        m_escBeh = ScriptHelper::FindFirstBB(script, "Key Event");
    }
    else if (strcmp(script->GetName(), "Menu_Pause") == 0) {
        m_restartBeh = ScriptHelper::FindFirstBB(script, "Restart Level");
    }
    /*else if (strcmp(script->GetName(), "Gameplay_Ingame") == 0) {
        CKBehavior* ballMgr = ScriptHelper::FindFirstBB(script, "BallManager");
        CKBehavior* newBall = ScriptHelper::FindFirstBB(ballMgr, "New Ball");
        dynamic_pos = ScriptHelper::FindNextBB(script, ballMgr, "TT Set Dynamic Position");
        phy_new_ball = ScriptHelper::FindFirstBB(newBall, "physicalize new Ball");
    }*/
}

void BallrainIO::OnProcess() {
    if (m_BML->IsIngame()) {
        // UI
        if (ImGui::Begin("BallrainIO Inspector", nullptr,
                ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoResize)) {
            if (ImGui::TreeNode("State")) {
                ImGui::Text("Ball Type: %d", gameState.ballType);
                ImGui::Text("Position: (%3.2f, %3.2f, %3.2f)", gameState.position.x, gameState.position.y, gameState.position.z);
                ImGui::Text("Quaternion: (%3.2f, %3.2f, %3.2f, %3.2f)", gameState.quaternion.x, gameState.quaternion.y, gameState.quaternion.z, gameState.quaternion.w);
                ImGui::Text("Now at sector #%d", gameState.currentSector);
                ImGui::Text("Next Sector: (%.2f, %.2f, %.2f)", gameState.nextSectorPosition.x, gameState.nextSectorPosition.y, gameState.nextSectorPosition.z);
                ImGui::Text("Last Sector: (%.2f, %.2f, %.2f)", gameState.lastSectorPosition.x, gameState.lastSectorPosition.y, gameState.lastSectorPosition.z);
                ImGui::TreePop();
            }
            if (ImGui::TreeNode("Action")) {
                ImGui::Text("%s %s %s %s",
                    currentkeyState.states[KeyState::BR_UP] ? "U" : " ",
                    currentkeyState.states[KeyState::BR_DOWN] ? "D" : " ",
                    currentkeyState.states[KeyState::BR_LEFT] ? "L" : " ",
                    currentkeyState.states[KeyState::BR_RIGHT] ? "R" : " ");
                ImGui::TreePop();
            }
        }
        ImGui::End();
    }

    if (!m_ballNavActive)
        return;

    auto* ball = GetCurrentBall();
    gameState.ballType = GetBallID(ball);
    ball->GetPosition(&gameState.position);
    ball->GetQuaternion(&gameState.quaternion);
    gameState.currentSector = GetCurrentSector();
    gameState.nextSectorPosition = GetNextSectorPosition(gameState.currentSector);
    gameState.lastSectorPosition = GetLastSectorPosition(gameState.currentSector);

    if (!m_tcpClient->Connected())
        return;
    //GetLogger()->Info("next: %d, last: %d", nextSector->GetName(), lastSector->GetName());
    //nextSector->GetPosition(&gameState.nextSectorPosition);
    //lastSector->GetPosition(&gameState.lastSectorPosition);
    auto sentsz = m_tcpClient->SendMsg(MessageType::BRM_GameState, &gameState);
    assert(sentsz == sizeof(MessageType) + sizeof(MsgGameState));
    //GetLogger()->Info("GameState");

    sentsz = m_tcpClient->SendMsg(MessageType::BRM_DepthImage, depthImage.image.data());
    assert(sentsz == sizeof(MessageType) + depthImage.image.size() * 1);
    //GetLogger()->Info("DepthImage");

    sentsz = m_tcpClient->SendMsg(MessageType::BRM_Tick);
    assert(sentsz == sizeof(MessageType));
    //GetLogger()->Info("Tick");

    MessageType msgType = m_tcpClient->ReceiveMsg();
    if (msgType == MessageType::BRM_ResetInput) {
        RestartLevel();
        return;
    }
    else if (msgType == MessageType::BRM_KbdInput) {
        auto* msg = m_tcpClient->GetMessageFromBuf<MsgKbdInput>();
        currentkeyState = msg->keyState;
        m_inputSystem->Process(msg->keyState);
    }
}

void BallrainIO::OnRender(CK_RENDER_FLAGS flags) {
    if (m_ballNavActive) {
        VxImageDescEx image;
        int size = m_BML->GetRenderContext()->DumpToMemory(nullptr, VXBUFFER_ZBUFFER, image);
        //GetLogger()->Info("size: %d", size);
        if (size > 0) {
            image.Image = (XBYTE*)depthImage.image.data();
            m_BML->GetRenderContext()->DumpToMemory(nullptr, VXBUFFER_ZBUFFER, image);

            /*static int cnt = 0;
            static char buf[100];
            sprintf(buf, "zdumps/d%06d.zdp", cnt++);
            std::ofstream of(buf, std::ios::binary);
            of.write((char*)depthImage.image.data(), size);
            of.flush();
            of.close();*/
        }
    }
}

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

void BallrainIO::OnStartLevel() {
    for (auto bbShowInfo : m_bbShowInformation) {
    bbShowInfo->ActivateInput(0);
    bbShowInfo->Execute(0);
  }
}

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
    MsgSceneRep msg;
    msg.floorBoxes = m_floorBoxes;
    m_tcpClient->SendMsg(MessageType::BRM_MsgSceneRep, &msg);
    m_tcpClient->SendMsg(MessageType::BRM_BallNavActive);
    m_ballNavActive = true;
}

void BallrainIO::OnBallNavInactive() {
    m_tcpClient->SendMsg(MessageType::BRM_BallNavInactive);
    m_ballNavActive = false;
}

void BallrainIO::OnCamNavActive() {}
void BallrainIO::OnCamNavInactive() {}

void BallrainIO::OnBallOff() {
    m_tcpClient->SendMsg(MessageType::BRM_BallOff);
  m_ballNavActive = false;
}

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

VxVector BallrainIO::GetNextSectorPosition(int sector)
{
    return m_sectorPositions[sector];
}

VxVector BallrainIO::GetLastSectorPosition(int sector)
{
    return m_sectorPositions[sector - 1];
}

void BallrainIO::RestartLevel()
{
    if (!m_BML->IsIngame())
        return;

    m_ballNavActive = false;
    m_escBeh->GetOutput(0)->Activate();
    m_BML->AddTimer(CKDWORD(3), [this]() {
        // std::thread([this] {
        CKMessageManager* mm = m_BML->GetMessageManager();
        // std::this_thread::sleep_for(std::chrono::milliseconds(120));

        CKMessageType reset_level_msg = mm->AddMessageType("Reset Level");
        mm->SendMessageSingle(reset_level_msg, static_cast<CKBeObject*>(m_BML->GetCKContext()->GetObjectByNameAndParentClass("Level", CKCID_BEOBJECT, nullptr)));
        mm->SendMessageSingle(reset_level_msg, static_cast<CKBeObject*>(m_BML->GetCKContext()->GetObjectByNameAndParentClass("All_Balls", CKCID_BEOBJECT, nullptr)));

        auto* output = m_restartBeh->GetOutput(0);
        output->Activate();
    });
}

CKBehavior* BallrainIO::CreateShowObjectInformation(CKBehavior* script, CK3dEntity* target, CKBOOL showBoundingBox)
{
    CKBehavior* beh = ScriptHelper::CreateBB(script, VT_VISUALS_SHOWOBJECTINFORMATION, true);
    beh->GetTargetParameter()->SetDirectSource(
        ScriptHelper::CreateParamObject(script, "Target", CKPGUID_3DENTITY, target));
    beh->GetInputParameter(0)->SetDirectSource(
        ScriptHelper::CreateParamValue(script, "Show Bounding Box", CKPGUID_BOOL, showBoundingBox));
    //beh->ActivateInput(0);
    //beh->Execute(0);
    return beh;
}

void BallrainIO::ShowBoundingBox(CK3dEntity* target, CKBOOL show)
{
    //ScriptHelper::SetParamObject(m_bbShowObjectInfo->GetTargetParameter()->GetDirectSource(), target);
    //ScriptHelper::SetParamValue(m_bbShowObjectInfo->GetInputParameter(0)->GetDirectSource(), show);
   // m_bbShowObjectInfo->ActivateInput(0);
 //   m_bbShowObjectInfo->Execute(0);
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
