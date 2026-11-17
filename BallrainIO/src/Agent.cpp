// BallrainIO agent mode: docs/agent-protocol.md (transport, messages, timing) and docs/state-spec.md
// (what is read and how). The injection path is the replay path of BallrainIO.cpp; this file only
// supplies the frames (prefix, then Python's action blocks), blocks for actions, and exports state.
//
// Read-only rules (state-spec.md section 0): CK parameters are read with update=FALSE, attributes by
// name only, no CK object is created, no IVP function is called; raw IVP memory is read with SafeRead
// (SEH + memcpy) and only after VerifyPhysicsDll accepted the 2002 physics_RT.dll.

#include "BallrainIO.h"

#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <unordered_set>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#include "Ivp2002.h"
#include "Wire.h"

using namespace ivp2002;
typedef unsigned char u8;

bool SafeRead(const void *p, void *out, size_t n) {
    if (!p)
        return false;
    __try {
        memcpy(out, p, n);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

namespace {

// ---------------------------------------------------------------- shared-memory header (protocol 2)
constexpr uint32_t SHM_MAGIC = 0x4D535242;   // 'BRSM'
constexpr uint32_t SHM_VERSION = 1;
constexpr uint32_t H_MAGIC = 0x00, H_VERSION = 0x04, H_SIZE = 0x08, H_OBS_OFF = 0x0C, H_OBS_CAP = 0x10,
                   H_ACT_OFF = 0x14, H_ACT_CAP = 0x18, H_MOD_SEQ = 0x20, H_PY_SEQ = 0x28, H_STATE = 0x30,
                   H_END = 0x34, H_MOD_PID = 0x38, H_FRAMES = 0x40, H_ERROR = 0x44, H_OBSVERS = 0x48,
                   H_VERSTR = 0x4C, H_VERSTR_LEN = 180, H_END_OF_HEADER = 0x100;

enum : uint32_t { ST_INIT, ST_HELLO, ST_MENU, ST_LOADING, ST_RUNNING, ST_WAITING, ST_ENDED, ST_ERROR };
enum : uint32_t { END_NONE, END_FINISH, END_GAMEOVER, END_MAXFRAMES, END_STOP, END_RESET, END_TIMEOUT, END_ERROR };
enum : uint32_t { ERR_NONE, ERR_SHM, ERR_ENV, ERR_DLL, ERR_ACT, ERR_SEQ };

constexpr uint32_t ACT_MAGIC = 0x31544341;   // 'ACT1'
constexpr uint32_t ACT_HEADER = 0x24;
constexpr uint32_t ACT_STOP = 1, ACT_TRAC = 2, ACT_LEVELMSG = 4;

// Observation flags.
constexpr uint16_t OBS_FINAL = 1, OBS_PREFIX = 2, OBS_DROPPED = 4, OBS_MINIMAL = 8;
// obs_mask: a set bit disables that optional section.
// (bits as ballrain/protocol.py OBSMASK_*; BALL and ARRS are extensions)
constexpr uint32_t M_BODY = 1u << 0, M_MIND = 1u << 1, M_FRIC = 1u << 2, M_CAMS = 1u << 3, M_GLCT = 1u << 4,
                   M_WTCH = 1u << 5, M_BALL = 1u << 6, M_ARRS = 1u << 7;

// Event codes (protocol 2.3).
enum : uint16_t {
    EV_PRELOAD = 1, EV_POSTLOAD, EV_START, EV_CAMNAV_ON, EV_CAMNAV_OFF, EV_BALLNAV_ON, EV_BALLNAV_OFF,
    EV_COUNTER_ON, EV_COUNTER_OFF, EV_PAUSE, EV_UNPAUSE, EV_BALLOFF, EV_DEAD, EV_PRESUBLIFE, EV_POSTSUBLIFE,
    EV_PRECHECK, EV_POSTCHECK, EV_EXTRAPOINT, EV_PRELIFEUP, EV_POSTLIFEUP, EV_FINISH, EV_GAMEOVER,
    EV_PREENDLEVEL, EV_POSTENDLEVEL, EV_PHYS, EV_UNPHYS, EV_ACTIVEBALL, EV_PRERESET, EV_PREEXIT, EV_RUNEND
};

const char *const kTrackedArrays[] = {"Energy", "CurrentLevel", "IngameParameter", "PH", "Checkpoints", "ResetPoints"};
constexpr int kTrackedCount = 6;
const char *const kCamNames[5] = {"Cam_OrientRef", "Cam_Orient", "Cam_Pos", "Cam_Target", "InGameCam"};
const char *const kAttrNames[3] = {"Continuous Contact ID", "Coll Detection ID", "Floor"};
constexpr uint32_t kNoString = 0xFFFFFFFFu;

std::string GetEnvS(const char *name) {
    const char *v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

std::wstring Widen(const std::string &s) {
    if (s.empty())
        return std::wstring();
    const int n = ::MultiByteToWideChar(CP_ACP, 0, s.c_str(), (int) s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    ::MultiByteToWideChar(CP_ACP, 0, s.c_str(), (int) s.size(), &w[0], n);
    return w;
}

inline volatile uint32_t &H32(uint8_t *shm, uint32_t off) { return *reinterpret_cast<volatile uint32_t *>(shm + off); }
inline volatile uint64_t &H64(uint8_t *shm, uint32_t off) { return *reinterpret_cast<volatile uint64_t *>(shm + off); }

bool StartsWith(const char *s, const char *p) { return s && !strncmp(s, p, strlen(p)); }
bool ContainsI(const char *s, const char *p) {
    if (!s)
        return false;
    std::string a(s), b(p);
    std::transform(a.begin(), a.end(), a.begin(), ::tolower);
    std::transform(b.begin(), b.end(), b.begin(), ::tolower);
    return a.find(b) != std::string::npos;
}

void PutMatrix12(Wire &w, const VxMatrix &m) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 3; ++c)
            w.F32(m[r][c]);
}

bool CoreUnmoveable(const u8 *core) {
    uint32_t v = 0;
    Rd(core, CO_FLAGS, v);
    return ((v >> 2) & 3) != 0;
}

// The partner triangle of a mindist / contact feature (state-spec.md 3.2).
struct Tri {
    const u8 *ledge = nullptr;
    int triIndex = -1, edgeIndex = -1;
    float v[3][3] = {};
};

bool DecodeTri(const u8 *edge, Tri &t) {
    if (!edge)
        return false;
    const u8 *tri = reinterpret_cast<const u8 *>(reinterpret_cast<uintptr_t>(edge) & ~uintptr_t(15));
    uint32_t hdr = 0;
    if (!Rd(tri, 0, hdr))
        return false;
    t.triIndex = (int) (hdr & 0xFFF);
    t.ledge = tri - 16 * t.triIndex - 16;
    int32_t pointOffset = 0;
    int16_t nTri = 0;
    if (!Rd(t.ledge, LEDGE_POINT_OFF, pointOffset) || !Rd(t.ledge, LEDGE_NTRI, nTri))
        return false;
    if (nTri <= 0 || t.triIndex >= nTri)
        return false;
    t.edgeIndex = (int) ((reinterpret_cast<uintptr_t>(edge) & 0xC) >> 2) - 1;
    const u8 *points = t.ledge + pointOffset;
    for (int k = 0; k < 3; ++k) {
        uint32_t e = 0;
        if (!Rd(tri, 4 + 4 * k, e) || !SafeRead(points + 16 * (e & 0xFFFF), t.v[k], 12))
            return false;
    }
    return true;
}

const u8 *OppositeEdge(const u8 *edge) {
    uint32_t e = 0;
    if (!Rd(edge, 0, e))
        return nullptr;
    const int opp = ((int) (e << 1)) >> 17;   // signed 15 bits at 16..30
    return edge + 4 * opp;
}

// Raw bytes of one data-array cell (state-spec.md 0: no parameter operation is pulled).
// int/float/object: the 4-byte element; string: the text (<= 255 bytes); parameter: raw data (<= 64 bytes).
size_t CellBytes(CKDataArray *a, int r, int c, u8 *out /* >= 256 */) {
    static char sbuf[1 << 16];
    const CK_ARRAYTYPE t = a->GetColumnType(c);
    CKDWORD *e = a->GetElement(r, c);
    if (!e)
        return 0;
    switch (t) {
        case CKARRAYTYPE_INT:
        case CKARRAYTYPE_FLOAT:
        case CKARRAYTYPE_OBJECT:
            std::memcpy(out, e, 4);
            return 4;
        case CKARRAYTYPE_STRING: {
            sbuf[0] = 0;
            const int need = a->GetElementStringValue(r, c, nullptr);
            if (need > 0 && need < (int) sizeof(sbuf))
                a->GetElementStringValue(r, c, sbuf);
            else
                return 0;
            const size_t n = std::min<size_t>(strnlen(sbuf, sizeof(sbuf)), 255);
            std::memcpy(out, sbuf, n);
            return n;
        }
        case CKARRAYTYPE_PARAMETER: {
            // The element holds the cell's CKParameterOut pointer (read this way by the probe on all
            // arrays of 5 records; the SDK header comment calls it an id).
            auto *p = (CKParameter *) *e;
            if (!p)
                return 0;
            const void *data = p->GetReadDataPtr(FALSE);
            const int size = std::min(p->GetDataSize(), 64);
            if (!data || size <= 0 || !SafeRead(data, out, (size_t) size))
                return 0;
            return (size_t) size;
        }
        default:
            return 0;
    }
}

// "P_Extra_Point_Ball<k>" or "P_Extra_Point_Ball<k><nnn>" (k 0..6): the score ball (k 0) and its six
// small balls, template or numbered instance (state-spec.md 2.9).
bool ParseFollowerName(const char *n, std::string &source, int &k) {
    static const char kPrefix[] = "P_Extra_Point_Ball";
    const size_t L = sizeof(kPrefix) - 1;
    if (!n || strncmp(n, kPrefix, L) != 0 || n[L] < '0' || n[L] > '6')
        return false;
    const char *rest = n + L + 1;
    const size_t r = strlen(rest);
    if (r != 0 && !(r == 3 && isdigit((unsigned char) rest[0]) && isdigit((unsigned char) rest[1]) &&
                    isdigit((unsigned char) rest[2])))
        return false;
    source.assign(rest, r);
    k = n[L] - '0';
    return true;
}

// Text form of a cell for the post-finish log.
void CellText(CKDataArray *a, int r, int c, char *out, size_t cap) {
    u8 cell[256];
    const size_t len = CellBytes(a, r, c, cell);
    const CK_ARRAYTYPE t = a->GetColumnType(c);
    if (len == 4 && t == CKARRAYTYPE_INT) {
        int32_t v;
        std::memcpy(&v, cell, 4);
        std::snprintf(out, cap, "%d", v);
    } else if (len == 4 && t == CKARRAYTYPE_FLOAT) {
        float v;
        std::memcpy(&v, cell, 4);
        std::snprintf(out, cap, "%.9g", v);
    } else if (len == 4 && t == CKARRAYTYPE_OBJECT) {
        uint32_t v;
        std::memcpy(&v, cell, 4);
        std::snprintf(out, cap, "#%u", v);
    } else if (t == CKARRAYTYPE_STRING) {
        std::snprintf(out, cap, "\"%.*s\"", (int) std::min<size_t>(len, 120), (const char *) cell);
    } else {
        size_t o = 0;
        out[0] = 0;
        for (size_t i = 0; i < len && i < 32 && o + 3 < cap; ++i)
            o += std::snprintf(out + o, cap - o, "%02x", cell[i]);
    }
}

// Per-message string table of the level message.
struct StrTab {
    std::unordered_map<std::string, uint32_t> index;
    std::vector<std::string> list;
    uint32_t Id(const char *s) {
        if (!s)
            return 0xFFFFFFFFu;   // no string
        std::string k = s;
        auto it = index.find(k);
        if (it != index.end())
            return it->second;
        const auto id = (uint32_t) list.size();
        index.emplace(k, id);
        list.push_back(std::move(k));
        return id;
    }
    void Write(Wire &w) const {
        w.Begin("STRS", 1);
        w.U32((uint32_t) list.size());
        for (const std::string &s : list) {
            const size_t n = std::min<size_t>(s.size(), 0xFFFF);
            w.U16((uint32_t) n);
            w.Raw(s.data(), n);
        }
        w.End();
    }
};

void BeginMessage(Wire &w, const char magic[4], uint16_t flags, uint32_t frames) {
    w.Raw(magic, 4);
    w.U16(1);
    w.U16(flags);
    w.U32(0);   // byte_len
    w.U32(frames);
    w.U32(0);   // section_count
    w.U32(0);
}

void EndMessage(Wire &w) {
    w.PatchU32(8, (uint32_t) w.Size());
    w.PatchU32(16, w.Sections());
}

}  // namespace

// ================================================================ environment / transport

bool BallrainIO::AgentReadEnvironment() {
    m_ShmName = GetEnvS("BALLRAIN_SHM");
    if (m_ShmName.empty()) {
        GetLogger()->Error("agent: BALLRAIN_SHM is not set: inactive");
        return false;
    }
    if (!AgentOpenTransport())
        return false;
    if (!GetEnvS("BALLRAIN_FRAMES").empty()) {
        AgentFail(ERR_ENV, "agent: BALLRAIN_FRAMES is set (replay and agent mode are exclusive)");
        return false;
    }
    m_TracePath = GetEnvS("BALLRAIN_TRACE");
    m_FrameLogPath = GetEnvS("BALLRAIN_FRAMELOG");
    m_EventsPath = GetEnvS("BALLRAIN_EVENTS");
    m_LevelMsgPath = GetEnvS("BALLRAIN_LEVELMSG");
    m_PrefixPath = GetEnvS("BALLRAIN_PREFIX");
    m_PrefixEvents = GetEnvS("BALLRAIN_PREFIX_EVENTS") == "1";
    m_Level = std::atoi(GetEnvS("BALLRAIN_LEVEL").c_str());
    m_SkipRender = GetEnvS("BALLRAIN_SKIP_RENDER") == "1";
    m_ExitWhenDone = GetEnvS("BALLRAIN_EXIT") == "1";
    const std::string loads = GetEnvS("BALLRAIN_LOADS");
    if (!loads.empty() && loads != "1") {
        AgentFail(ERR_ENV, "agent: BALLRAIN_LOADS=%s (agent mode supports one load)", loads.c_str());
        return false;
    }
    m_Loads = 1;
    if (m_Level < 1 || m_Level > 13) {
        AgentFail(ERR_ENV, "agent: BALLRAIN_LEVEL=%d is out of 1..13", m_Level);
        return false;
    }
    const std::string timeout = GetEnvS("BALLRAIN_TIMEOUT_MS");
    if (!timeout.empty())
        m_TimeoutMs = (uint32_t) std::strtoul(timeout.c_str(), nullptr, 10);
    m_ParentPid = (uint32_t) std::strtoul(GetEnvS("BALLRAIN_PARENT_PID").c_str(), nullptr, 10);
    m_MaxFrames = (uint32_t) std::strtoul(GetEnvS("BALLRAIN_MAX_FRAMES").c_str(), nullptr, 10);
    const std::string radius = GetEnvS("BALLRAIN_BODY_RADIUS");
    if (!radius.empty())
        m_BodyRadius = (float) std::atof(radius.c_str());
    const std::string tail = GetEnvS("BALLRAIN_SR_TAIL");
    if (!tail.empty())
        m_SrTail = (uint32_t) std::strtoul(tail.c_str(), nullptr, 10);
    m_PostFinish = (uint32_t) std::strtoul(GetEnvS("BALLRAIN_POST_FINISH").c_str(), nullptr, 10);
    m_PostLogPath = GetEnvS("BALLRAIN_POST_LOG");

    if (!m_PrefixPath.empty()) {
        std::ifstream file(m_PrefixPath, std::ios::binary | std::ios::ate);
        if (!file) {
            AgentFail(ERR_ENV, "agent: cannot open BALLRAIN_PREFIX %s", m_PrefixPath.c_str());
            return false;
        }
        const auto size = (size_t) file.tellg();
        file.seekg(0);
        m_Prefix.resize(size / sizeof(Frame));
        if (!m_Prefix.empty() &&
            !file.read(reinterpret_cast<char *>(m_Prefix.data()), (std::streamsize) (m_Prefix.size() * sizeof(Frame)))) {
            AgentFail(ERR_ENV, "agent: cannot read BALLRAIN_PREFIX %s", m_PrefixPath.c_str());
            return false;
        }
    }
    if (m_ParentPid)
        m_ParentProc = ::OpenProcess(SYNCHRONIZE, FALSE, m_ParentPid);
    GetLogger()->Info("agent: shm %s, level %d, prefix %u frames%s, timeout %u ms, max frames %u", m_ShmName.c_str(),
                      m_Level, (unsigned) m_Prefix.size(), m_PrefixEvents ? " (with observations)" : "", m_TimeoutMs,
                      m_MaxFrames);
    return true;
}

bool BallrainIO::AgentOpenTransport() {
    const std::wstring name = Widen(m_ShmName);
    HANDLE map = ::OpenFileMappingW(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, name.c_str());
    if (!map) {
        GetLogger()->Error("agent: cannot open shared memory %s (error %lu)", m_ShmName.c_str(), ::GetLastError());
        return false;
    }
    auto *view = (uint8_t *) ::MapViewOfFile(map, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
    MEMORY_BASIC_INFORMATION mbi = {};
    if (!view || !::VirtualQuery(view, &mbi, sizeof(mbi))) {
        GetLogger()->Error("agent: cannot map shared memory %s", m_ShmName.c_str());
        if (view)
            ::UnmapViewOfFile(view);
        ::CloseHandle(map);
        return false;
    }
    m_ShmMap = map;
    m_Shm = view;
    const size_t mapped = mbi.RegionSize;
    if (mapped < H_END_OF_HEADER || H32(m_Shm, H_MAGIC) != SHM_MAGIC) {
        GetLogger()->Error("agent: shared memory %s has no BRSM header", m_ShmName.c_str());
        ::UnmapViewOfFile(m_Shm);
        ::CloseHandle(map);
        m_Shm = nullptr;
        m_ShmMap = nullptr;
        return false;
    }
    m_ShmSize = H32(m_Shm, H_SIZE);
    m_ObsOff = H32(m_Shm, H_OBS_OFF);
    m_ObsCap = H32(m_Shm, H_OBS_CAP);
    m_ActOff = H32(m_Shm, H_ACT_OFF);
    m_ActCap = H32(m_Shm, H_ACT_CAP);
    H32(m_Shm, H_MOD_PID) = ::GetCurrentProcessId();
    H32(m_Shm, H_OBSVERS) = 1u << 1;
    const bool sizesOk = m_ShmSize <= mapped && m_ObsOff >= H_END_OF_HEADER && m_ActOff >= H_END_OF_HEADER &&
                         (uint64_t) m_ObsOff + m_ObsCap <= m_ShmSize && (uint64_t) m_ActOff + m_ActCap <= m_ShmSize &&
                         m_ObsCap >= 0x10000 && m_ActCap >= ACT_HEADER + 8 &&
                         (m_ObsOff + m_ObsCap <= m_ActOff || m_ActOff + m_ActCap <= m_ObsOff);
    if (H32(m_Shm, H_VERSION) != SHM_VERSION || !sizesOk) {
        AgentFail(ERR_SHM, "agent: shared memory version %u / layout (size %u obs %u+%u act %u+%u, mapped %u) refused",
                  (unsigned) H32(m_Shm, H_VERSION), m_ShmSize, m_ObsOff, m_ObsCap, m_ActOff, m_ActCap, (unsigned) mapped);
        return false;
    }
    m_EvObs = ::OpenEventW(EVENT_MODIFY_STATE | SYNCHRONIZE, FALSE, (name + L".obs").c_str());
    m_EvAct = ::OpenEventW(EVENT_MODIFY_STATE | SYNCHRONIZE, FALSE, (name + L".act").c_str());
    if (!m_EvObs || !m_EvAct) {
        AgentFail(ERR_SHM, "agent: cannot open events %s.obs / .act", m_ShmName.c_str());
        return false;
    }
    m_PySeq = H64(m_Shm, H_PY_SEQ);
    m_ModSeq = H64(m_Shm, H_MOD_SEQ);
    return true;
}

void BallrainIO::AgentHello() {
    if (!m_Shm || m_ModState >= ST_HELLO)
        return;
    char ver[H_VERSTR_LEN] = {};
    std::snprintf(ver, sizeof(ver), "BallrainIO %s physics_RT %.16s", GetVersion(), m_PhysicsHex.c_str());
    std::memcpy(m_Shm + H_VERSTR, ver, sizeof(ver));
    AgentSetState(ST_HELLO);
}

void BallrainIO::AgentSetState(uint32_t state) {
    m_ModState = state;
    if (m_Shm) {
        ::MemoryBarrier();
        H32(m_Shm, H_STATE) = state;
    }
}

void BallrainIO::AgentFail(uint32_t code, const char *fmt, ...) {
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    GetLogger()->Error("%s", msg);
    m_Active = false;
    if (!m_Shm)
        return;
    H32(m_Shm, H_ERROR) = code;
    H32(m_Shm, H_END) = END_ERROR;
    AgentSetState(ST_ERROR);
    if (m_EvObs)
        ::SetEvent(m_EvObs);   // wakes a waiting client; mod_seq is not advanced
}

// ================================================================ run start / events

void BallrainIO::AgentStart() {
    m_Phase = Phase::Running;
    m_Evs.clear();
    m_Evs.reserve(4096);
    m_StateBuf.assign(m_ObsCap, 0);
    m_StateLen = 0;
    m_Trac.clear();
    m_Trac.reserve(std::min<size_t>(m_ObsCap / sizeof(TracRec), 1 << 16));
    m_ActFrames.reserve(m_ActCap / sizeof(Frame));
    m_FrameLog.reserve(m_MaxFrames ? m_MaxFrames : 1 << 18);
    m_Reals.reserve(4096);
    m_Phys.reserve(1024);
    m_InPrefix = !m_Prefix.empty();
    m_BlockStart = 0;
    m_BlockEnd = m_InPrefix ? (m_PrefixEvents ? 1 : m_Prefix.size()) : 0;
    m_Covered = 0;
    m_FrameEvFrame = 0;
    m_FrameEvMask = 0;
    if (!m_LevelMsgPath.empty() && !(m_LevelInc = std::fopen((m_LevelMsgPath + ".inc").c_str(), "wb")))
        GetLogger()->Error("agent: cannot open %s.inc", m_LevelMsgPath.c_str());
    AgentSetState(ST_LOADING);
    AgentEvent(EV_PRELOAD);   // its events-file line was written by Start()
}

void BallrainIO::AgentEvent(uint16_t code, uint16_t param, const char *name) {
    if (m_Mode != Mode::Agent || m_Phase == Phase::Idle || m_Phase == Phase::Ended)
        return;
    const auto frame = (uint32_t) m_FrameIndex;
    if (frame != m_FrameEvFrame) {
        m_FrameEvFrame = frame;
        m_FrameEvMask = 0;
    }
    if (code < 64)
        m_FrameEvMask |= 1ull << code;
    m_Evs.push_back(Ev{frame, code, param});
    if (name && m_Events) {
        std::fprintf(m_Events, "%u %s\n", frame, name);
        std::fflush(m_Events);
    }
    switch (code) {
        case EV_START:
            // BML+ 0.3.12 BMLMod::OnStartLevel: ResetSRTimer (zero, stopped).
            m_Sr = 0.0;
            m_Sr32 = 0.0f;
            m_SrRunning = false;
            m_LevelMsgPending = true;
            AgentSetState(ST_RUNNING);
            break;
        case EV_COUNTER_ON:
        case EV_UNPAUSE:   // StartSRTimer
            m_SrRunning = true;
            AgentEventsLine("%u SR sr=%.17g f32sum=%.9g running=1", frame, m_Sr, m_Sr32);
            break;
        case EV_COUNTER_OFF:
        case EV_PAUSE:   // PauseSRTimer
            m_SrRunning = false;
            if (m_LevelFinished)
                m_SrInactiveSeen = true;
            AgentEventsLine("%u SR sr=%.17g f32sum=%.9g running=0", frame, m_Sr, m_Sr32);
            break;
        case EV_BALLOFF:
            ++m_BallOffs;
            m_LastBallOff = frame;
            break;
        case EV_BALLNAV_ON:
            m_LastBallNav = frame;
            break;
        case EV_FINISH: {
            m_LevelFinished = true;
            AgentSnapHs(m_HsAtFinish);
            m_FollowersAtFinish = AgentCountFollowers();
            const HsSnap &h = m_HsAtFinish;
            AgentEventsLine("%u HS points=%d lifes=%d lifebonus=%d levelbonus=%d hs=%d bml=%d followers=%u", frame,
                            (int) h.energy[0], (int) h.energy[1], (int) h.energy[5], (int) h.levelBonus, h.hs, h.bmlHs,
                            (unsigned) m_FollowersAtFinish);
            if (m_PostFinish && !m_PostLogPath.empty() && !m_PostLog) {
                m_PostLog = std::fopen(m_PostLogPath.c_str(), "w");
                AgentPostLog("H");   // at the hook, inside the LevelFinish frame
            }
            break;
        }
        default:
            break;
    }
}

void BallrainIO::AgentEventsLine(const char *fmt, ...) {
    if (!m_Events)
        return;
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(m_Events, fmt, ap);
    va_end(ap);
    std::fputc('\n', m_Events);
    std::fflush(m_Events);
}

// ================================================================ per-frame callbacks

// Time PreProcess (after the original). True when this callback was fully handled here.
bool BallrainIO::AgentPreTime() {
    if (m_Phase == Phase::Tail || m_Phase == Phase::Post) {
        // SR tail / post-finish window after LevelFinish: not exported, not traced; same delta as the last
        // exported frame.
        m_TimeManager->SetLastDeltaTime(m_LastDelta);
        return true;
    }
    if (!m_Playing)
        return false;
    if (m_EndAtNextTime && !m_GameOverPending) {
        m_EndReason = m_EndAtNextTime;
        m_EndAtNextTime = 0;
        Stop(StopReason::RecordEnd);
        return true;
    }
    return false;
}

// Input PreProcess (after the original). True when handled here (tail frames only).
bool BallrainIO::AgentPreInput() {
    if (m_Phase != Phase::Tail && m_Phase != Phase::Post)
        return false;
    SetKeyboardState(m_InputHook->GetKeyboardState(), 0);
    ++m_FrameIndex;
    if (m_Phase == Phase::Tail)
        ++m_TailFrames;
    else
        ++m_PostFrames;
    return true;
}

const Frame *BallrainIO::GetFrame(size_t index) const {
    if (m_Mode != Mode::Agent)
        return index < m_Frames.size() ? &m_Frames[index] : nullptr;
    if (index >= m_BlockEnd)
        return nullptr;
    if (m_InPrefix)
        return index < m_Prefix.size() ? &m_Prefix[index] : nullptr;
    if (index < m_BlockStart)
        return nullptr;
    const size_t k = (index - m_BlockStart) / m_Repeat;
    return k < m_ActFrames.size() ? &m_ActFrames[k] : nullptr;
}

// Blocks in the Time callback of frame m_FrameIndex until Python supplies it. False: the run ended.
bool BallrainIO::AgentWaitAction() {
    for (;;) {
        const uint32_t resume = m_ModState == ST_WAITING ? ST_RUNNING : m_ModState;
        H32(m_Shm, H_FRAMES) = (uint32_t) m_FrameLog.size();
        AgentSetState(ST_WAITING);
        HANDLE handles[2] = {m_EvAct, m_ParentProc};
        const DWORD n = m_ParentProc ? 2 : 1;
        const DWORD r = ::WaitForMultipleObjects(n, handles, FALSE, m_TimeoutMs ? m_TimeoutMs : INFINITE);
        AgentSetState(resume);
        if (r != WAIT_OBJECT_0) {
            GetLogger()->Error("agent: %s while waiting for the action of frame %u",
                               r == WAIT_TIMEOUT ? "timeout" : r == WAIT_OBJECT_0 + 1 ? "parent process gone" : "wait failed",
                               (unsigned) m_FrameIndex);
            m_EndReason = END_TIMEOUT;
            Stop(StopReason::RecordEnd);
            return false;
        }
        ::MemoryBarrier();
        const uint64_t seq = H64(m_Shm, H_PY_SEQ);
        const uint8_t *act = m_Shm + m_ActOff;
        uint32_t magic = 0, pySeq = 0, nFrames = 0, repeat = 0, maskLo = 0, maskHi = 0, obsMask = 0;
        uint16_t version = 0, flags = 0;
        std::memcpy(&magic, act + 0x00, 4);
        std::memcpy(&version, act + 0x04, 2);
        std::memcpy(&flags, act + 0x06, 2);
        std::memcpy(&pySeq, act + 0x08, 4);
        std::memcpy(&nFrames, act + 0x0C, 4);
        std::memcpy(&repeat, act + 0x10, 4);
        std::memcpy(&maskLo, act + 0x14, 4);
        std::memcpy(&maskHi, act + 0x18, 4);
        std::memcpy(&obsMask, act + 0x1C, 4);
        const char *bad = nullptr;
        if (seq != m_PySeq + 1 || pySeq != (uint32_t) seq)
            bad = "py_seq is not the previous one + 1";
        else if (magic != ACT_MAGIC || version != 1)
            bad = "bad ACT1 magic / version";
        else if ((uint64_t) nFrames * sizeof(Frame) + ACT_HEADER > m_ActCap)
            bad = "n_frames exceeds the act region";
        if (bad) {
            AgentFail(ERR_ACT, "agent: action refused at frame %u: %s (py_seq %llu, expected %llu)", (unsigned) m_FrameIndex,
                      bad, (unsigned long long) seq, (unsigned long long) (m_PySeq + 1));
            m_EndReason = END_ERROR;
            Stop(StopReason::RecordEnd);   // outputs are still written
            return false;
        }
        m_PySeq = seq;
        m_ObsMask = obsMask;
        m_WantTrac = (flags & ACT_TRAC) != 0;
        if (flags & ACT_LEVELMSG)
            AgentWriteLevelMessage();
        if (flags & ACT_STOP) {
            m_EndReason = END_STOP;
            Stop(StopReason::RecordEnd);
            return false;
        }
        if (nFrames == 0) {
            if (m_InPrefix && m_PrefixEvents) {   // acknowledgement of a prefix-frame observation
                m_BlockEnd = std::min(m_FrameIndex + 1, m_Prefix.size());
                return true;
            }
            AgentPublish(false);   // nothing to run: the same state again, with a new mod_seq
            continue;
        }
        m_ActFrames.resize(nFrames);
        std::memcpy(m_ActFrames.data(), act + ACT_HEADER, nFrames * sizeof(Frame));
        m_Repeat = repeat ? repeat : 1;
        m_StopMask = (uint64_t) maskHi << 32 | maskLo;
        m_InPrefix = false;
        m_BlockStart = m_FrameIndex;
        m_BlockEnd = m_BlockStart + (size_t) nFrames * m_Repeat;
        return true;
    }
}

// OnProcess (every frame while the mod is active in agent mode), after the replay trace line.
void BallrainIO::AgentProcess() {
    if (m_Phase == Phase::Idle || m_Phase == Phase::Ended)
        return;
    // BML+ 0.3.12 SRTimer::Update in ModManager::PostProcess: m_Time += (double) GetLastDeltaTime() while running.
    if (m_SrRunning && m_Phase != Phase::Post) {   // the SR value is final at the end of the SR tail
        const float d = m_TimeManager->GetLastDeltaTime();
        m_Sr += (double) d;
        m_Sr32 += d;
    }
    CK3dEntity *ball = GetActiveBall();
    if (ball != m_LastBall) {
        m_LastBall = ball;
        AgentEvent(EV_ACTIVEBALL, 0, "ActiveBallChanged");
    }
    if (m_PhysExported < m_Phys.size())
        AgentExportPhysicalized();
    if (m_LevelMsgPending) {
        m_LevelMsgPending = false;
        AgentLevelTables();
        AgentWriteLevelMessage();
    }

    switch (m_Phase) {
        case Phase::Running: {
            if (!m_Playing)
                return;
            if (m_WantTrac && ball) {
                if (m_Trac.size() < m_Trac.capacity()) {
                    TracRec t;
                    t.frame = (uint32_t) m_FrameIndex;
                    const VxMatrix &m = ball->GetWorldMatrix();
                    for (int r = 0; r < 4; ++r)
                        for (int c = 0; c < 3; ++c)
                            t.m[r * 3 + c] = m[r][c];
                    m_Trac.push_back(t);
                } else {
                    m_TracOverflow = true;
                }
            }
            bool end = m_FrameIndex >= m_BlockEnd;
            if (!m_InPrefix && m_Covered > 0 && m_FrameEvFrame == (uint32_t) m_FrameIndex && (m_FrameEvMask & m_StopMask)) {
                m_BlockEnd = m_FrameIndex;   // stop-on-event: the rest of the block is dropped
                end = true;
            }
            if (m_MaxFrames && m_FrameIndex >= m_MaxFrames && !m_EndAtNextTime) {
                m_EndAtNextTime = END_MAXFRAMES;
                end = true;
            }
            if (m_GameOverPending)
                end = true;
            if (!end)
                return;
            if (m_InPrefix && m_FrameIndex >= m_Prefix.size())
                m_InPrefix = false;   // hand-over: this is the first ordinary observation
            m_PrefixObs = m_InPrefix;
            AgentBuildState(m_FrameIndex == 0);
            if (!m_GameOverPending && !m_EndAtNextTime)
                AgentPublish(false);   // GameOver / frame limit: published as the final one at the next Time callback
            return;
        }
        case Phase::Tail:
            if (m_TailFrames == 0) {   // the LevelFinish frame itself
                m_SrAtFinish = m_Sr;
                m_Sr32AtFinish = m_Sr32;
            }
            AgentPostLog("T");
            if (!m_SrRunning || m_TailFrames >= m_SrTail) {
                if (m_PostFinish) {   // diagnostic window: keep running idle frames, SR frozen
                    m_Phase = Phase::Post;
                    m_PostFrames = 0;
                    if (m_SkipRender)
                        m_BML->SkipRenderForNextTick();
                    return;
                }
                m_PrefixObs = false;
                AgentBuildState(false);
                AgentFinish(END_FINISH);
            } else if (m_SkipRender) {
                m_BML->SkipRenderForNextTick();
            }
            return;
        case Phase::Post:
            AgentPostLog("P");
            if (m_PostFrames >= m_PostFinish) {
                m_PrefixObs = false;
                AgentBuildState(false);
                AgentFinish(END_FINISH);
            } else if (m_SkipRender) {
                m_BML->SkipRenderForNextTick();
            }
            return;
        case Phase::FinalPending:
            m_PrefixObs = false;
            AgentBuildState(false);
            AgentFinish(END_RESET);
            return;
        default:
            return;
    }
}

// End of BallrainIO::Stop (outputs already closed, events file kept open).
void BallrainIO::AgentAfterStop(StopReason reason) {
    switch (reason) {
        case StopReason::LevelFinish:
            m_Phase = Phase::Tail;
            m_TailFrames = 0;
            break;
        case StopReason::GameOver:
            AgentFinish(END_GAMEOVER);
            break;
        case StopReason::RecordEnd:
            AgentFinish(m_EndReason ? m_EndReason : END_STOP);
            break;
        case StopReason::ResetLevel:
        case StopReason::ExitLevel:
            m_Phase = Phase::FinalPending;
            break;
    }
}

void BallrainIO::AgentFinish(uint32_t endReason) {
    if (m_Phase == Phase::Ended)
        return;
    AgentEvent(EV_RUNEND, (uint16_t) endReason, "RunEnd");
    AgentEventsLine("%u SRFinal sr=%.17g sr_ms=%lld f32sum=%.9g bml=%.9g running=%d tail=%u inactive=%d finished=%d "
                    "sr_at_finish=%.17g end_reason=%u post=%u",
                    (unsigned) m_FrameIndex, m_Sr, (long long) m_Sr, m_Sr32, m_BML->GetSRScore(), (int) m_SrRunning,
                    m_TailFrames, (int) m_SrInactiveSeen, (int) m_LevelFinished, m_SrAtFinish, endReason, m_PostFrames);
    if (m_PostLog) {
        std::fclose(m_PostLog);
        m_PostLog = nullptr;
    }
    if (m_Events) {
        std::fclose(m_Events);
        m_Events = nullptr;
    }
    if (m_LevelInc) {
        std::fclose(m_LevelInc);
        m_LevelInc = nullptr;
    }
    m_EndReason = endReason;
    if (m_Shm) {
        H32(m_Shm, H_END) = endReason;
        if (endReason == END_ERROR)
            AgentSetState(ST_ERROR);
        else
            AgentSetState(ST_ENDED);
    }
    AgentPublish(true);
    m_Phase = Phase::Ended;
    GetLogger()->Info("agent: run ended (reason %u) after %u exported frames, SR %.3f ms", endReason,
                      (unsigned) m_FrameLog.size(), m_Sr);
    if (endReason == END_FINISH && CaptureTailStart())
        return;   // capture only: idle frames for the video after the final observation, then the exit (Capture.cpp)
    if (m_ExitWhenDone || endReason == END_TIMEOUT || endReason == END_ERROR)
        m_BML->ExitGame();
}

// ================================================================ observation

void BallrainIO::AgentPublish(bool final) {
    if (!m_Shm)
        return;
    Wire w(m_Shm + m_ObsOff, m_ObsCap);
    uint16_t flags = 0;
    if (final)
        flags |= OBS_FINAL;
    if (m_PrefixObs)
        flags |= OBS_PREFIX;
    BeginMessage(w, "OBS1", 0, (uint32_t) m_FrameLog.size());
    w.Raw(m_StateBuf.data(), m_StateLen);   // state sections built at OnProcess
    uint32_t sections = m_StateSections;

    w.Begin("EVTS", 1);
    const size_t nev = std::min<size_t>(m_Evs.size(), 0xFFFF);
    w.U16((uint32_t) nev);
    for (size_t i = 0; i < nev; ++i) {
        w.U32(m_Evs[i].frame);
        w.U16(m_Evs[i].code);
        w.U16(m_Evs[i].param);
    }
    sections += w.End();
    if (!m_PhysNew.empty()) {
        w.Begin("PHYS", 1);
        const size_t n = std::min<size_t>(m_PhysNew.size(), 0xFFFF);
        w.U16((uint32_t) n);
        for (size_t i = 0; i < n; ++i)
            w.U32(m_PhysNew[i]);
        sections += w.End();
    }
    if (m_WantTrac || !m_Trac.empty()) {
        w.Begin("TRAC", 1);
        if (m_TracOverflow)
            w.SetSectionFlags(1);
        w.U32((uint32_t) m_Trac.size());
        w.Raw(m_Trac.data(), m_Trac.size() * sizeof(TracRec));
        sections += w.End();
    }
    if (final || m_LevelFinished) {
        w.Begin("SREN", 1);
        w.F64(m_Sr);
        w.F32(m_Sr32);
        w.F32(m_BML->GetSRScore());
        w.U32(m_TailFrames);
        w.U8(m_LevelFinished);
        w.U8(m_SrInactiveSeen);
        w.U8(m_SrRunning);
        w.U8(0);
        w.F64(m_SrAtFinish);
        w.F32(m_Sr32AtFinish);
        const HsSnap &h = m_HsAtFinish;
        w.U8(h.valid);
        w.Raw(h.types, 6);
        w.U8(h.bonusType);
        w.Raw(h.energy, sizeof(h.energy));
        w.U32(h.levelBonus);
        w.I32(h.bmlHs);
        w.I32(h.hs);
        // appended (0.3.1): small balls following at the LevelFinish hook, following now, post-finish frames
        w.U16(m_FollowersAtFinish);
        w.U16(AgentCountFollowers());
        w.U32(m_PostFrames);
        sections += w.End();
    }
    if (m_StateDropped || w.Dropped())
        flags |= OBS_DROPPED;
    if (m_StateMinimal)
        flags |= OBS_MINIMAL;
    w.PatchU16(6, flags);
    w.PatchU32(8, (uint32_t) w.Size());
    w.PatchU32(16, sections);

    m_Evs.clear();
    m_PhysNew.clear();
    m_Trac.clear();
    m_TracOverflow = false;
    m_Covered = 0;

    H32(m_Shm, H_FRAMES) = (uint32_t) m_FrameLog.size();
    ::MemoryBarrier();
    H64(m_Shm, H_MOD_SEQ) = ++m_ModSeq;
    ::MemoryBarrier();
    ::SetEvent(m_EvObs);
}

void BallrainIO::AgentSnapHs(HsSnap &h) {
    h = HsSnap();
    CKContext *ctx = m_BML->GetCKContext();
    if (!m_EnergyId)
        if (CKDataArray *a = m_BML->GetArrayByName("Energy"))
            m_EnergyId = a->GetID();
    if (!m_AllLevelId)
        if (CKDataArray *a = m_BML->GetArrayByName("AllLevel"))
            m_AllLevelId = a->GetID();
    auto *energy = (CKDataArray *) ctx->GetObject(m_EnergyId);
    if (!energy || energy->GetRowCount() < 1 || energy->GetColumnCount() < 6)
        return;
    for (int c = 0; c < 6; ++c) {
        h.types[c] = (uint8_t) energy->GetColumnType(c);
        if (CKDWORD *e = energy->GetElement(0, c))
            h.energy[c] = (uint32_t) *e;
    }
    // Points, Lifes and LifeBonus (columns 0, 1, 5) are the ones the HS formula uses.
    const bool ints = h.types[0] == CKARRAYTYPE_INT && h.types[1] == CKARRAYTYPE_INT && h.types[5] == CKARRAYTYPE_INT;
    int level = m_Level;
    if (m_CurLevel && m_CurLevel->GetColumnType(0) == CKARRAYTYPE_INT)
        if (CKDWORD *e = m_CurLevel->GetElement(0, 0))
            level = (int) *e;
    auto *all = (CKDataArray *) ctx->GetObject(m_AllLevelId);
    if (all && level >= 1 && level <= all->GetRowCount() && all->GetColumnCount() > 6) {
        h.bonusType = (uint8_t) all->GetColumnType(6);
        if (CKDWORD *e = all->GetElement(level - 1, 6))
            h.levelBonus = (uint32_t) *e;
    }
    h.bmlHs = m_BML->GetHSScore();
    const bool bonusInt = h.bonusType == CKARRAYTYPE_INT;
    h.hs = (int32_t) h.energy[0] + (int32_t) h.energy[1] * (int32_t) h.energy[5] + (bonusInt ? (int32_t) h.levelBonus : 0);
    h.valid = ints && bonusInt ? 1 : 2;   // 1: all fields are ints; 2: raw values only
}

// Small score balls still following the player ball: P_Extra_Point_Ball1..6 of an instance whose score
// ball (P_Extra_Point_Ball0) is hidden and that are still visible (they hide when they reach the ball or
// are lost at a checkpoint). Derived from visibility only; checked against the WTCH history of every TAS
// record (docs/observations.md). m_SourceMainVis: -1 unknown, 0 hidden, 1 visible, 2 hidden with a
// visible small ball.
uint16_t BallrainIO::AgentCountFollowers() {
    CKContext *ctx = m_BML->GetCKContext();
    std::fill(m_SourceMainVis.begin(), m_SourceMainVis.end(), (int8_t) -1);
    for (const FollowerSlot &f : m_FollowerSlots)
        if (f.k == 0) {
            CKObject *o = ctx->GetObject(m_Watch[f.watch]);
            if (o && CKIsChildClassOf(o, CKCID_3DENTITY))
                m_SourceMainVis[f.source] = ((CK3dEntity *) o)->IsVisible() ? 1 : 0;
        }
    uint32_t n = 0;
    for (const FollowerSlot &f : m_FollowerSlots) {
        if (f.k == 0 || m_SourceMainVis[f.source] < 0 || m_SourceMainVis[f.source] == 1)
            continue;
        CKObject *o = ctx->GetObject(m_Watch[f.watch]);
        if (o && CKIsChildClassOf(o, CKCID_3DENTITY) && ((CK3dEntity *) o)->IsVisible()) {
            ++n;
            m_SourceMainVis[f.source] = 2;
        }
    }
    return (uint16_t) std::min<uint32_t>(n, 0xFFFF);
}

// BALLRAIN_POST_LOG (diagnostic, only with BALLRAIN_POST_FINISH): one block per frame from the LevelFinish
// hook (tag H) through the SR tail (T) and the post-finish window (P). Lines: "<tag> <frame> tail= post=
// e=<Energy row 0> hs= fol=<followers> sr= running=", then "A <frame> <array> <row> <col> <value>" for
// every data-array cell that changed (first block: Energy, CurrentLevel, IngameParameter in full),
// "B <frame> <ball> vis= x y z", "W <frame> <name> vis= d=<distance to the ball>" for visible or changed
// score-ball objects.
void BallrainIO::AgentPostLog(const char *tag) {
    if (!m_PostLog)
        return;
    CKContext *ctx = m_BML->GetCKContext();
    const auto frame = (unsigned) m_FrameIndex;
    HsSnap h;
    AgentSnapHs(h);
    std::fprintf(m_PostLog, "%s %u tail=%u post=%u e=%d,%d,%d,%d,%d,%d hs=%d fol=%u sr=%.17g running=%d\n", tag, frame,
                 m_TailFrames, m_PostFrames, (int) h.energy[0], (int) h.energy[1], (int) h.energy[2], (int) h.energy[3],
                 (int) h.energy[4], (int) h.energy[5], h.hs, (unsigned) AgentCountFollowers(), m_Sr, (int) m_SrRunning);
    char buf[512];
    const bool first = m_PostArrs.empty();
    const int na = ctx->GetObjectsCountByClassID(CKCID_DATAARRAY);
    CK_ID *ids = ctx->GetObjectsListByClassID(CKCID_DATAARRAY);
    for (int k = 0; k < na; ++k) {
        auto *a = (CKDataArray *) ctx->GetObject(ids[k]);
        if (!a)
            continue;
        auto it = std::find_if(m_PostArrs.begin(), m_PostArrs.end(), [&](const ArrSnap &s) { return s.id == ids[k]; });
        if (it == m_PostArrs.end()) {
            m_PostArrs.push_back(ArrSnap{ids[k]});
            it = m_PostArrs.end() - 1;
        }
        ArrSnap &snap = *it;
        const int rows = a->GetRowCount(), cols = a->GetColumnCount();
        const bool fresh = rows != snap.rows || cols != snap.cols;
        if (fresh) {
            snap.rows = rows;
            snap.cols = cols;
            snap.cells.assign((size_t) rows * cols, std::string());
        }
        const char *an = a->GetName() ? a->GetName() : "?";
        const bool dump = !first || !strcmp(an, "Energy") || !strcmp(an, "CurrentLevel") || !strcmp(an, "IngameParameter");
        for (int r = 0; r < rows && r < 4096; ++r)
            for (int c = 0; c < cols; ++c) {
                CellText(a, r, c, buf, sizeof(buf));
                std::string &old = snap.cells[(size_t) r * cols + c];
                if (old == buf)
                    continue;
                old = buf;
                if (dump)
                    std::fprintf(m_PostLog, "A %u %s %d %d %s\n", frame, an, r, c, buf);
            }
    }
    CK3dEntity *ball = GetActiveBall();
    VxVector bp(0, 0, 0);
    if (ball) {
        ball->GetPosition(&bp);
        std::fprintf(m_PostLog, "B %u %s vis=%d %.4f %.4f %.4f\n", frame, ball->GetName() ? ball->GetName() : "?",
                     ball->IsVisible() ? 1 : 0, bp.x, bp.y, bp.z);
    }
    if (m_PostWatch.size() != m_FollowerSlots.size())
        m_PostWatch.assign(m_FollowerSlots.size(), WatchLast{255, {0, 0, 0}});
    for (size_t i = 0; i < m_FollowerSlots.size(); ++i) {
        CKObject *o = ctx->GetObject(m_Watch[m_FollowerSlots[i].watch]);
        if (!o || !CKIsChildClassOf(o, CKCID_3DENTITY))
            continue;
        auto *e = (CK3dEntity *) o;
        const uint8_t vis = e->IsVisible() ? 1 : 0;
        if (vis == m_PostWatch[i].vis && !vis)
            continue;
        VxVector p;
        e->GetPosition(&p);
        std::fprintf(m_PostLog, "W %u %s vis=%d d=%.3f\n", frame, e->GetName() ? e->GetName() : "?", vis,
                     (p - bp).Magnitude());
        m_PostWatch[i].vis = vis;
    }
    std::fflush(m_PostLog);
}

unsigned char *BallrainIO::AgentFindBallReal(CK3dEntity *ball) {
    // The cache is dropped by OnPhysicalize / OnUnphysicalize of the cached entity (both run before the
    // block replaces or destroys the IVP object): a freed object can still hold client_data == ball for
    // a few frames (SR09, after each death), so client_data alone does not prove it is alive. The cache
    // is what still finds the frozen old ball during a ball transformation (it leaves the live vector).
    if (!ball || m_PhysicsOk <= 0)
        return nullptr;
    if (m_BallReal && m_BallRealOf == ball) {
        void *client = nullptr;
        if (Rd(m_BallReal, RO_CLIENT, client) && client == ball)
            return m_BallReal;
    }
    m_BallReal = nullptr;
    m_BallRealOf = nullptr;
    // CKIpionManager +0x28: IVP_U_Vector<IVP_Real_Object> of the movable objects (state-spec.md 2.1).
    auto *mgr = reinterpret_cast<u8 *>(m_IpionManager);
    uint16_t memsize = 0, n = 0;
    void **elems = nullptr;
    if (!Rd(mgr, MGR_MOVABLE_VEC, memsize) || !Rd(mgr, MGR_MOVABLE_VEC + 2, n) || !Rd(mgr, MGR_MOVABLE_VEC + 4, elems))
        return nullptr;
    if (!elems || n > memsize || n > 8000)
        return nullptr;
    for (int i = 0; i < n; ++i) {
        u8 *ro = nullptr;
        void *client = nullptr;
        if (!SafeRead(elems + i, &ro, sizeof(ro)) || !ro || !Rd(ro, RO_CLIENT, client))
            continue;
        if (client == ball) {
            m_BallReal = ro;
            m_BallRealOf = ball;
            return ro;
        }
    }
    return nullptr;
}

void BallrainIO::AgentWalkCluster(u8 *anyReal) {
    // All real objects of the ball's cluster, following next_in_cluster then prev_in_cluster
    // (state-spec.md 2.5); cluster nodes (object_type 1) are skipped.
    m_Reals.clear();
    if (!anyReal)
        return;
    m_Reals.push_back(anyReal);
    bool circular = false;
    for (int dir = 0; dir < 2 && !circular; ++dir) {
        u8 *o = nullptr;
        Rd(anyReal, dir == 0 ? RO_NEXT : RO_PREV, o);
        for (int guard = 0; o && guard < 20000; ++guard) {
            if (o == anyReal) {
                circular = true;
                break;
            }
            int32_t type = 0;
            if (!Rd(o, RO_TYPE, type))
                break;
            if (type != TYPE_CLUSTER && m_Reals.size() < 20000)
                m_Reals.push_back(o);
            u8 *nx = nullptr;
            if (!Rd(o, dir == 0 ? RO_NEXT : RO_PREV, nx))
                break;
            o = nx;
        }
    }
}

void BallrainIO::AgentRebuildEntityMap() {
    m_EntOf.clear();
    const XObjectPointerArray &list = m_BML->GetCKContext()->GetObjectListByType(CKCID_3DENTITY, TRUE);
    m_EntOf.reserve((size_t) list.Size() * 2);
    for (int k = 0; k < list.Size(); ++k)
        if (CKObject *o = list[k])
            m_EntOf.emplace(o, o->GetID());
}

CK_ID BallrainIO::AgentCkidOf(const void *client) const {
    if (!client)
        return 0;
    auto it = m_EntOf.find(client);
    if (it == m_EntOf.end())
        return 0;
    return m_BML->GetCKContext()->GetObject(it->second) == client ? it->second : 0;
}

void BallrainIO::AgentBuildState(bool minimal) {
    CKContext *ctx = m_BML->GetCKContext();
    Wire w(m_StateBuf.data(), m_StateBuf.size());
    auto *mgr = reinterpret_cast<u8 *>(m_IpionManager);
    const bool raw = m_PhysicsOk > 0;

    // ---- TIME
    {
        u8 *env = nullptr;
        double cur = 0, next = 0, last = 0;
        float c8 = 0, cc = 0, d0 = 0;
        if (raw) {
            Rd(mgr, MGR_ENV, env);
            if (env) {
                Rd(env, ENV_CURRENT_TIME, cur);
                Rd(env, ENV_NEXT_PSI, next);
                Rd(env, ENV_LAST_PSI, last);
            }
            Rd(mgr, MGR_DELTA, c8);
            Rd(mgr, MGR_PHYS_DELTA, cc);
            Rd(mgr, MGR_TIME_FACTOR, d0);
        }
        w.Begin("TIME", 1);
        w.F64(cur);
        w.F64(next);
        w.F64(last);
        w.F32(c8);
        w.F32(cc);
        w.F32(d0);
        w.F32(m_TimeManager->GetLastDeltaTime());
        w.F64(m_Sr);
        w.F32(m_BML->GetSRScore());
        w.U8(m_SrRunning);
        w.U32(m_Covered);
        // appended (v1): float32 sum, raw frame counter (tail frames included), tail frames
        w.F32(m_Sr32);
        w.U32((uint32_t) m_FrameIndex);
        w.U32(m_TailFrames);
        w.End();
    }
    m_StateMinimal = minimal;
    if (minimal) {
        m_StateLen = w.Size();
        m_StateSections = w.Sections();
        m_StateDropped = w.Dropped();
        return;
    }

    CK3dEntity *ball = GetActiveBall();
    u8 *real = raw ? AgentFindBallReal(ball) : nullptr;
    u8 *core = nullptr;
    if (real)
        Rd(real, RO_CORE, core);

    // ---- BALL
    if (!(m_ObsMask & M_BALL)) {
        w.Begin("BALL", 1);
        w.U32(ball ? ball->GetID() : 0);
        uint8_t type = 255;
        if (ball) {
            const char *n = ball->GetName();
            type = !n ? 254 : !strcmp(n, "Ball_Wood") ? 0 : !strcmp(n, "Ball_Paper") ? 1 : !strcmp(n, "Ball_Stone") ? 2 : 254;
        }
        w.U8(type);
        w.U8(real ? 1 : 0);
        if (ball)
            PutMatrix12(w, ball->GetWorldMatrix());
        else
            w.Zeros(48);
        double pos[3] = {}, ql[4] = {}, qn[4] = {}, tpsi = 0;
        float s[3] = {}, sc[3] = {}, rs[3] = {}, rsc[3] = {}, axis[3] = {}, curSpeed = 0, absOmega = 0, xrad = 0;
        uint32_t cflags = 0, psif = 0;
        if (core) {
            SafeRead(core + CO_POS, pos, sizeof(pos));
            SafeRead(core + CO_QL, ql, sizeof(ql));
            SafeRead(core + CO_QN, qn, sizeof(qn));
            Rd(core, CO_TLPSI, tpsi);
            SafeRead(core + CO_S, s, sizeof(s));
            SafeRead(core + CO_SC, sc, sizeof(sc));
            SafeRead(core + CO_RS, rs, sizeof(rs));
            SafeRead(core + CO_RSC, rsc, sizeof(rsc));
            SafeRead(core + CO_AXIS, axis, sizeof(axis));
            Rd(core, CO_CURSPEED, curSpeed);
            Rd(core, CO_ABSOMEGA, absOmega);
            Rd(core, CO_FLAGS, cflags);
            Rd(core, CO_PSIFLAGS, psif);
        }
        if (real)
            Rd(real, RO_XRAD, xrad);
        w.Raw(pos, sizeof(pos));
        w.Raw(ql, sizeof(ql));
        w.Raw(qn, sizeof(qn));
        w.F64(tpsi);
        w.Raw(s, sizeof(s));
        w.Raw(sc, sizeof(sc));
        w.Raw(rs, sizeof(rs));
        w.Raw(rsc, sizeof(rsc));
        w.Raw(axis, sizeof(axis));
        w.F32(curSpeed);
        w.F32(absOmega);
        w.U32(cflags);
        w.U32(psif);
        w.F32(xrad);
        w.End();
    }

    // ---- FRIC: friction synapses of the ball -> contact points
    if (!(m_ObsMask & M_FRIC)) {
        w.Begin("FRIC", 1);
        const size_t at = w.Size();
        w.U16(0);
        uint32_t n = 0;
        u8 *head = nullptr;
        if (real)
            Rd(real, RO_FRIC, head);
        for (u8 *s = head; s && n < 32;) {
            int16_t cpOff = 0;
            uint8_t st = 0;
            Rd(s, SF_CPOFF, cpOff);
            Rd(s, SF_STATUS, st);
            u8 *cp = s + cpOff;
            u8 *ps = s == cp + CP_SYN0 ? cp + CP_SYN1 : s == cp + CP_SYN1 ? cp + CP_SYN0 : nullptr;
            u8 *po = nullptr;
            uint8_t pst = 255, cpst = 0;
            void *pclient = nullptr;
            if (ps) {
                Rd(ps, SF_LOBJ, po);
                Rd(ps, SF_STATUS, pst);
                if (po)
                    Rd(po, RO_CLIENT, pclient);
            }
            float gap = 0, fric = 0, normal[3] = {};
            double point[3] = {};
            Rd(cp, CP_STATUS, cpst);
            Rd(cp, CP_GAP, gap);
            Rd(cp, CP_FRICTION, fric);
            u8 *lt = nullptr;
            const bool ltOk = Rd(cp, CP_LT, lt) && lt && SafeRead(lt + LT_NORMAL, normal, sizeof(normal)) &&
                              SafeRead(lt + LT_POINT, point, sizeof(point));
            if (!ltOk) {
                std::memset(normal, 0, sizeof(normal));
                std::memset(point, 0, sizeof(point));
            }
            w.U32(AgentCkidOf(pclient));
            w.U8(st);
            w.U8(pst);
            w.U8(cpst);
            w.U8(0);
            w.F32(gap);
            w.F32(fric);
            w.Raw(normal, sizeof(normal));
            w.Raw(point, sizeof(point));
            w.U8(ltOk ? 1 : 0);
            ++n;
            u8 *nx = nullptr;
            if (!Rd(s, SF_NEXT, nx) || nx == s || nx == head)
                break;
            s = nx;
        }
        w.PatchU16(at, (uint16_t) n);
        w.End();
    }

    // ---- MIND: exact synapses of the ball -> mindists
    if (!(m_ObsMask & M_MIND)) {
        w.Begin("MIND", 1);
        const size_t at = w.Size();
        w.U16(0);
        uint32_t n = 0;
        u8 *head = nullptr;
        if (real)
            Rd(real, RO_EXACT, head);
        int guard = 0;
        for (u8 *s = head; s && n < 64 && guard < 256; ++guard) {
            int16_t mdOff = 0, st = 0;
            Rd(s, SR_MDOFF, mdOff);
            Rd(s, SR_STATUS, st);
            u8 *md = s + mdOff;
            u8 *ps = s == md + MD_SYN0 ? md + MD_SYN1 : s == md + MD_SYN1 ? md + MD_SYN0 : nullptr;
            if (ps) {
                u8 *po = nullptr, *pedge = nullptr;
                int16_t pst = -1;
                void *pclient = nullptr;
                Rd(ps, SR_LOBJ, po);
                Rd(ps, SR_STATUS, pst);
                Rd(ps, SR_EDGE, pedge);
                if (po)
                    Rd(po, RO_CLIENT, pclient);
                float len = 0, plane[4] = {}, sumx = 0;
                Rd(md, MD_LEN, len);
                SafeRead(md + MD_PLANE, plane, sizeof(plane));
                Rd(md, MD_SUMXRAD, sumx);
                Tri t, o;
                uint8_t ntri = 0;
                if (pedge && pst >= 0 && pst <= 2 && DecodeTri(pedge, t)) {
                    ntri = 1;
                    const u8 *oe = pst == 1 ? OppositeEdge(pedge) : nullptr;
                    if (oe && DecodeTri(oe, o))
                        ntri = 2;
                }
                w.U32(AgentCkidOf(pclient));
                w.U8((uint8_t) st);
                w.U8((uint8_t) pst);
                w.U16(0);
                w.F32(len);
                w.Raw(plane, sizeof(plane));
                w.F32(sumx);
                w.U32(ntri ? (uint32_t) reinterpret_cast<uintptr_t>(t.ledge) : 0);
                w.U16(ntri ? (uint32_t) t.triIndex : 0xFFFF);
                w.Put<int8_t>((int8_t) (ntri ? t.edgeIndex : -1));
                w.U8(ntri);
                if (ntri >= 1)
                    w.Raw(t.v, sizeof(t.v));
                else
                    w.Zeros(36);
                if (ntri >= 2)
                    w.Raw(o.v, sizeof(o.v));
                else
                    w.Zeros(36);
                ++n;
            }
            u8 *nx = nullptr;
            if (!Rd(s, SR_NEXT, nx) || nx == s || nx == head)
                break;
            s = nx;
        }
        w.PatchU16(at, (uint16_t) n);
        w.End();
    }

    // ---- BODY: movable bodies near the ball (PSI poses)
    if (!(m_ObsMask & M_BODY)) {
        w.Begin("BODY", 1);
        const size_t at = w.Size();
        w.U16(0);
        uint32_t n = 0;
        if (real && ball) {
            AgentWalkCluster(real);
            const VxVector bp = ball->GetWorldMatrix()[3];
            const float r2 = m_BodyRadius * m_BodyRadius;
            for (u8 *ro : m_Reals) {
                if (ro == real || n >= 0xFFFF)
                    continue;
                u8 *bcore = nullptr;
                void *client = nullptr;
                if (!Rd(ro, RO_CORE, bcore) || !bcore || CoreUnmoveable(bcore) || !Rd(ro, RO_CLIENT, client))
                    continue;
                const CK_ID id = AgentCkidOf(client);
                if (!id)
                    continue;
                auto *ent = (CK3dEntity *) client;
                const VxMatrix &wm = ent->GetWorldMatrix();
                const float dx = wm[3][0] - bp.x, dy = wm[3][1] - bp.y, dz = wm[3][2] - bp.z;
                if (dx * dx + dy * dy + dz * dz > r2)
                    continue;
                double pos[3] = {}, q[4] = {}, qcf[4] = {0, 0, 0, 1};
                float shift[3] = {}, s[3] = {}, rs[3] = {};
                uint32_t psif = 0;
                Rd(bcore, CO_PSIFLAGS, psif);
                SafeRead(bcore + CO_POS, pos, sizeof(pos));
                SafeRead(bcore + CO_QL, q, sizeof(q));
                u8 *qp = nullptr;
                if (Rd(ro, RO_QCF, qp) && qp)
                    SafeRead(qp, qcf, sizeof(qcf));
                SafeRead(ro + RO_SHIFT, shift, sizeof(shift));
                SafeRead(bcore + CO_S, s, sizeof(s));
                SafeRead(bcore + CO_RS, rs, sizeof(rs));
                w.U32(id);
                w.U32(psif);
                w.Raw(pos, sizeof(pos));
                w.Raw(q, sizeof(q));
                w.Raw(qcf, sizeof(qcf));
                w.Raw(shift, sizeof(shift));
                w.Raw(s, sizeof(s));
                w.Raw(rs, sizeof(rs));
                PutMatrix12(w, wm);
                ++n;
            }
        }
        w.PatchU16(at, (uint16_t) n);
        w.End();
    }

    // ---- CAMS
    if (!(m_ObsMask & M_CAMS)) {
        w.Begin("CAMS", 1);
        for (CK_ID id : m_CamIds) {
            CKObject *o = id ? ctx->GetObject(id) : nullptr;
            if (o && CKIsChildClassOf(o, CKCID_3DENTITY)) {
                w.U32(id);
                PutMatrix12(w, ((CK3dEntity *) o)->GetWorldMatrix());
            } else {
                w.U32(0);
                w.Zeros(48);
            }
        }
        CKRenderContext *rc = ctx->GetPlayerRenderContext();
        CKCamera *cam = rc ? rc->GetAttachedCamera() : nullptr;
        w.U32(cam ? cam->GetID() : 0);
        w.End();
    }

    // ---- GLCT: the PhysicsContinuousContact block in use (Sound_Manager)
    if (!(m_ObsMask & M_GLCT)) {
        CK_ID bbId = 0;
        uint16_t outMask = 0;
        int32_t groups[6] = {};
        for (size_t k = 0; k < m_ContactBBs.size() && !bbId; ++k) {
            auto *b = (CKBehavior *) ctx->GetObject(m_ContactBBs[k]);
            auto *h = (CKParameter *) ctx->GetObject(m_ContactHandles[k]);
            if (!b || !h)
                continue;
            const void *hp = h->GetReadDataPtr(FALSE);
            u8 *data = nullptr, *gout = nullptr;
            if (!hp || !SafeRead(hp, &data, sizeof(data)) || !data)
                continue;
            if (!Rd(data, 0x10, gout) || !gout || !SafeRead(gout, groups, sizeof(groups)))
                std::memset(groups, 0, sizeof(groups));
            bbId = b->GetID();
            for (int i = 0; i < b->GetOutputCount() && i < 16; ++i)
                if (b->IsOutputActive(i))
                    outMask |= (uint16_t) (1u << i);
        }
        w.Begin("GLCT", 1);
        w.U32(bbId);
        w.U16(outMask);
        w.Raw(groups, sizeof(groups));
        w.End();
    }

    // ---- ARRS: changed cells of the watched arrays since the last observation
    if (!(m_ObsMask & M_ARRS)) {
        if (m_Arrs.size() != kTrackedCount)
            m_Arrs.resize(kTrackedCount);
        w.Begin("ARRS", 1);
        const size_t at = w.Size();
        w.U16(0);
        uint32_t n = 0;
        u8 cell[256];
        for (int k = 0; k < kTrackedCount; ++k) {
            ArrSnap &snap = m_Arrs[k];
            if (!snap.id)
                if (CKDataArray *a = m_BML->GetArrayByName(kTrackedArrays[k]))
                    snap.id = a->GetID();
            auto *a = snap.id ? (CKDataArray *) ctx->GetObject(snap.id) : nullptr;
            if (!a)
                continue;
            const int rows = a->GetRowCount(), cols = a->GetColumnCount();
            if (rows != snap.rows || cols != snap.cols) {
                snap.rows = rows;
                snap.cols = cols;
                snap.cells.assign((size_t) rows * cols, std::string("\x01\x02\x03", 3));   // never equals a cell
            }
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c) {
                    const size_t len = CellBytes(a, r, c, cell);
                    std::string &old = snap.cells[(size_t) r * cols + c];
                    if (old.size() == len && !std::memcmp(old.data(), cell, len))
                        continue;
                    if (n >= 0xFFFF)
                        continue;
                    old.assign((const char *) cell, len);
                    w.U32(snap.id);
                    w.U16((uint32_t) r);
                    w.U16((uint32_t) c);
                    w.U8((uint32_t) a->GetColumnType(c));
                    w.U8((uint32_t) len);
                    w.Raw(cell, len);
                    ++n;
                }
        }
        w.PatchU16(at, (uint16_t) n);
        if (!w.End())   // did not fit: report everything again next time
            for (ArrSnap &s : m_Arrs)
                s.rows = -1;
    }

    // ---- WTCH: watched objects whose visibility or position changed since the last observation
    if (!(m_ObsMask & M_WTCH)) {
        w.Begin("WTCH", 1);
        const size_t at = w.Size();
        w.U16(0);
        uint32_t n = 0;
        for (size_t i = 0; i < m_Watch.size() && n < 0xFFFF; ++i) {
            CKObject *o = ctx->GetObject(m_Watch[i]);
            if (!o || !CKIsChildClassOf(o, CKCID_3DENTITY))
                continue;
            auto *e = (CK3dEntity *) o;
            const uint8_t vis = e->IsVisible() ? 1 : 0;
            const VxMatrix &m = e->GetWorldMatrix();
            WatchLast &last = m_WatchLast[i];
            if (last.vis == vis && last.pos[0] == m[3][0] && last.pos[1] == m[3][1] && last.pos[2] == m[3][2])
                continue;
            last.vis = vis;
            last.pos[0] = m[3][0];
            last.pos[1] = m[3][1];
            last.pos[2] = m[3][2];
            w.U32(m_Watch[i]);
            w.U8(vis);
            w.Raw(last.pos, sizeof(last.pos));
            ++n;
        }
        w.PatchU16(at, (uint16_t) n);
        if (!w.End())
            for (WatchLast &l : m_WatchLast)
                l.vis = 255;
    }

    // ---- SCOR: HS readout and life bookkeeping
    {
        HsSnap h;
        AgentSnapHs(h);
        w.Begin("SCOR", 1);
        w.U8(h.valid);
        w.Raw(h.types, 6);
        w.U8(h.bonusType);
        w.Raw(h.energy, sizeof(h.energy));
        w.U32(h.levelBonus);
        w.I32(h.bmlHs);
        w.I32(h.hs);
        w.U32(m_BallOffs);
        w.U32(m_LastBallOff);
        w.U32(m_LastBallNav);
        // appended (0.3.1): small score balls following the player ball; touched score balls that still
        // have a following small ball
        const uint16_t followers = AgentCountFollowers();
        uint32_t sources = 0;
        for (int8_t v : m_SourceMainVis)
            sources += v == 2;
        w.U16(followers);
        w.U16(sources);
        w.End();
    }

    m_StateLen = w.Size();
    m_StateSections = w.Sections();
    m_StateDropped = w.Dropped();
}

// ================================================================ physicalize records

void BallrainIO::OnPhysicalize(CK3dEntity *target, CKBOOL fixed, float friction, float elasticity, float mass,
                               const char *collGroup, CKBOOL startFrozen, CKBOOL enableColl, CKBOOL calcMassCenter,
                               float linearDamp, float rotDamp, const char *collSurface, VxVector massCenter,
                               int convexCnt, CKMesh **convexMesh, int ballCnt, VxVector *ballCenter, float *ballRadius,
                               int concaveCnt, CKMesh **concaveMesh) {
    // Called before the Physicalize block runs: the IVP object does not exist yet and the mesh arrays
    // are freed afterwards, so only ids and values are copied here; meshes are exported at OnProcess.
    if (m_Mode != Mode::Agent || !m_Active || !target || (m_Phase != Phase::Running && m_Phase != Phase::Tail))
        return;
    if (target == m_BallRealOf) {   // its IVP object is about to be replaced
        m_BallReal = nullptr;
        m_BallRealOf = nullptr;
    }
    PhysRec r;
    r.seq = (uint32_t) m_Phys.size();
    r.frame = (uint32_t) m_FrameIndex;
    r.ent = target->GetID();
    r.kind = 0;
    r.fixed = fixed ? 1 : 0;
    r.frozen = startFrozen ? 1 : 0;
    r.coll = enableColl ? 1 : 0;
    r.calcmc = calcMassCenter ? 1 : 0;
    r.friction = friction;
    r.elasticity = elasticity;
    r.mass = mass;
    r.linDamp = linearDamp;
    r.rotDamp = rotDamp;
    r.massCenter[0] = massCenter.x;
    r.massCenter[1] = massCenter.y;
    r.massCenter[2] = massCenter.z;
    r.collGroup = collGroup ? collGroup : "";
    r.collSurface = collSurface ? collSurface : "";
    for (int i = 0; i < convexCnt && convexMesh; ++i)
        if (convexMesh[i])
            r.convex.push_back(convexMesh[i]->GetID());
    for (int i = 0; i < concaveCnt && concaveMesh; ++i)
        if (concaveMesh[i])
            r.concave.push_back(concaveMesh[i]->GetID());
    for (int i = 0; i < ballCnt && ballCenter && ballRadius; ++i) {
        r.balls.push_back(ballCenter[i].x);
        r.balls.push_back(ballCenter[i].y);
        r.balls.push_back(ballCenter[i].z);
        r.balls.push_back(ballRadius[i]);
    }
    m_Phys.push_back(std::move(r));
    AgentEvent(EV_PHYS, (uint16_t) m_Phys.back().seq, "Physicalize");
}

void BallrainIO::OnUnphysicalize(CK3dEntity *target) {
    if (m_Mode != Mode::Agent || !m_Active || !target || (m_Phase != Phase::Running && m_Phase != Phase::Tail))
        return;
    if (target == m_BallRealOf) {   // its IVP object is about to be destroyed
        m_BallReal = nullptr;
        m_BallRealOf = nullptr;
    }
    PhysRec r;
    r.seq = (uint32_t) m_Phys.size();
    r.frame = (uint32_t) m_FrameIndex;
    r.ent = target->GetID();
    r.kind = 1;
    m_Phys.push_back(std::move(r));
    AgentEvent(EV_UNPHYS, (uint16_t) m_Phys.back().seq, "Unphysicalize");
}

// Writes PHYS / MESH sections for records [from, to) (MESH: (entity, mesh) pairs not in `done`) and an
// IVPO section with every (physicalized entity, IVP object) pair of the cluster walk not in `ivpoDone`.
static void WritePhysSections(Wire &w, StrTab &strs, CKContext *ctx, const std::vector<BallrainIO::PhysRec> &recs,
                              size_t from, size_t to, std::vector<std::pair<CK_ID, CK_ID>> &done,
                              const std::vector<unsigned char *> &reals, bool physOk,
                              std::unordered_set<uint64_t> &ivpoDone);

unsigned char *BallrainIO::AgentClusterStart() {
    if (m_PhysicsOk <= 0)
        return nullptr;
    u8 *start = AgentFindBallReal(GetActiveBall());
    if (!start) {   // no ball yet: any movable object is in the same (root) cluster
        auto *mgr = reinterpret_cast<u8 *>(m_IpionManager);
        uint16_t memsize = 0, n = 0;
        void **elems = nullptr;
        if (Rd(mgr, MGR_MOVABLE_VEC, memsize) && Rd(mgr, MGR_MOVABLE_VEC + 2, n) && Rd(mgr, MGR_MOVABLE_VEC + 4, elems) &&
            elems && n > 0 && n <= memsize && n <= 8000)
            SafeRead(elems, &start, sizeof(start));
    }
    return start;
}

void BallrainIO::AgentExportPhysicalized() {
    AgentRebuildEntityMap();
    AgentWalkCluster(AgentClusterStart());
    const size_t from = m_PhysExported, to = m_Phys.size();
    if (m_LevelInc) {
        std::vector<uint8_t> buf;
        Wire w(&buf);
        StrTab strs;
        BeginMessage(w, "LVL1", 1, (uint32_t) m_FrameIndex);
        WritePhysSections(w, strs, m_BML->GetCKContext(), m_Phys, from, to, m_MeshesDone, m_Reals, m_PhysicsOk > 0,
                          m_IvpoDone);
        strs.Write(w);
        EndMessage(w);
        std::fwrite(buf.data(), 1, buf.size(), m_LevelInc);
        std::fflush(m_LevelInc);
    }
    for (size_t i = from; i < to; ++i)
        m_PhysNew.push_back(m_Phys[i].seq);
    m_PhysExported = to;
}

static void WritePhysSections(Wire &w, StrTab &strs, CKContext *ctx, const std::vector<BallrainIO::PhysRec> &recs,
                              size_t from, size_t to, std::vector<std::pair<CK_ID, CK_ID>> &done,
                              const std::vector<unsigned char *> &reals, bool physOk,
                              std::unordered_set<uint64_t> &ivpoDone) {
    w.Begin("PHYS", 1);
    w.U32((uint32_t) (to - from));
    for (size_t i = from; i < to; ++i) {
        const BallrainIO::PhysRec &r = recs[i];
        const bool phys = r.kind == 0;
        w.U32(r.seq);
        w.U32(r.frame);
        w.U32(r.ent);
        w.U8(phys ? 0 : 1);   // unphysicalize
        w.U8(r.fixed);
        w.F32(r.friction);
        w.F32(r.elasticity);
        w.F32(r.mass);
        w.U32(phys ? strs.Id(r.collGroup.c_str()) : kNoString);
        w.U8(r.frozen);
        w.U8(r.coll);
        w.U8(r.calcmc);
        w.F32(r.linDamp);
        w.F32(r.rotDamp);
        w.U32(phys ? strs.Id(r.collSurface.c_str()) : kNoString);
        w.Raw(r.massCenter, sizeof(r.massCenter));
        w.U16((uint32_t) r.convex.size());
        for (CK_ID id : r.convex)
            w.U32(id);
        w.U16((uint32_t) (r.balls.size() / 4));
        w.Raw(r.balls.data(), r.balls.size() * sizeof(float));
        w.U16((uint32_t) r.concave.size());
        for (CK_ID id : r.concave)
            w.U32(id);
    }
    w.End();

    // MESH: every (entity, mesh) pair once, local vertices + the entity's world matrix now.
    w.Begin("MESH", 1);
    const size_t at = w.Size();
    w.U32(0);
    uint32_t n = 0;
    for (size_t i = from; i < to; ++i) {
        const BallrainIO::PhysRec &r = recs[i];
        if (r.kind != 0)
            continue;
        auto *ent = (CK3dEntity *) ctx->GetObject(r.ent);
        if (!ent || !CKIsChildClassOf(ent, CKCID_3DENTITY))
            continue;
        for (int kind = 0; kind < 2; ++kind)
            for (CK_ID mid : kind == 0 ? r.convex : r.concave) {
                const std::pair<CK_ID, CK_ID> key(r.ent, mid);
                if (std::find(done.begin(), done.end(), key) != done.end())
                    continue;
                auto *mesh = (CKMesh *) ctx->GetObject(mid);
                if (!mesh || !CKIsChildClassOf(mesh, CKCID_MESH))
                    continue;
                done.push_back(key);
                const VxMatrix &wm = ent->GetWorldMatrix();
                const int nv = mesh->GetVertexCount(), nf = mesh->GetFaceCount();
                w.U32(r.ent);
                w.U32(mid);
                w.U8((uint32_t) kind);
                for (int a = 0; a < 4; ++a)
                    for (int b = 0; b < 4; ++b)
                        w.F32(wm[a][b]);
                w.U32((uint32_t) nv);
                for (int v = 0; v < nv; ++v) {
                    VxVector p;
                    mesh->GetVertexPosition(v, &p);
                    w.F32(p.x);
                    w.F32(p.y);
                    w.F32(p.z);
                }
                w.U32((uint32_t) nf);
                for (int f = 0; f < nf; ++f) {
                    int a = 0, b = 0, c = 0;
                    mesh->GetFaceVertexIndex(f, a, b, c);
                    w.U32((uint32_t) a);
                    w.U32((uint32_t) b);
                    w.U32((uint32_t) c);
                }
                ++n;
            }
    }
    w.PatchU32(at, n);
    w.End();

    // IVPO: IVP objects of the physicalized entities found by the cluster walk, each pair once.
    w.Begin("IVPO", 1);
    const size_t at2 = w.Size();
    w.U32(0);
    uint32_t n2 = 0;
    if (physOk) {
        std::unordered_map<const void *, CK_ID> want;
        for (size_t i = 0; i < recs.size(); ++i)
            if (recs[i].kind == 0)
                if (CKObject *o = ctx->GetObject(recs[i].ent))
                    want.emplace(o, recs[i].ent);
        for (unsigned char *ro : reals) {
            void *client = nullptr;
            if (!Rd(ro, RO_CLIENT, client) || !client)
                continue;
            auto it = want.find(client);
            if (it == want.end())
                continue;
            const CK_ID id = it->second;
            const uint64_t key = (uint64_t) id << 32 | (uint32_t) reinterpret_cast<uintptr_t>(ro);
            if (!ivpoDone.insert(key).second)
                continue;
            int32_t type = 0;
            float xrad = 0;
            u8 *core = nullptr, *mat = nullptr;
            u8 matRaw[32] = {};
            Rd(ro, RO_TYPE, type);
            Rd(ro, RO_XRAD, xrad);
            Rd(ro, RO_CORE, core);
            if (Rd(ro, RO_MAT, mat) && mat)
                SafeRead(mat, matRaw, sizeof(matRaw));
            w.U32(id);
            w.I32(type);
            w.U8(core && CoreUnmoveable(core) ? 1 : 0);
            w.F32(xrad);
            w.Raw(matRaw, sizeof(matRaw));
            ++n2;
        }
    }
    w.PatchU32(at2, n2);
    w.End();
}

// ================================================================ level message

void BallrainIO::AgentLevelTables() {
    CKContext *ctx = m_BML->GetCKContext();
    AgentRebuildEntityMap();
    for (int k = 0; k < 5; ++k) {
        CKObject *o = ctx->GetObjectByNameAndParentClass((CKSTRING) kCamNames[k], CKCID_3DENTITY, nullptr);
        m_CamIds[k] = o ? o->GetID() : 0;
    }
    m_ContactBBs.clear();
    m_ContactHandles.clear();
    {
        const int n = ctx->GetObjectsCountByClassID(CKCID_BEHAVIOR);
        CK_ID *ids = ctx->GetObjectsListByClassID(CKCID_BEHAVIOR);
        for (int k = 0; k < n; ++k) {
            auto *b = (CKBehavior *) ctx->GetObject(ids[k]);
            if (!b || !b->GetPrototypeName() || strcmp(b->GetPrototypeName(), "PhysicsContinuousContact") != 0)
                continue;
            CKParameter *handle = nullptr;
            for (int i = 0; i < b->GetLocalParameterCount() && !handle; ++i) {
                CKParameter *p = b->GetLocalParameter(i);
                if (p && p->GetName() && !strcmp(p->GetName(), "IVP_Handle"))
                    handle = p;
            }
            if (!handle)
                continue;
            m_ContactBBs.push_back(b->GetID());
            m_ContactHandles.push_back(handle->GetID());
        }
    }
    m_Watch.clear();
    m_WatchGroup.clear();
    {
        std::vector<std::pair<CK_ID, CK_ID>> watch;   // (object, group)
        const int n = ctx->GetObjectsCountByClassID(CKCID_GROUP);
        CK_ID *ids = ctx->GetObjectsListByClassID(CKCID_GROUP);
        for (int k = 0; k < n; ++k) {
            auto *g = (CKGroup *) ctx->GetObject(ids[k]);
            const char *gn = g ? g->GetName() : nullptr;
            if (!gn)
                continue;
            const bool watched = StartsWith(gn, "All_P_Extra") || StartsWith(gn, "All_P_Trafo") || StartsWith(gn, "All_PC_") ||
                               StartsWith(gn, "All_PS_") || StartsWith(gn, "All_PE_") || StartsWith(gn, "LifeBalls") ||
                               StartsWith(gn, "Trafos") || StartsWith(gn, "P_Extra") || StartsWith(gn, "P_Trafo") ||
                               StartsWith(gn, "PC_") || StartsWith(gn, "PR_") || StartsWith(gn, "PS_") ||
                               StartsWith(gn, "PE_") || ContainsI(gn, "Depth");
            if (!watched)
                continue;
            for (int i = 0; i < g->GetObjectCount(); ++i)
                if (CKBeObject *o = g->GetObject(i))
                    if (CKIsChildClassOf(o, CKCID_3DENTITY))
                        watch.emplace_back(o->GetID(), g->GetID());
        }
        std::stable_sort(watch.begin(), watch.end(),
                         [](const std::pair<CK_ID, CK_ID> &a, const std::pair<CK_ID, CK_ID> &b) { return a.first < b.first; });
        for (const auto &wg : watch)   // each object once, with the first group that matched
            if (m_Watch.empty() || m_Watch.back() != wg.first) {
                m_Watch.push_back(wg.first);
                m_WatchGroup.push_back(wg.second);
            }
        m_WatchLast.assign(m_Watch.size(), WatchLast{255, {0, 0, 0}});
    }
    m_FollowerSlots.clear();
    {
        std::vector<std::string> sources;
        for (size_t i = 0; i < m_Watch.size(); ++i) {
            CKObject *o = ctx->GetObject(m_Watch[i]);
            std::string src;
            int k = 0;
            if (!o || !ParseFollowerName(o->GetName(), src, k))
                continue;
            auto it = std::find(sources.begin(), sources.end(), src);
            const auto id = (uint16_t) (it - sources.begin());
            if (it == sources.end())
                sources.push_back(src);
            m_FollowerSlots.push_back(FollowerSlot{(uint32_t) i, id, (uint8_t) k});
        }
        m_SourceMainVis.assign(sources.size(), -1);
    }
    if (CKDataArray *a = m_BML->GetArrayByName("Energy"))
        m_EnergyId = a->GetID();
    if (CKDataArray *a = m_BML->GetArrayByName("AllLevel"))
        m_AllLevelId = a->GetID();
    m_TablesReady = true;
    GetLogger()->Info("agent: level tables: %u contact blocks, %u watched objects, cameras %u/%u/%u/%u/%u",
                      (unsigned) m_ContactBBs.size(), (unsigned) m_Watch.size(), m_CamIds[0], m_CamIds[1], m_CamIds[2],
                      m_CamIds[3], m_CamIds[4]);
}

void BallrainIO::AgentWriteLevelMessage() {
    if (m_LevelMsgPath.empty())
        return;
    CKContext *ctx = m_BML->GetCKContext();
    if (!m_TablesReady)
        AgentLevelTables();
    std::vector<uint8_t> buf;
    buf.reserve(8u << 20);
    Wire w(&buf);
    StrTab strs;
    BeginMessage(w, "LVL1", 0, (uint32_t) m_FrameIndex);
    CKAttributeManager *am = ctx->GetAttributeManager();
    CKParameterManager *pm = ctx->GetParameterManager();
    int attrTypes[3];
    for (int k = 0; k < 3; ++k)
        attrTypes[k] = am ? am->GetAttributeTypeByName((CKSTRING) kAttrNames[k]) : -1;

    // ENTS
    {
        const XObjectPointerArray &list = ctx->GetObjectListByType(CKCID_3DENTITY, TRUE);
        w.Begin("ENTS", 1);
        w.U32((uint32_t) list.Size());
        for (int k = 0; k < list.Size(); ++k) {
            auto *e = (CK3dEntity *) list[k];
            if (!e) {   // keeps the count: an empty entity record
                w.Zeros(4 * 4 + 1 + 48);
                w.I32(-1);
                w.I32(-1);
                w.U16(0);
                continue;
            }
            CK3dEntity *parent = e->GetParent();
            w.U32(e->GetID());
            w.U32(strs.Id(e->GetName()));
            w.U32((uint32_t) e->GetClassID());
            w.U32(parent ? parent->GetID() : 0);
            w.U8(e->IsVisible() ? 1 : 0);
            PutMatrix12(w, e->GetWorldMatrix());
            // Attributes by name only (state-spec.md 0). Contact / collision ids: the int value, a float
            // parameter's value truncated, -1 when the entity has no such attribute. Floor: raw bytes.
            for (int a = 0; a < 3; ++a) {
                const int t = attrTypes[a];
                const bool present = t >= 0 && e->HasAttribute(t);
                CKParameterOut *p = present ? e->GetAttributeParameter(t) : nullptr;
                u8 rawAttr[64] = {};
                int size = 0;
                if (p) {
                    const void *data = p->GetReadDataPtr(FALSE);
                    size = std::min(p->GetDataSize(), 64);
                    if (!data || size < 0 || !SafeRead(data, rawAttr, (size_t) size))
                        size = 0;
                }
                if (a < 2) {
                    int32_t v = -1;
                    if (present) {
                        v = 0;
                        if (size >= 4 && p->GetGUID() == CKPGUID_FLOAT) {
                            float f;
                            std::memcpy(&f, rawAttr, 4);
                            v = (int32_t) f;
                        } else if (size >= 4) {
                            std::memcpy(&v, rawAttr, 4);
                        }
                    }
                    w.I32(v);
                } else {
                    w.U16((uint32_t) size);
                    w.Raw(rawAttr, (size_t) size);
                }
            }
        }
        w.End();
        // EMSH: current mesh of every entity (extension; skipped by readers that do not know it)
        w.Begin("EMSH", 1);
        const size_t at = w.Size();
        w.U32(0);
        uint32_t cnt = 0;
        for (int k = 0; k < list.Size(); ++k) {
            auto *e = (CK3dEntity *) list[k];
            CKMesh *mesh = e ? e->GetCurrentMesh() : nullptr;
            if (!mesh)
                continue;
            w.U32(e->GetID());
            w.U32(mesh->GetID());
            ++cnt;
        }
        w.PatchU32(at, cnt);
        w.End();
    }
    // GRPS
    {
        const int n = ctx->GetObjectsCountByClassID(CKCID_GROUP);
        CK_ID *ids = ctx->GetObjectsListByClassID(CKCID_GROUP);
        w.Begin("GRPS", 1);
        const size_t at = w.Size();
        w.U32(0);
        uint32_t cnt = 0;
        for (int k = 0; k < n; ++k) {
            auto *g = (CKGroup *) ctx->GetObject(ids[k]);
            if (!g)
                continue;
            w.U32(g->GetID());
            w.U32(strs.Id(g->GetName()));
            w.U32((uint32_t) g->GetObjectCount());
            for (int i = 0; i < g->GetObjectCount(); ++i) {
                CKBeObject *o = g->GetObject(i);
                w.U32(o ? o->GetID() : 0);
            }
            ++cnt;
        }
        w.PatchU32(at, cnt);
        w.End();
    }
    // ARRS (full)
    {
        const int n = ctx->GetObjectsCountByClassID(CKCID_DATAARRAY);
        CK_ID *ids = ctx->GetObjectsListByClassID(CKCID_DATAARRAY);
        w.Begin("ARRS", 1);
        const size_t at = w.Size();
        w.U32(0);
        uint32_t cnt = 0;
        u8 cell[256];
        for (int k = 0; k < n; ++k) {
            auto *a = (CKDataArray *) ctx->GetObject(ids[k]);
            if (!a)
                continue;
            int rows = a->GetRowCount(), cols = a->GetColumnCount();
            if (cols > 0xFFFF)
                cols = 0;
            if (rows > 0xFFFF || (int64_t) rows * cols > 1000000)
                rows = 0;   // not exported (no such array in the game)
            w.U32(a->GetID());
            w.U32(strs.Id(a->GetName()));
            w.U16((uint32_t) rows);
            w.U16((uint32_t) cols);
            for (int c = 0; c < cols; ++c) {
                const CK_ARRAYTYPE type = a->GetColumnType(c);
                w.U32(strs.Id(a->GetColumnName(c)));
                w.U8((uint32_t) type);
                const char *ptype = nullptr;
                if (type == CKARRAYTYPE_PARAMETER && pm)
                    ptype = pm->ParameterGuidToName(a->GetColumnParameterGuid(c));
                w.U32(strs.Id(ptype));
            }
            for (int r = 0; r < rows; ++r)
                    for (int c = 0; c < cols; ++c) {
                        const size_t len = CellBytes(a, r, c, cell);
                        w.U8((uint32_t) len);
                        w.Raw(cell, len);
                    }
            ++cnt;
        }
        w.PatchU32(at, cnt);
        w.End();
    }
    // PHYS / MESH / IVPO: everything physicalized so far
    {
        AgentWalkCluster(AgentClusterStart());
        std::vector<std::pair<CK_ID, CK_ID>> done;
        std::unordered_set<uint64_t> ivpoDone;
        WritePhysSections(w, strs, ctx, m_Phys, 0, m_Phys.size(), done, m_Reals, m_PhysicsOk > 0, ivpoDone);
    }
    // DEPT: world AABB (from mesh vertices) of the members of groups whose name contains "Depth"
    {
        const int n = ctx->GetObjectsCountByClassID(CKCID_GROUP);
        CK_ID *ids = ctx->GetObjectsListByClassID(CKCID_GROUP);
        w.Begin("DEPT", 1);
        const size_t at = w.Size();
        w.U32(0);
        uint32_t cnt = 0;
        for (int k = 0; k < n; ++k) {
            auto *g = (CKGroup *) ctx->GetObject(ids[k]);
            if (!g || !ContainsI(g->GetName(), "Depth"))
                continue;
            for (int i = 0; i < g->GetObjectCount(); ++i) {
                CKBeObject *o = g->GetObject(i);
                if (!o || !CKIsChildClassOf(o, CKCID_3DENTITY))
                    continue;
                auto *e = (CK3dEntity *) o;
                CKMesh *mesh = e->GetCurrentMesh();
                if (!mesh || mesh->GetVertexCount() <= 0)
                    continue;
                VxVector lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
                const VxMatrix &wm = e->GetWorldMatrix();
                for (int v = 0; v < mesh->GetVertexCount(); ++v) {
                    VxVector p, t;
                    mesh->GetVertexPosition(v, &p);
                    Vx3DMultiplyMatrixVector(&t, wm, &p);
                    lo.x = std::min(lo.x, t.x);
                    lo.y = std::min(lo.y, t.y);
                    lo.z = std::min(lo.z, t.z);
                    hi.x = std::max(hi.x, t.x);
                    hi.y = std::max(hi.y, t.y);
                    hi.z = std::max(hi.z, t.z);
                }
                w.U32(e->GetID());
                w.F32(lo.x);
                w.F32(lo.y);
                w.F32(lo.z);
                w.F32(hi.x);
                w.F32(hi.y);
                w.F32(hi.z);
                ++cnt;
            }
        }
        w.PatchU32(at, cnt);
        w.End();
    }
    // WLST
    {
        w.Begin("WLST", 1);
        w.U32((uint32_t) m_Watch.size());
        for (size_t i = 0; i < m_Watch.size(); ++i) {
            CKObject *g = ctx->GetObject(m_WatchGroup[i]);
            w.U32(m_Watch[i]);
            w.U32(strs.Id(g ? g->GetName() : nullptr));
        }
        w.End();
    }
    // CAMS / contact blocks resolved for this level
    {
        w.Begin("LTAB", 1);
        for (CK_ID id : m_CamIds)
            w.U32(id);
        w.U32((uint32_t) m_ContactBBs.size());
        for (CK_ID id : m_ContactBBs)
            w.U32(id);
        w.End();
    }
    strs.Write(w);
    EndMessage(w);

    const std::string tmp = m_LevelMsgPath + ".tmp";
    FILE *f = std::fopen(tmp.c_str(), "wb");
    bool ok = f && std::fwrite(buf.data(), 1, buf.size(), f) == buf.size();
    if (f)
        ok = std::fclose(f) == 0 && ok;
    ok = ok && ::MoveFileExA(tmp.c_str(), m_LevelMsgPath.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
    if (!ok)
        GetLogger()->Error("agent: cannot write level message %s", m_LevelMsgPath.c_str());
    else
        GetLogger()->Info("agent: level message %s: %u bytes", m_LevelMsgPath.c_str(), (unsigned) buf.size());
}
