// Ballrain headless agent mode: the BallrainIO agent mode (docs/agent-protocol.md) on the headless
// x87-soft engine of BallanceMMO's SimTool, speaking the same OBS1 / ACT1 / LVL1 wire format over a
// socket instead of shared memory (docs/headless.md).
//
//   BallanceMMOSimTool --ballrain-agent --root <game dir> [--boot-ticks N] [SimTool's --tas-* flags]
//
// Configuration comes from the BALLRAIN_* environment variables of the BallrainIO agent mode
// (BALLRAIN_LEVEL, BALLRAIN_LEVELMSG, BALLRAIN_FRAMELOG, BALLRAIN_PREFIX, ...), plus BALLRAIN_SOCK
// (unix:<path> | tcp:<host>:<port>, Python listens), BALLRAIN_HL_BALL_ORDER (the ball-pair contact order,
// docs/headless.md) and BALLRAIN_HL_CRT_SEED / BALLRAIN_HL_VERBOSE.
//
// The frame path is SimTool's --tas playback (BallanceMMO x87-soft), with its flags and defaults
// (ballrain_agent.h): boot ticks, TASSupport's legacy edit (SimTool's apply_tas_legacy_edit), no x87
// precision reset at the level load, "Load Level", --tas-offset (34) hidden fixed ticks, then every
// frame = time manager delta + 9 legacy keys written raw into the keyboard state + one CKContext::Process;
// the physics clock reset before frame --tas-reset-frame (1) with the retail reset state
// (--tas-movement-check 4, --tas-seed 75^21). BML+'s event hooks are re-created on the same graph
// positions (BML+ 0.3.12 EventHook.cpp / ScriptHelper.cpp), and its Physicalize hook by wrapping the
// prototype function. Everything else follows BallrainIO/src/Agent.cpp; where the 2002 DLL's raw IVP
// memory was read, the open IVP classes are read by member name.
//
// The Windows x86 build of patch 0003 (BALLRAIN_PHYS2002_HOST) runs the ORIGINAL 2002 physics_RT.dll instead
// (phys2002.hpp; BALLRAIN_HL_PHYS2002 = its path | 0, default <root>/BuildingBlocks/physics_RT.dll): then TIME, BALL,
// FRIC, MIND, BODY, GLCT and IVPO read the DLL's memory exactly as BallrainIO/src/Agent.cpp does (Ivp2002.h), the
// clock reset writes it as TASSupport / SimTool --tas do, the ball-pair key goes to the DLL's hooked compare, and
// every field below except MIND ledge (an address), bml_hs and the CK ids is the retail one.
//
// What the engine cannot give (marked in the data: OBS1 flag OBS_HEADLESS, docs/headless.md):
// MIND ledge (an address), IVPO material bytes, GLCT group counters,
// BALL core_flags / core_state other than the unmoveable bits / movement state, SCOR / SREN bml_hs
// (computed as Points + Lifes x LifeBonus, BML+'s value), CK ids (allocated by this engine).

#include "headless_engine.hpp"
#include "ballrain_agent.h"

#include "CKAll.h"
#include "CKIpionManager.h"
#include "ivp_environment.hxx"

#include <physics/physics_state.hpp>

#include "../../BallanceMMOCommon/src/physics/physics_savestate_internal.hpp"
#include "ivp_compact_ledge.hxx"
#include "ballrain_wire.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <fstream>
#include <functional>
#include <memory>
#include <type_traits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET sock_t;
#define BR_BADSOCK INVALID_SOCKET
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
typedef int sock_t;
#define BR_BADSOCK (-1)
#endif

#ifndef IVP_PORTABLE_BMMO
#error "the agent reads IVP members of the BallanceMMO build (IVP_PORTABLE_BMMO)"
#endif

// Ballrain hl-win (patches/0003, Windows x86): the engine can run the ORIGINAL 2002 physics_RT.dll (phys2002.hpp);
// then the IVP sections read the DLL's raw memory exactly as BallrainIO/src/Agent.cpp does (layouts from its
// Ivp2002.h, copied next to this file by the build), and nothing of the source IVP classes is touched.
#ifdef BALLRAIN_PHYS2002_HOST
#include "phys2002.hpp"
#include "Ivp2002.h"
#include <windows.h>
bool SafeRead(const void *p, void *out, size_t n) {   // BallrainIO Agent.cpp: SEH + memcpy
    if (!p)
        return false;
    __try {
        std::memcpy(out, p, n);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}
#endif

int ivp_srand_read();      // ivu_os_dep.cxx
void ivp_srand(int seed);

namespace {

typedef unsigned char u8;

// ------------------------------------------------------------------------------------------- constants
constexpr const char *kVersion = "BallrainIO-headless 0.1";
constexpr uint32_t BR_PING = 0x100, BR_PONG = 0x101, BR_STATE = 0x102, BR_OBS = 0x103, BR_ACT = 0x104;

enum : uint32_t { ST_INIT, ST_HELLO, ST_MENU, ST_LOADING, ST_RUNNING, ST_WAITING, ST_ENDED, ST_ERROR };
enum : uint32_t { END_NONE, END_FINISH, END_GAMEOVER, END_MAXFRAMES, END_STOP, END_RESET, END_TIMEOUT, END_ERROR };
enum : uint32_t { ERR_NONE, ERR_SHM, ERR_ENV, ERR_DLL, ERR_ACT, ERR_SEQ };

constexpr uint32_t ACT_MAGIC = 0x31544341;   // 'ACT1'
constexpr uint32_t ACT_HEADER = 0x24;
constexpr uint32_t ACT_STOP = 1, ACT_TRAC = 2, ACT_LEVELMSG = 4;
constexpr size_t ACT_MAX_FRAMES = 1u << 20;

constexpr uint16_t OBS_FINAL = 1, OBS_PREFIX = 2, OBS_DROPPED = 4, OBS_MINIMAL = 8, OBS_HEADLESS = 16;
constexpr uint16_t SECF_APPROX = 0x8000;   // section with fields the headless engine cannot give
constexpr uint32_t M_BODY = 1u << 0, M_MIND = 1u << 1, M_FRIC = 1u << 2, M_CAMS = 1u << 3, M_GLCT = 1u << 4,
                   M_WTCH = 1u << 5, M_BALL = 1u << 6, M_ARRS = 1u << 7;

enum : uint16_t {
    EV_PRELOAD = 1, EV_POSTLOAD, EV_START, EV_CAMNAV_ON, EV_CAMNAV_OFF, EV_BALLNAV_ON, EV_BALLNAV_OFF,
    EV_COUNTER_ON, EV_COUNTER_OFF, EV_PAUSE, EV_UNPAUSE, EV_BALLOFF, EV_DEAD, EV_PRESUBLIFE, EV_POSTSUBLIFE,
    EV_PRECHECK, EV_POSTCHECK, EV_EXTRAPOINT, EV_PRELIFEUP, EV_POSTLIFEUP, EV_FINISH, EV_GAMEOVER,
    EV_PREENDLEVEL, EV_POSTENDLEVEL, EV_PHYS, EV_UNPHYS, EV_ACTIVEBALL, EV_PRERESET, EV_PREEXIT, EV_RUNEND
};
const char *const kEventNames[] = {
    "-", "PreLoadLevel", "PostLoadLevel", "StartLevel", "CamNavActive", "CamNavInactive", "BallNavActive",
    "BallNavInactive", "CounterActive", "CounterInactive", "PauseLevel", "UnpauseLevel", "BallOff", "Dead",
    "PreSubLife", "PostSubLife", "PreCheckpoint", "PostCheckpoint", "ExtraPoint", "PreLifeUp", "PostLifeUp",
    "LevelFinish", "GameOver", "PreEndLevel", "PostEndLevel", "Physicalize", "Unphysicalize",
    "ActiveBallChanged", "PreResetLevel", "PreExitLevel", "RunEnd"};

const char *const kTrackedArrays[] = {"Energy", "CurrentLevel", "IngameParameter", "PH", "Checkpoints", "ResetPoints"};
constexpr int kTrackedCount = 6;
const char *const kCamNames[5] = {"Cam_OrientRef", "Cam_Orient", "Cam_Pos", "Cam_Target", "InGameCam"};
const char *const kAttrNames[3] = {"Continuous Contact ID", "Coll Detection ID", "Floor"};
constexpr uint32_t kNoString = 0xFFFFFFFFu;
const CKGUID kPhysicalizeGuid(0x7522370e, 0x37ec15ec);
const CKGUID kNopGuid(0x302561c4, 0xd282980);

std::string GetEnvS(const char *name) {
    const char *v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

void Log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fputs("ballrain: ", stderr);
    std::vfprintf(stderr, fmt, ap);
    std::fputc('\n', stderr);
    va_end(ap);
}

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

#ifdef BALLRAIN_PHYS2002_HOST
// ---- 2002 DLL raw reads (BallrainIO Agent.cpp, unchanged)
using namespace ivp2002;
// f32 now_friction_pressure of the contact point (not in Ivp2002.h). ivp_friction.hxx IVP_Contact_Point_Fast:
// tmp_contact_info 0x40 (CP_LT), real_friction_factor 0x44 (CP_FRICTION), integrated_destroyed_energy 0x48,
// inv_triangle_det 0x4C, old_energy_dynamic_fr 0x50, now_friction_pressure 0x54, last_gap_len 0x58 (CP_GAP).
constexpr ptrdiff_t CP_PRESSURE = 0x54;

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
#endif

#ifdef BALLRAIN_PHYS2002_HOST
const char *Phys2002Sha() { return ballrain::phys2002::kSha256; }
#else
const char *Phys2002Sha() { return ""; }
#endif

// Raw bytes of one data-array cell, as BallrainIO's CellBytes.
size_t CellBytes(CKDataArray *a, int r, int c, u8 *out /* >= 256 */) {
    static char sbuf[1 << 16];
    const CK_ARRAYTYPE t = a->GetColumnType(c);
    auto *e = a->GetElement(r, c);   // CKUINTPTR cells on 64-bit builds: int / float / object in the low 4 bytes
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
            if (a->GetElementStringValue(r, c, sbuf, (int) sizeof(sbuf)) <= 0)
                return 0;
            sbuf[sizeof(sbuf) - 1] = 0;
            const size_t n = std::min<size_t>(strnlen(sbuf, sizeof(sbuf)), 255);
            std::memcpy(out, sbuf, n);
            return n;
        }
        case CKARRAYTYPE_PARAMETER: {
            auto *p = (CKParameter *) (uintptr_t) *e;
            if (!p)
                return 0;
            const void *data = p->GetReadDataPtr(FALSE);
            const int size = std::min(p->GetDataSize(), 64);
            if (!data || size <= 0)
                return 0;
            std::memcpy(out, data, (size_t) size);
            return (size_t) size;
        }
        default:
            return 0;
    }
}

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

struct StrTab {
    std::unordered_map<std::string, uint32_t> index;
    std::vector<std::string> list;
    uint32_t Id(const char *s) {
        if (!s)
            return kNoString;
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
    w.U32(0);
    w.U32(frames);
    w.U32(0);
    w.U32(0);
}

void EndMessage(Wire &w) {
    w.PatchU32(8, (uint32_t) w.Size());
    w.PatchU32(16, w.Sections());
}

// FRCP v1 (headless only, optional): u16 n, then n x (u32 partner ckid, f32 now_friction_pressure), the same items in
// the same order as the FRIC section just written (the contact's normal and point are FRIC's lt_* fields). IVP keeps the
// pressure on each friction contact point; airborne / impact frames have no friction contact (n = 0). Readers that do
// not know the tag skip it.
struct FrcpItem {
    uint32_t partner = 0;
    float pressure = 0;
};
void WriteFrcp(Wire &w, const FrcpItem *items, uint32_t n) {
    w.Begin("FRCP", 1);
    w.U16((uint16_t) n);
    for (uint32_t i = 0; i < n; ++i) {
        w.U32(items[i].partner);
        w.F32(items[i].pressure);
    }
    w.End();
}

// ------------------------------------------------------------------------------------------- socket
class Channel {
public:
    ~Channel() { Close(); }
    bool Connect(const std::string &addr) {
#ifdef _WIN32
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
        if (addr.rfind("tcp:", 0) == 0) {
            const std::string rest = addr.substr(4);
            const size_t colon = rest.rfind(':');
            if (colon == std::string::npos)
                return false;
            const std::string host = rest.substr(0, colon), port = rest.substr(colon + 1);
            addrinfo hints = {}, *res = nullptr;
            hints.ai_family = AF_UNSPEC;
            hints.ai_socktype = SOCK_STREAM;
            if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || !res)
                return false;
            for (addrinfo *p = res; p && m_Sock == BR_BADSOCK; p = p->ai_next) {
                sock_t s = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
                if (s == BR_BADSOCK)
                    continue;
                if (connect(s, p->ai_addr, (int) p->ai_addrlen) == 0) {
                    m_Sock = s;
                } else {
#ifdef _WIN32
                    closesocket(s);
#else
                    ::close(s);
#endif
                }
            }
            freeaddrinfo(res);
            if (m_Sock == BR_BADSOCK)
                return false;
            int one = 1;
            setsockopt(m_Sock, IPPROTO_TCP, TCP_NODELAY, (const char *) &one, sizeof(one));
            return true;
        }
#ifndef _WIN32
        if (addr.rfind("unix:", 0) == 0) {
            const std::string path = addr.substr(5);
            sockaddr_un sa = {};
            sa.sun_family = AF_UNIX;
            if (path.size() >= sizeof(sa.sun_path))
                return false;
            std::memcpy(sa.sun_path, path.c_str(), path.size() + 1);
            sock_t s = socket(AF_UNIX, SOCK_STREAM, 0);
            if (s == BR_BADSOCK)
                return false;
            if (connect(s, (sockaddr *) &sa, sizeof(sa)) != 0) {
                ::close(s);
                return false;
            }
            m_Sock = s;
            return true;
        }
#endif
        return false;
    }
    void Close() {
        if (m_Sock != BR_BADSOCK) {
#ifdef _WIN32
            closesocket(m_Sock);
#else
            ::close(m_Sock);
#endif
            m_Sock = BR_BADSOCK;
        }
    }
    bool Ok() const { return m_Sock != BR_BADSOCK; }

    bool Send(uint32_t type, const void *a, size_t na, const void *b = nullptr, size_t nb = 0) {
        if (m_Sock == BR_BADSOCK)
            return false;
        uint32_t hdr[2] = {type, (uint32_t) (na + nb)};
        return SendAll(hdr, sizeof(hdr)) && SendAll(a, na) && SendAll(b, nb);
    }
    // 1: message, 0: timeout, -1: closed / error.
    int Recv(uint32_t &type, std::vector<u8> &payload, uint32_t timeout_ms) {
        uint32_t hdr[2];
        int r = RecvAll(hdr, sizeof(hdr), timeout_ms);
        if (r <= 0)
            return r;
        type = hdr[0];
        if (hdr[1] > (64u << 20))
            return -1;
        payload.resize(hdr[1]);
        r = RecvAll(payload.data(), hdr[1], 30000);
        return r <= 0 ? -1 : 1;
    }

private:
    bool SendAll(const void *p, size_t n) {
        const char *c = (const char *) p;
        while (n > 0) {
#ifdef _WIN32
            const int k = send(m_Sock, c, (int) n, 0);
#else
            const ssize_t k = send(m_Sock, c, n, MSG_NOSIGNAL_COMPAT);
#endif
            if (k <= 0)
                return false;
            c += k;
            n -= (size_t) k;
        }
        return true;
    }
    int RecvAll(void *p, size_t n, uint32_t timeout_ms) {
        char *c = (char *) p;
        bool first = true;
        while (n > 0) {
            {
#ifdef _WIN32
                WSAPOLLFD pf = {m_Sock, POLLRDNORM, 0};
                const int pr = WSAPoll(&pf, 1, timeout_ms == 0 ? -1 : (int) timeout_ms);
#else
                pollfd pf = {m_Sock, POLLIN, 0};
                const int pr = poll(&pf, 1, timeout_ms == 0 ? -1 : (int) timeout_ms);
#endif
                if (pr == 0)
                    return first ? 0 : -1;
                if (pr < 0)
                    return -1;
            }
#ifdef _WIN32
            const int k = recv(m_Sock, c, (int) n, 0);
#else
            const ssize_t k = recv(m_Sock, c, n, 0);
#endif
            if (k <= 0)
                return -1;
            c += k;
            n -= (size_t) k;
            first = false;
        }
        return 1;
    }
#ifndef _WIN32
#ifdef MSG_NOSIGNAL
    static constexpr int MSG_NOSIGNAL_COMPAT = MSG_NOSIGNAL;
#else
    static constexpr int MSG_NOSIGNAL_COMPAT = 0;
#endif
#endif
    sock_t m_Sock = BR_BADSOCK;
};

// ------------------------------------------------------------------------------------------- ball-pair order
// Which of two touching balls IVP takes as the first of their contact: the sign of the contact normal and the roles
// of the two cores in the impact and friction solve. The 2002 DLL orders the pair by the CK3dEntity heap addresses,
// which differ from one Player.exe process to the next (on L1 the pooled P_Ball_Stone_MF001-004 at the level load);
// the product build orders by CK id (IVP_Environment::bmmo_ball_order_key, installed by CKIpionManager::
// CreateEnvironment). BALLRAIN_HL_BALL_ORDER picks another key (smaller key = lower address = first):
//   ckid (default)         the product's key, untouched
//   reverse                the CK ids reversed
//   perm:<seed>            the pooled balls (P_Ball_*_MF*) swap their CK-id places by a seeded permutation; every other
//                          ball (the player ball included) keeps its place
//   table:<name=rank,...>  the named entities, ordered by rank (distinct ranks), among the CK-id places they occupy
//                          under ckid; every other ball keeps its CK-id place
//   retail:<seed>          a draw of retail's measured per-process order, among the named balls' CK-id places (L1:
//                          Ball_Wood, then P_Ball_Stone_MF / P_Ball_Wood_MF with P(Stone_MF first) = 0.66, then a
//                          uniform permutation of P_Ball_Stone_MF001-004); other levels: ckid (not measured)
// The draws are splitmix64 (ballrain/headless.py makes the same ones).
struct BallOrder {
    enum Mode { CKID, REVERSE, PERM, TABLE, RETAIL } mode = CKID;
    std::string spec = "ckid";
    uint64_t seed = 0;
    std::vector<std::pair<std::string, uint32_t>> ranks;   // TABLE / RETAIL: entity name -> rank
    // built from the level's entities (lazily, at the first key asked for)
    CKContext *ctx = nullptr;
    bool built = false;
    std::unordered_map<CK_ID, uint32_t> keys;   // PERM / TABLE / RETAIL
};
BallOrder g_BallOrder;
constexpr uint32_t kRankLimit = 1u << 20;   // TABLE: ranks 0 .. 2^20-1

uint64_t SplitMix64(uint64_t &state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
uint32_t DrawBelow(uint64_t &state, uint32_t n) {   // uniform in [0, n), by rejection
    const uint64_t limit = UINT64_MAX - UINT64_MAX % n;
    uint64_t x;
    do
        x = SplitMix64(state);
    while (x >= limit);
    return (uint32_t) (x % n);
}
template <class T>
void Shuffle(std::vector<T> &v, uint64_t &state) {   // Fisher-Yates from the back
    for (size_t i = v.size(); i > 1; --i)
        std::swap(v[i - 1], v[DrawBelow(state, (uint32_t) i)]);
}

bool IsPooledBall(const char *name) {   // P_Ball_*_MF*
    return StartsWith(name, "P_Ball_") && std::strstr(name + 7, "_MF") != nullptr;
}

bool ParseU64(const std::string &text, uint64_t &value) {
    if (text.empty() || text.size() > 20 || text.find_first_not_of("0123456789") != std::string::npos)
        return false;
    errno = 0;
    value = std::strtoull(text.c_str(), nullptr, 10);
    return errno == 0;
}

// retail:<seed> on L1 (50 fresh Player.exe processes, runtime/analysis/ball_order/summary.json).
std::vector<std::string> RetailDrawL1(uint64_t seed) {
    uint64_t st = seed;
    std::vector<std::string> order{"Ball_Wood"};
    if (DrawBelow(st, 100) < 66)
        order.insert(order.end(), {"P_Ball_Stone_MF", "P_Ball_Wood_MF"});
    else
        order.insert(order.end(), {"P_Ball_Wood_MF", "P_Ball_Stone_MF"});
    std::vector<std::string> mf{"P_Ball_Stone_MF001", "P_Ball_Stone_MF002", "P_Ball_Stone_MF003", "P_Ball_Stone_MF004"};
    Shuffle(mf, st);
    order.insert(order.end(), mf.begin(), mf.end());
    return order;
}

bool ParseBallOrder(const std::string &spec, int level, BallOrder &out, std::string &err) {
    out = BallOrder();
    out.spec = spec.empty() ? "ckid" : spec;
    const std::string &s = out.spec;
    const size_t colon = s.find(':');
    const std::string mode = s.substr(0, colon), arg = colon == std::string::npos ? "" : s.substr(colon + 1);
    if (mode == "ckid" || mode == "reverse") {
        if (colon != std::string::npos) {
            err = mode + " takes no argument";
            return false;
        }
        out.mode = mode == "ckid" ? BallOrder::CKID : BallOrder::REVERSE;
        return true;
    }
    if (mode == "perm" || mode == "retail") {
        if (!ParseU64(arg, out.seed)) {
            err = mode + ":<seed> needs a decimal seed (0 .. 2^64-1)";
            return false;
        }
        if (mode == "perm") {
            out.mode = BallOrder::PERM;
            return true;
        }
        if (level != 1) {
            Log("warning: ball order %s: retail's order is measured on level 1 only; level %d keeps ckid", s.c_str(),
                level);
            out.mode = BallOrder::CKID;
            return true;
        }
        out.mode = BallOrder::RETAIL;
        const std::vector<std::string> order = RetailDrawL1(out.seed);
        for (size_t i = 0; i < order.size(); ++i)
            out.ranks.emplace_back(order[i], (uint32_t) i);
        return true;
    }
    if (mode == "table") {
        out.mode = BallOrder::TABLE;
        std::unordered_set<std::string> names;
        std::unordered_set<uint64_t> ranks;
        size_t at = 0;
        while (at <= arg.size()) {
            size_t end = arg.find(',', at);
            if (end == std::string::npos)
                end = arg.size();
            const std::string item = arg.substr(at, end - at);
            const size_t eq = item.rfind('=');
            uint64_t rank = 0;
            if (eq == std::string::npos || eq == 0 || !ParseU64(item.substr(eq + 1), rank) || rank >= kRankLimit) {
                err = "table entry '" + item + "' is not <name>=<rank> (rank 0 .. 2^20-1)";
                return false;
            }
            const std::string name = item.substr(0, eq);
            if (!names.insert(name).second || !ranks.insert(rank).second) {
                err = "table: '" + item + "' repeats a name or a rank";
                return false;
            }
            out.ranks.emplace_back(name, (uint32_t) rank);
            at = end + 1;
        }
        return true;
    }
    err = "unknown mode '" + mode + "' (ckid | reverse | perm:<seed> | table:<name=rank,...> | retail:<seed>)";
    return false;
}

// Every 3D entity of the context, by CK id.
std::vector<CK3dEntity *> AllEntities(CKContext *ctx) {
    std::vector<CK3dEntity *> out;
    for (CK_CLASSID cid : {CKCID_3DENTITY, CKCID_3DOBJECT, CKCID_CHARACTER, CKCID_SPRITE3D}) {
        const int n = ctx->GetObjectsCountByClassID(cid);
        CK_ID *ids = ctx->GetObjectsListByClassID(cid);
        for (int i = 0; i < n; ++i)
            if (auto *e = CK3dEntity::Cast(ctx->GetObject(ids[i])))
                out.push_back(e);
    }
    std::sort(out.begin(), out.end(), [](CK3dEntity *a, CK3dEntity *b) { return a->GetID() < b->GetID(); });
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

void BuildBallOrder() {
    BallOrder &o = g_BallOrder;
    o.keys.clear();
    o.built = true;
    if (!o.ctx)
        return;
    const std::vector<CK3dEntity *> all = AllEntities(o.ctx);
    if (o.mode == BallOrder::PERM) {
        std::vector<CK_ID> slots;   // the pooled balls' CK ids, ascending
        for (CK3dEntity *e : all)
            if (IsPooledBall(e->GetName()))
                slots.push_back(e->GetID());
        std::vector<CK_ID> perm = slots;
        uint64_t st = o.seed;
        Shuffle(perm, st);
        for (size_t i = 0; i < slots.size(); ++i)
            o.keys[perm[i]] = (uint32_t) slots[i];   // perm[i] takes the i-th place
    } else if (o.mode == BallOrder::TABLE || o.mode == BallOrder::RETAIL) {
        // the named entities take the CK-id places they occupy under ckid, in rank order (ties: by CK id); every
        // other ball keeps its own CK id as its key
        std::unordered_map<std::string, uint32_t> rank(o.ranks.begin(), o.ranks.end());
        std::unordered_set<std::string> seen;
        std::vector<std::pair<uint32_t, CK_ID>> named;   // (rank, CK id)
        std::vector<CK_ID> slots;                        // their CK ids, ascending (all is by CK id)
        for (CK3dEntity *e : all) {
            auto it = e->GetName() ? rank.find(e->GetName()) : rank.end();
            if (it == rank.end())
                continue;
            if (!seen.insert(it->first).second)
                Log("warning: ball order %s: more than one entity named %s", o.spec.c_str(), it->first.c_str());
            named.emplace_back(it->second, e->GetID());
            slots.push_back(e->GetID());
        }
        std::sort(named.begin(), named.end());
        for (size_t i = 0; i < named.size(); ++i)
            o.keys[named[i].second] = (uint32_t) slots[i];   // the i-th ranked takes the i-th place
    }
}

std::string DescribeBallOrder(CKContext *ctx, unsigned int (*key)(void *));

unsigned int BallOrderKey(void *client_data) {
    auto *e = static_cast<CK3dEntity *>(client_data);
    if (!e)
        return 0;
    const CK_ID id = e->GetID();
    BallOrder &o = g_BallOrder;
    if (o.mode == BallOrder::REVERSE)
        return 0xFFFFFFFFu - (uint32_t) id;
    auto it = o.keys.find(id);
    if (it == o.keys.end()) {
        // a pooled (PERM) or named (TABLE / RETAIL) ball the last build did not see: the entities changed
        const char *name = e->GetName();
        bool missing = !o.built;
        if (!missing && name && o.mode == BallOrder::PERM)
            missing = IsPooledBall(name);
        if (!missing && name && o.mode != BallOrder::PERM)
            for (const auto &r : o.ranks)
                missing = missing || r.first == name;
        if (missing) {
            const bool rebuilt = o.built;
            BuildBallOrder();
            if (rebuilt)
                Log("ball order %s: %s (id %u) appeared after the order was built: rebuilt: %s", o.spec.c_str(),
                    name ? name : "?", (unsigned) id, DescribeBallOrder(o.ctx, &BallOrderKey).c_str());
            it = o.keys.find(id);
        }
    }
    if (it != o.keys.end())
        return it->second;
    return (uint32_t) id;   // an unnamed / unpooled ball keeps its CK-id place
}

// The balls that can take part (pooled, named, player balls) in the order a key gives them, for the log.
std::string DescribeBallOrder(CKContext *ctx, unsigned int (*key)(void *)) {
    std::vector<std::pair<uint32_t, std::string>> rows;
    for (CK3dEntity *e : AllEntities(ctx)) {
        const char *name = e->GetName();
        if (!name)
            continue;
        bool show = IsPooledBall(name) || !std::strcmp(name, "Ball_Wood") || !std::strcmp(name, "Ball_Stone") ||
                    !std::strcmp(name, "Ball_Paper");
        for (const auto &r : g_BallOrder.ranks)
            show = show || r.first == name;
        if (show)
            rows.emplace_back(key ? key(e) : (uint32_t) e->GetID(),
                              std::string(name) + "(" + std::to_string((unsigned) e->GetID()) + ")");
    }
    std::sort(rows.begin(), rows.end());
    std::string out;
    for (const auto &r : rows)
        out += (out.empty() ? "" : " < ") + r.second;
    return out;
}

// ------------------------------------------------------------------------------------------- agent
struct Frame {
    float delta;
    uint32_t keys;
};
static_assert(sizeof(Frame) == 8, "legacy .tas frame");

struct Ev {
    uint32_t frame;
    uint16_t code, param;
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
    std::vector<float> balls;   // x y z r
};

struct HsSnap {
    uint8_t valid = 0;
    uint8_t types[6] = {};
    uint8_t bonusType = 0;
    uint32_t energy[6] = {};
    uint32_t levelBonus = 0;
    int32_t bmlHs = 0;
    int32_t hs = 0;
};

struct TracRec {
    uint32_t frame;
    float m[12];
};
static_assert(sizeof(TracRec) == 52, "TRAC item");

struct ArrSnap {
    CK_ID id = 0;
    int rows = -1, cols = -1;
    std::vector<std::string> cells;
};
struct WatchLast {
    uint8_t vis;
    float pos[3];
};
struct FollowerSlot {
    uint32_t watch;
    uint16_t source;
    uint8_t k;
};

enum class Phase { Idle, Running, Tail, FinalPending, Ended };
enum class StopReason { RecordEnd, LevelFinish, GameOver, ResetLevel, ExitLevel };

class Agent;
Agent *g_Agent = nullptr;

class Agent {
public:
    bool Configure(const ballrain_agent_args &args);
    int Run();
    void ApplyBallOrder();

    // hooks
    void Event(uint16_t code, uint16_t param = 0);
    void OnPhysicalize(CKBehavior *beh, bool physicalize);
    void OnHook(uint16_t code);

private:
    bool Fail(uint32_t code, const char *fmt, ...);
    bool Boot();
    bool InstallHooks();
    void ResetClock();
    void SetKeys(uint32_t keys);
    bool TickFrame(float delta, uint32_t keys, std::string &err);
    void AfterTick();
    const Frame *GetFrame(size_t index) const;
    bool WaitAction();
    void Stop(StopReason reason);
    void Finish(uint32_t endReason);
    void Publish(bool final);
    void SendState();
    void BuildState(bool minimal);
    void SnapHs(HsSnap &h);
    uint16_t CountFollowers();
    void LevelTables();
    void WriteLevelMessage();
    void ExportPhysicalized();
    void WritePhysSections(Wire &w, StrTab &strs, size_t from, size_t to, std::vector<std::pair<CK_ID, CK_ID>> &done,
                           std::unordered_set<uint64_t> &ivpoDone);
    CK3dEntity *GetActiveBall() const;
    IVP_Real_Object *BallReal(CK3dEntity *ball) const;
    // the 2002 DLL (m_P2): BallrainIO Agent.cpp's raw readers
    void ResetClock2002();
    void WriteTime2002(Wire &w);
    void WriteIvp2002(Wire &w, CK3dEntity *ball);   // BALL, FRIC, MIND, BODY
    void WriteGlct2002(Wire &w);
    void WriteIvpo2002(Wire &w, std::unordered_set<uint64_t> &ivpoDone);
    u8 *FindBallReal2002(CK3dEntity *ball);
    void WalkCluster2002(u8 *anyReal);
    u8 *ClusterStart2002();
    void RebuildEntityMap();
    CK_ID CkidOf2002(const void *client) const;
    void EventsLine(const char *fmt, ...);
    void WriteFrameLog();
    void SetState(uint32_t s) { m_ModState = s; }

    CKContext *Ctx() const { return m_Engine->context(); }
    CKDataArray *Array(const char *name) const { return m_Engine->data_array(name); }

    // configuration
    std::string m_Root, m_Sock, m_LevelMsgPath, m_FrameLogPath, m_TracePath, m_EventsPath, m_PrefixPath;
    int m_Level = 0;
    ballrain_agent_args m_Recipe{};   // SimTool's --tas recipe (ballrain_agent.h): offset, reset, legacy, precision
    std::string m_BallOrderSpec;      // BALLRAIN_HL_BALL_ORDER
    bool m_PrefixEvents = false;
    unsigned m_CrtSeed = 1;
    uint32_t m_TimeoutMs = 60000, m_MaxFrames = 0, m_SrTail = 32;
    float m_BodyRadius = 60.0f;

    std::unique_ptr<bmmo::sim::headless_engine> m_Engine;
    CKIpionManager *m_Physics = nullptr;
    Channel m_Chan;
    // the 2002 DLL (BALLRAIN_PHYS2002_HOST builds): its manager, the ball's IVP object cache, the cluster walk, the
    // entity pointer -> CK id map, the contact blocks' IVP_Handle parameters
    bool m_P2 = false, m_P2Hooked = false;
    u8 *m_Mgr2 = nullptr;
    u8 *m_BallReal2 = nullptr;
    CK3dEntity *m_BallRealOf2 = nullptr;
    std::vector<u8 *> m_Reals;
    std::unordered_map<const void *, CK_ID> m_EntOf;
    std::vector<CK_ID> m_ContactHandles;

    // run state
    Phase m_Phase = Phase::Idle;
    uint32_t m_ModState = ST_INIT, m_EndReason = END_NONE, m_LastError = 0;
    uint64_t m_ModSeq = 0, m_PySeq = 0;
    size_t m_FrameIndex = 0;
    std::vector<Frame> m_FrameLog, m_Prefix, m_ActFrames;
    bool m_InPrefix = false, m_PrefixObs = false;
    size_t m_BlockStart = 0, m_BlockEnd = 0;
    uint32_t m_Repeat = 1;
    uint64_t m_StopMask = 0;
    uint32_t m_ObsMask = 0;
    bool m_WantTrac = false, m_TracOverflow = false;
    uint32_t m_Covered = 0;
    uint32_t m_FrameEvFrame = 0;
    uint64_t m_FrameEvMask = 0;
    bool m_ClockReset = false;
    bool m_GameOverPending = false;
    uint32_t m_EndAtNextTime = 0;
    float m_LastDelta = 1000.0f / 132.0f;
    uint32_t m_TailFrames = 0;
    int m_KeyCodes[6] = {CKKEY_UP, CKKEY_DOWN, CKKEY_LEFT, CKKEY_RIGHT, CKKEY_LSHIFT, CKKEY_SPACE};

    std::vector<Ev> m_Evs;
    std::vector<uint32_t> m_PhysNew;
    std::vector<TracRec> m_Trac;
    std::vector<u8> m_StateBuf;
    size_t m_StateLen = 0;
    uint32_t m_StateSections = 0;
    bool m_StateDropped = false, m_StateMinimal = false;

    double m_Sr = 0.0, m_SrAtFinish = 0.0;
    float m_Sr32 = 0.0f, m_Sr32AtFinish = 0.0f;
    bool m_SrRunning = false, m_LevelFinished = false, m_SrInactiveSeen = false;
    uint32_t m_BallOffs = 0, m_LastBallOff = 0, m_LastBallNav = 0;
    HsSnap m_HsAtFinish;
    uint16_t m_FollowersAtFinish = 0;
    uint64_t m_TraceHash = 1469598103934665603ull;   // FNV-1a 64 of every exported frame's ball CK matrix
    uint32_t m_TraceHashed = 0;

    CKParameter *m_ActiveBallParam = nullptr;
    CK3dEntity *m_LastBall = nullptr;
    bool m_LevelMsgPending = false, m_TablesReady = false;
    std::vector<PhysRec> m_Phys;
    size_t m_PhysExported = 0;
    std::vector<std::pair<CK_ID, CK_ID>> m_MeshesDone;
    std::unordered_set<uint64_t> m_IvpoDone;
    FILE *m_LevelInc = nullptr, *m_Events = nullptr, *m_Trace = nullptr;

    CK_ID m_CamIds[5] = {};
    std::vector<CK_ID> m_ContactBBs;
    std::vector<CK_ID> m_Watch, m_WatchGroup;
    std::vector<WatchLast> m_WatchLast;
    std::vector<FollowerSlot> m_FollowerSlots;
    std::vector<int8_t> m_SourceMainVis;
    std::vector<ArrSnap> m_Arrs;
    CK_ID m_EnergyId = 0, m_AllLevelId = 0;
    int m_HooksInstalled = 0;
};

// ------------------------------------------------------------------------------------------- BML+ hooks
// ScriptHelper (BML+ 0.3.12 src/ScriptHelper.cpp), same search order.
namespace sh {
bool FindBB(CKBehavior *script, const std::function<bool(CKBehavior *)> &callback, const char *name,
            bool hierarchically = false, int inputCnt = -1, int outputCnt = -1, int inputParamCnt = -1,
            int outputParamCnt = -1) {
    const int cnt = script->GetSubBehaviorCount();
    for (int i = 0; i < cnt; i++) {
        CKBehavior *beh = script->GetSubBehavior(i);
        if (hierarchically && beh->GetSubBehaviorCount() > 0)
            if (!FindBB(beh, callback, name, hierarchically, inputCnt, outputCnt, inputParamCnt, outputParamCnt))
                return false;
        const char *bn = beh->GetName();
        if ((!name || (bn && !strcmp(bn, name))) && (inputCnt < 0 || beh->GetInputCount() == inputCnt) &&
            (outputCnt < 0 || beh->GetOutputCount() == outputCnt) &&
            (inputParamCnt < 0 || beh->GetInputParameterCount() == inputParamCnt) &&
            (outputParamCnt < 0 || beh->GetOutputParameterCount() == outputParamCnt))
            if (!callback(beh))
                return false;
    }
    return true;
}
CKBehavior *FindFirstBB(CKBehavior *script, const char *name, bool hierarchically = false, int inputCnt = -1,
                        int outputCnt = -1, int inputParamCnt = -1, int outputParamCnt = -1) {
    if (!script)
        return nullptr;
    CKBehavior *res = nullptr;
    FindBB(script, [&res](CKBehavior *beh) { res = beh; return false; }, name, hierarchically, inputCnt, outputCnt,
           inputParamCnt, outputParamCnt);
    return res;
}
CKBehaviorLink *CreateLink(CKBehavior *script, CKBehaviorIO *in, CKBehaviorIO *out, int delay = 0) {
    auto *link = (CKBehaviorLink *) script->GetCKContext()->CreateObject(CKCID_BEHAVIORLINK);
    link->SetInitialActivationDelay(delay);
    link->ResetActivationDelay();
    link->SetInBehaviorIO(in);
    link->SetOutBehaviorIO(out);
    script->AddSubBehaviorLink(link);
    return link;
}
CKBehaviorLink *CreateLink(CKBehavior *script, CKBehavior *inBeh, CKBehavior *outBeh, int inPos = 0, int outPos = 0) {
    return CreateLink(script, inBeh->GetOutput(inPos), outBeh->GetInput(outPos));
}
void InsertBB(CKBehavior *script, CKBehaviorLink *link, CKBehavior *beh, int inPos = 0, int outPos = 0) {
    CreateLink(script, beh->GetOutput(outPos), link->GetOutBehaviorIO());
    link->SetOutBehaviorIO(beh->GetInput(inPos));
}
CKBehaviorLink *FindNextLink(CKBehavior *script, CKBehavior *beh, const char *name = nullptr, int inPos = -1,
                             int outPos = -1) {
    if (!script || !beh)
        return nullptr;
    const int linkCnt = script->GetSubBehaviorLinkCount();
    for (int i = 0; i < linkCnt; i++) {
        CKBehaviorLink *link = script->GetSubBehaviorLink(i);
        CKBehaviorIO *in = link->GetInBehaviorIO();
        if (in->GetOwner() == beh && (inPos < 0 || beh->GetOutput(inPos) == in)) {
            CKBehaviorIO *out = link->GetOutBehaviorIO();
            CKBehavior *outBeh = out->GetOwner();
            const char *on = outBeh->GetName();
            if ((!name || (on && !strcmp(on, name))) && (outPos < 0 || outBeh->GetInput(outPos) == out))
                return link;
        }
    }
    return nullptr;
}
CKBehaviorLink *FindNextLink(CKBehavior *script, CKBehaviorIO *io) {
    if (!script || !io)
        return nullptr;
    const int linkCnt = script->GetSubBehaviorLinkCount();
    for (int i = 0; i < linkCnt; i++) {
        CKBehaviorLink *link = script->GetSubBehaviorLink(i);
        if (link->GetInBehaviorIO() == io)
            return link;
    }
    return nullptr;
}
CKBehavior *FindNextBB(CKBehavior *script, CKBehavior *beh, const char *name = nullptr, int inPos = -1, int outPos = -1) {
    CKBehaviorLink *link = FindNextLink(script, beh, name, inPos, outPos);
    return link ? link->GetOutBehaviorIO()->GetOwner() : nullptr;
}
CKBehavior *FindNextBB(CKBehavior *script, CKBehaviorIO *io) {
    CKBehaviorLink *link = FindNextLink(script, io);
    return link ? link->GetOutBehaviorIO()->GetOwner() : nullptr;
}
CKBehavior *FindEndOfChain(CKBehavior *script, CKBehavior *beh) {
    for (int guard = 0; beh && guard < 10000; ++guard) {
        CKBehavior *next = FindNextBB(script, beh);
        if (!next)
            break;
        beh = next;
    }
    return beh;
}
int MessageOf(CKBehavior *beh) {
    CKParameterIn *pin = beh->GetInputParameter(0);
    CKParameter *src = pin ? pin->GetDirectSource() : nullptr;
    int msg = -1;
    if (src)
        src->GetValue(&msg);
    return msg;
}
}  // namespace sh

// The hook block: BML+'s HookBlock (activate the outputs, call back) as a Nop instance with its own
// function; the event code of each instance is kept here.
std::unordered_map<CKBehavior *, uint16_t> g_HookCodes;
int HookFunction(const CKBehaviorContext &behcontext) {
    CKBehavior *beh = behcontext.Behavior;
    for (int i = 0; i < beh->GetInputCount(); ++i)
        beh->ActivateInput(i, FALSE);
    auto it = g_HookCodes.find(beh);
    if (it != g_HookCodes.end() && g_Agent)
        g_Agent->OnHook(it->second);
    for (int i = 0; i < beh->GetOutputCount(); ++i)
        beh->ActivateOutput(i);
    return CKBR_OK;
}
CKBehavior *Hook(CKBehavior *script, uint16_t code) {
    auto *beh = (CKBehavior *) script->GetCKContext()->CreateObject(CKCID_BEHAVIOR);
    beh->InitFromGuid(kNopGuid);
    beh->SetFunction(&HookFunction);
    script->AddSubBehavior(beh);
    g_HookCodes[beh] = code;
    return beh;
}

CKBEHAVIORFCT g_Physicalize = nullptr;
int PhysicalizeHook(const CKBehaviorContext &behcontext) {
    CKBehavior *beh = behcontext.Behavior;
#ifdef BALLRAIN_PHYS2002_HOST
    // Older Physicalize instances store [convex, ball, shift].  The 2002 BB
    // expects [convex, ball, concave, shift].  CK2 may already have renamed
    // the locals by prototype index; identify the old shape by type, not name.
    if (ballrain::phys2002::loaded() && beh->GetLocalParameterCount() == 3) {
        auto *shift = beh->GetLocalParameter(2);
        if (shift && shift->GetGUID() == CKPGUID_VECTOR) {
            // Keep the vector object (and any links to it). Allocate before
            // moving it so an allocation failure leaves the old layout intact.
            auto *count = beh->CreateLocalParameter("Concave Count", CKPGUID_INT);
            if (!count) return CKBR_PARAMETERERROR;
            const int zero = 0;
            count->SetValue(&zero);
            beh->RemoveLocalParameter(2);
            shift->SetName("Shift Mass Center");
            beh->AddLocalParameter(shift);
            Log("physicalize: migrated legacy shift vector on block %u (concave count 0)", beh->GetID());
        }
    }
#endif
    if (g_Agent) {
        g_Agent->ApplyBallOrder();
        g_Agent->OnPhysicalize(beh, beh->IsInputActive(0) != FALSE);
    }
    return g_Physicalize ? g_Physicalize(behcontext) : CKBR_OK;
}
int PatchPhysicalizeBlocks(CKBehavior *b) {
    if (!b)
        return 0;
    int n = 0;
    if (b->IsUsingFunction() && b->GetPrototypeGuid() == kPhysicalizeGuid && b->GetFunction() != &PhysicalizeHook) {
        b->SetFunction(&PhysicalizeHook);
        ++n;
    }
    for (int i = 0; i < b->GetSubBehaviorCount(); ++i)
        n += PatchPhysicalizeBlocks(b->GetSubBehavior(i));
    return n;
}

enum HookCode : uint16_t {
    H_NONE = 0,
    H_RESET = 200,
    H_EXIT,
};

// ------------------------------------------------------------------------------------------- Agent impl
bool Agent::Fail(uint32_t code, const char *fmt, ...) {
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    Log("error: %s", msg);
    m_LastError = code;
    m_EndReason = END_ERROR;
    SetState(ST_ERROR);
    SendState();
    return false;
}

bool Agent::Configure(const ballrain_agent_args &args) {
    m_Recipe = args;
    m_Root = args.root;
    m_Sock = args.sock;
    m_Level = args.level;
    if (m_Sock.empty())
        m_Sock = GetEnvS("BALLRAIN_SOCK");
    if (m_Root.empty())
        m_Root = GetEnvS("BALLRAIN_HL_ROOT");
    if (!m_Level)
        m_Level = std::atoi(GetEnvS("BALLRAIN_LEVEL").c_str());
    m_LevelMsgPath = GetEnvS("BALLRAIN_LEVELMSG");
    m_FrameLogPath = GetEnvS("BALLRAIN_FRAMELOG");
    m_TracePath = GetEnvS("BALLRAIN_TRACE");
    m_EventsPath = GetEnvS("BALLRAIN_EVENTS");
    m_PrefixPath = GetEnvS("BALLRAIN_PREFIX");
    m_PrefixEvents = GetEnvS("BALLRAIN_PREFIX_EVENTS") == "1";
    auto num = [](const char *name, auto &dst) {
        const std::string s = GetEnvS(name);
        if (!s.empty())
            dst = (std::remove_reference_t<decltype(dst)>) std::strtoll(s.c_str(), nullptr, 10);
    };
    num("BALLRAIN_TIMEOUT_MS", m_TimeoutMs);
    num("BALLRAIN_MAX_FRAMES", m_MaxFrames);
    num("BALLRAIN_SR_TAIL", m_SrTail);
    num("BALLRAIN_HL_CRT_SEED", m_CrtSeed);
    // The recipe is SimTool's flags now (--boot-ticks, --tas-offset, --tas-movement-check, --tas-seed, --tas-no-legacy,
    // --tas-reset-precision, --tas-orig-random); the variables of the hx1c builds are ignored.
    for (const char *old : {"BALLRAIN_HL_BOOT_TICKS", "BALLRAIN_HL_HIDDEN_TICKS", "BALLRAIN_HL_NMC", "BALLRAIN_HL_SEED",
                            "BALLRAIN_HL_LEGACY", "BALLRAIN_HL_KEEP_PRECISION", "BALLRAIN_HL_ORIG_RANDOM"})
        if (!GetEnvS(old).empty())
            Log("warning: %s is ignored (SimTool's --tas-* / --boot-ticks flags set the recipe)", old);
    m_BallOrderSpec = GetEnvS("BALLRAIN_HL_BALL_ORDER");
#ifdef BALLRAIN_PHYS2002_HOST
    // The 2002 physics_RT.dll: BALLRAIN_HL_PHYS2002 = its path, or 0 for the source physics; by default the game
    // root's own BuildingBlocks/physics_RT.dll (the engine refuses any other file than the 2002 one: no fallback).
    {
        std::string dll = GetEnvS("BALLRAIN_HL_PHYS2002");
        if (dll.empty())
            dll = GetEnvS("BALLRAIN_PHYS2002");
        if (dll == "0")
            dll.clear();
        else if (dll.empty() && !m_Root.empty())
            dll = m_Root + "/BuildingBlocks/physics_RT.dll";
        ballrain::phys2002::select(dll);
    }
#endif
    const std::string radius = GetEnvS("BALLRAIN_BODY_RADIUS");
    if (!radius.empty())
        m_BodyRadius = (float) std::atof(radius.c_str());
    if (!GetEnvS("BALLRAIN_POST_FINISH").empty() && GetEnvS("BALLRAIN_POST_FINISH") != "0")
        Log("BALLRAIN_POST_FINISH is not supported by the headless backend: ignored");
    if (!m_PrefixPath.empty()) {
        std::ifstream file(m_PrefixPath, std::ios::binary | std::ios::ate);
        if (!file) {
            Log("cannot open BALLRAIN_PREFIX %s", m_PrefixPath.c_str());
            return false;
        }
        const auto size = (size_t) file.tellg();
        file.seekg(0);
        m_Prefix.resize(size / sizeof(Frame));
        if (!m_Prefix.empty() && !file.read(reinterpret_cast<char *>(m_Prefix.data()),
                                            (std::streamsize) (m_Prefix.size() * sizeof(Frame)))) {
            Log("cannot read BALLRAIN_PREFIX %s", m_PrefixPath.c_str());
            return false;
        }
    }
    return true;
}

void Agent::SendState() {
    uint32_t st[4] = {m_ModState, m_EndReason, (uint32_t) m_FrameLog.size(), m_LastError};
    m_Chan.Send(BR_STATE, st, sizeof(st));
}

void Agent::EventsLine(const char *fmt, ...) {
    if (!m_Events)
        return;
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(m_Events, fmt, ap);
    va_end(ap);
    std::fputc('\n', m_Events);
}

CK3dEntity *Agent::GetActiveBall() const {
    return m_ActiveBallParam ? (CK3dEntity *) m_ActiveBallParam->GetValueObject() : nullptr;
}

IVP_Real_Object *Agent::BallReal(CK3dEntity *ball) const {
    if (!ball || !m_Physics)
        return nullptr;
    PhysicsObject *po = m_Physics->GetPhysicsObject(ball);
    return po ? po->m_RealObject : nullptr;
}

#ifdef BALLRAIN_PHYS2002_HOST
// ---- the 2002 DLL: BallrainIO Agent.cpp's readers (AgentFindBallReal, AgentWalkCluster, AgentRebuildEntityMap,
// AgentCkidOf, AgentClusterStart, and the raw parts of AgentBuildState / WritePhysSections / ResetPhysicsTime)
u8 *Agent::FindBallReal2002(CK3dEntity *ball) {
    // The cache is dropped by OnPhysicalize of the cached entity (it runs before the block replaces or destroys the
    // IVP object): a freed object can still hold client_data == ball for a few frames (SR09, after each death), so
    // client_data alone does not prove it is alive. The cache is what still finds the frozen old ball during a ball
    // transformation (it leaves the live vector).
    if (!ball || !m_Mgr2)
        return nullptr;
    if (m_BallReal2 && m_BallRealOf2 == ball) {
        void *client = nullptr;
        if (Rd(m_BallReal2, RO_CLIENT, client) && client == ball)
            return m_BallReal2;
    }
    m_BallReal2 = nullptr;
    m_BallRealOf2 = nullptr;
    uint16_t memsize = 0, n = 0;
    void **elems = nullptr;
    if (!Rd(m_Mgr2, MGR_MOVABLE_VEC, memsize) || !Rd(m_Mgr2, MGR_MOVABLE_VEC + 2, n) || !Rd(m_Mgr2, MGR_MOVABLE_VEC + 4, elems))
        return nullptr;
    if (!elems || n > memsize || n > 8000)
        return nullptr;
    for (int i = 0; i < n; ++i) {
        u8 *ro = nullptr;
        void *client = nullptr;
        if (!SafeRead(elems + i, &ro, sizeof(ro)) || !ro || !Rd(ro, RO_CLIENT, client))
            continue;
        if (client == ball) {
            m_BallReal2 = ro;
            m_BallRealOf2 = ball;
            return ro;
        }
    }
    return nullptr;
}

void Agent::WalkCluster2002(u8 *anyReal) {
    // All real objects of the ball's cluster, following next_in_cluster then prev_in_cluster (state-spec.md 2.5);
    // cluster nodes (object_type 1) are skipped.
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

u8 *Agent::ClusterStart2002() {
    if (!m_Mgr2)
        return nullptr;
    u8 *start = FindBallReal2002(GetActiveBall());
    if (!start) {   // no ball yet: any movable object is in the same (root) cluster
        uint16_t memsize = 0, n = 0;
        void **elems = nullptr;
        if (Rd(m_Mgr2, MGR_MOVABLE_VEC, memsize) && Rd(m_Mgr2, MGR_MOVABLE_VEC + 2, n) &&
            Rd(m_Mgr2, MGR_MOVABLE_VEC + 4, elems) && elems && n > 0 && n <= memsize && n <= 8000)
            SafeRead(elems, &start, sizeof(start));
    }
    return start;
}

void Agent::RebuildEntityMap() {
    m_EntOf.clear();
    const XObjectPointerArray &list = Ctx()->GetObjectListByType(CKCID_3DENTITY, TRUE);
    m_EntOf.reserve((size_t) list.Size() * 2);
    for (int k = 0; k < list.Size(); ++k)
        if (CKObject *o = list[k])
            m_EntOf.emplace(o, o->GetID());
}

CK_ID Agent::CkidOf2002(const void *client) const {
    if (!client)
        return 0;
    auto it = m_EntOf.find(client);
    if (it == m_EntOf.end())
        return 0;
    return Ctx()->GetObject(it->second) == client ? it->second : 0;
}

void Agent::ResetClock2002() {
    std::string error;
    if (!ballrain::phys2002::reset_clock(Ctx(), m_Engine->time_manager()->GetLastDeltaTime(), m_Recipe.movement_check,
                                         m_Recipe.seed, error))
        Log("clock reset: %s", error.c_str());
}

void Agent::WriteTime2002(Wire &w) {
    u8 *env = nullptr;
    double cur = 0, next = 0, last = 0;
    float c8 = 0, cc = 0, d0 = 0;
    if (m_Mgr2) {
        Rd(m_Mgr2, MGR_ENV, env);
        if (env) {
            Rd(env, ENV_CURRENT_TIME, cur);
            Rd(env, ENV_NEXT_PSI, next);
            Rd(env, ENV_LAST_PSI, last);
        }
        Rd(m_Mgr2, MGR_DELTA, c8);
        Rd(m_Mgr2, MGR_PHYS_DELTA, cc);
        Rd(m_Mgr2, MGR_TIME_FACTOR, d0);
    }
    w.Begin("TIME", 1);
    w.F64(cur);
    w.F64(next);
    w.F64(last);
    w.F32(c8);
    w.F32(cc);
    w.F32(d0);
    w.F32(m_Engine->time_manager()->GetLastDeltaTime());
    w.F64(m_Sr);
    w.F32((float) m_Sr);
    w.U8(m_SrRunning);
    w.U32(m_Covered);
    w.F32(m_Sr32);
    w.U32((uint32_t) m_FrameIndex);
    w.U32(m_TailFrames);
    w.End();
}

void Agent::WriteIvp2002(Wire &w, CK3dEntity *ball) {
    u8 *real = FindBallReal2002(ball);
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
    // FRCP (after FRIC, same items in the same order): partner ckid + now_friction_pressure per contact point
    FrcpItem frcp[32];
    uint32_t frcpN = 0;
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
            FrcpItem &fi = frcp[n];
            fi.partner = CkidOf2002(pclient);
            Rd(cp, CP_PRESSURE, fi.pressure);
            frcpN = n + 1;
            u8 *lt = nullptr;
            const bool ltOk = Rd(cp, CP_LT, lt) && lt && SafeRead(lt + LT_NORMAL, normal, sizeof(normal)) &&
                              SafeRead(lt + LT_POINT, point, sizeof(point));
            if (!ltOk) {
                std::memset(normal, 0, sizeof(normal));
                std::memset(point, 0, sizeof(point));
            }
            w.U32(CkidOf2002(pclient));
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
        WriteFrcp(w, frcp, frcpN);
    }

    // ---- MIND: exact synapses of the ball -> mindists
    if (!(m_ObsMask & M_MIND)) {
        w.Begin("MIND", 1);
        w.SetSectionFlags(SECF_APPROX);   // ledge: an address of this process
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
                w.U32(CkidOf2002(pclient));
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
            WalkCluster2002(real);
            const VxVector bp = ball->GetWorldMatrix()[3];
            const float r2 = m_BodyRadius * m_BodyRadius;
            for (u8 *ro : m_Reals) {
                if (ro == real || n >= 0xFFFF)
                    continue;
                u8 *bcore = nullptr;
                void *client = nullptr;
                if (!Rd(ro, RO_CORE, bcore) || !bcore || CoreUnmoveable(bcore) || !Rd(ro, RO_CLIENT, client))
                    continue;
                const CK_ID id = CkidOf2002(client);
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
}

void Agent::WriteGlct2002(Wire &w) {
    // the PhysicsContinuousContact block in use (Sound_Manager), its group counters behind the IVP_Handle
    CKContext *ctx = Ctx();
    CK_ID bbId = 0;
    uint16_t outMask = 0;
    int32_t groups[6] = {};
    for (size_t k = 0; k < m_ContactBBs.size() && k < m_ContactHandles.size() && !bbId; ++k) {
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

void Agent::WriteIvpo2002(Wire &w, std::unordered_set<uint64_t> &ivpoDone) {
    // IVPO: IVP objects of the physicalized entities found by the cluster walk (m_Reals), each pair once.
    CKContext *ctx = Ctx();
    w.Begin("IVPO", 1);
    const size_t at2 = w.Size();
    w.U32(0);
    uint32_t n2 = 0;
    std::unordered_map<const void *, CK_ID> want;
    for (size_t i = 0; i < m_Phys.size(); ++i)
        if (m_Phys[i].kind == 0)
            if (CKObject *o = ctx->GetObject(m_Phys[i].ent))
                want.emplace(o, m_Phys[i].ent);
    for (u8 *ro : m_Reals) {
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
    w.PatchU32(at2, n2);
    w.End();
}
#else
void Agent::ResetClock2002() {}
void Agent::WriteTime2002(Wire &) {}
void Agent::WriteIvp2002(Wire &, CK3dEntity *) {}
void Agent::WriteGlct2002(Wire &) {}
void Agent::WriteIvpo2002(Wire &, std::unordered_set<uint64_t> &) {}
u8 *Agent::FindBallReal2002(CK3dEntity *) { return nullptr; }
void Agent::WalkCluster2002(u8 *) {}
u8 *Agent::ClusterStart2002() { return nullptr; }
void Agent::RebuildEntityMap() {}
CK_ID Agent::CkidOf2002(const void *) const { return 0; }
#endif

void Agent::OnHook(uint16_t code) {
    if (code == H_NONE)
        return;
    switch (code) {
        case EV_PRELOAD:   // the run's PreLoadLevel is its own (frame 0)
            return;
        case H_RESET:
            Event(EV_PRERESET);
            Stop(StopReason::ResetLevel);
            return;
        case H_EXIT:
            Event(EV_PREEXIT);
            Stop(StopReason::ExitLevel);
            return;
        case EV_FINISH:
            Event(EV_FINISH);
            Stop(StopReason::LevelFinish);
            return;
        case EV_GAMEOVER:
            Event(EV_GAMEOVER);
            if (m_Phase == Phase::Running)
                m_GameOverPending = true;
            return;
        default:
            Event(code);
    }
}

void Agent::Event(uint16_t code, uint16_t param) {
    if (m_Phase == Phase::Idle || m_Phase == Phase::Ended)
        return;
    const auto frame = (uint32_t) m_FrameIndex;
    if (frame != m_FrameEvFrame) {
        m_FrameEvFrame = frame;
        m_FrameEvMask = 0;
    }
    if (code < 64)
        m_FrameEvMask |= 1ull << code;
    m_Evs.push_back(Ev{frame, code, param});
    if (m_Events && code < sizeof(kEventNames) / sizeof(kEventNames[0]))
        std::fprintf(m_Events, "%u %s\n", frame, kEventNames[code]);
    switch (code) {
        case EV_START:
            m_Sr = 0.0;
            m_Sr32 = 0.0f;
            m_SrRunning = false;
            m_LevelMsgPending = true;
            SetState(ST_RUNNING);
            break;
        case EV_COUNTER_ON:
        case EV_UNPAUSE:
            m_SrRunning = true;
            EventsLine("%u SR sr=%.17g f32sum=%.9g running=1", frame, m_Sr, m_Sr32);
            break;
        case EV_COUNTER_OFF:
        case EV_PAUSE:
            m_SrRunning = false;
            if (m_LevelFinished)
                m_SrInactiveSeen = true;
            EventsLine("%u SR sr=%.17g f32sum=%.9g running=0", frame, m_Sr, m_Sr32);
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
            SnapHs(m_HsAtFinish);
            m_FollowersAtFinish = CountFollowers();
            const HsSnap &h = m_HsAtFinish;
            EventsLine("%u HS points=%d lifes=%d lifebonus=%d levelbonus=%d hs=%d bml=%d followers=%u", frame,
                       (int) h.energy[0], (int) h.energy[1], (int) h.energy[5], (int) h.levelBonus, h.hs, h.bmlHs,
                       (unsigned) m_FollowersAtFinish);
            break;
        }
        default:
            break;
    }
}

void Agent::OnPhysicalize(CKBehavior *beh, bool physicalize) {
    if (m_Phase != Phase::Running && m_Phase != Phase::Tail)
        return;
    auto *target = (CK3dEntity *) beh->GetTarget();
    if (!target)
        return;
    if (target == m_BallRealOf2) {   // 2002 DLL: its IVP object is about to be replaced / destroyed
        m_BallReal2 = nullptr;
        m_BallRealOf2 = nullptr;
    }
    PhysRec r;
    r.seq = (uint32_t) m_Phys.size();
    r.frame = (uint32_t) m_FrameIndex;
    r.ent = target->GetID();
    if (!physicalize) {
        r.kind = 1;
        m_Phys.push_back(std::move(r));
        Event(EV_UNPHYS, (uint16_t) m_Phys.back().seq);
        return;
    }
    // BML+ 0.3.12 PhysicsHook.cpp: the block's parameters, read the same way.
    CKBOOL fixed = FALSE, startFrozen = FALSE, enableCollision = TRUE, autoCalc = TRUE;
    float friction = 0.4f, elasticity = 0.5f, mass = 1.0f, linDamp = 0.1f, rotDamp = 0.1f;
    beh->GetInputParameterValue(0, &fixed);
    beh->GetInputParameterValue(1, &friction);
    beh->GetInputParameterValue(2, &elasticity);
    beh->GetInputParameterValue(3, &mass);
    auto collisionGroup = (CKSTRING) beh->GetInputParameterReadDataPtr(4);
    beh->GetInputParameterValue(5, &startFrozen);
    beh->GetInputParameterValue(6, &enableCollision);
    beh->GetInputParameterValue(7, &autoCalc);
    beh->GetInputParameterValue(8, &linDamp);
    beh->GetInputParameterValue(9, &rotDamp);
    auto collisionSurface = (CKSTRING) beh->GetInputParameterReadDataPtr(10);
    int convexCount = 1, ballCount = 0, concaveCount = 0;
    beh->GetLocalParameterValue(0, &convexCount);
    beh->GetLocalParameterValue(1, &ballCount);
    beh->GetLocalParameterValue(2, &concaveCount);
    int pos = 11;
    for (int i = 0; i < convexCount; ++i)
        if (auto *m = (CKMesh *) beh->GetInputParameterObject(pos + i))
            r.convex.push_back(m->GetID());
    pos += convexCount;
    for (int j = 0; j < ballCount; ++j) {
        VxVector c;
        float rad = 0;
        beh->GetInputParameterValue(pos + 2 * j, &c);
        beh->GetInputParameterValue(pos + 2 * j + 1, &rad);
        r.balls.insert(r.balls.end(), {c.x, c.y, c.z, rad});
    }
    pos += ballCount * 2;
    for (int k = 0; k < concaveCount; ++k)
        if (auto *m = (CKMesh *) beh->GetInputParameterObject(pos + k))
            r.concave.push_back(m->GetID());
    VxVector shift(0, 0, 0);
    beh->GetLocalParameterValue(3, &shift);
    r.kind = 0;
    r.fixed = fixed ? 1 : 0;
    r.frozen = startFrozen ? 1 : 0;
    r.coll = enableCollision ? 1 : 0;
    r.calcmc = autoCalc ? 1 : 0;
    r.friction = friction;
    r.elasticity = elasticity;
    r.mass = mass;
    r.linDamp = linDamp;
    r.rotDamp = rotDamp;
    r.massCenter[0] = shift.x;
    r.massCenter[1] = shift.y;
    r.massCenter[2] = shift.z;
    r.collGroup = collisionGroup ? collisionGroup : "";
    r.collSurface = collisionSurface ? collisionSurface : "";
    m_Phys.push_back(std::move(r));
    Event(EV_PHYS, (uint16_t) m_Phys.back().seq);
}

bool Agent::InstallHooks() {
    CKContext *ctx = Ctx();
    auto script = [&](const char *name) -> CKBehavior * {
        CKBehavior *found = nullptr;
        const int n = ctx->GetObjectsCountByClassID(CKCID_BEHAVIOR);
        CK_ID *ids = ctx->GetObjectsListByClassID(CKCID_BEHAVIOR);
        for (int i = 0; i < n && !found; ++i) {
            auto *b = (CKBehavior *) ctx->GetObject(ids[i]);
            if (b && !b->GetParent() && b->GetName() && !strcmp(b->GetName(), name))
                found = b;
        }
        return found;
    };
    int ok = 0;
    using namespace sh;
    // ---- Event_handler (BML+ EventHookRegistrar::RegisterBaseEventHandler; End Level's post hook and the
    // Highscore edit left out: they run after the run's end)
    if (CKBehavior *s = script("Event_handler")) {
        CKBehavior *som = FindFirstBB(s, "Switch On Message", false, 2, 11, 11, 0);
        if (som) {
            InsertBB(s, FindNextLink(s, FindNextBB(s, som, nullptr, 0, 0)), Hook(s, H_NONE));   // PreStartMenu
            CreateLink(s, FindEndOfChain(s, FindNextBB(s, som, nullptr, 0, 0)), Hook(s, H_NONE));   // PostStartMenu
            InsertBB(s, FindNextLink(s, FindNextBB(s, som, nullptr, 1, 0)), Hook(s, H_NONE));   // ExitGame
            CKBehaviorLink *link = FindNextLink(s, FindNextBB(s, FindNextBB(s, FindNextBB(s, som, nullptr, 2, 0))));
            if (link)
                InsertBB(s, link, Hook(s, EV_PRELOAD));
            CreateLink(s, FindEndOfChain(s, FindNextBB(s, som, nullptr, 2, 0)), Hook(s, EV_POSTLOAD));
            CreateLink(s, FindEndOfChain(s, FindNextBB(s, som, nullptr, 3, 0)), Hook(s, EV_START));
            if (CKBehavior *rl = FindFirstBB(s, "reset Level")) {
                link = FindNextLink(rl, FindNextBB(rl, FindNextBB(rl, rl->GetInput(0))));
                if (link)
                    InsertBB(s, link, Hook(s, H_RESET));
            }
            CreateLink(s, FindEndOfChain(s, FindNextBB(s, som, nullptr, 4, 0)), Hook(s, H_NONE));   // PostResetLevel
            CreateLink(s, FindEndOfChain(s, FindNextBB(s, som, nullptr, 5, 0)), Hook(s, EV_PAUSE));
            CreateLink(s, FindEndOfChain(s, FindNextBB(s, som, nullptr, 6, 0)), Hook(s, EV_UNPAUSE));
            CKBehavior *bs = FindNextBB(s, FindFirstBB(s, "DeleteCollisionSurfaces"));
            link = FindNextLink(s, FindNextBB(s, FindNextBB(s, FindNextBB(s, FindNextBB(s, FindNextBB(s, som, nullptr, 7, 0))))));
            if (link)
                InsertBB(s, link, Hook(s, H_EXIT));
            if (bs && (link = FindNextLink(s, FindNextBB(s, bs, nullptr, 0, 0))))
                InsertBB(s, link, Hook(s, H_NONE));   // PostExitLevel
            link = FindNextLink(s, FindNextBB(s, FindNextBB(s, FindNextBB(s, FindNextBB(s, FindNextBB(s, som, nullptr, 8, 0))))));
            if (link)
                InsertBB(s, link, Hook(s, H_NONE));   // PreNextLevel
            if (bs && (link = FindNextLink(s, FindNextBB(s, bs, nullptr, 1, 0))))
                InsertBB(s, link, Hook(s, H_NONE));   // PostNextLevel
            CreateLink(s, FindEndOfChain(s, FindNextBB(s, som, nullptr, 9, 0)), Hook(s, EV_DEAD));
            if ((link = FindNextLink(s, som, nullptr, 10, 0)))
                InsertBB(s, link, Hook(s, EV_PREENDLEVEL));
            ++ok;
        } else {
            Log("hooks: Event_handler has no Switch On Message");
        }
    }
    // ---- Gameplay_Ingame: CamNav / BallNav on/off; the ActiveBall parameter (BallrainIO OnLoadScript)
    if (CKBehavior *s = script("Gameplay_Ingame")) {
        CKMessageManager *mm = ctx->GetMessageManager();
        const int camon = mm->AddMessageType((CKSTRING) "CamNav activate"),
                  camoff = mm->AddMessageType((CKSTRING) "CamNav deactivate"),
                  ballon = mm->AddMessageType((CKSTRING) "BallNav activate"),
                  balloff = mm->AddMessageType((CKSTRING) "BallNav deactivate");
        CKBehavior *camonoff = FindFirstBB(s, "CamNav On/Off");
        CKBehavior *ballonoff = FindFirstBB(s, "BallNav On/Off");
        CKBehavior *con = nullptr, *coff = nullptr, *bon = nullptr, *boff = nullptr;
        if (camonoff)
            FindBB(camonoff, [&](CKBehavior *beh) {
                const int msg = MessageOf(beh);
                if (msg == camon) con = beh;
                if (msg == camoff) coff = beh;
                return true;
            }, "Wait Message");
        if (ballonoff)
            FindBB(ballonoff, [&](CKBehavior *beh) {
                const int msg = MessageOf(beh);
                if (msg == ballon) bon = beh;
                if (msg == balloff) boff = beh;
                return true;
            }, "Wait Message");
        if (con && coff && bon && boff) {
            CreateLink(camonoff, con, Hook(camonoff, EV_CAMNAV_ON), 0, 0);
            CreateLink(camonoff, coff, Hook(camonoff, EV_CAMNAV_OFF), 0, 0);
            CreateLink(ballonoff, bon, Hook(ballonoff, EV_BALLNAV_ON), 0, 0);
            CreateLink(ballonoff, boff, Hook(ballonoff, EV_BALLNAV_OFF), 0, 0);
            ++ok;
        } else {
            Log("hooks: Gameplay_Ingame nav blocks not found");
        }
        for (int i = 0; i < s->GetLocalParameterCount(); ++i) {
            CKParameter *p = s->GetLocalParameter(i);
            if (p && p->GetName() && !strcmp(p->GetName(), "ActiveBall")) {
                m_ActiveBallParam = p;
                break;
            }
        }
    }
    // ---- Gameplay_Energy: counter, life, ball off, extra point
    if (CKBehavior *s = script("Gameplay_Energy")) {
        CKBehavior *som = FindFirstBB(s, "Switch On Message");
        CKMessageManager *mm = ctx->GetMessageManager();
        const int lifeup = mm->AddMessageType((CKSTRING) "Life_Up"), balloff = mm->AddMessageType((CKSTRING) "Ball Off"),
                  sublife = mm->AddMessageType((CKSTRING) "Sub Life"), extrapoint = mm->AddMessageType((CKSTRING) "Extrapoint");
        CKBehavior *lu = nullptr, *bo = nullptr, *sl = nullptr, *ep = nullptr;
        FindBB(s, [&](CKBehavior *beh) {
            const int msg = MessageOf(beh);
            if (msg == lifeup) lu = beh;
            if (msg == balloff) bo = beh;
            if (msg == sublife) sl = beh;
            if (msg == extrapoint) ep = beh;
            return true;
        }, "Wait Message");
        CKBehaviorLink *l3 = som ? FindNextLink(s, som, nullptr, 3) : nullptr;
        CKBehaviorLink *l1 = som ? FindNextLink(s, som, nullptr, 1) : nullptr;
        if (l3 && l1 && lu && bo && sl && ep) {
            InsertBB(s, l3, Hook(s, EV_COUNTER_ON));
            InsertBB(s, l1, Hook(s, EV_COUNTER_OFF));
            CKBehavior *luhook = Hook(s, EV_PRELIFEUP);
            InsertBB(s, FindNextLink(s, lu, "add Life"), luhook);
            CreateLink(s, FindEndOfChain(s, luhook), Hook(s, EV_POSTLIFEUP));
            InsertBB(s, FindNextLink(s, bo, "Delayer"), Hook(s, EV_BALLOFF));
            CKBehavior *slhook = Hook(s, EV_PRESUBLIFE);
            InsertBB(s, FindNextLink(s, sl, "sub Life"), slhook);
            CreateLink(s, FindEndOfChain(s, slhook), Hook(s, EV_POSTSUBLIFE));
            InsertBB(s, FindNextLink(s, ep, "Show"), Hook(s, EV_EXTRAPOINT));
            ++ok;
        } else {
            Log("hooks: Gameplay_Energy blocks not found");
        }
    }
    // ---- Gameplay_Events: checkpoint, game over, level finish
    if (CKBehavior *s = script("Gameplay_Events")) {
        CKMessageManager *mm = ctx->GetMessageManager();
        const int checkpoint = mm->AddMessageType((CKSTRING) "Checkpoint reached"),
                  gameover = mm->AddMessageType((CKSTRING) "Game Over"),
                  levelfinish = mm->AddMessageType((CKSTRING) "Level_Finish");
        CKBehavior *cp = nullptr, *go = nullptr, *lf = nullptr;
        FindBB(s, [&](CKBehavior *beh) {
            const int msg = MessageOf(beh);
            if (msg == checkpoint) cp = beh;
            if (msg == gameover) go = beh;
            if (msg == levelfinish) lf = beh;
            return true;
        }, "Wait Message");
        if (cp && go && lf) {
            CKBehavior *hook = Hook(s, EV_PRECHECK);
            InsertBB(s, FindNextLink(s, cp, "set Resetpoint"), hook);
            CreateLink(s, FindEndOfChain(s, hook), Hook(s, EV_POSTCHECK));
            InsertBB(s, FindNextLink(s, go, "Send Message"), Hook(s, EV_GAMEOVER));
            InsertBB(s, FindNextLink(s, lf, "Send Message"), Hook(s, EV_FINISH));
            ++ok;
        } else {
            Log("hooks: Gameplay_Events blocks not found");
        }
    }
    // ---- Physicalize (BML+ HookPhysicalize: the prototype's function, plus the blocks already made)
    if (CKBehaviorPrototype *proto = CKGetPrototypeFromGuid(kPhysicalizeGuid)) {
        if (!g_Physicalize)
            g_Physicalize = proto->GetFunction();
        proto->SetFunction(&PhysicalizeHook);
        int patched = 0;
        const int n = ctx->GetObjectsCountByClassID(CKCID_BEHAVIOR);
        CK_ID *ids = ctx->GetObjectsListByClassID(CKCID_BEHAVIOR);
        for (int i = 0; i < n; ++i) {
            auto *b = (CKBehavior *) ctx->GetObject(ids[i]);
            if (b && !b->GetParent())
                patched += PatchPhysicalizeBlocks(b);
        }
        Log("hooks: Physicalize prototype wrapped, %d existing blocks", patched);
        ++ok;
    }
    m_HooksInstalled = ok;
    Log("hooks: %d of 5 groups installed, ActiveBall %s", ok, m_ActiveBallParam ? "found" : "missing");
    return ok == 5 && m_ActiveBallParam;
}

bool Agent::Boot() {
    bmmo::sim::engine_options options;
    options.game_root = m_Root;
    options.log = [](const std::string &text) {
        if (std::getenv("BALLRAIN_HL_VERBOSE"))
            std::fprintf(stderr, "%s\n", text.c_str());
    };
    std::string error;
    m_Engine = bmmo::sim::headless_engine::create(options, error);
    if (!m_Engine)
        return Fail(ERR_ENV, "engine: %s", error.c_str());
    // CKStartUp seeds the C runtime's rand() with time(): the Random blocks the engine does not reroute draw from
    // it, which made runs of the same frames differ (docs/headless.md). A fixed seed makes every run repeatable.
    std::srand(m_CrtSeed);
    if (!m_Engine->load_composition(error))
        return Fail(ERR_ENV, "composition: %s", error.c_str());
    // SimTool's --tas recipe (run_tas): the Random blocks, and no x87 precision reset at the level load (the 2002
    // process runs at 24 bits only until its first hull, built during the boot, then at 64 bits for good).
#ifdef BALLRAIN_PHYS2002_HOST
    // the 2002 DLL: SimTool --tas's hl-win recipe (no BMMO script patch: its blocks would reach the source manager)
    m_P2 = ballrain::phys2002::loaded();
    if (ballrain::phys2002::requested() && !m_P2)
        return Fail(ERR_DLL, "the 2002 physics_RT.dll was requested but is not loaded");
    if (m_P2) {
        m_Mgr2 = ballrain::phys2002::manager(Ctx());
        if (!m_Mgr2)
            return Fail(ERR_DLL, "2002 physics_RT.dll: no physics manager");
        m_P2Hooked = ballrain::phys2002::ball_hook_installed();
        if (!m_P2Hooked) {
            // BALLRAIN_HL_PHYS2002_HOOK=0 (diagnostic): the DLL's own compare, i.e. this process's heap-address order
            if (g_BallOrder.mode != BallOrder::CKID)
                return Fail(ERR_ENV, "BALLRAIN_HL_BALL_ORDER=%s needs the ball-pair hook (BALLRAIN_HL_PHYS2002_HOOK=0)",
                            g_BallOrder.spec.c_str());
            Log("warning: ball-pair compare not hooked: pairs follow this process's heap addresses, not the CK id");
        }
        ballrain::phys2002::set_ball_key(g_BallOrder.mode == BallOrder::CKID ? &ballrain::phys2002::ball_key_ckid
                                                                             : &BallOrderKey);
    }
#endif
    m_Engine->set_random_patch(!m_Recipe.orig_random && !m_P2);
    m_Engine->set_reset_precision_on_load(m_Recipe.reset_precision);
    m_Physics = m_Engine->physics();
    g_BallOrder.ctx = Ctx();
    m_Engine->set_frame_ms(0.0f);
    for (int i = 0; i < m_Recipe.boot_ticks; ++i) {
        ApplyBallOrder();
        if (!m_Engine->tick(error))
            return Fail(ERR_ENV, "boot tick %d: %s", i, error.c_str());
    }
    if (!InstallHooks())
        return Fail(ERR_DLL, "BML+ event hooks could not be installed");
    if (m_Recipe.legacy && m_Recipe.legacy_edit)
        m_Recipe.legacy_edit(*m_Engine);
    return true;
}

// BALLRAIN_HL_BALL_ORDER other than ckid: our key on the physics environment (CKIpionManager::Reset re-creates it with
// the product's key, so before every tick and every Physicalize block). ckid leaves the product's key alone.
void Agent::ApplyBallOrder() {
    if (g_BallOrder.mode == BallOrder::CKID || !m_Physics)
        return;
    if (IVP_Environment *env = m_Physics->GetEnvironment())
        if (env->bmmo_ball_order_key != &BallOrderKey)
            env->bmmo_ball_order_key = &BallOrderKey;
}

void Agent::ResetClock() {
    // TASSupport OnStart's ResetPhysicsTime (a 1-tick timer after PreLoadLevel: before frame 1), plus the
    // IVP state Player.exe has at that point.
    if (m_P2) {
        ResetClock2002();
        return;
    }
    bmmo_physics_clock_state clock{};
    std::string error;
    if (!m_Physics || !bmmo::physics::get_clock_state(m_Physics, clock, error)) {
        Log("clock reset: %s", error.c_str());
        return;
    }
    clock.base_time = 0;
    clock.current_time = 0;
    clock.time_of_last_psi = 0;
    clock.time_of_next_psi = 1.0 / 66;
    // TASSupport resets the clock only: the movement-check counter and the ivp_rand seed are what the process left,
    // in Player.exe the same on every level (SimTool --tas-movement-check / --tas-seed)
    if (m_Recipe.movement_check >= 0)
        clock.next_movement_check = m_Recipe.movement_check;
    ivp_srand((int) m_Recipe.seed);
    if (!bmmo::physics::set_clock_state(m_Physics, clock, error))
        Log("clock reset: %s", error.c_str());
    m_Physics->m_DeltaTime = m_Engine->time_manager()->GetLastDeltaTime();
}

void Agent::SetKeys(uint32_t keys) {
    unsigned char kb[256] = {};
    for (int k = 0; k < 6; ++k)
        kb[m_KeyCodes[k] & 0xFF] = (keys >> k) & 1u;
    kb[CKKEY_Q] = (keys >> 6) & 1u;
    kb[CKKEY_ESCAPE] = (keys >> 7) & 1u;
    kb[CKKEY_RETURN] = (keys >> 8) & 1u;
    m_Engine->set_keyboard_state(kb);
}

const Frame *Agent::GetFrame(size_t index) const {
    if (index >= m_BlockEnd)
        return nullptr;
    if (m_InPrefix)
        return index < m_Prefix.size() ? &m_Prefix[index] : nullptr;
    if (index < m_BlockStart)
        return nullptr;
    const size_t k = (index - m_BlockStart) / m_Repeat;
    return k < m_ActFrames.size() ? &m_ActFrames[k] : nullptr;
}

bool Agent::WaitAction() {
    std::vector<u8> msg;
    for (;;) {
        const uint32_t resume = m_ModState == ST_WAITING ? ST_RUNNING : m_ModState;
        SetState(ST_WAITING);
        uint32_t type = 0;
        int r = 0;
        do {
            r = m_Chan.Recv(type, msg, m_TimeoutMs);
        } while (r > 0 && type != BR_ACT);
        SetState(resume);
        if (r <= 0) {
            Log("%s while waiting for the action of frame %u", r == 0 ? "timeout" : "client gone", (unsigned) m_FrameIndex);
            m_EndReason = END_TIMEOUT;
            Stop(StopReason::RecordEnd);
            return false;
        }
        uint32_t magic = 0, pySeq = 0, nFrames = 0, repeat = 0, maskLo = 0, maskHi = 0, obsMask = 0;
        uint16_t version = 0, flags = 0;
        const char *bad = nullptr;
        if (msg.size() < ACT_HEADER) {
            bad = "short ACT1 message";
        } else {
            const u8 *a = msg.data();
            std::memcpy(&magic, a + 0x00, 4);
            std::memcpy(&version, a + 0x04, 2);
            std::memcpy(&flags, a + 0x06, 2);
            std::memcpy(&pySeq, a + 0x08, 4);
            std::memcpy(&nFrames, a + 0x0C, 4);
            std::memcpy(&repeat, a + 0x10, 4);
            std::memcpy(&maskLo, a + 0x14, 4);
            std::memcpy(&maskHi, a + 0x18, 4);
            std::memcpy(&obsMask, a + 0x1C, 4);
            if (pySeq != (uint32_t) (m_PySeq + 1))
                bad = "py_seq is not the previous one + 1";
            else if (magic != ACT_MAGIC || version != 1)
                bad = "bad ACT1 magic / version";
            else if (nFrames > ACT_MAX_FRAMES || (size_t) nFrames * sizeof(Frame) + ACT_HEADER > msg.size())
                bad = "n_frames exceeds the message";
        }
        if (bad) {
            Log("action refused at frame %u: %s", (unsigned) m_FrameIndex, bad);
            m_LastError = ERR_ACT;
            m_EndReason = END_ERROR;
            Stop(StopReason::RecordEnd);
            return false;
        }
        m_PySeq = pySeq;
        m_ObsMask = obsMask;
        m_WantTrac = (flags & ACT_TRAC) != 0;
        if (flags & ACT_LEVELMSG)
            WriteLevelMessage();
        if (flags & ACT_STOP) {
            m_EndReason = END_STOP;
            Stop(StopReason::RecordEnd);
            return false;
        }
        if (nFrames == 0) {
            if (m_InPrefix && m_PrefixEvents) {
                m_BlockEnd = std::min(m_FrameIndex + 1, m_Prefix.size());
                return true;
            }
            Publish(false);
            continue;
        }
        m_ActFrames.resize(nFrames);
        std::memcpy(m_ActFrames.data(), msg.data() + ACT_HEADER, nFrames * sizeof(Frame));
        m_Repeat = repeat ? repeat : 1;
        m_StopMask = (uint64_t) maskHi << 32 | maskLo;
        m_InPrefix = false;
        m_BlockStart = m_FrameIndex;
        m_BlockEnd = m_BlockStart + (size_t) nFrames * m_Repeat;
        return true;
    }
}

void Agent::WriteFrameLog() {
    if (m_FrameLogPath.empty())
        return;
    FILE *f = std::fopen(m_FrameLogPath.c_str(), "wb");
    if (!f) {
        Log("cannot write %s", m_FrameLogPath.c_str());
        return;
    }
    if (!m_FrameLog.empty())
        std::fwrite(m_FrameLog.data(), sizeof(Frame), m_FrameLog.size(), f);
    std::fclose(f);
}

void Agent::Stop(StopReason reason) {
    if (m_Phase != Phase::Running)
        return;
    WriteFrameLog();
    if (m_Trace) {
        std::fclose(m_Trace);
        m_Trace = nullptr;
    }
    switch (reason) {
        case StopReason::LevelFinish:
            m_Phase = Phase::Tail;
            m_TailFrames = 0;
            break;
        case StopReason::GameOver:
            Finish(END_GAMEOVER);
            break;
        case StopReason::RecordEnd:
            Finish(m_EndReason ? m_EndReason : END_STOP);
            break;
        case StopReason::ResetLevel:
        case StopReason::ExitLevel:
            m_Phase = Phase::FinalPending;
            break;
    }
}

void Agent::Finish(uint32_t endReason) {
    if (m_Phase == Phase::Ended)
        return;
    // the pooled balls are created during the run (L1: when their sector loads): the order over all of them
    Log("ball order %s at the end: %s", g_BallOrder.spec.c_str(),
        DescribeBallOrder(Ctx(), g_BallOrder.mode == BallOrder::CKID ? nullptr : &BallOrderKey).c_str());
    if (m_P2)   // what the unhooked 2002 compare would use: this process's CK3dEntity addresses
        Log("ball heap-address order at the end: %s",
            DescribeBallOrder(Ctx(), [](void *e) { return (unsigned int) (uintptr_t) e; }).c_str());
    if (m_Phase == Phase::Running || m_Phase == Phase::Tail || m_Phase == Phase::FinalPending)
        WriteFrameLog();
    Event(EV_RUNEND, (uint16_t) endReason);
    EventsLine("%u TraceHash %016llx frames=%u", (unsigned) m_FrameIndex, (unsigned long long) m_TraceHash, m_TraceHashed);
    EventsLine("%u SRFinal sr=%.17g sr_ms=%lld f32sum=%.9g bml=%.9g running=%d tail=%u inactive=%d finished=%d "
               "sr_at_finish=%.17g end_reason=%u post=0",
               (unsigned) m_FrameIndex, m_Sr, (long long) m_Sr, m_Sr32, (float) m_Sr, (int) m_SrRunning, m_TailFrames,
               (int) m_SrInactiveSeen, (int) m_LevelFinished, m_SrAtFinish, endReason);
    if (m_Events) {
        std::fclose(m_Events);
        m_Events = nullptr;
    }
    if (m_LevelInc) {
        std::fclose(m_LevelInc);
        m_LevelInc = nullptr;
    }
    m_EndReason = endReason;
    SetState(endReason == END_ERROR ? ST_ERROR : ST_ENDED);
    Publish(true);
    m_Phase = Phase::Ended;
    Log("run ended (reason %u) after %u exported frames, SR %.3f ms", endReason, (unsigned) m_FrameLog.size(), m_Sr);
}

void Agent::Publish(bool final) {
    std::vector<u8> buf;
    buf.reserve(m_StateLen + 4096 + m_Trac.size() * sizeof(TracRec) + m_Evs.size() * 8);
    Wire w(&buf);
    uint16_t flags = OBS_HEADLESS;
    if (final)
        flags |= OBS_FINAL;
    if (m_PrefixObs)
        flags |= OBS_PREFIX;
    BeginMessage(w, "OBS1", 0, (uint32_t) m_FrameLog.size());
    w.Raw(m_StateBuf.data(), m_StateLen);
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
        w.SetSectionFlags(SECF_APPROX);   // bml_hs computed, not read from BML+
        w.F64(m_Sr);
        w.F32(m_Sr32);
        w.F32((float) m_Sr);   // BML+ GetSRScore(): the float of its double sum
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
        w.U16(m_FollowersAtFinish);
        w.U16(CountFollowers());
        w.U32(0);   // no post-finish window
        sections += w.End();
    }
    if (final) {
        w.Begin("HASH", 1);   // headless extension: the run fingerprint
        w.Put<uint64_t>(m_TraceHash);
        w.U32(m_TraceHashed);
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

    ++m_ModSeq;
    struct {
        uint32_t state, end, frames, error;
        uint64_t seq;
    } hdr = {m_ModState, m_EndReason, (uint32_t) m_FrameLog.size(), m_LastError, m_ModSeq};
    static_assert(sizeof(hdr) == 24, "OBS header");
    m_Chan.Send(BR_OBS, &hdr, sizeof(hdr), buf.data(), buf.size());
}

void Agent::SnapHs(HsSnap &h) {
    h = HsSnap();
    CKContext *ctx = Ctx();
    if (!m_EnergyId)
        if (CKDataArray *a = Array("Energy"))
            m_EnergyId = a->GetID();
    if (!m_AllLevelId)
        if (CKDataArray *a = Array("AllLevel"))
            m_AllLevelId = a->GetID();
    auto *energy = (CKDataArray *) ctx->GetObject(m_EnergyId);
    if (!energy || energy->GetRowCount() < 1 || energy->GetColumnCount() < 6)
        return;
    for (int c = 0; c < 6; ++c) {
        h.types[c] = (uint8_t) energy->GetColumnType(c);
        if (auto *e = energy->GetElement(0, c))
            h.energy[c] = (uint32_t) *e;
    }
    const bool ints = h.types[0] == CKARRAYTYPE_INT && h.types[1] == CKARRAYTYPE_INT && h.types[5] == CKARRAYTYPE_INT;
    int level = m_Level;
    if (CKDataArray *cur = Array("CurrentLevel"))
        if (cur->GetColumnType(0) == CKARRAYTYPE_INT)
            if (auto *e = cur->GetElement(0, 0))
                level = (int) *e;
    auto *all = (CKDataArray *) ctx->GetObject(m_AllLevelId);
    if (all && level >= 1 && level <= all->GetRowCount() && all->GetColumnCount() > 6) {
        h.bonusType = (uint8_t) all->GetColumnType(6);
        if (auto *e = all->GetElement(level - 1, 6))
            h.levelBonus = (uint32_t) *e;
    }
    // BML+ GetHSScore(): Points + Lifes x LifeBonus (checked on every tas-v1 record: hs_bml = hs_mod - LevelBonus)
    h.bmlHs = (int32_t) h.energy[0] + (int32_t) h.energy[1] * (int32_t) h.energy[5];
    const bool bonusInt = h.bonusType == CKARRAYTYPE_INT;
    h.hs = (int32_t) h.energy[0] + (int32_t) h.energy[1] * (int32_t) h.energy[5] + (bonusInt ? (int32_t) h.levelBonus : 0);
    h.valid = ints && bonusInt ? 1 : 2;
}

uint16_t Agent::CountFollowers() {
    CKContext *ctx = Ctx();
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

namespace ivpx {
// IVP_Real_Object::q_core_f_object is protected (explicit instantiation is exempt from access checks).
template <class Tag, typename Tag::type Member>
struct thief {
    friend typename Tag::type steal(Tag) { return Member; }
};
struct ro_qcf {
    using type = IVP_U_Quat *IVP_Real_Object_Fast_Static::*;
    friend type steal(ro_qcf);
};
template struct thief<ro_qcf, &IVP_Real_Object_Fast_Static::q_core_f_object>;
inline IVP_U_Quat *qcf(IVP_Real_Object *o) { return o->*steal(ro_qcf()); }
struct ro_shift {
    using type = IVP_U_Float_Point IVP_Real_Object_Fast_Static::*;
    friend type steal(ro_shift);
};
template struct thief<ro_shift, &IVP_Real_Object_Fast_Static::shift_core_f_object>;
inline IVP_U_Float_Point &shift(IVP_Real_Object *o) { return o->*steal(ro_shift()); }
}  // namespace ivpx

CK_ID CkidOf(IVP_Real_Object *o) {
    auto *e = o ? static_cast<CK3dEntity *>(o->client_data) : nullptr;
    return e ? e->GetID() : 0;
}

void PutTri(Wire &w, const IVP_Compact_Edge *edge) {
    const IVP_Compact_Triangle *tri = edge->get_triangle();
    const IVP_Compact_Ledge *ledge = edge->get_compact_ledge();
    const IVP_Compact_Poly_Point *pts = ledge->get_point_array();
    for (int k = 0; k < 3; ++k) {
        const IVP_Compact_Poly_Point &p = pts[tri->get_edge(k)->get_start_point_index()];
        w.F32((float) p.k[0]);
        w.F32((float) p.k[1]);
        w.F32((float) p.k[2]);
    }
}

void Agent::BuildState(bool minimal) {
    CKContext *ctx = Ctx();
    if (m_StateBuf.size() < (1u << 20))
        m_StateBuf.resize(1u << 20);
    Wire w(m_StateBuf.data(), m_StateBuf.size());
    // ---- TIME
    if (m_P2)
        WriteTime2002(w);
    else {
        bmmo_physics_clock_state ck{};
        std::string err;
        if (m_Physics)
            bmmo::physics::get_clock_state(m_Physics, ck, err);
        w.Begin("TIME", 1);
        w.F64(ck.current_time);
        w.F64(ck.time_of_next_psi);
        w.F64(ck.time_of_last_psi);
        w.F32(m_Physics ? m_Physics->m_DeltaTime : 0.0f);
        w.F32(m_Physics ? m_Physics->m_PhysicsDeltaTime : 0.0f);
        w.F32(m_Physics ? m_Physics->m_PhysicsTimeFactor : 0.0f);
        w.F32(m_Engine->time_manager()->GetLastDeltaTime());
        w.F64(m_Sr);
        w.F32((float) m_Sr);
        w.U8(m_SrRunning);
        w.U32(m_Covered);
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
    if (m_P2)
        WriteIvp2002(w, ball);   // BALL, FRIC, MIND, BODY from the 2002 DLL's memory
    IVP_Real_Object *real = m_P2 ? nullptr : BallReal(ball);
    IVP_Core *core = real ? real->get_core() : nullptr;

    // ---- BALL
    if (!m_P2 && !(m_ObsMask & M_BALL)) {
        w.Begin("BALL", 1);
        w.SetSectionFlags(SECF_APPROX);   // core_flags / core_state: only the unmoveable bits / movement state
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
        if (core) {
            for (int k = 0; k < 3; ++k)
                w.F64(core->pos_world_f_core_last_psi.k[k]);
            const IVP_U_Quat &ql = core->q_world_f_core_last_psi, &qn = core->q_world_f_core_next_psi;
            w.F64(ql.x), w.F64(ql.y), w.F64(ql.z), w.F64(ql.w);
            w.F64(qn.x), w.F64(qn.y), w.F64(qn.z), w.F64(qn.w);
            w.F64(core->time_of_last_psi.get_seconds());
            for (const IVP_U_Float_Point *v : {&core->speed, &core->speed_change, &core->rot_speed, &core->rot_speed_change,
                                               &core->rotation_axis_world_space})
                for (int k = 0; k < 3; ++k)
                    w.F32((float) v->k[k]);
            w.F32(core->current_speed);
            w.F32(core->abs_omega);
            w.U32(((uint32_t) (core->physical_unmoveable ? 1 : 0) & 3u) << 2);
            w.U32((uint32_t) (uint8_t) core->movement_state);
        } else {
            w.Zeros(24 + 32 + 32 + 8 + 60 + 8 + 8);
        }
        w.F32(real ? real->get_extra_radius() : 0.0f);
        w.End();
    }

    // ---- FRIC (+ FRCP: partner ckid + now_friction_pressure per item, same order)
    FrcpItem frcp[32];
    uint32_t frcpN = 0;
    if (!m_P2 && !(m_ObsMask & M_FRIC)) {
        w.Begin("FRIC", 1);
        // lt_*: the long-term contact info (tmp_contact_info) is transaction memory, cleared after the PSI; but
        // IVP_Contact_Point::recalc_friction_s_vals copies its contact_point_ws / surf_normal onto the point itself
        // (last_contact_point_ws, surf_normal_ws: "cache surface normal as needed by external code"), so those are the
        // values the 2002 host reads through lt (the point as f32 here, f64 there). lt_valid 0 until the first recalc.
        auto contactLt = [](IVP_Contact_Point *c, float nrm[3], double pt[3]) -> bool {
            const IVP_U_Float_Point &n = IVP_BMMO_State::cp_normal_ws(*c);
            const IVP_U_Float_Point &q = IVP_BMMO_State::cp_contact_ws(*c);
            for (int k = 0; k < 3; ++k) {
                nrm[k] = n.k[k];
                pt[k] = q.k[k];
            }
            return n.k[0] != 0.0f || n.k[1] != 0.0f || n.k[2] != 0.0f;
        };
        const size_t at = w.Size();
        w.U16(0);
        uint32_t n = 0;
        IVP_Synapse_Friction *head = real ? real->get_first_friction_synapse() : nullptr;
        for (IVP_Synapse_Friction *s = head; s && n < 32;) {
            IVP_Contact_Point *cp = s->get_contact_point();
            IVP_Synapse_Friction *s0 = &IVP_BMMO_State::cp_synapse(*cp, 0), *s1 = &IVP_BMMO_State::cp_synapse(*cp, 1);
            IVP_Synapse_Friction *ps = s == s0 ? s1 : s == s1 ? s0 : nullptr;
            w.U32(ps ? CkidOf(ps->get_object()) : 0);
            w.U8((uint32_t) (uint8_t) s->get_status());
            w.U8(ps ? (uint32_t) (uint8_t) ps->get_status() : 255);
            w.U8((uint32_t) (uint8_t) IVP_BMMO_State::cp_status(*cp));
            w.U8(0);
            w.F32(IVP_BMMO_State::cp_gap(*cp));
            w.F32(IVP_BMMO_State::cp_friction_factor(*cp));
            FrcpItem &fi = frcp[n];
            fi.partner = ps ? CkidOf(ps->get_object()) : 0;
            fi.pressure = IVP_BMMO_State::cp_pressure(*cp);
            frcpN = n + 1;
            float nrm[3] = {};
            double pt[3] = {};
            const bool ltOk = contactLt(cp, nrm, pt);
            w.Raw(nrm, sizeof(nrm));
            w.Raw(pt, sizeof(pt));
            w.U8(ltOk ? 1 : 0);
            ++n;
            IVP_Synapse_Friction *nx = s->get_next();
            if (nx == s || nx == head)
                break;
            s = nx;
        }
        w.PatchU16(at, (uint16_t) n);
        w.End();
        WriteFrcp(w, frcp, frcpN);
    }

    // ---- MIND
    if (!m_P2 && !(m_ObsMask & M_MIND)) {
        w.Begin("MIND", 1);
        w.SetSectionFlags(SECF_APPROX);   // ledge: 0 (an address in the 2002 DLL)
        const size_t at = w.Size();
        w.U16(0);
        uint32_t n = 0;
        IVP_Synapse_Real *head = real ? real->get_first_exact_synapse() : nullptr;
        int guard = 0;
        for (IVP_Synapse *s = head; s && n < 64 && guard < 256; ++guard) {
            IVP_Mindist_Base *md = s->get_synapse_mindist();
            const IVP_Synapse *s0 = md->get_mindist_synapse(0), *s1 = md->get_mindist_synapse(1);
            const IVP_Synapse *ps = s == s0 ? s1 : s == s1 ? s0 : nullptr;
            if (ps) {
                const int pst = (int) ps->get_status();
                const IVP_Compact_Edge *pedge = ps->get_edge();
                uint8_t ntri = 0;
                const IVP_Compact_Edge *oe = nullptr;
                if (pedge && pst >= 0 && pst <= 2) {
                    ntri = 1;
                    if (pst == 1 && (oe = pedge->get_opposite()))
                        ntri = 2;
                }
                w.U32(CkidOf(const_cast<IVP_Synapse *>(ps)->get_object()));
                w.U8((uint32_t) (uint8_t) s->get_status());
                w.U8((uint32_t) (uint8_t) pst);
                w.U16(0);
                w.F32(md->len_numerator);
                w.F32((float) md->contact_plane.k[0]);
                w.F32((float) md->contact_plane.k[1]);
                w.F32((float) md->contact_plane.k[2]);
                w.F32((float) md->contact_plane.hesse_val);
                w.F32(md->sum_extra_radius);
                w.U32(0);
                w.U16(ntri ? (uint32_t) pedge->get_triangle()->get_tri_index() : 0xFFFF);
                w.Put<int8_t>((int8_t) (ntri ? pedge->get_edge_index() : -1));
                w.U8(ntri);
                if (ntri >= 1)
                    PutTri(w, pedge);
                else
                    w.Zeros(36);
                if (ntri >= 2)
                    PutTri(w, oe);
                else
                    w.Zeros(36);
                ++n;
            }
            IVP_Synapse *nx = s->next;
            if (nx == s || nx == head)
                break;
            s = nx;
        }
        w.PatchU16(at, (uint16_t) n);
        w.End();
    }

    // ---- BODY: movable bodies near the ball (PSI poses)
    if (!m_P2 && !(m_ObsMask & M_BODY)) {
        w.Begin("BODY", 1);
        w.SetSectionFlags(SECF_APPROX);   // core_state: movement state only; order of the movable-object list
        const size_t at = w.Size();
        w.U16(0);
        uint32_t n = 0;
        if (real && ball && m_Physics) {
            const VxVector bp = ball->GetWorldMatrix()[3];
            const float r2 = m_BodyRadius * m_BodyRadius;
            // every physicalized entity (the 2002 mod walks the whole cluster): bodies off the movable list (asleep)
            // count too; unmoveable cores are skipped as there
            for (auto it = m_Physics->m_PhysicsObjects.Begin(); it != m_Physics->m_PhysicsObjects.End() && n < 0xFFFF; ++it) {
                IVP_Real_Object *ro = (*it).m_RealObject;
                if (!ro || ro == real)
                    continue;
                IVP_Core *bc = ro->get_core();
                if (!bc || bc->physical_unmoveable)
                    continue;
                auto *ent = static_cast<CK3dEntity *>(ro->client_data);
                if (!ent)
                    continue;
                const VxMatrix &wm = ent->GetWorldMatrix();
                const float dx = wm[3][0] - bp.x, dy = wm[3][1] - bp.y, dz = wm[3][2] - bp.z;
                if (dx * dx + dy * dy + dz * dz > r2)
                    continue;
                w.U32(ent->GetID());
                w.U32((uint32_t) (uint8_t) bc->movement_state);
                for (int k = 0; k < 3; ++k)
                    w.F64(bc->pos_world_f_core_last_psi.k[k]);
                const IVP_U_Quat &q = bc->q_world_f_core_last_psi;
                w.F64(q.x), w.F64(q.y), w.F64(q.z), w.F64(q.w);
                if (IVP_U_Quat *qp = ivpx::qcf(ro))
                    w.F64(qp->x), w.F64(qp->y), w.F64(qp->z), w.F64(qp->w);
                else
                    w.F64(0), w.F64(0), w.F64(0), w.F64(1);
                const IVP_U_Float_Point &sh = ivpx::shift(ro);
                for (int k = 0; k < 3; ++k)
                    w.F32((float) sh.k[k]);
                for (int k = 0; k < 3; ++k)
                    w.F32((float) bc->speed.k[k]);
                for (int k = 0; k < 3; ++k)
                    w.F32((float) bc->rot_speed.k[k]);
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

    // ---- GLCT: block and output pulses only (the group counters are behind a 2002 pointer)
    if (m_P2 && !(m_ObsMask & M_GLCT))
        WriteGlct2002(w);
    else if (!(m_ObsMask & M_GLCT)) {
        CK_ID bbId = 0;
        uint16_t outMask = 0;
        for (size_t k = 0; k < m_ContactBBs.size() && !bbId; ++k) {
            auto *b = (CKBehavior *) ctx->GetObject(m_ContactBBs[k]);
            if (!b || !b->IsActive())
                continue;
            bbId = b->GetID();
            for (int i = 0; i < b->GetOutputCount() && i < 16; ++i)
                if (b->IsOutputActive(i))
                    outMask |= (uint16_t) (1u << i);
        }
        w.Begin("GLCT", 1);
        w.SetSectionFlags(SECF_APPROX);
        w.U32(bbId);
        w.U16(outMask);
        w.Zeros(24);
        w.End();
    }

    // ---- ARRS
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
                if (CKDataArray *a = Array(kTrackedArrays[k]))
                    snap.id = a->GetID();
            auto *a = snap.id ? (CKDataArray *) ctx->GetObject(snap.id) : nullptr;
            if (!a)
                continue;
            const int rows = a->GetRowCount(), cols = a->GetColumnCount();
            if (rows != snap.rows || cols != snap.cols) {
                snap.rows = rows;
                snap.cols = cols;
                snap.cells.assign((size_t) rows * cols, std::string("\x01\x02\x03", 3));
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
        if (!w.End())
            for (ArrSnap &s : m_Arrs)
                s.rows = -1;
    }

    // ---- WTCH
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

    // ---- SCOR
    {
        HsSnap h;
        SnapHs(h);
        w.Begin("SCOR", 1);
        w.SetSectionFlags(SECF_APPROX);   // bml_hs computed
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
        const uint16_t followers = CountFollowers();
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

void Agent::WritePhysSections(Wire &w, StrTab &strs, size_t from, size_t to, std::vector<std::pair<CK_ID, CK_ID>> &done,
                              std::unordered_set<uint64_t> &ivpoDone) {
    CKContext *ctx = Ctx();
    w.Begin("PHYS", 1);
    w.U32((uint32_t) (to - from));
    for (size_t i = from; i < to; ++i) {
        const PhysRec &r = m_Phys[i];
        const bool phys = r.kind == 0;
        w.U32(r.seq);
        w.U32(r.frame);
        w.U32(r.ent);
        w.U8(phys ? 0 : 1);
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

    w.Begin("MESH", 1);
    const size_t at = w.Size();
    w.U32(0);
    uint32_t n = 0;
    for (size_t i = from; i < to; ++i) {
        const PhysRec &r = m_Phys[i];
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

    if (m_P2) {   // the 2002 DLL: the cluster walk itself (m_Reals, walked by the caller), material bytes included
        WriteIvpo2002(w, ivpoDone);
        return;
    }
    // IVPO: as the 2002 mod's cluster walk, nothing before the first movable object exists, then every
    // (physicalized entity, IVP object) pair once. Material bytes are not available (zeros).
    w.Begin("IVPO", 1);
    w.SetSectionFlags(SECF_APPROX);
    const size_t at2 = w.Size();
    w.U32(0);
    uint32_t n2 = 0;
    if (m_Physics && m_Physics->m_MovableObjects.len() > 0) {
        std::vector<CK_ID> ents;
        for (const PhysRec &r : m_Phys)
            if (r.kind == 0 && std::find(ents.begin(), ents.end(), r.ent) == ents.end())
                ents.push_back(r.ent);
        for (CK_ID id : ents) {
            auto *ent = (CK3dEntity *) ctx->GetObject(id);
            if (!ent || !CKIsChildClassOf(ent, CKCID_3DENTITY))
                continue;
            IVP_Real_Object *ro = BallReal(ent);
            if (!ro)
                continue;
            const uint64_t key = (uint64_t) id << 32 | (uint32_t) (reinterpret_cast<uintptr_t>(ro) >> 4);
            if (!ivpoDone.insert(key).second)
                continue;
            IVP_Core *c = ro->get_core();
            w.U32(id);
            w.I32((int32_t) ro->get_type());
            w.U8(c && c->physical_unmoveable ? 1 : 0);
            w.F32(ro->get_extra_radius());
            w.Zeros(32);
            ++n2;
        }
    }
    w.PatchU32(at2, n2);
    w.End();
}

void Agent::ExportPhysicalized() {
    if (m_P2) {
        RebuildEntityMap();
        WalkCluster2002(ClusterStart2002());
    }
    const size_t from = m_PhysExported, to = m_Phys.size();
    if (m_LevelInc) {
        std::vector<uint8_t> buf;
        Wire w(&buf);
        StrTab strs;
        BeginMessage(w, "LVL1", 1, (uint32_t) m_FrameIndex);
        WritePhysSections(w, strs, from, to, m_MeshesDone, m_IvpoDone);
        strs.Write(w);
        EndMessage(w);
        std::fwrite(buf.data(), 1, buf.size(), m_LevelInc);
        std::fflush(m_LevelInc);
    }
    for (size_t i = from; i < to; ++i)
        m_PhysNew.push_back(m_Phys[i].seq);
    m_PhysExported = to;
}

void Agent::LevelTables() {
    CKContext *ctx = Ctx();
    if (m_P2) {   // the 2002 DLL: BallrainIO AgentLevelTables's contact blocks (with their IVP_Handle) and entity map
        RebuildEntityMap();
        m_ContactBBs.clear();
        m_ContactHandles.clear();
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
    for (int k = 0; k < 5; ++k) {
        CKObject *o = ctx->GetObjectByNameAndParentClass((CKSTRING) kCamNames[k], CKCID_3DENTITY, nullptr);
        m_CamIds[k] = o ? o->GetID() : 0;
    }
    if (!m_P2) {
        m_ContactBBs.clear();
        const int n = ctx->GetObjectsCountByClassID(CKCID_BEHAVIOR);
        CK_ID *ids = ctx->GetObjectsListByClassID(CKCID_BEHAVIOR);
        for (int k = 0; k < n; ++k) {
            auto *b = (CKBehavior *) ctx->GetObject(ids[k]);
            if (!b || !b->IsUsingFunction() || !b->GetPrototypeName() ||
                strcmp(b->GetPrototypeName(), "PhysicsContinuousContact") != 0)
                continue;
            bool handle = false;
            for (int i = 0; i < b->GetLocalParameterCount() && !handle; ++i) {
                CKParameter *p = b->GetLocalParameter(i);
                handle = p && p->GetName() && !strcmp(p->GetName(), "IVP_Handle");
            }
            if (handle)
                m_ContactBBs.push_back(b->GetID());
        }
    }
    m_Watch.clear();
    m_WatchGroup.clear();
    {
        std::vector<std::pair<CK_ID, CK_ID>> watch;
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
        for (const auto &wg : watch)
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
    if (CKDataArray *a = Array("Energy"))
        m_EnergyId = a->GetID();
    if (CKDataArray *a = Array("AllLevel"))
        m_AllLevelId = a->GetID();
    m_TablesReady = true;
}

void Agent::WriteLevelMessage() {
    if (m_LevelMsgPath.empty())
        return;
    CKContext *ctx = Ctx();
    if (!m_TablesReady)
        LevelTables();
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
    {
        const XObjectPointerArray &list = ctx->GetObjectListByType(CKCID_3DENTITY, TRUE);
        w.Begin("ENTS", 1);
        w.U32((uint32_t) list.Size());
        for (int k = 0; k < list.Size(); ++k) {
            auto *e = (CK3dEntity *) list[k];
            if (!e) {
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
            for (int a = 0; a < 3; ++a) {
                const int t = attrTypes[a];
                const bool present = t >= 0 && e->HasAttribute(t);
                CKParameterOut *p = present ? e->GetAttributeParameter(t) : nullptr;
                u8 rawAttr[64] = {};
                int size = 0;
                if (p) {
                    const void *data = p->GetReadDataPtr(FALSE);
                    size = std::min(p->GetDataSize(), 64);
                    if (!data || size < 0)
                        size = 0;
                    else
                        std::memcpy(rawAttr, data, (size_t) size);
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
                rows = 0;
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
    {
        if (m_P2)
            WalkCluster2002(ClusterStart2002());
        std::vector<std::pair<CK_ID, CK_ID>> done;
        std::unordered_set<uint64_t> ivpoDone;
        WritePhysSections(w, strs, 0, m_Phys.size(), done, ivpoDone);
    }
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
#ifdef _WIN32
    ok = ok && MoveFileExA(tmp.c_str(), m_LevelMsgPath.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;   // rename won't replace
#else
    ok = ok && std::rename(tmp.c_str(), m_LevelMsgPath.c_str()) == 0;
#endif
    if (!ok)
        Log("cannot write level message %s", m_LevelMsgPath.c_str());
}

bool Agent::TickFrame(float delta, uint32_t keys, std::string &err) {
    m_Engine->set_frame_ms(delta);
    SetKeys(keys);
    ApplyBallOrder();
    return m_Engine->tick(err);
}

// BallrainIO AgentProcess (OnProcess), after the frame's CKContext::Process.
void Agent::AfterTick() {
    // TASSupport's 1-tick timer from PreLoadLevel: the clock reset lands inside frame 0, before its observation
    // (the retail frames_done 1 observation has env time 0); frame 1 is the first physics step from it
    // (SimTool --tas-reset-frame R: before frame R, default 1).
    if (m_Phase == Phase::Running && (int) m_FrameIndex == m_Recipe.reset_frame && !m_ClockReset) {
        ResetClock();
        m_ClockReset = true;
    }
    static const bool perturb = std::getenv("BALLRAIN_HL_HEAP_PERTURB") != nullptr;   // diagnostic
    if (perturb && m_FrameIndex % 10 == 0) {
        static std::vector<void *> keep;
        keep.push_back(std::malloc(16 + (m_FrameIndex * 37) % 512));
    }
    if (m_SrRunning) {
        const float d = m_Engine->time_manager()->GetLastDeltaTime();
        m_Sr += (double) d;
        m_Sr32 += d;
    }
    CK3dEntity *ball = GetActiveBall();
    if (ball != m_LastBall) {
        m_LastBall = ball;
        Event(EV_ACTIVEBALL);
    }
    if (m_PhysExported < m_Phys.size())
        ExportPhysicalized();
    if (m_LevelMsgPending) {
        m_LevelMsgPending = false;
        LevelTables();
        WriteLevelMessage();
    }
    switch (m_Phase) {
        case Phase::Running: {
            if (ball) {   // run fingerprint (HASH section of the final observation): same frames, same engine => same hash
                const VxMatrix &hm = ball->GetWorldMatrix();
                const uint32_t fi = (uint32_t) m_FrameIndex;
                auto mix = [&](const void *p, size_t n) {
                    for (size_t i = 0; i < n; ++i)
                        m_TraceHash = (m_TraceHash ^ ((const u8 *) p)[i]) * 1099511628211ull;
                };
                mix(&fi, 4);
                for (int r = 0; r < 4; ++r)
                    mix(&hm[r][0], 12);
                ++m_TraceHashed;
            }
            if (m_Trace && ball) {
                const VxMatrix &m = ball->GetWorldMatrix();
                std::fprintf(m_Trace, "%u %s", (unsigned) m_FrameIndex, ball->GetName() ? ball->GetName() : "-");
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 3; ++c)
                        std::fprintf(m_Trace, " %.9g", m[r][c]);
                std::fputc('\n', m_Trace);
            }
            if (m_WantTrac && ball) {
                if (m_Trac.size() < (1u << 16)) {
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
                m_BlockEnd = m_FrameIndex;
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
                m_InPrefix = false;
            m_PrefixObs = m_InPrefix;
            BuildState(m_FrameIndex == 0);
            if (!m_GameOverPending && !m_EndAtNextTime)
                Publish(false);
            return;
        }
        case Phase::Tail:
            if (m_TailFrames == 0) {
                m_SrAtFinish = m_Sr;
                m_Sr32AtFinish = m_Sr32;
            }
            if (!m_SrRunning || m_TailFrames >= m_SrTail) {
                m_PrefixObs = false;
                BuildState(false);
                Finish(END_FINISH);
            }
            return;
        case Phase::FinalPending:
            m_PrefixObs = false;
            BuildState(false);
            Finish(END_RESET);
            return;
        default:
            return;
    }
}

int Agent::Run() {
    if (m_Sock.empty() || !m_Chan.Connect(m_Sock)) {
        Log("cannot connect to %s", m_Sock.c_str());
        return 2;
    }
    // Handshake (legacy prototype convention: Python pings, the game pongs).
    {
        uint32_t type = 0;
        std::vector<u8> msg;
        if (m_Chan.Recv(type, msg, 60000) <= 0 || type != BR_PING) {
            Log("no ping from the client");
            return 2;
        }
    }
    auto pong = [&](uint32_t state) {
        struct {
            uint32_t pid, obsVersions, state;
            char version[180];
        } p = {};
#ifdef _WIN32
        p.pid = (uint32_t) GetCurrentProcessId();
#else
        p.pid = (uint32_t) getpid();
#endif
        p.obsVersions = 1u << 1;
        p.state = state;
        if (m_P2)   // the 2002 DLL: its hash prefix (the retail hello carries the physics_RT hash too)
            std::snprintf(p.version, sizeof(p.version), "%s phys2002 %.16s engine headless level %d", kVersion,
                          Phys2002Sha(), m_Level);
        else
            std::snprintf(p.version, sizeof(p.version), "%s x87soft engine headless (approximate backend) level %d",
                          kVersion, m_Level);
        m_Chan.Send(BR_PONG, &p, sizeof(p));
    };
    if (m_Level < 1 || m_Level > 13 || m_Root.empty()) {
        pong(ST_HELLO);
        return Fail(ERR_ENV, "BALLRAIN_LEVEL=%d / --root missing", m_Level) ? 0 : 2;
    }
    {
        std::string err;
        if (!ParseBallOrder(m_BallOrderSpec, m_Level, g_BallOrder, err)) {
            pong(ST_HELLO);
            return Fail(ERR_ENV, "BALLRAIN_HL_BALL_ORDER=%s: %s", m_BallOrderSpec.c_str(), err.c_str()) ? 0 : 2;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!Boot())
        return 2;
    g_Agent = this;
    SetState(ST_MENU);
    pong(ST_MENU);
    Log("booted in %.2f s", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    // Run start (BallrainIO Start / AgentStart at PreLoadLevel).
    if (CKDataArray *kb = Array("Keyboard"))
        for (int k = 0; k < 6; ++k) {
            int code = 0;
            if (kb->GetElementValue(0, k, &code) && code)
                m_KeyCodes[k] = code;
        }
    if (!m_EventsPath.empty())
        m_Events = std::fopen(m_EventsPath.c_str(), "w");
    if (!m_TracePath.empty())
        m_Trace = std::fopen(m_TracePath.c_str(), "w");
    if (!m_LevelMsgPath.empty())
        m_LevelInc = std::fopen((m_LevelMsgPath + ".inc").c_str(), "wb");
    m_Phase = Phase::Running;
    m_FrameIndex = 0;
    m_InPrefix = !m_Prefix.empty();
    m_BlockStart = 0;
    m_BlockEnd = m_InPrefix ? (m_PrefixEvents ? 1 : m_Prefix.size()) : 0;
    SetState(ST_LOADING);
    Event(EV_PRELOAD);
    std::string err;
    ApplyBallOrder();
    if (!m_Engine->load_level(m_Level, err))
        return Fail(ERR_ENV, "load_level: %s", err.c_str()) ? 0 : 2;
    m_Engine->set_frame_ms(0.0f);
    SetKeys(0);
    for (int i = 0; i < m_Recipe.offset; ++i) {
        ApplyBallOrder();
        if (!m_Engine->tick(err))
            return Fail(ERR_ENV, "hidden tick %d: %s", i, err.c_str()) ? 0 : 2;
    }
    {
        // the product build's key (VX_X87_GAME): without it two balls are ordered by heap address, which changes from
        // run to run
        IVP_Environment *env = m_Physics ? m_Physics->GetEnvironment() : nullptr;
        if (m_P2) {
            // the 2002 DLL: its one ball-pair compare runs through our key (phys2002.hpp); checked in Boot
            Log("2002 physics_RT.dll: ball-pair key %s",
                !m_P2Hooked ? "none (heap addresses)"
                : g_BallOrder.mode == BallOrder::CKID ? "CK id" : g_BallOrder.spec.c_str());
        } else if (!env || !env->bmmo_ball_order_key)
            return Fail(ERR_ENV, "the physics environment has no ball-pair key (a build without VX_X87_GAME?)") ? 0 : 2;
        if (g_BallOrder.mode != BallOrder::CKID && !g_BallOrder.built)
            BuildBallOrder();
        Log("ball order %s: %s", g_BallOrder.spec.c_str(),
            DescribeBallOrder(Ctx(), g_BallOrder.mode == BallOrder::CKID ? nullptr : &BallOrderKey).c_str());
    }
    if (m_Recipe.reset_frame == 0 && !m_ClockReset) {   // --tas-reset-frame 0: before frame 0
        ResetClock();
        m_ClockReset = true;
    }
    // frame 0's observation (minimal) unless a prefix runs first
    if (m_PhysExported < m_Phys.size())
        ExportPhysicalized();
    if (m_BlockEnd == 0) {
        BuildState(true);
        Publish(false);
    }

    while (m_Phase != Phase::Ended) {
        float delta = m_LastDelta;
        uint32_t keys = 0;
        if (m_Phase == Phase::Tail) {
            ++m_FrameIndex;
            ++m_TailFrames;
        } else if (m_Phase == Phase::Running) {
            if (m_EndAtNextTime && !m_GameOverPending) {
                m_EndReason = m_EndAtNextTime;
                m_EndAtNextTime = 0;
                Stop(StopReason::RecordEnd);
                break;
            }
            if (m_GameOverPending) {
                Stop(StopReason::GameOver);
                break;
            }
            const Frame *frame = GetFrame(m_FrameIndex);
            if (!frame) {
                if (!WaitAction())
                    break;
                frame = GetFrame(m_FrameIndex);
            }
            if (!frame) {
                Stop(StopReason::RecordEnd);
                break;
            }
            delta = frame->delta;
            keys = frame->keys;
            m_FrameLog.push_back(Frame{delta, keys});
            ++m_FrameIndex;
            ++m_Covered;
            m_LastDelta = delta;
        } else {
            break;   // FinalPending is resolved inside AfterTick
        }
        if (!TickFrame(delta, keys, err)) {
            Log("frame %u: %s", (unsigned) m_FrameIndex, err.c_str());
            m_LastError = ERR_ENV;
            m_EndReason = END_ERROR;
            if (m_Phase == Phase::Running)
                Stop(StopReason::RecordEnd);
            else
                Finish(END_ERROR);
            break;
        }
        AfterTick();
    }
    if (m_Phase != Phase::Ended)
        Finish(m_EndReason ? m_EndReason : END_STOP);
    g_Agent = nullptr;
    m_Chan.Close();
    return 0;
}

}  // namespace

int ballrain_agent_main(const ballrain_agent_args &args) {
#ifndef _WIN32
    std::signal(SIGPIPE, SIG_IGN);
#endif
#ifdef _WIN32
    std::setvbuf(stderr, nullptr, _IONBF, 0);   // the MSVC CRT has no line buffering (_IOLBF = full): keep lines up to a crash
#else
    std::setvbuf(stderr, nullptr, _IOLBF, 1 << 14);
#endif
    auto *agent = new Agent;   // never freed: the engine is torn down by process exit
    if (!agent->Configure(args))
        return 2;
    const int rc = agent->Run();
    std::fflush(nullptr);
    std::_Exit(rc);   // skip the engine teardown (one run per process)
}
