// Wire formats of the agent mode (docs/agent-protocol.md; a port of ballrain/protocol.py): OBS1 observations, ACT1
// actions, LVL1 level messages. Field types follow the numpy dtypes there (f4 stays float, f8 double), so the
// feature code does the same arithmetic as the Python side.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ballrain/util.h"

namespace ballrain {

constexpr uint32_t OBS_VERSION = 1;
constexpr uint32_t ACT_VERSION = 1;

namespace ModState { enum { INIT, HELLO, MENU, LOADING, RUNNING, WAITING, ENDED, ERROR }; }
namespace EndReason { enum { NONE, LEVEL_FINISH, GAME_OVER, FRAME_LIMIT, PY_STOP, RESET_EXIT, TIMEOUT, ERROR }; }
const char *end_reason_name(uint32_t r);

namespace Ev {
enum : uint16_t {
    PreLoadLevel = 1, PostLoadLevel, StartLevel, CamNavActive, CamNavInactive, BallNavActive, BallNavInactive,
    CounterActive, CounterInactive, PauseLevel, UnpauseLevel, BallOff, Dead, PreSubLife, PostSubLife, PreCheckpoint,
    PostCheckpoint, ExtraPoint, PreLifeUp, PostLifeUp, LevelFinish, GameOver, PreEndLevel, PostEndLevel, Physicalize,
    Unphysicalize, ActiveBallChanged, PreResetLevel, PreExitLevel, RunEnd
};
}
const char *event_name(uint16_t code);
uint64_t event_mask(std::initializer_list<uint16_t> codes);

// ACT1 flags / obs_mask bits
constexpr uint16_t ACT_STOP = 1 << 0;
constexpr uint16_t ACT_WANT_TRAC = 1 << 1;
constexpr uint16_t ACT_REWRITE_LEVEL = 1 << 2;
// OBS1 header flags
constexpr uint16_t OBS_FINAL = 1, OBS_PREFIX = 2, OBS_DROPPED = 4, OBS_MINIMAL = 8, OBS_HEADLESS = 16;

// keys (ballrain/tas.py)
constexpr uint32_t KEY_UP = 1 << 0, KEY_DOWN = 1 << 1, KEY_LEFT = 1 << 2, KEY_RIGHT = 1 << 3, KEY_SHIFT = 1 << 4,
                   KEY_SPACE = 1 << 5, KEY_Q = 1 << 6, KEY_ESC = 1 << 7, KEY_ENTER = 1 << 8;

// CK_ARRAYTYPE
constexpr uint8_t ARR_INT = 1, ARR_FLOAT = 2, ARR_STRING = 3, ARR_OBJECT = 4, ARR_PARAMETER = 5;
constexpr uint32_t NO_STRING = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------------------------- observation

struct TimeSec {
    double env_time, next_psi, last_psi;
    float mgr_delta_ms, mgr_step_s, time_factor, delta_ms;
    double sr_double;
    float sr_float;
    uint8_t sr_running;
    uint32_t frames_covered;
};
struct BallSec {
    uint32_t ckid;
    uint8_t type, found;
    float ck_matrix[12];
    double pos[3], q_last[4], q_next[4], tpsi;
    float speed[3], speed_change[3], rot_speed[3], rot_speed_change[3], rot_axis[3];
    float current_speed, abs_omega;
    uint32_t core_flags, core_state;
    float extra_radius;
};
struct FricItem {
    uint32_t partner;
    uint8_t self_status, partner_status, cp_status;
    float gap, friction, lt_normal[3];
    double lt_point[3];
    uint8_t lt_valid;
};
struct MindItem {
    uint32_t partner;
    uint8_t self_status, partner_status;
    float gap, plane[4], sum_extra_radius;
    uint32_t ledge;
    uint16_t tri_index;
    int8_t edge_index;
    uint8_t n_tris;
    float tri[9], tri_adj[9];
};
struct BodyItem {
    uint32_t ckid, core_state;
    double pos[3], q[4], q_cfo[4];
    float shift[3], speed[3], rot_speed[3], ck_matrix[12];
};
struct CamsSec {
    struct Cam {
        uint32_t ckid;
        float matrix[12];  // rows x, y, z axis, pos (4 x 3)
    } cams[5];
    uint32_t attached;
};
struct HsFields {
    uint8_t hs_valid, types[6], bonus_type;
    uint32_t energy[6], level_bonus;
    int32_t bml_hs, hs;
};
struct ScorSec {
    HsFields hs;
    uint32_t ball_offs, last_balloff, last_ballnav;
};
struct ScorExt { uint16_t followers, sources; };
struct SrenSec {
    double sr_double;
    float sr_f32sum, sr_bml;
    uint32_t tail_frames;
    uint8_t level_finished, counter_inactive_seen, sr_running;
    double sr_double_at_finish;
    float sr_f32sum_at_finish;
    HsFields hs;
};
struct SrenExt { uint16_t followers_at_finish, followers_end; uint32_t post_frames; };
struct HashSec { uint64_t trace_hash; uint32_t frames; };
struct EvtItem { uint32_t frame; uint16_t code, param; };
struct WtchItem { uint32_t ckid; uint8_t visible; float pos[3]; };
struct ArrCell {
    uint32_t array;
    uint16_t row, col;
    uint8_t col_type;
    Bytes data;
};

struct Observation {
    uint32_t frames_done = 0;
    uint16_t flags = 0;
    std::optional<TimeSec> time;
    std::optional<BallSec> ball;
    std::vector<FricItem> fric;
    std::vector<MindItem> mind;
    std::vector<BodyItem> body;
    std::optional<CamsSec> cams;
    std::optional<ScorSec> scor;
    std::optional<ScorExt> scor_ext;
    std::optional<SrenSec> sren;
    std::optional<SrenExt> sren_ext;
    std::optional<HashSec> hash;
    std::vector<ArrCell> arrs;
    std::vector<EvtItem> events;
    std::vector<WtchItem> wtch;
    std::vector<uint32_t> phys;
    std::vector<std::string> unknown;

    bool final() const { return flags & OBS_FINAL; }
    bool has_event(uint16_t code) const;
};

// One OBS1 message (the payload after the socket's OBS header). ProtocolError on malformed input.
Observation parse_observation(const char *data, size_t n);
inline Observation parse_observation(const Bytes &b) { return parse_observation(b.data(), b.size()); }

// ---------------------------------------------------------------------------------------------- actions

struct Frame {
    float delta;
    uint32_t keys;
};
struct Action {
    std::vector<Frame> frames;
    uint32_t repeat = 1;
    uint64_t stop_mask = 0;
    uint16_t flags = 0;
    uint32_t obs_mask = 0;
    uint64_t total_frames() const { return uint64_t(frames.size()) * repeat; }
};
Bytes encode_action(const Action &a, uint32_t py_seq);

// ---------------------------------------------------------------------------------------------- level messages

struct Entity {
    uint32_t ckid = 0;
    std::optional<std::string> name;
    uint32_t class_id = 0, parent = 0;
    bool visible = false;
    float matrix[12] = {};
    int32_t contact_id = -1, coll_id = -1;
    Bytes floor;
};
struct Group {
    uint32_t ckid = 0;
    std::optional<std::string> name;
    std::vector<uint32_t> members;
};
struct ArrayColumn {
    std::optional<std::string> name;
    uint8_t type = 0;
    std::optional<std::string> param_type;
};
struct ArrayDef {
    uint32_t ckid = 0;
    std::optional<std::string> name;
    std::vector<ArrayColumn> columns;
    std::vector<std::vector<Bytes>> cells;  // rows x cols
};
struct PhysRecord {
    uint32_t seq = 0, frame = 0, ckid = 0;
    bool unphysicalize = false, fixed = false;
    float friction = 0, elasticity = 0, mass = 0;
    std::optional<std::string> collision_group;
    bool frozen = false, collision = true, calc_mass_center = false;
    float linear_damping = 0, rot_damping = 0;
    std::optional<std::string> surface;
    float mass_center[3] = {};
    std::vector<uint32_t> convex, concave;
    struct Ball { float center[3]; float radius; };
    std::vector<Ball> balls;
};
struct Mesh {
    uint32_t ckid = 0, mesh = 0;
    uint8_t kind = 0;
    float matrix[16] = {};       // (4,4) world matrix at export
    std::vector<float> vertices;  // (nv,3) local
    std::vector<uint32_t> faces;  // (nf,3)
};
struct IvpObject {
    uint32_t ckid = 0;
    int32_t object_type = 0;
    bool unmoveable = false;
    float extra_radius = 0;
    Bytes material;
};
struct DepthBox {
    uint32_t ckid;
    float lo[3], hi[3];
};
struct LevelMessage {
    uint32_t frames_done = 0;
    std::vector<std::string> strings;
    std::vector<Entity> entities;
    std::vector<Group> groups;
    std::vector<ArrayDef> arrays;
    std::vector<PhysRecord> phys;
    std::vector<Mesh> meshes;
    std::vector<IvpObject> ivp;
    std::vector<DepthBox> depth;
    std::vector<std::pair<uint32_t, std::optional<std::string>>> watch;
    std::vector<std::pair<uint32_t, uint32_t>> entity_meshes;
    std::vector<std::string> unknown;
};

// One LVL1 message at data[offset]; returns its byte_len through *len.
LevelMessage parse_level_message(const char *data, size_t n, size_t *len);
// Concatenated LVL1 messages (the .inc file); a trailing incomplete message is left; *used = bytes consumed.
std::vector<LevelMessage> parse_level_stream(const char *data, size_t n, size_t *used);

// ---------------------------------------------------------------------------------------------- cells

struct CellValue {
    enum Kind { None, Int, Float, String, Matrix, Bool, Raw } kind = None;
    int64_t i = 0;
    double f = 0;
    float m[16] = {};
    Bytes s;
    // Python int(value): Int / Bool / Float (truncated); false for the rest (the caller's error / default)
    bool to_int(int64_t &out) const;
};
// protocol.decode_cell
CellValue decode_cell(const Bytes &data, uint8_t col_type, const std::optional<std::string> &param_type);

}  // namespace ballrain
