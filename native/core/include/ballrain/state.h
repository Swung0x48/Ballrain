// Accumulated game state (a port of ballrain/state.py): level message contents, data arrays, event-driven flags.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "ballrain/wire.h"

namespace ballrain {

extern const char *const G_FLOORS;
extern const char *const G_RAILS;
extern const char *const G_STOPPER;
extern const char *const G_DEPTH;
constexpr int SMALL_BALL_POINTS = 20;

class ArrayStore {
public:
    void define(const ArrayDef &a, bool seed_cells = true);
    void apply(const std::vector<ArrCell> &cells);
    std::optional<uint32_t> ckid(const std::string &name) const;
    int rows(const std::string &name) const;
    // decoded cell; nullopt when the array / cell is unknown (Python's default)
    std::optional<CellValue> get(const std::string &name, int row, int col) const;
    int64_t get_int(const std::string &name, int row, int col, int64_t dflt) const;

private:
    static uint64_t key(uint32_t a, uint32_t r, uint32_t c) { return (uint64_t(a) << 32) | (uint64_t(r) << 16) | c; }
    struct Cell { uint8_t type; Bytes data; };
    std::unordered_map<uint64_t, Cell> cells_;
    std::map<std::string, uint32_t> names_;
    std::unordered_map<uint32_t, std::vector<ArrayColumn>> columns_;
    std::unordered_map<uint32_t, int> nrows_;
};

struct MeshEntry {
    Mesh mesh;
    uint64_t serial = 0;  // bumped when the entry is replaced (identity for caches)
};

class LevelData {
public:
    std::unordered_map<uint32_t, Entity> entities;
    std::map<std::string, uint32_t> by_name;
    std::map<std::string, Group> groups;
    std::unordered_map<uint32_t, std::set<std::string>> ent_groups;
    std::map<uint32_t, PhysRecord> phys;               // seq -> record
    std::unordered_map<uint32_t, PhysRecord> phys_by_ent;  // ckid -> latest record
    std::map<std::pair<uint32_t, uint32_t>, MeshEntry> meshes;  // (ckid, mesh) -> last merged export
    std::unordered_map<uint32_t, std::vector<uint32_t>> mesh_order;  // ckid -> mesh ids in first-insertion order
    std::map<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, Mesh>> phys_meshes;  // exported with Physicalize
    std::unordered_map<uint32_t, IvpObject> ivp;
    std::vector<DepthBox> depth;
    std::vector<std::pair<uint32_t, std::optional<std::string>>> watch;
    int version = 0;

    void merge(const LevelMessage &lm);

    const std::string *name(uint32_t ckid) const;
    bool in_group(uint32_t ckid, const std::string &group) const;
    int32_t contact_id(uint32_t ckid) const;
    bool is_physicalized(uint32_t ckid) const;
    bool is_fixed(uint32_t ckid) const;
    std::vector<const MeshEntry *> meshes_of(uint32_t ckid) const;
    std::vector<uint32_t> static_entities() const;
    uint64_t static_key() const;  // content hash of the static collision geometry (cache key)
    bool entity_position(const std::string &name, double out[3]) const;
    bool physical_position(const std::string &name, double out[3]) const;

private:
    uint64_t serial_ = 0;
};

struct Flags {
    bool ball_nav = false, cam_nav = false, counter = false, paused = false, started = false, finished = false,
         game_over = false, ball_off = false;
};

struct FollowerState {
    int total = 0;
    std::map<std::string, int> per_source;
    std::vector<std::array<double, 3>> positions;
};

class GameTracker {
public:
    explicit GameTracker(int level);

    int level_no;
    LevelData level;
    ArrayStore arrays;
    Flags flags;
    struct Event { uint32_t frame; uint16_t code, param; };
    std::vector<Event> events;
    uint32_t frames_done = 0;
    std::optional<Observation> obs;
    int checkpoints = 0, extra_points = 0, life_ups = 0, deaths = 0;
    std::optional<uint32_t> levelfinish_frame, last_checkpoint_frame, control_start_frame, paused_since;
    std::map<uint32_t, std::pair<int, std::array<float, 3>>> watch;  // ckid -> (visible, position)
    uint64_t serial;  // identity of this tracker (one per game process)

    void merge_level(const LevelMessage &lm);
    void update(Observation obs);  // takes ownership

    int points() const;
    int lives() const;
    int life_bonus() const;
    int sector() const;
    bool transforming() const { return trafo_; }
    std::optional<uint32_t> active_ball() const;
    std::optional<double> sr_ms() const;
    int ball_type() const;
    bool control_valid() const;
    FollowerState follower_state() const;
    int followers() const;
    std::optional<double> nearest_follower(const double pos[3]) const;
    uint32_t frames_since_checkpoint() const;
    uint32_t paused_frames() const;

private:
    bool is_ball_record(uint32_t seq) const;
    const std::map<uint32_t, std::pair<std::string, int>> &follower_slots() const;
    bool trafo_ = false;
    mutable std::map<uint32_t, std::pair<std::string, int>> fslots_;
    mutable int fslots_version_ = -1;
};

}  // namespace ballrain
