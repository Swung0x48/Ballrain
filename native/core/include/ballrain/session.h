// One headless engine process in agent mode (a port of ballrain/client.py AgentSession, ballrain/headless.py
// HeadlessLauncher and the socket half of ballrain/transport.py SocketChannel).
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ballrain/state.h"
#include "ballrain/stream.h"
#include "ballrain/wire.h"

namespace ballrain {

namespace platform {
class Listener;
class Stream;
class Process;
}  // namespace platform

struct PeerGone : Error {
    explicit PeerGone(const std::string &m) : Error(m) {}
};
struct Timeout : Error {
    explicit Timeout(const std::string &m) : Error(m) {}
};

// SocketChannel.header()
struct ChanHeader {
    uint32_t mod_state = 0, end_reason = 0, frames_done = 0, last_error = 0;
    uint64_t mod_seq = 0;
    uint32_t mod_pid = 0, obs_versions = 0;
    std::string mod_version;
};

// Where the level messages come from: the engine's files (live) or a recording (offline replay).
class LevelSource {
public:
    virtual ~LevelSource() = default;
    // AgentSession._load_level: the full file, or nullopt when it is absent or not needed (level_loaded && !full)
    virtual std::optional<Bytes> load_full(bool level_loaded, bool full) = 0;
    // AgentSession._load_increments: the unread tail from inc_pos, or nullopt when absent / not grown
    virtual std::optional<Bytes> load_inc(uint64_t inc_pos) = 0;
};

class FileLevelSource : public LevelSource {
public:
    explicit FileLevelSource(std::string levelmsg_path) : path_(std::move(levelmsg_path)) {}
    std::optional<Bytes> load_full(bool level_loaded, bool full) override;
    std::optional<Bytes> load_inc(uint64_t inc_pos) override;

private:
    std::string path_;
};

// AgentSession._ingest: level message first, then the tracker; end detection. Records the level reads when a
// writer is given.
class Ingest {
public:
    Ingest(int level, LevelSource *src, StreamWriter *rec) : tracker(level), src_(src), rec_(rec) {}
    void ingest(Observation obs, const ChanHeader &h);
    void mark_ended(const ChanHeader &h);

    GameTracker tracker;
    bool level_loaded = false;
    uint64_t inc_pos = 0;
    uint32_t frames_done = 0;
    bool ended = false;
    std::optional<uint32_t> end_reason;
    std::optional<uint64_t> trace_hash;
    std::optional<double> sr_at_levelfinish;

private:
    void load_level(bool full);
    void load_increments();
    LevelSource *src_;
    StreamWriter *rec_;
};

// The engine side of the socket protocol (docs/headless.md section 2), Python as the server.
class SocketChannel {
public:
    SocketChannel();
    ~SocketChannel();
    std::string name() const;  // "tcp:127.0.0.1:<port>" (BALLRAIN_SOCK)
    const ChanHeader &header() const { return hdr_; }
    ChanHeader wait_hello(double timeout_s);
    std::optional<Bytes> wait_obs(double timeout_s);  // the OBS payload (24-byte header + OBS1); nullopt = ended
    Bytes send_action(const Action &a);              // returns the ACT1 bytes sent
    void close();

private:
    bool recv(double deadline, uint32_t &type, Bytes &payload);
    void send(uint32_t type, const Bytes &payload);
    void apply_state(const Bytes &p);
    std::unique_ptr<platform::Listener> listener_;
    std::unique_ptr<platform::Stream> conn_;
    ChanHeader hdr_;
    uint32_t py_seq_ = 0;
    bool closed_peer_ = false;
};

// HeadlessConfig: the engine binary, the game folder, SimTool's frame recipe flags (only where they differ from the
// defaults), the ball-pair order, extra environment (e.g. BALLRAIN_HL_PHYS2002=0).
struct EngineConfig {
    std::string binary, root;
    std::string ball_order = "ckid";
    std::optional<int64_t> boot_ticks, hidden_ticks, reset_nmc, reset_seed, reset_frame;
    bool legacy_edit = true, reset_precision = false, orig_random = false;
    bool log = true;
    std::vector<std::pair<std::string, std::string>> extra_env;
};

struct SessionConfig {
    int level = 1;
    std::string out_dir;
    uint32_t timeout_ms = 30 * 60 * 1000;
    std::optional<uint32_t> max_frames;
    double body_radius = 60.0;
    int sr_tail = 32;
    bool skip_render = true, exit_game = true, events_txt = true;
    std::string prefix;  // raw frames replayed before hand-over ("" = none)
    double hello_timeout_s = 120.0, obs_timeout_s = 300.0;
};

std::vector<std::string> engine_argv(const EngineConfig &e);
// the child environment: the current one without BALLRAIN_*, then the session's variables, then extra_env
std::vector<std::pair<std::string, std::string>> engine_env(const SessionConfig &s, const EngineConfig &e,
                                                            const std::string &sock_name);
std::string py_float_str(double v);  // Python repr(float) (BALLRAIN_BODY_RADIUS)

class Session {
public:
    Session(SessionConfig cfg, EngineConfig eng, StreamWriter *rec = nullptr);
    ~Session();
    const Observation &start();
    const Observation *step(const Action &a);
    const Observation *act(const std::vector<Frame> &frames, uint32_t repeat, uint64_t stop_mask, uint16_t flags = 0,
                           uint32_t obs_mask = 0);
    const Observation *run_until(uint32_t keys, uint64_t stop_mask, float delta, uint32_t chunk = 4096,
                                 uint32_t limit = 200000);
    void stop(double wait_s = 30.0);
    void close(bool kill = false);

    GameTracker &tracker() { return ing_->tracker; }
    const GameTracker &tracker() const { return ing_->tracker; }
    bool ended() const { return ing_->ended; }
    uint32_t frames_done() const { return ing_->frames_done; }
    const Observation *obs() const { return ing_->tracker.obs ? &*ing_->tracker.obs : nullptr; }
    std::optional<uint64_t> trace_hash() const { return ing_->trace_hash; }
    std::optional<uint32_t> end_reason() const { return ing_->end_reason; }
    const ChanHeader &hello() const { return hello_; }
    int pid() const;

    struct Timing { double launch = 0, hello = 0, first_obs = 0; } timing;

private:
    const Observation *next_obs();
    void finish_process(double wait_s);
    SessionConfig cfg_;
    EngineConfig eng_;
    StreamWriter *rec_;
    std::string levelmsg_;
    std::unique_ptr<FileLevelSource> src_;
    std::unique_ptr<Ingest> ing_;
    std::unique_ptr<SocketChannel> chan_;
    std::unique_ptr<platform::Process> proc_;
    ChanHeader hello_;
};

}  // namespace ballrain
