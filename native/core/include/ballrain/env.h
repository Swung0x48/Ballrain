// One game process as RL episodes (a port of the parts of ballrain/env.py BallanceEnv a pool task uses: SR mode,
// arrow actions, world actions, held-key re-press, the camera guard, the level-1 intro, per-life respawns).
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ballrain/engine_chain.h"
#include "ballrain/features.h"
#include "ballrain/json.h"
#include "ballrain/reward.h"
#include "ballrain/session.h"
#include "ballrain/starts.h"

namespace ballrain {

extern const uint32_t ARROWS9[9];

struct EnvConfig {
    int level = 1;
    float delta = float(1000.0 / 132.0);  // np.float32(delta_ms)
    int action_repeat = 1;
    bool camera_actions = false, world_actions = false;
    std::optional<bool> repress_stale_keys;
    double camera_quarter_tol = 1e-4, camera_turn_grace_ms = 400.0;
    int esc_frames = 4;
    bool mask_keys_while_paused = true;
    uint32_t pause_keys = 0;
    std::optional<int64_t> max_pause_frames = 3000, max_episode_pause_frames = 20000;
    bool per_life = true;
    std::optional<int64_t> max_episode_frames, max_process_frames;
    uint32_t idle_keys = 0;
    bool skip_tutorial = true;
    uint32_t timeout_ms = 30 * 60 * 1000;
    double obs_timeout_s = 300.0, hello_timeout_s = 120.0;
    int sr_tail = 32;
    uint32_t respawn_limit_frames = 20000;
    double body_radius = 60.0;
    bool skip_render = true;
    bool start_check = true;  // compare the hand-over state of a TAS start with the record; relaunch on a mismatch
    int start_retries = 3;
    std::vector<std::pair<int64_t, int64_t>> exclude_starts;  // record frames [lo, hi) never used as start states
    double rest_speed = 0.5;       // at-rest starts: braked until the horizontal speed is below this
    int64_t rest_max_frames = 600;  // the brake's (and by default the drive's) frame limit

    // The manifest's "env" block (ballrain/pool/weights.py ENV_FIELDS). Refuses what this port does not support
    // (camera / Esc / collect actions, HS mode).
    static EnvConfig from_manifest(const Json &env);
};

// An at-rest start after the hand-over at the start frame (protocol TASK "rest"; env.py _drive / _brake): with a target,
// drive the ball to (x, z) within max_frames (default rest_max_frames), then brake it below rest_speed
struct RestStart {
    bool drive = false;
    double x = 0.0, z = 0.0;
    std::optional<int64_t> max_frames;
    static std::optional<RestStart> from_json(const Json *rest);  // nullptr / null: no at-rest start
};

// info["start"]: how the current episode started
struct StartInfo {
    std::string kind = "tas";  // "tas" (a new process: TAS start k, k = 0 from level load) | "tas_rest" | "respawn"
    int64_t frame = 0;
    bool new_process = true;
    int handover_mismatches = 0;
    int64_t drive_frames = 0, rest_frames = 0;
};

struct StepInfo {
    uint32_t frames_done = 0;
    std::optional<double> sr_ms;
    int sector = 0;
    std::vector<GameTracker::Event> events;
    std::string episode_end;  // "" while the episode goes on
    bool camera_off_quarter = false;
    bool process_ended = false;
    std::optional<std::string> error;
    FeatureInfo feat;
};

class BallanceEnv {
public:
    // starts: the task's start record (nullptr: starts from level load without a record); start_frame: k (0 = from
    // level load), one of starts->valid when > 0 (an at-rest start with a drive target: any hand-over state, excluded
    // ranges too). reward: the manifest's reward (env.step's reward). rest: every process starts at rest after k.
    BallanceEnv(EnvConfig cfg, FeatureConfig features, EngineConfig engine, std::string run_root,
                StreamWriter *rec = nullptr, const TasStarts *starts = nullptr, int64_t start_frame = 0,
                RewardConfig reward = RewardConfig(), std::optional<RestStart> rest = std::nullopt);
    ~BallanceEnv();

    const std::vector<uint32_t> &actions() const { return actions_; }
    FeatureExtractor &features() { return fx_; }
    Session *session() { return sess_.get(); }
    bool after_death() const { return after_death_; }
    const StartInfo &start_info() const { return start_; }
    int64_t ep_frames() const { return ep_frames_; }
    Reward &reward() { return reward_; }
    const EnvConfig &config() const { return cfg_; }
    // pool tasks: the digest every session of this env feeds (reset at each launch)
    void set_digest(EngineDigest *d) { digest_ = d; }

    // A new process (or, after a death in a per-life run, the respawn in the same one). relaunch=false: when the
    // process ended (e.g. GameOver during the respawn), returns nullopt instead of starting another.
    std::optional<std::vector<float>> reset(bool relaunch = true, StepInfo *info = nullptr);
    // info.reward: the env's reward of this step (rewards.Reward + the camera-guard penalty)
    std::vector<float> step(int action, bool &terminated, bool &truncated, StepInfo &info, double *reward = nullptr);
    void close();

    uint64_t processes = 0;

private:
    void start_process();
    bool launch(int64_t k);  // one process brought to control; false: the hand-over differs from the record
    bool handover_ok(int64_t k) const;
    void init_held(int64_t k);
    void drive(const RestStart &r);
    void brake();
    void intro();
    void finish_process();
    std::vector<float> features(StepInfo *info);
    std::optional<int> cam_quarter() const;
    double camera_dev() const;
    bool camera_off_quarter() const;
    void invalidate_held();
    void set_held(uint32_t keys);
    uint32_t stale(uint32_t keys) const;
    std::string esc();
    bool pause_budget_spent() const;

    EnvConfig cfg_;
    EngineConfig eng_;
    std::string run_root_;
    StreamWriter *rec_;
    FeatureExtractor fx_;
    std::vector<uint32_t> actions_;
    std::unique_ptr<Session> sess_;
    bool repress_;
    bool after_death_ = false;
    int64_t ep_steps_ = 0, ep_frames_ = 0, ep_pause_frames_ = 0;
    uint32_t held_ = 0;
    std::optional<int> press_q_[4];  // per arrow bit: the camera quarter when it went down (nullopt = stale)
    bool press_known_[4] = {false, false, false, false};
    double ms_since_turn_;
    const TasStarts *starts_;
    int64_t start_frame_;
    std::optional<RestStart> rest_;
    Reward reward_;
    StartInfo start_;
    EngineDigest *digest_ = nullptr;
};

// env.world_to_camera_keys
uint32_t world_to_camera_keys(uint32_t keys, const float cam_matrix[12]);

}  // namespace ballrain
