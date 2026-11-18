// One pool task on the native env (a port of ballrain/pool/rollout.py: TaskRunner, ChunkAssembler, sample_task --
// the code the gateway's verifier replays with): the worker's training reward (split bonus, gates, frame cap, stall
// limit with stall_terminal / stall_compensate), the task-level ends, the engine-state chain, the chunks.
#pragma once

#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "ballrain/chunk.h"
#include "ballrain/engine_chain.h"
#include "ballrain/env.h"
#include "ballrain/policy.h"

namespace ballrain {

struct Task {  // protocol.TASK
    std::string task_id, kind = "sample", run, policy_sha, record, record_sha256, engine;
    int level = 1;
    int64_t start_frame = 0, ball_order_seed = 0, chunk_steps = 256, max_steps = 4096, max_frames = 60000;
    int64_t deadline_s = 3600, issued_at = 0;
    bool greedy = false;
    uint64_t sample_seed = 0;
    std::string asset_starts, asset_path;
    std::optional<RestStart> rest;  // format 3: an at-rest start after the hand-over (server-drawn)
    uint64_t life_seed = 0;          // format 3: the per-life draws (ballrain/pool/counters.py)
    std::optional<int64_t> respawn_frontier;  // format 3: the curriculum frontier of the early-respawn rule
    static Task from_json(const Json &t);
};

// WorkerSpec fields of the manifest's "rollout" (weights.ROLLOUT_FIELDS)
struct RolloutSpec {
    int64_t chunk = 128;
    double gamma = 0.997, cap_mult = 2.5, cap_extra_s = 10.0, stall_s = 10.0;
    std::optional<double> respawn_stall_s;
    double stall_progress = 5.0;
    bool stall_terminal = false;
    double stall_extra = 0.0;
    bool stall_compensate = false;
    std::optional<double> stall_speed;
    double stall_max_s = 15.0;
    Json gates = Json::object();
    double split_k = 0.0, split_min_ref_s = 5.0, split_rest_allow_s = 1.0;
    std::vector<std::array<double, 3>> split_extra;
    Json counter_rand = Json::object();  // WorkerSpec.counter_rand (CounterRand)
    double p_early_respawn = 1.0;        // WorkerSpec.p_early_respawn (counters.skip_respawn)
    bool respawn_level_starts = false;
    static RolloutSpec from_json(const Json &r);
};

// ballrain/pool/counters.py: one set of counter-feature values per life in processes not started from level load,
// drawn from the task's life_seed (u_j = sample_u(seed, 4 life + j): j = 0 lives, 1 sr_s, 2 points)
class CounterRand {
public:
    CounterRand(const Json &spec, const FeatureExtractor &fx);
    bool active(int64_t start_frame) const { return any_ && start_frame > 0; }
    void draw(uint64_t seed, int64_t life);  // the set of life `life`
    void clear() { ops_.clear(); }
    void apply(std::vector<float> &o) const;

private:
    struct Op { int index; bool set; double value; };
    bool any_ = false;
    int i_lives_ = -1, i_sr_ = -1, i_points_ = -1;
    std::vector<double> values_, cdf_;
    double sr_lo_ = 0, sr_hi_ = 0, pt_lo_ = 0, pt_hi_ = 0;
    bool has_lives_ = false, has_sr_ = false, has_pt_ = false;
    std::vector<Op> ops_;
};

struct EngineFailure : Error {
    explicit EngineFailure(const std::string &m) : Error(m) {}
};

struct StepResult {
    int action = 0;
    float reward = 0, env_reward = 0;
    uint8_t flags = 0;
    int64_t frames = 0;
    int sector = 0;
    double sr_ms = 0;
    double arc = 0;  // NaN: no projection
    std::vector<float> next_obs;
    std::string end;  // "" while the task goes on
    Bytes digest;     // engine digest of the next row
};

class TaskRunner {
public:
    // env: built for the task (start record, start frame, ball order, the manifest's env / features / reward)
    TaskRunner(BallanceEnv &env, const TasStarts *starts, const Task &task, const RolloutSpec &spec, double delta_ms);
    std::vector<float> start();  // the first row (EngineFailure when the engine fails)
    // one step; stop: "" or a client end ("stopped" / "deadline") consulted after the step
    StepResult step(int a, const std::function<std::string()> &stop = nullptr);
    int64_t steps = 0;
    std::string end;
    Bytes digest;  // engine digest of the current row
    int64_t frames = 0;
    int episodes = 0;

private:
    struct Episode {
        std::string kind;
        int64_t start_frame = 0;
        int start_sector = 0;
        double start_sr_ms = 0;
        std::optional<int64_t> cap_frames;
        double mark_s = -INFINITY;
        int64_t mark_frame = 0, move_frame = 0;
        int max_sector_seen = 0;
        bool has_seg = false;
        double seg_agent = 0, seg_ref = 0;
        std::optional<int64_t> seg_k0;
        std::set<std::string> gates_hit;
        std::optional<bool> cap_branch;  // the branch flag the cap was last computed for (worker ep.d["cap_branch"])
        bool stall_branch = false;       // the route the stall odometer (mark_s) is measured on (worker ep.stall_branch)
    };
    void new_episode(const StepInfo &info);
    std::optional<int64_t> cap_frames(const StepInfo &info) const;
    double tas_igt(int64_t k) const;
    std::optional<double> tas_cross(int sector) const;
    std::optional<int64_t> tas_cross_frame(int sector) const;
    void split_start(Episode &ep, const StepInfo &info);
    std::vector<float> new_life(std::vector<float> obs);
    bool skip_respawn(int sector) const;
    double split_bonus(Episode &ep, const StepInfo &info, bool finished);
    double gate(Episode &ep, const double pos[3], const double vel[3]);

    BallanceEnv &env_;
    const TasStarts *st_;
    Task task_;
    RolloutSpec sp_;
    double delta_ms_;
    int64_t lf_ = 0;
    double rec_ms_ = 1000.0 / 264.0;
    int64_t stall_all_, stall_rsp_, stall_max_;
    int64_t f_hand_ = 0;
    Episode ep_;
    EngineDigest dg_;
    StepInfo info_;
    CounterRand cr_;
    int64_t lives_ = 0;
};

// rollout.ChunkAssembler: a task's steps cut into encoded chunks of task.chunk_steps steps
class ChunkAssembler {
public:
    ChunkAssembler(const Task &task, int n_obs, int n_act, const std::string &policy_sha);
    void begin(const std::vector<float> &obs, float value, const Bytes &digest);
    void add(float logp, const StepResult &st, float next_value);
    size_t pending() const { return actions_.size(); }
    bool ready() const { return int64_t(pending()) >= task_.chunk_steps; }
    Bytes cut(const std::string &end = "");
    int64_t seq = 0, step0 = 0;

private:
    Task task_;
    int n_obs_, n_act_;
    std::string sha_;
    Bytes prev_;
    std::vector<std::vector<float>> rows_;
    std::vector<Bytes> digests_;
    std::vector<float> values_, rewards_, logp_, env_rewards_, arc_;
    std::vector<uint8_t> actions_, flags_, sector_;
    std::vector<uint16_t> frames_;
    std::vector<double> sr_ms_;
};

// rollout.sample_task: play the task with the policy (the task's deterministic draws), hand every encoded chunk to
// on_chunk(data, final) as soon as it is complete; returns the end. stop(): "" or "stopped" / "deadline". An engine
// failure right after a chunk was cut leaves no final chunk (end "error", no final on_chunk): the caller closes the
// task with task_end.
// choose (tests only): the action of step t instead of the policy's draw (forced actions; a gateway refuses them)
std::string sample_task(TaskRunner &runner, const Policy &policy, const Task &task,
                        const std::function<void(const Bytes &, bool final)> &on_chunk,
                        const std::function<std::string()> &stop = nullptr,
                        const std::function<int(int64_t, const std::vector<float> &)> &choose = nullptr);

}  // namespace ballrain
