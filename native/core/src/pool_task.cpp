#include "ballrain/pool_task.h"

#include <filesystem>

#include "platform.h"

namespace ballrain {

static void keep_engine_log(const std::string &run_dir) {
    // the newest run's headless.log next to the run directory (the run directory itself is removed)
    std::string best;
    for (auto &n : platform::list_dir(run_dir))
        if (n.rfind("run-", 0) == 0 && file_exists(run_dir + "/" + n + "/headless.log") && n > best) best = n;
    if (best.empty()) return;
    Bytes b;
    if (read_file(run_dir + "/" + best + "/headless.log", b)) {
        if (b.size() > 256 * 1024) b = b.substr(b.size() - 256 * 1024);
        write_file(run_dir + ".headless.log", b);
    }
}

static void remove_tree(const std::string &d) {
    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::u8path(d), ec);  // a task keeps nothing locally
}

TaskOutcome play_task(const Task &task, PoolCache &cache, std::mutex &cache_mu, const LocalEngine &le,
                      const std::string &run_dir, const std::function<void(int64_t, const Bytes &, bool)> &on_chunk,
                      const std::function<std::string()> &stop, const std::function<void(int64_t)> &on_step) {
    if (task.engine != le.kind) throw Error("task engine " + task.engine + ", local engine " + le.kind);
    Policy pol;
    TasStarts starts;
    std::string path_dir;
    {
        std::lock_guard<std::mutex> g(cache_mu);
        pol = cache.policy(task.policy_sha);
        starts = cache.starts(task.asset_starts);
        path_dir = cache.path_dir(task.asset_path, task.level);
    }
    const Json &m = pol.manifest;
    EnvConfig ec = EnvConfig::from_manifest(m.at("env"));
    if (ec.level != task.level || m.at("env").get_string("start_record", "") != task.record)
        throw Error("task level / record differ from the policy's");
    if (starts.record != task.record || starts.record_sha256 != task.record_sha256)
        throw Error("the starts pack is not the task's record");
    starts.apply_camera_filter(ec.camera_quarter_tol, ec.camera_turn_grace_ms);
    starts.exclude(ec.exclude_starts);
    FeatureConfig fc = FeatureConfig::from_json(m.at("features"));
    if (fc.path_sha256 != task.asset_path) throw Error("the task's reference line is not the policy's");
    fc.path_dir = path_dir;
    bool drive = task.rest && task.rest->drive;  // an at-rest start with a target may hand over in an excluded range
    if (task.start_frame > 0 && !(drive ? starts.expected(task.start_frame).has_value()
                                        : starts.index_of(task.start_frame).has_value()))
        throw Error("start frame is not a valid start frame of the record");
    EngineConfig eng;
    eng.binary = le.binary, eng.root = le.root, eng.background = le.background;
    eng.ball_order = "retail:" + std::to_string(task.ball_order_seed);
    const Json &eo = m.at("engine_options");
    auto opt = [&](const char *k, std::optional<int64_t> &dst) {
        const Json *j = eo.find(k);
        if (j && !j->is_null()) dst = j->as_int();
    };
    opt("boot_ticks", eng.boot_ticks), opt("hidden_ticks", eng.hidden_ticks), opt("reset_nmc", eng.reset_nmc);
    opt("reset_seed", eng.reset_seed), opt("reset_frame", eng.reset_frame);
    eng.legacy_edit = eo.get_bool("legacy_edit", true);
    eng.reset_precision = eo.get_bool("reset_precision", false);
    eng.orig_random = eo.get_bool("orig_random", false);
    for (auto &kv : le.extra_env) {
        if (kv.first.compare(0, 9, "BALLRAIN_") != 0 || kv.first.find("BALL_ORDER") != std::string::npos)
            throw Error("engine environment: BALLRAIN_* only, not the ball order");
        eng.extra_env.push_back(kv);
    }
    remove_tree(run_dir);
    TaskOutcome out;
    double t0 = platform::monotonic_s();
    bool failed = false;
    try {
        BallanceEnv env(ec, fc, eng, run_dir, nullptr, &starts, task.start_frame,
                        RewardConfig::from_json(m.at("reward")), task.rest);
        if (env.features().size() != pol.n_obs) throw Error("the env's observation size differs from the policy's");
        for (size_t i = 0; i < env.actions().size(); ++i)
            if (env.actions()[i] != pol.action_keys[i]) throw Error("the env's action set differs from the policy's");
        TaskRunner runner(env, &starts, task, RolloutSpec::from_json(m.at("rollout")),
                          m.at("env").at("delta_ms").as_double());
        int64_t seq = 0;
        out.end = sample_task(
            runner, pol, task,
            [&](const Bytes &data, bool final) {
                on_chunk(seq++, data, final);
                out.final_chunk = final;
            },
            [&]() -> std::string {
                if (on_step) on_step(runner.steps);
                return stop ? stop() : "";
            });
        env.close();
        out.steps = runner.steps, out.frames = runner.frames, out.episodes = runner.episodes, out.chunks = seq;
        failed = out.end == "error";
        if (failed) out.error = "the engine failed during the task";
    } catch (...) {
        keep_engine_log(run_dir);
        remove_tree(run_dir);
        throw;
    }
    if (failed) keep_engine_log(run_dir);
    remove_tree(run_dir);
    out.wall_s = platform::monotonic_s() - t0;
    return out;
}

}  // namespace ballrain
