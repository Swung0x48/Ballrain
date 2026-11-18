#include "ballrain/pool_task.h"

#include <cstdlib>
#include <filesystem>

#include "ballrain/sha256.h"
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

bool plain_ascii(const std::string &s) {
    for (unsigned char c : s)
        if (c >= 0x80) return false;
    return true;
}

static std::string parent_of(const std::string &p) {
    size_t i = p.find_last_of("/\\");
    return i == std::string::npos ? std::string(".") : p.substr(0, i);
}

static void copy_tree(const std::string &from, const std::string &to) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(fs::u8path(to), ec);
    fs::copy(fs::u8path(from), fs::u8path(to), fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
    if (ec) throw Error("cannot copy " + from + " for the engine: " + ec.message());
}

std::string engine_work_dir(const std::string &data_dir) {
    std::string d = data_dir + "/engine-work";  // a subdirectory of the root (BALLRAIN_WORK_DIR moves all of it)
    platform::make_dirs(d);
    if (plain_ascii(d)) return d;
    std::string sp = platform::short_path(d);
    if (plain_ascii(sp)) return sp;
    throw Error("你的 Ballance 目录或客户端路径里有中文，客户端要把引擎用的文件复制到一个纯英文路径，但 " + d +
                " 没有纯英文的形式。请把 Ballance 和客户端放到纯英文路径下，或者设置环境变量 BALLRAIN_WORK_DIR，"
                "让客户端的整个数据目录放到一个纯英文的文件夹（例如 C:\\ballrain-work）。");
}

LocalEngine engine_view(const LocalEngine &le, const Json &game_files, const std::string &data_dir,
                        std::string *note) {
    static std::mutex mu;  // instances share the copies: one copies, the others wait and reuse it
    std::lock_guard<std::mutex> g(mu);
    LocalEngine out = le;
    std::string why;
    if (!plain_ascii(le.binary)) {
        std::string sp = platform::short_path(le.binary);
        if (plain_ascii(sp)) {
            out.binary = sp;
        } else {  // the engine's own directory (its DLLs, ENGINE_VERSION), keyed by the binary's sha256
            std::string sha = file_sha256(le.binary);
            std::string d = engine_work_dir(data_dir) + "/engine-" + sha.substr(0, 16);
            std::string name = le.binary.substr(le.binary.find_last_of("/\\") + 1);
            if (!file_exists(d + "/" + name) || file_sha256(d + "/" + name) != sha) copy_tree(parent_of(le.binary), d);
            out.binary = d + "/" + name;
            why += "引擎路径含非 ASCII 字符，已复制到 " + d + "；";
        }
        if (!plain_ascii(out.binary)) throw Error("引擎没有纯英文路径可用（可以设置 BALLRAIN_WORK_DIR，把客户端的数据目录放到纯英文路径）");
    }
    if (!plain_ascii(le.root)) {
        std::string sp = platform::short_path(le.root);
        if (plain_ascii(sp)) {
            out.root = sp;
        } else {  // the entries the engine reads, keyed by their hashes (hello's game_files)
            std::string key = Sha256::hex(game_files.dump(true));
            std::string d = engine_work_dir(data_dir) + "/game-" + key.substr(0, 16);
            if (!file_exists(d + "/.complete")) {
                platform::make_dirs(d);
                for (const char *n : {"base.cmo", "Database.tdb"}) {
                    Bytes b;
                    if (!read_file(le.root + "/" + n, b)) throw Error(std::string("cannot read ") + n);
                    write_file(d + "/" + n, b);
                }
                for (const char *n : {"3D Entities", "Text"}) copy_tree(le.root + "/" + n, d + "/" + n);
                if (le.kind == "2002") {
                    platform::make_dirs(d + "/Bin"), platform::make_dirs(d + "/BuildingBlocks");
                    for (const char *n : {"Bin/VxMath.dll", "BuildingBlocks/physics_RT.dll"}) {
                        Bytes b;
                        if (!read_file(le.root + "/" + n, b)) throw Error(std::string("cannot read ") + n);
                        write_file(d + "/" + n, b);
                    }
                }
                write_file(d + "/.complete", key);
            }
            out.root = d;
            why += "Ballance 目录路径含非 ASCII 字符，引擎用的四项游戏文件已复制到 " + d + "；";
        }
        if (!plain_ascii(out.root)) throw Error("游戏文件没有纯英文路径可用（可以设置 BALLRAIN_WORK_DIR，把客户端的数据目录放到纯英文路径）");
    }
    if (note) *note = why;
    return out;
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
    std::string path_dir, branch_dir, reach_dir;
    {
        std::lock_guard<std::mutex> g(cache_mu);
        pol = cache.policy(task.policy_sha);
        starts = cache.starts(task.asset_starts);
        path_dir = cache.path_dir(task.asset_path, task.level);
        FeatureConfig pins = FeatureConfig::from_json(pol.manifest.at("features"));
        if (pins.has_branch() != !task.asset_branch.empty())
            throw Error(pins.has_branch() ? "the policy follows a route branch: the task has no branch pack"
                                          : "a branch pack for a policy without a route branch");
        if (pins.has_branch()) branch_dir = cache.branch_dir(task.asset_branch, task.level, pins);
        if (pins.reach) {  // f7: the fields of the level's geometries (a geometry without one fails the task)
            if (task.asset_reach.empty()) throw Error("the policy has the reach block: the task has no reach fields");
            reach_dir = cache.reach_dir(task.asset_reach, task.level);
        }
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
    fc.reach_dir = reach_dir;
    if (fc.has_branch()) {  // the learner's route branch, from the task's pack (never a local directory)
        check_branch_dir(branch_dir, task.level, fc);
        fc.branch_dir = branch_dir;
    }
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
