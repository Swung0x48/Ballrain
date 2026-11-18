// One pool task offline (tests): the task JSON, the policy, the starts pack and the reference line from local files,
// the chunks written to <out>/<seq:06d>.brpc. tools/native/task_diff.py replays them with the Python verifier.
//
//   nc_task --task task.json --manifest <sha>.json --starts <pack>.npz --path-dir <dir of L01.npz>
//           --engine BallanceMMOSimTool.exe --root <game> --out <dir>
//           [--actions policy|idle|sticky:<seed>] [--stop-after N] [--engine-env NAME=VALUE ...]
//
// --actions: policy = the task's seeded draws (what a gateway accepts); idle = action 0 every step; sticky:<seed> =
// random actions held 2-39 steps (a test-only source: deaths, respawns, stalls; the verifier's draw check refuses it).
// --stop-after N: the player stops after step N (end "stopped").
#include <cstdio>
#include <cstring>
#include <string>

#include "ballrain/sha256.h"
#include "ballrain/pool_task.h"
#include "ballrain/task_runner.h"
#include "platform.h"

using namespace ballrain;

static std::string arg(int argc, char **argv, const char *name, const char *dflt = nullptr) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    if (!dflt) throw Error(std::string("missing ") + name);
    return dflt;
}

int main(int argc, char **argv) {
    try {
        Bytes tj;
        if (!read_file(arg(argc, argv, "--task"), tj)) throw Error("cannot read the task");
        Task task = Task::from_json(Json::parse(tj));
        std::string mpath = arg(argc, argv, "--manifest");
        Json m = Policy::read_manifest(mpath);
        Policy pol = Policy::load(mpath.substr(0, mpath.size() - 5) + ".npz", m, task.policy_sha);
        std::string pack = arg(argc, argv, "--starts");
        TasStarts starts = TasStarts::load_pack(pack, file_sha256(pack));
        EnvConfig ec = EnvConfig::from_manifest(m.at("env"));
        starts.apply_camera_filter(ec.camera_quarter_tol, ec.camera_turn_grace_ms);
        starts.exclude(ec.exclude_starts);
        FeatureConfig fc = FeatureConfig::from_json(m.at("features"));
        fc.path_dir = arg(argc, argv, "--path-dir");
        fc.branch_dir = arg(argc, argv, "--branch-dir", "");  // the ball-type diversion assets (tests: out of band)
        fc.reach_dir = arg(argc, argv, "--reach-dir", "");  // the f7 fields (tests; the client: the task's assets.reach)
        std::string bpack = arg(argc, argv, "--branch-pack", "");  // as the client: the task's branch pack
        if (fc.has_branch() != !bpack.empty())
            throw Error(fc.has_branch() ? "the policy follows a route branch: give --branch-pack"
                                        : "--branch-pack for a policy without a route branch");
        if (!bpack.empty()) {
            fc.branch_dir = install_branch_pack(bpack, arg(argc, argv, "--out") + "/cache", task.level, file_sha256(bpack), fc);
            check_branch_dir(fc.branch_dir, task.level, fc);
        }
        EngineConfig eng;
        eng.binary = platform::absolute_path(arg(argc, argv, "--engine"));
        eng.root = platform::absolute_path(arg(argc, argv, "--root"));
        {  // as the client: non-ASCII paths through the engine work directory
            LocalEngine le;
            le.binary = eng.binary, le.root = eng.root, le.kind = task.engine;
            std::string note;
            le = engine_view(le, hash_game_files(eng.root, task.engine), arg(argc, argv, "--out"), &note);
            if (!note.empty()) std::fprintf(stderr, "%s\n", note.c_str());
            eng.binary = le.binary, eng.root = le.root;
        }
        eng.ball_order = "retail:" + std::to_string(task.ball_order_seed);
        for (int i = 1; i + 1 < argc; ++i)
            if (std::strcmp(argv[i], "--engine-env") == 0) {
                std::string kv = argv[i + 1];
                eng.extra_env.push_back({kv.substr(0, kv.find('=')), kv.substr(kv.find('=') + 1)});
            }
        std::string out = arg(argc, argv, "--out");
        platform::make_dirs(out + "/runs");
        BallanceEnv env(ec, fc, eng, out + "/runs", nullptr, &starts, task.start_frame, RewardConfig::from_json(m.at("reward")),
                        task.rest);
        TaskRunner runner(env, &starts, task, RolloutSpec::from_json(m.at("rollout")), m.at("env").at("delta_ms").as_double());
        std::string mode = arg(argc, argv, "--actions", "policy");
        int64_t stop_after = std::atoll(arg(argc, argv, "--stop-after", "0").c_str());
        uint64_t rng = 0;
        int sticky = 0, left = 0;
        if (mode.compare(0, 7, "sticky:") == 0) rng = std::strtoull(mode.c_str() + 7, nullptr, 10) * 2654435761ull + 1;
        auto next = [&]() {
            rng = splitmix64(rng);
            return rng;
        };
        std::function<int(int64_t, const std::vector<float> &)> choose;
        if (mode == "idle") choose = [](int64_t, const std::vector<float> &) { return 0; };
        else if (mode.compare(0, 7, "sticky:") == 0)
            choose = [&](int64_t, const std::vector<float> &) {
                if (left <= 0) sticky = int(next() % 9), left = 2 + int(next() % 38);
                --left;
                return sticky;
            };
        else if (mode != "policy") throw Error("--actions policy|idle|sticky:<seed>");
        int64_t seq = 0;
        std::string end = sample_task(
            runner, pol, task,
            [&](const Bytes &data, bool) {
                char n[32];
                std::snprintf(n, sizeof n, "/%06lld.brpc", (long long)seq++);
                write_file(out + n, data);
            },
            [&]() -> std::string { return stop_after > 0 && runner.steps >= stop_after ? "stopped" : ""; }, choose);
        env.close();
        Json r = Json::object();
        r.set("end", Json::string(end));
        r.set("steps", Json::integer(runner.steps));
        r.set("chunks", Json::integer(seq));
        r.set("episodes", Json::integer(runner.episodes));
        r.set("frames", Json::integer(runner.frames));
        std::printf("%s\n", r.dump(true).c_str());
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "nc_task: %s\n", e.what());
        return 2;
    }
}
