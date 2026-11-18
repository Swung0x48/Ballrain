// Episode runner: plays one pool-style task from level load (spawn) with a policy export on the headless engine.
// Deaths respawn in the same process; the task ends when the process ends or after --max-steps. The action of step t
// is the inverse-cdf draw at u_t = sample_u(seed, t), as ballrain/pool/rollout.py sample_task.
//
//   nc_runner --manifest <sha>.json --engine BallanceMMOSimTool.exe --root <game folder> --path-dir <dir of L01.npz>
//             --out <dir> [--seed 1] [--ball-seed 12] [--max-steps 20000] [--x87soft 1]
//
// Writes <out>/stream.bin (tools/native/golden.py format: diff it against a Python run with stream_diff.py) and
// <out>/result.json (steps, ends, steps/s).
#include <cstdio>
#include <cstring>
#include <string>

#include "ballrain/env.h"
#include "ballrain/policy.h"
#include "ballrain/stream.h"
#include "platform.h"

using namespace ballrain;

static std::string arg(int argc, char **argv, const char *name, const char *dflt = nullptr) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    if (!dflt) throw Error(std::string("missing ") + name);
    return dflt;
}

static void engine_options(EngineConfig &e, const Json &o) {
    auto opt = [&](const char *k, std::optional<int64_t> &dst) {
        const Json *j = o.find(k);
        if (j && !j->is_null()) dst = j->as_int();
    };
    opt("boot_ticks", e.boot_ticks);
    opt("hidden_ticks", e.hidden_ticks);
    opt("reset_nmc", e.reset_nmc);
    opt("reset_seed", e.reset_seed);
    opt("reset_frame", e.reset_frame);
    e.legacy_edit = o.get_bool("legacy_edit", true);
    e.reset_precision = o.get_bool("reset_precision", false);
    e.orig_random = o.get_bool("orig_random", false);
}

int main(int argc, char **argv) {
    try {
        std::string mpath = arg(argc, argv, "--manifest");
        std::string out = arg(argc, argv, "--out");
        uint64_t seed = std::strtoull(arg(argc, argv, "--seed", "1").c_str(), nullptr, 10);
        std::string ball_seed = arg(argc, argv, "--ball-seed", "12");
        int64_t max_steps = std::atoll(arg(argc, argv, "--max-steps", "20000").c_str());
        bool x87soft = arg(argc, argv, "--x87soft", "1") == "1";
        Json manifest = Policy::read_manifest(mpath);
        Policy pol = Policy::load(mpath.substr(0, mpath.size() - 5) + ".npz", manifest);
        EnvConfig ec = EnvConfig::from_manifest(manifest.at("env"));
        FeatureConfig fc = FeatureConfig::from_json(manifest.at("features"));
        fc.path_dir = arg(argc, argv, "--path-dir");
        EngineConfig eng;
        eng.binary = arg(argc, argv, "--engine");
        eng.root = arg(argc, argv, "--root");
        eng.ball_order = "retail:" + ball_seed;
        if (x87soft) eng.extra_env.push_back({"BALLRAIN_HL_PHYS2002", "0"});
        engine_options(eng, manifest.at("engine_options"));
        if (!platform::make_dirs(out)) throw Error("cannot create " + out);
        StreamWriter rec(out + "/stream.bin");
        BallanceEnv env(ec, fc, eng, out + "/runs", &rec);
        if (env.features().size() != pol.n_obs) throw Error("feature size differs from the policy's n_obs");
        for (size_t i = 0; i < env.actions().size(); ++i)
            if (env.actions()[i] != pol.action_keys[i]) throw Error("action set differs from the policy's");
        // --actions-from <stream.bin>: take step t's action from a recorded stream's POLICY records instead of the
        // draw (replays e.g. golden.py's random-action runs, deaths and respawns included, on the native env)
        std::vector<int> forced;
        std::string from = arg(argc, argv, "--actions-from", "");
        if (!from.empty())
            for (auto &r : read_stream(from))
                if (r.kind == K_POLICY) {
                    Reader rd(r.data);
                    rd.u32(), rd.f64();
                    forced.push_back(int(rd.u32()));
                }
        double t0 = platform::monotonic_s();
        StepInfo info;
        auto obs = env.reset(true, &info);
        double t_loop0 = platform::monotonic_s();
        int64_t steps = 0, deaths = 0;
        Json ends = Json::array();
        double t_feat = 0, t_pol = 0;
        while (steps < max_steps) {
            double a0 = platform::monotonic_s();
            std::vector<float> x = pol.normalize(*obs);
            std::vector<float> lp = pol.log_probs(x);
            float v = pol.value(x);
            double u = sample_u(seed, uint64_t(steps));
            int a = sample_action(lp, u);
            if (!from.empty()) {
                if (steps >= int64_t(forced.size())) break;
                a = forced[size_t(steps)];
            }
            t_pol += platform::monotonic_s() - a0;
            Bytes p;
            put<uint32_t>(p, uint32_t(steps));
            put<double>(p, u);
            put<uint32_t>(p, uint32_t(a));
            put<float>(p, v);
            put_bytes(p, lp.data(), lp.size() * 4);
            rec.put(K_POLICY, p);
            bool term, trunc;
            obs = env.step(a, term, trunc, info);
            ++steps;
            if (info.error) {
                ends.push(Json::string("error"));
                std::fprintf(stderr, "engine failure: %s\n", info.error->c_str());
                break;
            }
            if (term || trunc) {
                ends.push(Json::string(info.episode_end));
                Json note = Json::object();
                note.set("episode_end", Json::string(info.episode_end));
                note.set("frames", Json::integer(info.frames_done));
                note.set("step", Json::integer(steps));
                rec.put(K_NOTE, note.dump(true));
                if (info.episode_end == "BallOff") ++deaths;
                if (env.after_death()) {
                    auto o2 = env.reset(false, &info);
                    if (!o2) {
                        ends.push(Json::string("process_ended"));
                        break;
                    }
                    obs = o2;
                } else {
                    break;
                }
            }
        }
        double loop_s = platform::monotonic_s() - t_loop0;
        env.close();
        rec.flush();
        Json r = Json::object();
        r.set("steps", Json::integer(steps));
        r.set("deaths", Json::integer(deaths));
        r.set("ends", ends);
        r.set("wall_s", Json::number(platform::monotonic_s() - t0));
        r.set("loop_s", Json::number(loop_s));
        r.set("steps_per_s", Json::number(double(steps) / std::max(loop_s, 1e-9)));
        r.set("policy_ms_per_step", Json::number(1000.0 * t_pol / std::max<int64_t>(steps, 1)));
        r.set("policy_sha", Json::string(pol.sha));
        write_file(out + "/result.json", r.dump(true) + "\n");
        std::printf("%s\n", r.dump(true).c_str());
        (void)t_feat;
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "nc_runner: %s\n", e.what());
        return 2;
    }
}
