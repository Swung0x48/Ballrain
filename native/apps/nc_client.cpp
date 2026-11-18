// The native volunteer client (M2: command line; the TUI, several instances and the browser login are M3).
//
//   nc_client login --gateway URL [--data-dir D]
//       login_start, shows the login code and the browser link, polls until the gateway hands out the token, which
//       is saved to <D>/token (D defaults to ./ballrain-pool-data)
//   nc_client run --gateway URL --ballance-dir G --engine BallanceMMOSimTool[.exe] [--data-dir D] [--tasks N]
//                 [--engine-kind x87soft|2002] [--engine-env NAME=VALUE ...] [--leaderboard on|off]
//       hello (engine sha256, game-file hashes), then tasks one after another: the policy, starts pack and reference
//       line from the sha-addressed cache, the task played on the local engine, every chunk posted as soon as it is
//       complete. A refusal is printed verbatim; a token-fatal one stops the client.
//
// Output: one JSON object per line on stdout (events), human-readable notes on stderr.
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "ballrain/pool_client.h"
#include "ballrain/task_runner.h"
#include "platform.h"

using namespace ballrain;

static std::atomic<bool> g_stop{false};
static void on_signal(int) { g_stop = true; }

static std::string arg(int argc, char **argv, const char *name, const char *dflt = nullptr) {
    for (int i = 2; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    if (!dflt) throw Error(std::string("missing ") + name);
    return dflt;
}

static std::vector<std::string> args_all(int argc, char **argv, const char *name) {
    std::vector<std::string> out;
    for (int i = 2; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) out.push_back(argv[i + 1]);
    return out;
}

static void event(const std::string &kind, Json fields = Json::object()) {
    Json e = Json::object();
    e.set("event", Json::string(kind));
    for (auto &kv : fields.members()) e.set(kv.first, kv.second);
    std::printf("%s\n", e.dump(true).c_str());
    std::fflush(stdout);
}

static void sleep_s(double s) {
    double end = platform::monotonic_s() + s;
    while (!g_stop && platform::monotonic_s() < end) std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

static int cmd_login(int argc, char **argv) {
    GatewayClient api(arg(argc, argv, "--gateway"));
    std::string dir = arg(argc, argv, "--data-dir", "ballrain-pool-data");
    platform::make_dirs(dir);
    Json ls = api.login_start();
    Json f = Json::object();
    f.set("user_code", ls.at("user_code"));
    f.set("browser_url", ls.at("browser_url"));
    event("login_started", f);
    std::fprintf(stderr, "登录代码：%s\n请在浏览器中打开：%s\n", ls.at("user_code").as_string().c_str(),
                 ls.at("browser_url").as_string().c_str());
    double end = platform::monotonic_s() + double(ls.at("expires_in_s").as_int());
    double poll = double(ls.at("poll_s").as_int());
    while (!g_stop && platform::monotonic_s() < end) {
        sleep_s(poll);
        Json st;
        try {
            st = api.login_poll(ls.at("login_id").as_string(), ls.at("poll_secret").as_string());
        } catch (const GatewayRefusal &e) {
            if (e.code == "rate_limited") continue;
            throw;
        }
        std::string s = st.at("status").as_string();
        if (s == "pending") continue;
        if (s == "ok") {
            write_file(dir + "/token", st.at("token").as_string());
#ifndef _WIN32
            std::error_code pe;  // the token is a credential: owner only (the Python controller writes it 0600)
            std::filesystem::permissions(dir + "/token", std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                                         std::filesystem::perm_options::replace, pe);
#endif
            Json g = Json::object();
            g.set("user", st.at("user"));
            event("login_ok", g);
            return 0;
        }
        Json g = Json::object();
        g.set("status", Json::string(s));
        if (const Json *r = st.find("reason")) {
            g.set("reason", *r);
            std::fprintf(stderr, "%s\n", r->get_string("message", "").c_str());
        }
        event("login_failed", g);
        return 1;
    }
    event("login_failed", make_message("x"));
    return 1;
}

struct Uploader {  // posts a task's chunks in order; network errors retried with backoff
    GatewayClient &api;
    std::string refusal_message;
    bool fatal = false, failed = false;
    int64_t accepted_steps = 0;
    int64_t posted = 0;   // chunks the gateway has (accepted, or a duplicate of an earlier attempt)
    bool final_posted = false;
    void post(const Bytes &data, int64_t seq, bool final = false) {
        if (failed) return;
        double delay = 2.0;
        for (int tried = 0;; ++tried) {
            try {
                Json r = api.chunk(data);
                Json f = Json::object();
                f.set("seq", r.at("seq"));
                f.set("status", r.at("status"));
                f.set("verify", r.at("verify"));
                f.set("steps", r.at("steps"));
                if (const Json *rs = r.find("reason")) f.set("reason", *rs);
                event("chunk", f);
                if (r.at("status").as_string() == "rejected" || r.get_bool("token_revoked", false)) {
                    failed = true;
                    fatal = r.get_bool("token_revoked", false);
                    if (const Json *rs = r.find("reason")) refusal_message = rs->get_string("message", "");
                } else {
                    accepted_steps += r.at("steps").as_int();
                    ++posted, final_posted = final;
                }
                return;
            } catch (const GatewayUnavailable &e) {
                if (tried >= 6) {
                    failed = true;
                    refusal_message = e.what();
                    return;
                }
                std::fprintf(stderr, "chunk %lld: %s; retrying\n", (long long)seq, e.what());
                sleep_s(delay);
                delay = std::min(delay * 2, 30.0);
            } catch (const GatewayRefusal &e) {
                if (tried && e.code == "chunk_duplicate") {  // our own earlier attempt arrived
                    ++posted, final_posted = final;
                    return;
                }
                failed = true;
                fatal = e.fatal;
                refusal_message = e.message;
                Json f = Json::object();
                f.set("seq", Json::integer(seq));
                f.set("reason", e.reason);
                event("chunk_refused", f);
                return;
            }
        }
    }
};

static int cmd_run(int argc, char **argv) {
    std::string dir = arg(argc, argv, "--data-dir", "ballrain-pool-data");
    Bytes token;
    if (!read_file(dir + "/token", token) || token.empty()) throw Error("no token: run `nc_client login` first");
    while (!token.empty() && (token.back() == '\n' || token.back() == '\r')) token.pop_back();
    GatewayClient api(arg(argc, argv, "--gateway"), token);
    std::string game = platform::absolute_path(arg(argc, argv, "--ballance-dir"));
    std::string engine_bin = platform::absolute_path(arg(argc, argv, "--engine"));
    std::string kind = arg(argc, argv, "--engine-kind", "x87soft");
    int64_t max_tasks = std::atoll(arg(argc, argv, "--tasks", "0").c_str());
    std::string lb = arg(argc, argv, "--leaderboard", "");
    platform::make_dirs(dir + "/cache");
    platform::make_dirs(dir + "/runs");
    Json engine_json = engine_info(engine_bin, kind), files_json = hash_game_files(game, kind);
    Json hello = api.hello(platform_name(), engine_json, files_json, 1);
    auto rehello = [&]() {  // session_expired: the gateway forgot the session (idle / restarted): a new hello
        hello = api.hello(platform_name(), engine_json, files_json, 1);
        event("rehello", make_message("x"));
    };
    bool fatal = false;
    // a refusal outside a chunk: fatal (token) stops the client, session_expired renews the session, others are shown
    auto refused = [&](const GatewayRefusal &e, const char *where) {
        Json f = Json::object();
        f.set("where", Json::string(where));
        f.set("reason", e.reason);
        event("refused", f);
        std::fprintf(stderr, "%s\n", e.message.c_str());
        if (e.fatal) fatal = true, g_stop = true;
        else if (e.code == "session_expired") {
            try {
                rehello();
            } catch (const GatewayRefusal &e2) {
                if (e2.fatal) fatal = true, g_stop = true;
            } catch (const GatewayUnavailable &) {
            }
        }
    };
    {
        Json f = Json::object();
        f.set("user", hello.at("user"));
        f.set("leaderboard", hello.at("leaderboard"));
        f.set("max_instances", hello.at("max_instances"));
        event("hello_ok", f);
    }
    if (!lb.empty() && (lb == "on") != hello.at("leaderboard").as_bool()) api.set_leaderboard(lb == "on");
    double heartbeat_s = double(hello.at("heartbeat_s").as_int());
    double last_hb = platform::monotonic_s();
    int64_t total_steps = 0;
    double t_start = platform::monotonic_s();
    PoolCache cache(dir + "/cache", api);
    auto heartbeat = [&](bool running, const std::string &task_id, int64_t steps) {
        if (platform::monotonic_s() - last_hb < heartbeat_s) return;
        last_hb = platform::monotonic_s();
        Json inst = Json::object();
        inst.set("slot", Json::integer(0));
        inst.set("state", Json::string(running ? "running" : "idle"));
        inst.set("steps", Json::integer(total_steps + steps));
        inst.set("task_id", task_id.empty() ? Json() : Json::string(task_id));
        Json list = Json::array();
        list.push(inst);
        double el = platform::monotonic_s() - t_start;
        Json r;
        try {
            r = api.heartbeat(running, list, el > 0 ? double(total_steps + steps) / el : 0.0);
        } catch (const GatewayUnavailable &e) {  // the next one will tell; the task goes on
            std::fprintf(stderr, "heartbeat: %s\n", e.what());
            return;
        } catch (const GatewayRefusal &e) {
            refused(e, "heartbeat");
            return;
        }
        event("heartbeat", make_message("x"));
        if (const Json *rv = r.find("revoked")) {
            std::fprintf(stderr, "%s\n", rv->get_string("message", "").c_str());
            Json f = Json::object();
            f.set("reason", *rv);
            event("revoked", f);
            g_stop = true;
        }
    };
    int64_t done = 0;
    int failures = 0;  // consecutive failed tasks / fetches: backoff, give up after 8
    auto backoff = [&](const std::string &why) {
        ++failures;
        Json f = Json::object();
        f.set("error", Json::string(why));
        f.set("failures", Json::integer(failures));
        event("task_failed", f);
        std::fprintf(stderr, "%s\n", why.c_str());
        if (failures >= 8) {
            g_stop = true;
            return;
        }
        sleep_s(std::min(300.0, 2.0 * double(1 << std::min(failures, 8))));
    };
    while (!g_stop && (max_tasks <= 0 || done < max_tasks)) {
        Json tm;
        try {
            tm = api.task();
        } catch (const GatewayUnavailable &e) {
            backoff(e.what());
            continue;
        } catch (const GatewayRefusal &e) {
            refused(e, "task");
            if (!fatal && e.code != "session_expired") sleep_s(30);
            continue;
        }
        if (tm.at("type").as_string() == "no_task") {
            Json f = Json::object();
            f.set("reason", tm.at("reason"));
            event("no_task", f);
            heartbeat(false, "", 0);
            sleep_s(double(tm.at("retry_s").as_int()));
            continue;
        }
        std::string open_task;  // the fetched task (task_end when it ends without a final chunk)
        Uploader up{api, std::string(), false, false, 0};
        // close a task that ended without a final chunk once everything it has was posted (best effort: the deadline
        // closes it otherwise), so the gateway frees its instance slot now
        auto close_open = [&](const std::string &why) {
            if (open_task.empty() || up.failed || up.final_posted) return;
            try {
                api.task_end(open_task, up.posted, why);
                Json f = Json::object();
                f.set("task_id", Json::string(open_task));
                f.set("chunks", Json::integer(up.posted));
                event("task_closed", f);
            } catch (const std::exception &e) {
                std::fprintf(stderr, "task_end %s: %s\n", open_task.c_str(), e.what());
            }
        };
        try {
        Task task = Task::from_json(tm.at("task"));
        open_task = task.task_id;
        double deadline = platform::monotonic_s() + double(task.deadline_s) - std::min(30.0, double(task.deadline_s) / 10);
        {
            Json f = Json::object();
            f.set("task_id", Json::string(task.task_id));
            f.set("start_frame", Json::integer(task.start_frame));
            f.set("ball_order_seed", Json::integer(task.ball_order_seed));
            event("task", f);
        }
        if (task.engine != kind) throw Error("task engine " + task.engine + ", local engine " + kind);
        Policy pol = cache.policy(task.policy_sha);
        const Json &m = pol.manifest;
        EnvConfig ec = EnvConfig::from_manifest(m.at("env"));
        if (ec.level != task.level || m.at("env").get_string("start_record", "") != task.record)
            throw Error("task level / record differ from the policy's");
        TasStarts starts = cache.starts(task.asset_starts);
        if (starts.record != task.record || starts.record_sha256 != task.record_sha256)
            throw Error("the starts pack is not the task's record");
        starts.apply_camera_filter(ec.camera_quarter_tol, ec.camera_turn_grace_ms);
        starts.exclude(ec.exclude_starts);
        FeatureConfig fc = FeatureConfig::from_json(m.at("features"));
        if (fc.path_sha256 != task.asset_path) throw Error("the task's reference line is not the policy's");
        fc.path_dir = cache.path_dir(task.asset_path, task.level);
        bool drive = task.rest && task.rest->drive;  // an at-rest start with a target may hand over in an excluded range
        if (task.start_frame > 0 && !(drive ? starts.expected(task.start_frame).has_value()
                                            : starts.index_of(task.start_frame).has_value()))
            throw Error("start frame is not a valid start frame of the record");
        EngineConfig eng;
        eng.binary = engine_bin, eng.root = game;
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
        for (auto &kv : args_all(argc, argv, "--engine-env")) {
            size_t eq = kv.find('=');
            if (eq == std::string::npos || kv.compare(0, 9, "BALLRAIN_") != 0 || kv.find("BALL_ORDER") != std::string::npos)
                throw Error("--engine-env NAME=VALUE (BALLRAIN_*, not the ball order)");
            eng.extra_env.push_back({kv.substr(0, eq), kv.substr(eq + 1)});
        }
        std::string runs = dir + "/runs/" + task.task_id;
        BallanceEnv env(ec, fc, eng, runs, nullptr, &starts, task.start_frame, RewardConfig::from_json(m.at("reward")),
                        task.rest);
        if (env.features().size() != pol.n_obs) throw Error("the env's observation size differs from the policy's");
        for (size_t i = 0; i < env.actions().size(); ++i)
            if (env.actions()[i] != pol.action_keys[i]) throw Error("the env's action set differs from the policy's");
        TaskRunner runner(env, &starts, task, RolloutSpec::from_json(m.at("rollout")), m.at("env").at("delta_ms").as_double());
        double t0 = platform::monotonic_s();
        std::string end = sample_task(
            runner, pol, task,
            [&](const Bytes &data, bool final) {
                up.post(data, -1, final);
            },
            [&]() -> std::string {
                heartbeat(true, task.task_id, runner.steps);
                if (g_stop) return "stopped";
                if (platform::monotonic_s() >= deadline) return "deadline";
                if (up.failed) return "stopped";
                return "";
            });
        env.close();
        std::error_code ec_rm;
        std::filesystem::remove_all(std::filesystem::u8path(runs), ec_rm);  // a task keeps nothing locally
        total_steps += runner.steps;
        ++done;
        Json f = Json::object();
        f.set("task_id", Json::string(task.task_id));
        f.set("end", Json::string(end));
        f.set("steps", Json::integer(runner.steps));
        f.set("accepted_steps", Json::integer(up.accepted_steps));
        f.set("frames", Json::integer(runner.frames));
        f.set("episodes", Json::integer(runner.episodes));
        double dt = platform::monotonic_s() - t0;
        f.set("wall_s", Json::number(dt));
        f.set("steps_per_s", Json::number(dt > 0 ? double(runner.steps) / dt : 0.0));
        event("task_end", f);
        close_open("end " + end + " without a final chunk");
        if (up.failed && !up.refusal_message.empty()) std::fprintf(stderr, "%s\n", up.refusal_message.c_str());
        if (up.fatal) fatal = true, g_stop = true;
        failures = 0;
        } catch (const GatewayRefusal &e) {  // a download refused, or the policy / assets of the task
            close_open(std::string("refused: ") + e.code);
            refused(e, "task");
            ++done;
        } catch (const std::exception &e) {  // an engine failure at the start, a mismatching asset, ...: next task
            close_open(std::string("failed: ") + e.what());
            ++done;
            backoff(std::string("task: ") + e.what());
        }
    }
    if (fatal) {
        event("stopped", make_message("token_fatal"));
        return 3;
    }
    return 0;
}

int main(int argc, char **argv) {
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    if (argc < 2) {
        std::fprintf(stderr, "usage: nc_client login|run [options] (see the source header)\n");
        return 2;
    }
    try {
        std::string cmd = argv[1];
        if (cmd == "login") return cmd_login(argc, argv);
        if (cmd == "run") return cmd_run(argc, argv);
        std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
        return 2;
    } catch (const GatewayRefusal &e) {
        std::fprintf(stderr, "%s\n", e.message.c_str());
        Json f = Json::object();
        f.set("reason", e.reason);
        event("refused", f);
        return e.fatal ? 3 : 1;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "nc_client: %s\n", e.what());
        Json f = Json::object();
        f.set("error", Json::string(e.what()));
        event("error", f);
        return 2;
    }
}
