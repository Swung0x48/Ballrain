// The native volunteer client (M2: command line; the TUI, several instances and the browser login are M3).
//
//   nc_client login --gateway URL [--data-dir D]
//       login_start, shows the login code and the browser link, polls until the gateway hands out the token, which
//       is saved to <D>/token (D defaults to ./ballrain-pool-data)
//   nc_client run --gateway URL --ballance-dir G --engine BallanceMMOSimTool[.exe] [--data-dir D] [--tasks N]
//                 [--instances K] [--engine-kind x87soft|2002] [--engine-env NAME=VALUE ...] [--leaderboard on|off]
//                 [--auto-pause on|off]
//       the controller (native/core/src/controller.cpp) without a UI: hello, K instances playing tasks, the upload
//       queue, heartbeats; with --tasks N it stops after N tasks once their chunks are posted. A refusal is printed
//       verbatim; a token-fatal one stops the client (exit code 3).
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

#include "ballrain/controller.h"
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

static int cmd_run(int argc, char **argv) {
    std::string dir = arg(argc, argv, "--data-dir", "ballrain-pool-data");
    Bytes token;
    if (!read_file(dir + "/token", token) || token.empty()) throw Error("no token: run `nc_client login` first");
    bool fatal = false;
    Controller ctl(dir, [&](const std::string &kind, const Json &f) {
        if (kind == "log") {
            std::fprintf(stderr, "%s\n", f.get_string("message", "").c_str());
            return;
        }
        event(kind, f);
    });
    ClientConfig c = ctl.config();
    c.gateway_url = arg(argc, argv, "--gateway");
    c.ballance_dir = platform::absolute_path(arg(argc, argv, "--ballance-dir"));
    c.engine_binary = platform::absolute_path(arg(argc, argv, "--engine"));
    c.engine_kind = arg(argc, argv, "--engine-kind", "x87soft");
    c.instances = std::atoi(arg(argc, argv, "--instances", "1").c_str());
    // the automatic pause is the TUI's (a test machine may run Player.exe for other reasons): off unless asked
    c.auto_pause_player = c.auto_pause_battery = std::string(arg(argc, argv, "--auto-pause", "off")) == "on";
    ctl.set_config(c);
    ctl.max_tasks = std::atoll(arg(argc, argv, "--tasks", "0").c_str());
    for (auto &kv : args_all(argc, argv, "--engine-env")) {
        size_t eq = kv.find('=');
        if (eq == std::string::npos || kv.compare(0, 9, "BALLRAIN_") != 0 || kv.find("BALL_ORDER") != std::string::npos)
            throw Error("--engine-env NAME=VALUE (BALLRAIN_*, not the ball order)");
        ctl.engine_env.push_back({kv.substr(0, eq), kv.substr(eq + 1)});
    }
    if (!ctl.connect()) {
        ControllerView v = ctl.view();
        if (v.banner) std::fprintf(stderr, "%s\n", v.banner->message.c_str());
        return v.token_dead ? 3 : 1;
    }
    std::string lb = arg(argc, argv, "--leaderboard", "");
    if (!lb.empty() && (lb == "on") != ctl.view().leaderboard) ctl.set_leaderboard(lb == "on");
    ctl.start_service();
    ctl.start();
    while (!g_stop) {
        if (ctl.finished()) break;
        ControllerView v = ctl.view();
        if (v.token_dead) {
            fatal = true;
            break;
        }
        bool gave_up = !v.inst.empty();
        for (auto &i : v.inst) gave_up = gave_up && i.state == "error";
        if (gave_up) break;  // every instance stopped retrying (a broken installation): the banner says why
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    ctl.shutdown();
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
