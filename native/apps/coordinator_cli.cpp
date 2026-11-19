// The native volunteer client (M2: command line; the TUI, several instances and the browser login are M3).
//
//   ballrain_coordinator login --gateway URL [--data-dir D]
//       the controller's login: login_start, shows the login code and the browser link, polls until the gateway hands
//       out the token, which is saved to <D>/token (D defaults to the client data dir: BALLRAIN_WORK_DIR or ~/.ballrain)
//   ballrain_coordinator diag [--data-dir D] [--out F.zip]
//       the scrubbed diagnostics zip (default <D>/diagnostics-<time>.zip)
//   ballrain_coordinator run [--gateway URL] [--ballance-dir G] [--engine BallanceMMOSimTool[.exe]] [--data-dir D] [--tasks N]
//                 (options default to the saved config.json; the engine to the one shipped next to ballrain_coordinator.exe)
//                 [--instances K] [--engine-kind x87soft|2002] [--engine-env NAME=VALUE ...] [--leaderboard on|off]
//                 [--auto-pause on|off] [--quiet-file PATH] [--phase-summary-s S] [--wait-upload]
//       the controller (native/core/src/controller.cpp) without a UI: hello, K instances playing tasks, the upload
//       queue, heartbeats; with --tasks N it stops after N tasks once their chunks are posted. A refusal is printed
//       verbatim; a token-fatal one stops the client (exit code 3).
//
//       --phase-summary-s: the phase summary interval (default 300 s; the first one after min(60 s, S));
//       --wait-upload: the 0.5.0 behaviour (an instance waits for its previous task's upload before the next fetch)
//
// Output: one JSON object per line on stdout (events, the log lines among them: "log"), the log lines also on stderr;
// <D>/logs/client.log gets the log lines too.
#include <algorithm>
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
#include "platform.h"
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

// the events on stdout; a log line (the same as client.log's) on stderr, and in `run` also as a "log" event on stdout
// (the events file of a headless run then has the log lines; login's stdout stays login_started ... login_ok)
static bool g_log_events = false;
static void print_event(const std::string &kind, const Json &f) {
    if (kind == "log") {
        std::fprintf(stderr, "%s\n", f.get_string("message", "").c_str());
        std::fflush(stderr);
        if (!g_log_events) return;
    }
    event(kind, f);
}

// the controller's login (the same code path as the TUI's)
static int cmd_login(int argc, char **argv) {
    std::string dir = arg(argc, argv, "--data-dir", platform::data_dir().c_str());
    Controller ctl(dir, print_event);
    ClientConfig c = ctl.config();
    c.gateway_url = arg(argc, argv, "--gateway");
    ctl.set_config(c);
    Json ls = ctl.login_start();
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
        std::string st = ctl.login_poll(ls);
        if (st == "pending") continue;
        if (st == "ok") return 0;  // the controller emitted login_ok (with the user) and saved the token
        Json g = Json::object();
        g.set("status", Json::string(st));
        if (ctl.view().banner) g.set("message", Json::string(ctl.view().banner->message));
        event("login_failed", g);
        return 1;
    }
    event("login_failed", make_message("x"));
    return 1;
}

// the diagnostics zip of a data directory (as the TUI's "导出诊断")
static int cmd_diag(int argc, char **argv) {
    Controller ctl(arg(argc, argv, "--data-dir", platform::data_dir().c_str()), print_event);
    std::string p = ctl.export_diagnostics(arg(argc, argv, "--out", ""));
    Json f = Json::object();
    f.set("path", Json::string(p));
    event("diagnostics", f);
    return 0;
}

static int cmd_run(int argc, char **argv) {
    std::string dir = arg(argc, argv, "--data-dir", platform::data_dir().c_str());
    Bytes token;
    if (!read_file(dir + "/token", token) || token.empty()) throw Error("no token: run `ballrain_coordinator login` first");
    bool fatal = false;
    g_log_events = true;
    Controller ctl(dir, print_event);
    ClientConfig c = ctl.config();
    // every option falls back to the saved config.json (a scheduled task or an SSH session runs `ballrain_coordinator run`
    // after one configured run); --engine absent and none saved: the engine shipped next to ballrain_coordinator.exe
    auto opt = [&](const char *name, const std::string &saved) { return std::string(arg(argc, argv, name, saved.c_str())); };
    c.gateway_url = opt("--gateway", c.gateway_url);
    std::string g = opt("--ballance-dir", c.ballance_dir), e = opt("--engine", c.engine_binary);
    c.ballance_dir = g.empty() ? g : platform::absolute_path(g);
    c.engine_binary = e.empty() ? e : platform::absolute_path(e);
    c.engine_kind = opt("--engine-kind", c.engine_kind);
    c.instances = std::atoi(opt("--instances", std::to_string(c.instances)).c_str());
    // the automatic pause is the TUI's (a test machine may run Player.exe for other reasons): off unless asked
    c.auto_pause_player = c.auto_pause_battery = opt("--auto-pause", "off") == "on";
    c.quiet_file = opt("--quiet-file", c.quiet_file);  // the stop switch (independent of --auto-pause)
    ctl.set_config(c);
    ctl.max_tasks = std::atoll(arg(argc, argv, "--tasks", "0").c_str());
    ctl.phase_summary_s = std::max(1.0, std::atof(arg(argc, argv, "--phase-summary-s", "300").c_str()));
    for (int i = 2; i < argc; ++i)
        if (std::strcmp(argv[i], "--wait-upload") == 0) ctl.wait_upload = true;
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
        std::fprintf(stderr, "usage: ballrain_coordinator login|run [options] (see the source header)\n");
        return 2;
    }
    try {
        std::string cmd = argv[1];
        if (cmd == "login") return cmd_login(argc, argv);
        if (cmd == "diag") return cmd_diag(argc, argv);
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
        std::fprintf(stderr, "ballrain_coordinator: %s\n", e.what());
        Json f = Json::object();
        f.set("error", Json::string(e.what()));
        event("error", f);
        return 2;
    }
}
