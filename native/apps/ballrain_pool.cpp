// Ballrain 算力池客户端: the volunteer client's terminal UI (FTXUI), a thin layer over ballrain::Controller
// (native/core/src/controller.cpp): every decision and every message to the gateway is the controller's; this file
// only draws ControllerView and calls the controller's actions. Pages: first-run setup (the Ballance folder, checked),
// login (the gateway's page in the system browser, then polling), the main screen (start / pause, instances,
// leaderboard switch, per-instance table, totals, log, the gateway's messages verbatim), diagnostics export.
//
// Build-time settings (CMake cache variables, set by the release build, not in the repository):
//   BALLRAIN_DEFAULT_GATEWAY  the gateway URL prefilled in the setup
//   BALLRAIN_UPDATE_URL       the signed release manifest (update.h), e.g. .../releases/latest/download/release.json
//   BALLRAIN_RELEASE_PUBKEY   the release key's public half (hex; tools: nc_release keygen)
//
//   ballrain_pool [--data-dir D]
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "ballrain/controller.h"
#include "ftxui/component/app.hpp"
#include "ftxui/component/component.hpp"
#include "ftxui/component/component_options.hpp"
#include "ftxui/component/event.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/dom/table.hpp"
#include "platform.h"

#ifdef _WIN32
#include <windows.h>
#endif

#ifndef BALLRAIN_DEFAULT_GATEWAY
#define BALLRAIN_DEFAULT_GATEWAY ""
#endif
#ifndef BALLRAIN_UPDATE_URL
#define BALLRAIN_UPDATE_URL ""
#endif
#ifndef BALLRAIN_RELEASE_PUBKEY
#define BALLRAIN_RELEASE_PUBKEY ""
#endif

using namespace ballrain;
using namespace ftxui;

namespace {

enum Page { SETUP = 0, LOGIN = 1, MAIN = 2, DIAG = 3 };

// a background job (network, hashing) so the UI never freezes; one at a time
struct Jobs {
    std::atomic<bool> busy{false};
    std::mutex mu;
    std::string status;  // the last job's message for the status line
    std::vector<std::thread> threads;
    bool run(const std::string &what, std::function<std::string()> fn) {
        if (busy.exchange(true)) return false;
        set(what + "……");
        threads.emplace_back([this, fn] {
            std::string r;
            try {
                r = fn();
            } catch (const std::exception &e) {
                r = e.what();
            }
            set(r);
            busy = false;
        });
        return true;
    }
    void set(const std::string &s) {
        std::lock_guard<std::mutex> g(mu);
        status = s;
    }
    std::string get() {
        std::lock_guard<std::mutex> g(mu);
        return status;
    }
    void join() {
        for (auto &t : threads)
            if (t.joinable()) t.join();
    }
};

std::string fmt_steps(int64_t n) {
    if (n >= 10000000) return std::to_string(n / 1000000) + "M";
    if (n >= 100000) return std::to_string(n / 1000) + "k";
    return std::to_string(n);
}

std::string fmt_rate(double r) {
    char b[32];
    std::snprintf(b, sizeof b, "%.0f", r);
    return b;
}

Element banner_box(const Banner &b) {
    // the gateway's reason.message exactly as received: text() draws it as plain text (no markup)
    Color c = b.kind == "revoked" ? Color::Red : b.kind == "rejected" ? Color::Yellow : Color::RedLight;
    return window(text(" " + b.title + (b.code.empty() ? "" : "（" + b.code + "）") + " ") | color(c),
                  paragraph(b.message)) |
           color(c);
}

}  // namespace

int main(int argc, char **argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    SetConsoleTitleW(L"Ballrain 算力池");
    {  // QuickEdit would freeze the output when the player clicks into the window
        HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
        DWORD mode = 0;
        if (GetConsoleMode(in, &mode)) SetConsoleMode(in, (mode | ENABLE_EXTENDED_FLAGS) & ~DWORD(ENABLE_QUICK_EDIT_MODE));
    }
#endif
    std::string data_dir = platform::data_dir();
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "--data-dir") == 0) data_dir = platform::absolute_path(argv[i + 1]);
    platform::make_dirs(data_dir);
    if (!platform::single_instance(data_dir)) {
        std::fprintf(stderr, "Ballrain 算力池客户端已经在运行了（同一个数据目录只能开一个）。\n");
        return 1;
    }
    Controller ctl(data_dir);
    ClientConfig cfg = ctl.config();
    if (cfg.gateway_url.empty()) cfg.gateway_url = BALLRAIN_DEFAULT_GATEWAY;
    Jobs jobs;
    std::atomic<bool> quitting{false};
    int page = cfg.setup_done && ctl.problems().empty() ? (ctl.has_token() ? MAIN : LOGIN) : SETUP;

    // ---------------------------------------------------------------------------------------- setup page
    std::string in_gateway = cfg.gateway_url, in_dir = cfg.ballance_dir, in_engine = cfg.engine_binary;
    bool bundled = !bundled_engine(cfg.engine_kind).empty();  // the release zip ships the engine: no field then
    std::vector<GameProblem> probs;
    std::string found_note;
    bool checked = false;
    auto do_check = [&] {
        probs = check_game_dir(in_dir, cfg.engine_kind);
        auto e = check_engine(in_engine.empty() ? bundled_engine(cfg.engine_kind) : platform::absolute_path(in_engine));
        probs.insert(probs.end(), e.begin(), e.end());
        try {
            check_gateway_url(in_gateway);
        } catch (const Error &ex) {
            probs.push_back({"gateway", "missing", ex.what()});
        }
        checked = true;
    };
    InputOption one_line;
    one_line.multiline = false;
    auto i_gateway = Input(&in_gateway, "https://…（网关地址）", one_line);
    auto i_dir = Input(&in_dir, "例如 D:\\Games\\Ballance", one_line);
    auto i_engine = Input(&in_engine, "BallanceMMOSimTool.exe 的完整路径", one_line);
    auto b_find = Button("查找常见位置", [&] {
        auto d = find_ballance_dirs();
        if (d.empty()) {
            found_note = "没有在常见位置找到 Ballance，请手动填写安装目录（含 base.cmo 的那个文件夹）。";
        } else {
            in_dir = d[0];
            found_note = "找到：" + d[0] + (d.size() > 1 ? "（另外还有 " + std::to_string(d.size() - 1) + " 处）" : "");
            do_check();
        }
    });
    auto b_check = Button("检查游戏文件", [&] { do_check(); });
    auto b_save = Button("保存并继续", [&] {
        do_check();
        if (!probs.empty()) return;
        ClientConfig c = ctl.config();
        c.gateway_url = in_gateway, c.ballance_dir = in_dir, c.engine_binary = in_engine, c.setup_done = true;
        ctl.set_config(c);
        cfg = c;
        page = ctl.has_token() ? MAIN : LOGIN;
        if (page == MAIN) jobs.run("连接网关", [&] { return ctl.connect() ? "已连接网关" : std::string("连接失败"); });
    });
    auto b_quit0 = Button("退出", [&] { quitting = true; });
    auto setup_c = Container::Vertical({i_gateway, i_dir, Maybe(i_engine, [&] { return !bundled; }),
                                        Container::Horizontal({b_find, b_check, b_save, b_quit0})});
    auto setup_r = Renderer(setup_c, [&] {
        Elements pl;
        if (checked && probs.empty()) pl.push_back(text("游戏文件检查通过。") | color(Color::Green));
        for (auto &p : probs) pl.push_back(paragraph("• " + p.message) | color(Color::Red));
        return vbox({
                   text(" Ballrain 算力池 · 首次设置 ") | bold | center,
                   separator(),
                   paragraph("用你自己电脑上的 Ballance 帮忙采集训练数据。客户端只读取游戏目录里的 base.cmo、Database.tdb、"
                             "3D Entities 和 Text，不会改动任何游戏文件。"),
                   text(""),
                   hbox({text("网关地址：    "), i_gateway->Render() | flex}) | border,
                   hbox({text("Ballance 目录："), i_dir->Render() | flex}) | border,
                   bundled ? text("") : hbox({text("引擎文件：    "), i_engine->Render() | flex}) | border,
                   found_note.empty() ? text("") : paragraph(found_note),
                   vbox(std::move(pl)),
                   text(""),
                   hbox({b_find->Render(), text(" "), b_check->Render(), text(" "), b_save->Render(), filler(),
                         b_quit0->Render()}),
               }) |
               border;
    });

    // ---------------------------------------------------------------------------------------- login page
    std::mutex login_mu;
    std::string login_code, login_url, login_state = "还没有登录。";
    std::atomic<bool> login_running{false};
    std::thread login_thread;
    auto start_login = [&] {
        if (login_running.exchange(true)) return;
        if (login_thread.joinable()) login_thread.join();
        login_thread = std::thread([&] {
            try {
                Json s = ctl.login_start();
                {
                    std::lock_guard<std::mutex> g(login_mu);
                    login_code = s.at("user_code").as_string();
                    login_url = s.at("browser_url").as_string();
                    login_state = "已在浏览器中打开登录页面，请用论坛账号登录并确认登录代码。";
                }
                if (!platform::open_url(s.at("browser_url").as_string())) {
                    std::lock_guard<std::mutex> g(login_mu);
                    login_state = "没能自动打开浏览器，请手动打开下面的地址。";
                }
                double end = platform::monotonic_s() + double(s.at("expires_in_s").as_int());
                double poll = std::max<double>(1.0, double(s.at("poll_s").as_int()));
                std::string st = "pending";
                while (!quitting && platform::monotonic_s() < end) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(int(poll * 1000)));
                    st = ctl.login_poll(s);
                    if (st != "pending") break;
                }
                std::lock_guard<std::mutex> g(login_mu);
                if (st == "ok") {
                    login_state = "登录成功。";
                    page = MAIN;
                    jobs.run("连接网关", [&] { return ctl.connect() ? "已连接网关" : std::string("连接失败"); });
                } else if (st == "refused") {
                    login_state = "登录被拒绝，原因见上方。";
                } else {
                    login_state = "登录已过期，请重新点“在浏览器中登录”。";
                }
            } catch (const std::exception &e) {
                std::lock_guard<std::mutex> g(login_mu);
                login_state = std::string("登录失败：") + e.what();
            }
            login_running = false;
        });
    };
    auto b_login = Button("在浏览器中登录", start_login);
    auto b_back = Button("返回设置", [&] { page = SETUP; });
    auto b_quit1 = Button("退出", [&] { quitting = true; });
    auto login_c = Container::Horizontal({b_login, b_back, b_quit1});
    auto login_r = Renderer(login_c, [&] {
        ControllerView v = ctl.view();
        std::lock_guard<std::mutex> g(login_mu);
        return vbox({
                   text(" Ballrain 算力池 · 登录 ") | bold | center,
                   separator(),
                   v.banner ? banner_box(*v.banner) : text(""),
                   paragraph("用 Ballance 论坛（forum.ballance.top）的账号登录。登录在浏览器里完成，客户端不会看到你的论坛密码；"
                             "登录成功后客户端会自动拿到算力池令牌。"),
                   text(""),
                   login_code.empty() ? text("") : hbox({text("登录代码："), text(login_code) | bold | color(Color::Cyan)}),
                   login_url.empty() ? text("") : paragraph("登录页面：" + login_url),
                   paragraph(login_state),
                   text(""),
                   hbox({b_login->Render(), text(" "), b_back->Render(), filler(), b_quit1->Render()}),
               }) |
               border;
    });

    // ---------------------------------------------------------------------------------------- main page
    std::string start_label = "开始", lb_label = "上榜：开";
    auto b_start = Button(&start_label, [&] {
        if (ctl.view().want_running) ctl.pause();
        else jobs.run("开始", [&] {
            ctl.start();
            return std::string("");
        });
    });
    auto b_minus = Button(" − ", [&] { ctl.set_instances(ctl.config().instances - 1); });
    auto b_plus = Button(" ＋ ", [&] { ctl.set_instances(ctl.config().instances + 1); });
    auto b_lb = Button(&lb_label, [&] {
        bool want = !ctl.view().leaderboard;
        jobs.run("修改上榜设置", [&ctl, want] {
            try {
                return std::string("上榜已") + (ctl.set_leaderboard(want) ? "打开" : "关闭");
            } catch (const GatewayRefusal &e) {
                return e.message;
            }
        });
    });
    DiagFiles diag;
    std::string diag_saved;
    auto b_diag = Button("导出诊断", [&] {
        diag = ctl.diagnostics();
        diag_saved.clear();
        page = DIAG;
    });
    auto b_setup = Button("设置", [&] {
        ClientConfig c = ctl.config();
        in_gateway = c.gateway_url, in_dir = c.ballance_dir, in_engine = c.engine_binary, checked = false, probs.clear();
        page = SETUP;
    });
    auto b_logout = Button("退出登录", [&] {
        ctl.logout();
        page = LOGIN;
    });
    auto b_quit2 = Button("退出", [&] { quitting = true; });
    auto main_c = Container::Horizontal({b_start, b_minus, b_plus, b_lb, b_diag, b_setup, b_logout, b_quit2});
    auto main_r = Renderer(main_c, [&] {
        ControllerView v = ctl.view();
        start_label = v.want_running ? "暂停" : "开始";
        lb_label = std::string("上榜：") + (v.leaderboard ? "开" : "关");
        std::string state = !v.has_token ? "未登录"
                            : v.token_dead ? "令牌已停用"
                            : !v.connected ? "未连接"
                            : !v.want_running ? "已暂停"
                            : !v.auto_pause.empty() ? "自动暂停"
                                                    : "采样中";
        Elements head{text(" Ballrain 算力池 ") | bold, text("  "), text(v.user_name.empty() ? "" : "玩家：" + v.user_name),
                      text("  状态："), text(state) | bold | color(state == "采样中" ? Color::Green : Color::Yellow),
                      filler(), text("客户端 " + std::string(CLIENT_VERSION) + " ")};
        std::vector<std::vector<std::string>> rows{{"实例", "状态", "任务", "本任务步数", "累计步数", "已接收", "步/秒", "说明"}};
        for (auto &i : v.inst)
            rows.push_back({std::to_string(i.slot + 1), i.state_text, i.task_id.empty() ? "-" : i.task_id.substr(0, 8),
                            fmt_steps(i.task_steps), fmt_steps(i.total_steps), fmt_steps(i.accepted_steps),
                            fmt_rate(i.steps_per_s), i.detail.substr(0, 40)});
        if (v.inst.empty()) rows.push_back({"-", "-", "-", "-", "-", "-", "-", v.want_running ? "准备中" : "按“开始”开始采样"});
        Table t(rows);
        t.SelectAll().Border(LIGHT);
        t.SelectRow(0).Decorate(bold);
        t.SelectRow(0).SeparatorVertical(LIGHT);
        t.SelectRow(0).Border(LIGHT);
        std::string totals = "待上传数据段：" + std::to_string(v.queued_chunks) + "    本次完成任务：" + std::to_string(v.tasks_done);
        if (v.stats.is_object())
            totals += "    网关统计：已接收 " + fmt_steps(v.stats.get_int("steps_accepted", 0)) + " 步，已验证 " +
                      fmt_steps(v.stats.get_int("steps_verified", 0)) + " 步，待验证 " +
                      fmt_steps(v.stats.get_int("steps_pending", 0)) + " 步";
        Elements logl;
        size_t from = v.log.size() > 12 ? v.log.size() - 12 : 0;
        for (size_t k = from; k < v.log.size(); ++k) logl.push_back(paragraph(v.log[k]));
        Elements body{hbox(std::move(head)), separator()};
        if (v.banner) body.push_back(banner_box(*v.banner));
        if (!v.auto_pause.empty()) body.push_back(paragraph(v.auto_pause) | color(Color::Yellow));
        if (!v.notice.empty()) body.push_back(window(text(" 网关通知 "), paragraph(v.notice)));
        if (!v.update_version.empty())
            body.push_back(window(text(" 有新版本 " + v.update_version + " ") | color(Color::Cyan),
                                  paragraph((v.update_notes.empty() ? "" : v.update_notes + "\n") + "下载地址：" +
                                            v.update_url + "（下载后解压，覆盖当前文件夹即可；数据和登录状态会保留）")));
        body.push_back(hbox({b_start->Render(), text("  实例数 "), b_minus->Render(),
                             text(" " + std::to_string(v.instances) + " ") | bold, b_plus->Render(),
                             text("（最多 " + std::to_string(v.max_instances) + "）  "), b_lb->Render(), text("  "),
                             b_diag->Render(), text(" "), b_setup->Render(), text(" "), b_logout->Render(), filler(),
                             b_quit2->Render()}));
        body.push_back(t.Render());
        body.push_back(paragraph(totals));
        std::string js = jobs.get();
        if (!js.empty()) body.push_back(text(js) | dim);
        body.push_back(window(text(" 日志 "), vbox(std::move(logl))) | flex);
        return vbox(std::move(body)) | border;
    });

    // ---------------------------------------------------------------------------------------- diagnostics page
    auto b_dsave = Button("保存为 zip", [&] {
        try {
            diag_saved = "已保存：" + ctl.export_diagnostics();
        } catch (const std::exception &e) {
            diag_saved = std::string("保存失败：") + e.what();
        }
    });
    auto b_dback = Button("返回", [&] { page = MAIN; });
    auto diag_c = Container::Horizontal({b_dsave, b_dback});
    auto diag_r = Renderer(diag_c, [&] {
        Elements files;
        std::string info;
        for (auto &f : diag) {
            files.push_back(text("• " + f.first + "（" + std::to_string(f.second.size()) + " 字节）"));
            if (f.first == "info.json") info = f.second;
        }
        return vbox({
                   text(" 导出诊断信息 ") | bold | center,
                   separator(),
                   paragraph("下面这些文件会打包成 zip，保存在客户端的数据目录里。令牌、你的用户名和家目录路径、论坛账号、"
                             "IP 地址都已经替换掉。把 zip 发给测试负责人即可。"),
                   vbox(std::move(files)),
                   window(text(" info.json "), paragraph(info.substr(0, 3000))) | flex,
                   diag_saved.empty() ? text("") : paragraph(diag_saved) | color(Color::Green),
                   hbox({b_dsave->Render(), text(" "), b_dback->Render()}),
               }) |
               border;
    });

    // ---------------------------------------------------------------------------------------- the app
    auto tabs = Container::Tab({setup_r, login_r, main_r, diag_r}, &page);
    auto root = CatchEvent(tabs, [&](Event e) {
        if (quitting) return false;
        return false;
    });
    std::string snapshot;  // tests: one frame of a page as text (no terminal needed), e.g. --snapshot setup
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "--snapshot") == 0) snapshot = argv[i + 1];
    if (!snapshot.empty()) {
        if (snapshot == "setup") do_check(), page = SETUP;
        else if (snapshot == "login") page = LOGIN;
        else if (snapshot == "diag") diag = ctl.diagnostics(), page = DIAG;
        else page = MAIN;
        // "live:S": the real controller (connect, start) for S seconds, then the main page as it is
        double live = snapshot.rfind("live:", 0) == 0 ? std::atof(snapshot.c_str() + 5) : 0.0;
        if (live > 0) {
            ctl.start_service();
            ctl.connect();
            ctl.start();
            std::this_thread::sleep_for(std::chrono::milliseconds(int(live * 1000)));
        }
        auto screen = Screen::Create(Dimension::Fixed(120), Dimension::Fixed(40));
        Render(screen, tabs->Render());
        std::printf("%s\n", screen.ToString().c_str());
        std::fflush(stdout);
        if (live > 0) ctl.shutdown();
        return 0;
    }
    auto app = App::Fullscreen();
    ctl.start_service();
    if (page == MAIN) jobs.run("连接网关", [&] { return ctl.connect() ? "已连接网关" : std::string("连接失败"); });
    std::thread update_thread([&] {
        if (std::strlen(BALLRAIN_UPDATE_URL) && std::strlen(BALLRAIN_RELEASE_PUBKEY))
            ctl.check_update(BALLRAIN_UPDATE_URL, BALLRAIN_RELEASE_PUBKEY);
    });
    std::thread ticker([&] {  // redraw while the controller works (4 per second), leave when the player quits
        while (!quitting) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            app.PostEvent(Event::Custom);
        }
        app.Post(app.ExitLoopClosure());
    });
    app.Loop(root);
    quitting = true;
    ticker.join();
    std::printf("正在停止所有实例并上传剩下的数据……\n");
    std::fflush(stdout);
    if (login_thread.joinable()) login_thread.join();
    ctl.shutdown();
    update_thread.join();
    jobs.join();
    return 0;
}
