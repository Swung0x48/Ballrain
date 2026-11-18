#include "ballrain/controller.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>

#include "ballrain/sha256.h"
#include "ballrain/update.h"
#include "platform.h"

namespace ballrain {

static const int CRASH_LIMIT = 8;         // failed tasks of one instance in a row before it stops retrying
static const double STABLE_S = 120.0;     // a task that ran this long resets its instance's failure count
static const double RECONNECT_S = 30.0;   // at most one automatic hello per this many seconds
static const double PROBE_S = 5.0;        // automatic-pause probes
static const double ME_S = 60.0;          // /pool/me (the totals) while connected
static const int64_t QUEUE_LIMIT = 256LL << 20;  // queued chunk bytes before the instances wait for the network
static const char *PAUSE_PLAYER = "检测到 Ballance（Player.exe）正在运行，已自动暂停；关闭游戏后会自动继续。";
static const char *PAUSE_BATTERY = "电脑正在使用电池，已自动暂停；接上电源后会自动继续。";
static const char *PHYSICS_RT_2002_SHA256 = "e72e4afcfa5c33a7d3d27776137f8c997b3c52d89d8a8a4745f1ca21e45893ec";

static std::string state_text(const std::string &s) {
    if (s == "idle") return "空闲";
    if (s == "starting") return "准备中";
    if (s == "running") return "采样中";
    if (s == "paused") return "已暂停";
    if (s == "error") return "出错";
    return s;
}

static std::string banner_title(const std::string &k) {
    if (k == "revoked") return "令牌已停用";
    if (k == "rejected") return "网关拒收";
    return "无法连接";
}

static void remove_tree(const std::string &d) {
    std::error_code ec;
    std::filesystem::remove_all(std::filesystem::u8path(d), ec);
}

static void write_atomic(const std::string &path, const Bytes &data) {
    std::string tmp = path + ".tmp";
    write_file(tmp, data);
    if (!platform::rename_file(tmp, path)) throw Error("cannot write " + path);
}

static std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    size_t i = 0;
    while (i < s.size() && s[i] == ' ') ++i;
    return s.substr(i);
}

// ---------------------------------------------------------------------------------------------- configuration

int ClientConfig::default_instances() { return std::max(1, std::min(MAX_INSTANCES, platform::cpu_count() / 2)); }

ClientConfig ClientConfig::load(const std::string &dir) {
    ClientConfig c;
    c.instances = default_instances();
    Bytes b;
    if (read_file(dir + "/config.json", b)) {
        try {
            Json j = Json::parse(b, 8);
            if (j.is_object()) {
                c.gateway_url = j.get_string("gateway_url", c.gateway_url);
                c.ballance_dir = j.get_string("ballance_dir", c.ballance_dir);
                c.engine_binary = j.get_string("engine_binary", c.engine_binary);
                c.engine_kind = j.get_string("engine_kind", c.engine_kind);
                c.instances = int(j.get_int("instances", c.instances));
                c.auto_pause_player = j.get_bool("auto_pause_player", c.auto_pause_player);
                c.auto_pause_battery = j.get_bool("auto_pause_battery", c.auto_pause_battery);
                c.setup_done = j.get_bool("setup_done", c.setup_done);
                c.quiet_file = j.get_string("quiet_file", c.quiet_file);
            }
        } catch (const Error &) {  // a broken file: the defaults (the setup asks again)
        }
    }
    c.instances = std::max(1, std::min(MAX_INSTANCES, c.instances));
    if (c.engine_kind != "x87soft" && c.engine_kind != "2002") c.engine_kind = "x87soft";
    return c;
}

void ClientConfig::save(const std::string &dir) const {
    platform::make_dirs(dir);
    Json j = Json::object();
    j.set("gateway_url", Json::string(gateway_url));
    j.set("ballance_dir", Json::string(ballance_dir));
    j.set("engine_binary", Json::string(engine_binary));
    j.set("engine_kind", Json::string(engine_kind));
    j.set("instances", Json::integer(instances));
    j.set("auto_pause_player", Json::boolean(auto_pause_player));
    j.set("auto_pause_battery", Json::boolean(auto_pause_battery));
    j.set("setup_done", Json::boolean(setup_done));
    j.set("quiet_file", Json::string(quiet_file));
    write_atomic(dir + "/config.json", j.dump());
}

// ---------------------------------------------------------------------------------------------- game folder

std::vector<GameProblem> check_game_dir(const std::string &root, const std::string &kind) {
    std::vector<GameProblem> out;
    if (root.empty()) return {{"", "missing", "还没有选择 Ballance 安装目录。"}};
    if (!platform::is_dir(root)) return {{"", "missing", "找不到 Ballance 安装目录：" + root}};
    std::vector<std::string> names{"base.cmo", "Database.tdb", "3D Entities/", "Text/"};
    if (kind == "2002") names.insert(names.end(), {"Bin/VxMath.dll", "BuildingBlocks/physics_RT.dll"});
    for (auto &n : names) {
        bool dir = n.back() == '/';
        std::string bare = dir ? n.substr(0, n.size() - 1) : n;
        std::string p = root + "/" + bare;
        if (dir) {
            std::vector<platform::DirEntry> files;
            if (!platform::is_dir(p)) out.push_back({n, "missing", "缺少文件夹：" + bare + "（应在 Ballance 安装目录下）"});
            else if (!platform::list_files(p, files, 1000000) || files.empty())
                out.push_back({n, "empty", "文件夹 " + bare + " 是空的，游戏文件不完整。"});
            continue;
        }
        if (!file_exists(p)) {
            std::string hint = n == "BuildingBlocks/physics_RT.dll"
                                   ? "Windows 版需要 2002 年原版的物理 DLL；没有的话可以改用全平台版。"
                                   : "游戏文件不完整，请检查安装目录。";
            out.push_back({n, "missing", "缺少文件：" + n + "。" + hint});
            continue;
        }
        if (n == "BuildingBlocks/physics_RT.dll") {
            std::string sha = file_sha256(p);
            if (sha != PHYSICS_RT_2002_SHA256)
                out.push_back({n, "wrong", n + " 不是 2002 年原版（sha256 " + sha.substr(0, 12) + "…，应为 " +
                                               std::string(PHYSICS_RT_2002_SHA256).substr(0, 12) +
                                               "…）。Windows 版只能加载原版物理 DLL，请换回原版文件，或者改用全平台版。"});
        }
    }
    return out;
}

std::vector<GameProblem> check_engine(const std::string &binary) {
    if (binary.empty()) return {{"engine", "engine", "没有找到无头引擎，请在设置里指定引擎文件（BallanceMMOSimTool）。"}};
    if (!file_exists(binary)) return {{"engine", "engine", "找不到无头引擎文件：" + binary}};
    return {};
}

static std::string parent_dir(const std::string &p) {
    size_t i = p.find_last_of("/\\");
    return i == std::string::npos ? std::string(".") : p.substr(0, i);
}

std::string bundled_engine(const std::string &kind) {
    std::string d = parent_dir(platform::executable_path()) + "/engine/" + kind + "/" + platform_name() + "/";
    for (const char *n : {"BallanceMMOSimTool.exe", "BallanceMMOSimTool"})
        if (file_exists(d + n)) return d + n;
    return "";
}

std::vector<std::string> find_ballance_dirs() {
    std::vector<std::string> cand;
#ifdef _WIN32
    for (const char *drive : {"C:", "D:", "E:", "F:"})
        for (const char *sub : {"/Ballance", "/Games/Ballance", "/Program Files (x86)/Ballance", "/Program Files/Ballance",
                                "/Program Files (x86)/Atari/Ballance", "/游戏/Ballance"})
            cand.push_back(std::string(drive) + sub);
    std::string h = platform::home_dir();
    for (const char *sub : {"/Desktop/Ballance", "/Documents/Ballance", "/Downloads/Ballance", "/Games/Ballance"})
        cand.push_back(h + sub);
#else
    std::string h = platform::home_dir();
    for (const char *sub : {"/Ballance", "/Games/Ballance", "/Desktop/Ballance", "/Downloads/Ballance"}) cand.push_back(h + sub);
#endif
    std::vector<std::string> out;
    for (auto &c : cand)
        if (file_exists(c + "/base.cmo")) out.push_back(c);
    return out;
}

// ---------------------------------------------------------------------------------------------- log

ClientLog::ClientLog(std::string dir) : dir_(std::move(dir)) { platform::make_dirs(dir_); }

void ClientLog::line(const std::string &msg) {
    std::time_t t = std::time(nullptr);
    char ts[32];
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", &tm);
    std::string l = std::string(ts) + " " + msg;
    std::lock_guard<std::mutex> g(mu_);
    mem_.push_back(l.substr(11));  // the UI shows the time of day
    while (mem_.size() > 500) mem_.pop_front();
    std::string f = dir_ + "/client.log";
    if (platform::file_size(f) > (2 << 20)) {
        platform::remove_file(f + ".3");
        platform::rename_file(f + ".2", f + ".3");
        platform::rename_file(f + ".1", f + ".2");
        platform::rename_file(f, f + ".1");
    }
    std::ofstream fp(u8_path(f), std::ios::binary | std::ios::app);
    fp << l << '\n';
}

std::vector<std::string> ClientLog::tail(size_t n) const {
    std::lock_guard<std::mutex> g(mu_);
    size_t k = std::min(n, mem_.size());
    return std::vector<std::string>(mem_.end() - std::ptrdiff_t(k), mem_.end());
}

// ---------------------------------------------------------------------------------------------- controller

struct Controller::Slot {
    int index = 0;
    std::thread th;
    std::atomic<bool> quit{false}, done{false};
    // under Controller::mu_
    std::string state = "idle", task_id, detail;
    int64_t task_steps = 0, total_steps = 0, accepted = 0;
    double task_t0 = 0, busy_s = 0;
    int fails = 0;
    double retry_at = 0;
    bool fetched = false;  // a task was fetched and is not counted in tasks_done_ yet (this slot's thread only)
    std::string path_note;  // the last engine_view note logged (this slot's thread only)
    std::string last_task;  // the slot's previous task (its upload is waited for before the next fetch)
};

Controller::Controller(std::string data_dir, EventFn on_event)
    : dir_(std::move(data_dir)), on_event_(std::move(on_event)), log_(dir_ + "/logs") {
    // the one root everything goes under: refuse clearly when it cannot be written (restricted accounts)
    bool ok = !dir_.empty() && platform::make_dirs(dir_);
    if (ok) {
        std::string probe = dir_ + "/.write-test";
        try {
            write_file(probe, "ok");
            platform::remove_file(probe);
        } catch (...) {
            ok = false;
        }
    }
    if (!ok)
        throw Error("客户端的数据目录 " + (dir_.empty() ? std::string("（找不到用户目录）") : dir_) +
                    " 无法写入。请设置环境变量 BALLRAIN_WORK_DIR，指向一个你有写权限的文件夹（例如 D:\\ballrain），"
                    "客户端的所有文件都会放在那里。");
    platform::make_dirs(dir_ + "/queue");
    for (auto &t : platform::list_dir(dir_ + "/queue"))  // a crash right at queue_open: nothing to upload there
        if (!file_exists(dir_ + "/queue/" + t + "/meta.json")) remove_tree(dir_ + "/queue/" + t);
    for (auto &t : platform::list_dir(dir_ + "/queue")) {  // tasks an earlier run did not finish: posted, then closed
        std::string d = dir_ + "/queue/" + t;
        bool final = false;
        for (auto &f : platform::list_dir(d)) final = final || f.find(".final.") != std::string::npos;
        if (final || file_exists(d + "/end.json")) continue;
        Json e = Json::object();
        e.set("chunks", Json::integer(-1));
        e.set("reason", Json::string("client restarted"));
        write_atomic(d + "/end.json", e.dump());
    }
    platform::make_dirs(dir_ + "/cache");
    platform::make_dirs(dir_ + "/runs");
    cfg_ = ClientConfig::load(dir_);
    Bytes t;
    if (read_file(dir_ + "/token", t)) token_ = trim(t);
    t_start_ = platform::monotonic_s();
}

Controller::~Controller() { shutdown(); }

void Controller::emit(const std::string &kind, Json f) {
    if (on_event_) on_event_(kind, f);
}

void Controller::say(const std::string &msg) {
    log_.line(msg);
    Json f = Json::object();
    f.set("message", Json::string(msg));
    emit("log", f);
}

void Controller::set_banner(const std::string &kind, const Json *reason, const std::string &message) {
    Banner b;
    b.kind = kind, b.title = banner_title(kind);
    if (reason) b.code = reason->get_string("code", ""), b.message = reason->get_string("message", "");
    else b.message = message;
    {
        std::lock_guard<std::mutex> g(mu_);
        banner_ = b;
    }
    say("【" + b.title + "】" + b.message);
}

ClientConfig Controller::config() const {
    std::lock_guard<std::mutex> g(mu_);
    return cfg_;
}

void Controller::set_config(const ClientConfig &c) {
    bool reconnect;
    {
        std::lock_guard<std::mutex> g(mu_);
        reconnect = c.gateway_url != cfg_.gateway_url || c.ballance_dir != cfg_.ballance_dir ||
                    c.engine_binary != cfg_.engine_binary || c.engine_kind != cfg_.engine_kind;
        cfg_ = c;
        cfg_.instances = std::max(1, std::min(MAX_INSTANCES, cfg_.instances));
        if (reconnect && connected_) reconnect_wanted_ = true, next_reconnect_ = 0;
    }
    c.save(dir_);
    cv_.notify_all();
}

std::string Controller::engine_binary() const {
    ClientConfig c = config();
    return c.engine_binary.empty() ? bundled_engine(c.engine_kind) : platform::absolute_path(c.engine_binary);
}

std::vector<GameProblem> Controller::problems() const {
    ClientConfig c = config();
    auto out = check_game_dir(c.ballance_dir, c.engine_kind);
    auto e = check_engine(engine_binary());
    out.insert(out.end(), e.begin(), e.end());
    return out;
}

GatewayClient Controller::api() const {
    std::lock_guard<std::mutex> g(mu_);
    GatewayClient c(gateway_.empty() ? cfg_.gateway_url : gateway_, token_);
    c.session_id = session_id_;
    return c;
}

bool Controller::wait_for(double seconds, const std::function<bool()> &until) {
    std::unique_lock<std::mutex> lk(wait_mu_);
    cv_.wait_for(lk, std::chrono::duration<double>(seconds), [&] { return closing_.load() || (until && until()); });
    return !closing_;
}

// ---------------------------------------------------------------------------------------------- login

Json Controller::login_start() {
    GatewayClient g(config().gateway_url);
    Json s = g.login_start();
    say("请在浏览器中用论坛账号登录（登录代码 " + s.at("user_code").as_string() + "）。");
    return s;
}

std::string Controller::login_poll(const Json &started) {
    GatewayClient g(config().gateway_url);
    Json st;
    try {
        st = g.login_poll(started.at("login_id").as_string(), started.at("poll_secret").as_string());
    } catch (const GatewayRefusal &e) {
        if (e.code == "rate_limited") return "pending";
        set_banner("rejected", &e.reason);
        return "refused";
    } catch (const GatewayUnavailable &e) {
        log_.line(std::string("登录状态查询失败，稍后重试：") + e.what());
        return "pending";
    }
    std::string s = st.at("status").as_string();
    if (s == "ok") {
        std::string tok = st.at("token").as_string();
        write_atomic(dir_ + "/token", tok);
#ifndef _WIN32
        std::error_code pe;  // a credential: owner only (Windows: the user profile is the user's own)
        std::filesystem::permissions(dir_ + "/token", std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                                     std::filesystem::perm_options::replace, pe);
#endif
        {
            std::lock_guard<std::mutex> g2(mu_);
            token_ = tok, token_dead_ = false;
            if (banner_ && banner_->kind == "revoked") banner_.reset();
        }
        say("论坛登录成功，已保存算力池令牌。");
        Json f = Json::object();
        f.set("user", st.at("user"));
        emit("login_ok", f);
        return "ok";
    }
    if (s == "refused" || s == "expired") {
        if (const Json *r = st.find("reason")) set_banner("rejected", r);
        return s;
    }
    return "pending";
}

bool Controller::has_token() const {
    std::lock_guard<std::mutex> g(mu_);
    return !token_.empty();
}

void Controller::logout() {
    platform::remove_file(dir_ + "/token");
    {
        std::lock_guard<std::mutex> g(mu_);
        token_.clear(), session_id_.clear();
        connected_ = false, want_running_ = false;
    }
    say("已退出登录。");
    cv_.notify_all();
}

// ---------------------------------------------------------------------------------------------- session

bool Controller::connect() {
    ClientConfig c = config();
    std::string tok;
    {
        std::lock_guard<std::mutex> g(mu_);
        tok = token_;
    }
    try {
        check_gateway_url(c.gateway_url);
    } catch (const Error &e) {
        set_banner("error", nullptr, e.what());
        return false;
    }
    if (tok.empty()) {
        set_banner("error", nullptr, "还没有登录：请先点“登录”，在浏览器里用论坛账号登录。");
        return false;
    }
    auto probs = problems();
    if (!probs.empty()) {
        set_banner("error", nullptr, probs[0].message);
        return false;
    }
    Json eng, files;
    try {
        eng = engine_info(engine_binary(), c.engine_kind);
        files = hash_game_files(c.ballance_dir, c.engine_kind);
    } catch (const std::exception &e) {
        set_banner("error", nullptr, std::string("读取游戏文件失败：") + e.what());
        return false;
    }
    GatewayClient g(c.gateway_url, tok);
    Json h;
    try {
        h = g.hello(platform_name(), eng, files, c.instances);
    } catch (const GatewayRefusal &e) {
        refused(e, "hello");
        return false;
    } catch (const GatewayUnavailable &e) {
        set_banner("error", nullptr, e.what());
        {
            std::lock_guard<std::mutex> lk(mu_);
            reconnect_wanted_ = true, next_reconnect_ = platform::monotonic_s() + RECONNECT_S;
        }
        return false;
    }
    std::string name;
    {
        std::lock_guard<std::mutex> lk(mu_);
        gateway_ = check_gateway_url(c.gateway_url);
        session_id_ = g.session_id;
        engine_ = eng, game_files_ = files;
        user_ = h.at("user");
        name = user_.get_string("nickname", "");
        if (name.empty()) name = user_.get_string("username", "");
        user_name_ = name;
        leaderboard_ = h.at("leaderboard").as_bool();
        max_instances_ = std::min<int>(MAX_INSTANCES, int(h.at("max_instances").as_int()));
        heartbeat_s_ = std::max<int>(5, int(h.at("heartbeat_s").as_int()));
        notice_ = h.get_string("notice", "");
        connected_ = true, token_dead_ = false, reconnect_wanted_ = false;
        if (banner_ && (banner_->kind == "error" || banner_->code == "session_expired" ||
                        banner_->code == "too_many_instances"))
            banner_.reset();
        last_me_ = 0;
    }
    {
        std::lock_guard<std::mutex> cg(cache_mu_);  // downloads are anonymous: a token-less client for the cache
        if (!cache_api_) {
            cache_api_ = std::make_unique<GatewayClient>(c.gateway_url);
            cache_ = std::make_unique<PoolCache>(dir_ + "/cache", *cache_api_);
        } else {
            *cache_api_ = GatewayClient(c.gateway_url);
        }
    }
    try {  // the forum account of this data directory: diagnostics scrub it even when exported offline
        write_atomic(dir_ + "/user.json", h.at("user").dump(true));
    } catch (const Error &) {
    }
    say("已连接网关：" + name + "，上榜" + (h.at("leaderboard").as_bool() ? "开" : "关") + "，最多 " +
        std::to_string(int(h.at("max_instances").as_int())) + " 个实例。");
    Json f = Json::object();
    f.set("user", h.at("user"));
    f.set("leaderboard", h.at("leaderboard"));
    f.set("max_instances", h.at("max_instances"));
    emit("hello_ok", f);
    cv_.notify_all();
    return true;
}

void Controller::refused(const GatewayRefusal &e, const char *where) {
    set_banner(e.fatal ? "revoked" : "rejected", &e.reason);
    Json f = Json::object();
    f.set("where", Json::string(where));
    f.set("reason", e.reason);
    emit("refused", f);
    {
        std::lock_guard<std::mutex> g(mu_);
        if (e.fatal) {
            token_dead_ = true, want_running_ = false, connected_ = false;
        } else if (e.code == "session_expired" || e.code == "too_many_instances") {
            reconnect_wanted_ = true;  // hello again (a gateway restart, a long sleep, a changed instance count)
            if (e.code == "session_expired") connected_ = false, next_reconnect_ = 0;
        }
    }
    cv_.notify_all();
}

bool Controller::is_running() const {
    return want_running_ && connected_ && !token_dead_ && auto_pause_.empty();
}

void Controller::start() {
    bool need_connect;
    {
        std::lock_guard<std::mutex> g(mu_);
        want_running_ = true;
        for (auto &s : slots_) s->fails = 0, s->retry_at = 0;  // the player asks again: instances that gave up retry
        if (banner_ && banner_->kind == "error") banner_.reset();
        need_connect = !connected_ && !token_dead_;
    }
    say("开始采样。");
    if (need_connect) connect();
    cv_.notify_all();
}

void Controller::pause() {
    {
        std::lock_guard<std::mutex> g(mu_);
        want_running_ = false;
    }
    say("已暂停。");
    cv_.notify_all();
}

int Controller::set_instances(int n) {
    ClientConfig c;
    {
        std::lock_guard<std::mutex> g(mu_);
        n = std::max(1, std::min({n, max_instances_ > 0 ? max_instances_ : MAX_INSTANCES, MAX_INSTANCES}));
        cfg_.instances = n;
        if (banner_ && banner_->code == "too_many_instances") reconnect_wanted_ = true, next_reconnect_ = 0;
        c = cfg_;
    }
    c.save(dir_);
    cv_.notify_all();
    return n;
}

bool Controller::set_leaderboard(bool on) {
    GatewayClient g = api();
    Json r;
    try {
        r = g.set_leaderboard(on);
    } catch (const GatewayRefusal &e) {
        refused(e, "me");
        throw;
    }
    bool now;
    {
        std::lock_guard<std::mutex> lk(mu_);
        leaderboard_ = now = r.at("leaderboard").as_bool();
        stats_ = r.at("stats");
        user_ = r.at("user");
    }
    say(std::string("上榜：") + (now ? "开" : "关") + "。");
    return now;
}

// ---------------------------------------------------------------------------------------------- service

void Controller::start_service() {
    std::lock_guard<std::mutex> g(mu_);
    if (!threads_.empty()) return;
    auto guarded = [this](void (Controller::*fn)(), const char *name) {
        return [this, fn, name] {
            for (;;) {
                try {
                    (this->*fn)();
                    return;
                } catch (const std::exception &e) {  // logged, the loop starts again
                    log_.line(std::string(name) + " 内部错误：" + e.what());
                    Json f = Json::object();
                    f.set("error", Json::string(std::string(name) + ": " + e.what()));
                    emit("error", f);
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            }
        };
    };
    threads_.emplace_back(guarded(&Controller::service_loop, "service"));
    threads_.emplace_back(guarded(&Controller::upload_loop, "upload"));
}

void Controller::shutdown() {
    if (closing_.exchange(true)) {
        if (stop_) return;
    }
    cv_.notify_all();
    std::vector<std::unique_ptr<Slot>> slots;
    {
        std::lock_guard<std::mutex> g(mu_);
        for (auto &s : slots_) s->quit = true;
    }
    for (;;) {  // the instances end their tasks ("stopped"), their engines stop
        bool all = true;
        {
            std::lock_guard<std::mutex> g(mu_);
            for (auto &s : slots_) all = all && s->done.load();
        }
        if (all) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    {
        std::lock_guard<std::mutex> g(mu_);
        slots.swap(slots_);
    }
    for (auto &s : slots)
        if (s->th.joinable()) s->th.join();
    double until = platform::monotonic_s() + 5.0;  // the uploader posts what the stop produced (the rest waits on disk)
    while (platform::monotonic_s() < until && !threads_.empty()) {
        bool empty = platform::list_dir(dir_ + "/queue").empty();
        bool ok;
        {
            std::lock_guard<std::mutex> g(mu_);
            ok = connected_ && !token_dead_;
        }
        if (empty || !ok) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    stop_ = true;
    cv_.notify_all();
    for (auto &t : threads_)
        if (t.joinable()) t.join();
    threads_.clear();
}

void Controller::service_loop() {
    double last_hb = -1e9;  // the first heartbeat right after connecting
    while (!closing_) {
        double now = platform::monotonic_s();
        try {
            if (now - last_probe_ >= PROBE_S) last_probe_ = now, probe();
            bool rec, conn;
            {
                std::lock_guard<std::mutex> g(mu_);
                rec = reconnect_wanted_ && !token_dead_ && !token_.empty() && now >= next_reconnect_;
                if (rec) next_reconnect_ = now + RECONNECT_S;
                conn = connected_;
            }
            if (rec) {
                say("重新连接网关……");
                emit("rehello");
                connect();
            }
            int hb;
            {
                std::lock_guard<std::mutex> g(mu_);
                hb = heartbeat_s_;
            }
            if (conn && now - last_hb >= hb) last_hb = now, heartbeat_once();
            if (conn && now - last_me_ >= ME_S) {
                last_me_ = now;
                try {
                    Json r = api().me();
                    std::lock_guard<std::mutex> g(mu_);
                    stats_ = r.at("stats");
                    leaderboard_ = r.at("leaderboard").as_bool();
                } catch (const GatewayRefusal &e) {
                    refused(e, "me");
                } catch (const GatewayUnavailable &) {
                }
            }
            reconcile();
        } catch (const std::exception &e) {
            log_.line(std::string("内部错误：") + e.what());
        }
        wait_for(0.25);
    }
}

void Controller::heartbeat_once() {
    Json list = Json::array();
    int64_t total = 0;
    bool running;
    {
        std::lock_guard<std::mutex> g(mu_);
        running = is_running();
        for (auto &s : slots_) {
            Json i = Json::object();
            i.set("slot", Json::integer(s->index));
            i.set("state", Json::string(s->state));
            i.set("steps", Json::integer(s->total_steps + s->task_steps));
            i.set("task_id", s->task_id.empty() ? Json() : Json::string(s->task_id));
            list.push(i);
            total += s->total_steps + s->task_steps;
        }
    }
    double el = platform::monotonic_s() - t_start_;
    Json r;
    try {
        r = api().heartbeat(running, list, el > 0 ? double(total) / el : 0.0);
    } catch (const GatewayUnavailable &e) {  // the next one will tell; the tasks go on
        log_.line(std::string("心跳失败：") + e.what());
        return;
    } catch (const GatewayRefusal &e) {
        refused(e, "heartbeat");
        return;
    }
    emit("heartbeat");
    if (const Json *rv = r.find("revoked")) {
        set_banner("revoked", rv);
        std::lock_guard<std::mutex> g(mu_);
        token_dead_ = true, want_running_ = false, connected_ = false;
        return;
    }
    std::lock_guard<std::mutex> g(mu_);
    max_instances_ = std::min<int>(MAX_INSTANCES, int(r.at("max_instances").as_int()));
    notice_ = r.get_string("notice", notice_);
}

void Controller::probe() {
    ClientConfig c = config();
    std::string reason;
    if (!c.quiet_file.empty() && (file_exists(c.quiet_file) || platform::is_dir(c.quiet_file)))
        reason = "暂停开关 " + c.quiet_file + " 存在，已自动暂停；删除它后会自动继续。";
    else if (c.auto_pause_player && platform::process_running("Player.exe")) reason = PAUSE_PLAYER;
    else if (c.auto_pause_battery && platform::on_battery()) reason = PAUSE_BATTERY;
    bool changed;
    {
        std::lock_guard<std::mutex> g(mu_);
        changed = reason != auto_pause_;
        auto_pause_ = reason;
    }
    if (changed) {
        say(reason.empty() ? std::string("自动暂停的条件已解除。") : reason);
        cv_.notify_all();
    }
}

void Controller::reconcile() {
    std::lock_guard<std::mutex> g(mu_);
    int want = std::min(cfg_.instances, std::max(1, max_instances_));
    while (int(slots_.size()) < want) {
        auto s = std::make_unique<Slot>();
        s->index = int(slots_.size());
        Slot *p = s.get();
        slots_.push_back(std::move(s));
        p->th = std::thread([this, p] { slot_loop(p); });
    }
    for (auto &s : slots_) s->quit = s->index >= want;
    while (!slots_.empty() && slots_.back()->quit && slots_.back()->done) {  // fewer instances: the last ones end
        if (slots_.back()->th.joinable()) slots_.back()->th.join();
        slots_.pop_back();
    }
}

// ---------------------------------------------------------------------------------------------- instances

void Controller::slot_loop(Slot *s) {
    while (!closing_ && !s->quit) {
        bool can;
        {
            std::lock_guard<std::mutex> g(mu_);
            double now = platform::monotonic_s();
            bool fetch_ok = max_tasks <= 0 || tasks_fetched_ < max_tasks;
            can = is_running() && fetch_ok && s->fails < CRASH_LIMIT && now >= s->retry_at;
            if (!can) {
                s->state = s->fails >= CRASH_LIMIT ? "error" : (want_running_ && connected_ && fetch_ok) ? "paused" : "idle";
                if (s->fails < CRASH_LIMIT && now < s->retry_at) s->state = "starting";
            }
        }
        if (!can) {
            wait_for(0.5);
            continue;
        }
        if (queue_bytes() > QUEUE_LIMIT) {
            {
                std::lock_guard<std::mutex> g(mu_);
                s->state = "paused", s->detail = "上传队列已满，等网络恢复";
            }
            wait_for(5.0);
            continue;
        }
        try {
            run_one_task(s);
        } catch (const std::exception &e) {  // never out of the thread (std::terminate): logged, the slot backs off
            if (s->fetched) ++tasks_done_, s->fetched = false;
            log_.line(std::string("实例 ") + std::to_string(s->index + 1) + " 内部错误：" + e.what());
            Json f = Json::object();
            f.set("error", Json::string(e.what()));
            emit("error", f);
            std::lock_guard<std::mutex> g(mu_);
            s->state = "idle", s->task_id.clear(), s->task_steps = 0;
            s->fails++, s->retry_at = platform::monotonic_s() + 30.0;
        }
    }
    std::lock_guard<std::mutex> g(mu_);
    s->state = "idle", s->task_id.clear(), s->task_steps = 0;
    s->done = true;
}

bool Controller::run_one_task(Slot *s) {
    if (max_tasks > 0 && tasks_fetched_.fetch_add(1) >= max_tasks) {
        --tasks_fetched_;
        return false;
    }
    auto undo = [&] {
        if (max_tasks > 0) --tasks_fetched_;
    };
    // the previous task of this slot is open at the gateway until its last chunk (or task_end) arrived: wait for the
    // uploader first, else a slot at the gateway's max_instances gets too_many_instances (uploads are asynchronous)
    if (!s->last_task.empty()) {
        std::string d = dir_ + "/queue/" + s->last_task;
        {
            std::lock_guard<std::mutex> g(mu_);
            s->state = "starting", s->detail = "等上一个任务的数据上传";
        }
        double until = platform::monotonic_s() + 600.0;
        while (platform::is_dir(d) && platform::monotonic_s() < until && !closing_ && !s->quit) wait_for(0.25);
        s->last_task.clear();
    }
    {
        std::lock_guard<std::mutex> g(mu_);
        s->state = "starting", s->detail = "取任务";
    }
    GatewayClient g = api();
    Json tm;
    try {
        tm = g.task();
    } catch (const GatewayUnavailable &e) {
        undo();
        log_.line(std::string("取任务失败：") + e.what());
        Json f = Json::object();
        f.set("error", Json::string(e.what()));
        emit("task_unavailable", f);
        wait_for(30.0);
        return false;
    } catch (const GatewayRefusal &e) {
        undo();
        refused(e, "task");
        if (e.code == "too_many_instances") wait_for(10.0);  // the slots of a closing task free up shortly
        else if (!e.fatal && e.code != "session_expired") wait_for(30.0);
        return false;
    }
    if (tm.at("type").as_string() == "no_task") {
        undo();
        Json f = Json::object();
        f.set("reason", tm.at("reason"));
        emit("no_task", f);
        {
            std::lock_guard<std::mutex> lk(mu_);
            s->state = "idle", s->detail = tm.at("reason").get_string("message", "");
        }
        wait_for(double(tm.at("retry_s").as_int()));
        return false;
    }
    s->fetched = true;
    Task task = Task::from_json(tm.at("task"));
    s->last_task = task.task_id;
    queue_open(task, s->index);
    double t0 = platform::monotonic_s();
    {
        std::lock_guard<std::mutex> lk(mu_);
        s->state = "running", s->task_id = task.task_id, s->task_steps = 0, s->task_t0 = t0, s->detail.clear();
    }
    {
        Json f = Json::object();
        f.set("task_id", Json::string(task.task_id));
        f.set("slot", Json::integer(s->index));
        f.set("start_frame", Json::integer(task.start_frame));
        f.set("ball_order_seed", Json::integer(task.ball_order_seed));
        emit("task", f);
    }
    ClientConfig c = config();
    LocalEngine le;
    le.binary = engine_binary(), le.root = c.ballance_dir, le.kind = c.engine_kind, le.extra_env = engine_env;
    Json files;
    {
        std::lock_guard<std::mutex> lk(mu_);
        files = game_files_;
    }
    double deadline = t0 + double(task.deadline_s) - std::min(30.0, double(task.deadline_s) / 10);
    char rn[32];
    std::snprintf(rn, sizeof rn, "/runs/slot-%02d", s->index + 1);
    // the engine writes its run files by narrow paths too: a non-ASCII data directory runs under the work directory
    std::string run_dir;
    TaskOutcome out;
    std::string error;
    bool threw = false;
    try {
        PoolCache *cache;
        {
            std::lock_guard<std::mutex> cg(cache_mu_);
            cache = cache_.get();
        }
        if (!cache) throw Error("not connected");
        run_dir = (plain_ascii(dir_) ? dir_ : engine_work_dir(dir_)) + rn;
        std::string note;
        le = engine_view(le, files, dir_, &note);
        if (!note.empty() && note != s->path_note) say(note), s->path_note = note;
        out = play_task(
            task, *cache, cache_mu_, le, run_dir,
            [&](int64_t seq, const Bytes &data, bool final) { queue_chunk(task.task_id, seq, data, final); },
            [&]() -> std::string {
                if (closing_ || s->quit) return "stopped";
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    if (!is_running() || refused_.count(task.task_id)) return "stopped";
                }
                return platform::monotonic_s() >= deadline ? "deadline" : "";
            },
            [&](int64_t st) {
                std::lock_guard<std::mutex> lk(mu_);
                s->task_steps = st;
            });
    } catch (const std::exception &e) {
        threw = true;
        error = e.what();
    }
    double now = platform::monotonic_s();
    bool failed = threw || out.end == "error";
    if (failed) {
        if (!threw) error = out.error;
        if (!out.final_chunk) queue_end(task.task_id, out.chunks, "failed: " + error);
        std::string kept = run_dir + ".headless.log";
        if (file_exists(kept)) {
            char ln[32];
            std::snprintf(ln, sizeof ln, "/engine-%02d.log", s->index + 1);
            platform::rename_file(kept, log_.dir() + ln);
        }
        int fails;
        {
            std::lock_guard<std::mutex> lk(mu_);
            s->fails = now - t0 >= STABLE_S ? 1 : s->fails + 1;
            fails = s->fails;
            if (fails < CRASH_LIMIT) s->retry_at = now + std::min(2.0 * double(1 << std::min(fails - 1, 10)), 300.0);
        }
        Json f = Json::object();
        f.set("task_id", Json::string(task.task_id));
        f.set("error", Json::string(error));
        f.set("failures", Json::integer(fails));
        emit("task_failed", f);
        if (fails >= CRASH_LIMIT) {
            set_banner("error", nullptr, "实例 " + std::to_string(s->index + 1) + " 连续 " + std::to_string(fails) +
                                             " 次出错，已停止重试。请查看数据目录下 logs 里的 client.log 和 engine-" +
                                             (s->index + 1 < 10 ? "0" : "") + std::to_string(s->index + 1) +
                                             ".log；按“开始”可以再试。");
        } else {
            say("实例 " + std::to_string(s->index + 1) + " 的任务出错（" + error.substr(0, 200) + "），稍后重试（连续第 " +
                std::to_string(fails) + " 次）。");
        }
    } else {
        std::lock_guard<std::mutex> lk(mu_);
        s->fails = 0;
    }
    if (!failed && !out.final_chunk) queue_end(task.task_id, out.chunks, "end " + out.end + " without a final chunk");
    if (!threw) {
        Json f = Json::object();
        f.set("task_id", Json::string(task.task_id));
        f.set("slot", Json::integer(s->index));
        f.set("end", Json::string(out.end));
        f.set("steps", Json::integer(out.steps));
        f.set("frames", Json::integer(out.frames));
        f.set("episodes", Json::integer(out.episodes));
        f.set("chunks", Json::integer(out.chunks));
        f.set("wall_s", Json::number(out.wall_s));
        f.set("steps_per_s", Json::number(out.wall_s > 0 ? double(out.steps) / out.wall_s : 0.0));
        emit("task_end", f);
    }
    ++tasks_done_;
    s->fetched = false;
    std::lock_guard<std::mutex> lk(mu_);
    s->total_steps += out.steps;
    s->busy_s += now - t0;
    s->state = "idle", s->task_id.clear(), s->task_steps = 0;
    refused_.erase(task.task_id);
    return true;
}

// ---------------------------------------------------------------------------------------------- upload queue
// <data dir>/queue/<task_id>/: meta.json {deadline (unix s)}, <seq:06>.brpc / <seq:06>.final.brpc, end.json {chunks,
// reason}. The uploader posts each task's chunks in order; a chunk survives a network loss or a client restart until
// the task's deadline. A task without a final chunk is closed with task_end once its chunks are all posted.

void Controller::queue_open(const Task &t, int slot) {
    std::string d = dir_ + "/queue/" + t.task_id;
    platform::make_dirs(d);
    Json m = Json::object();
    m.set("task_id", Json::string(t.task_id));
    m.set("deadline", Json::integer(t.issued_at + t.deadline_s));
    write_atomic(d + "/meta.json", m.dump());
    std::lock_guard<std::mutex> g(mu_);
    task_slot_[t.task_id] = slot;
}

void Controller::queue_chunk(const std::string &task_id, int64_t seq, const Bytes &data, bool final) {
    char n[40];
    std::snprintf(n, sizeof n, "/%06lld%s.brpc", (long long)seq, final ? ".final" : "");
    platform::make_dirs(dir_ + "/queue/" + task_id);
    write_atomic(dir_ + "/queue/" + task_id + n, data);
    cv_.notify_all();
}

void Controller::queue_end(const std::string &task_id, int64_t chunks, const std::string &reason) {
    Json e = Json::object();
    e.set("chunks", Json::integer(chunks));
    e.set("reason", Json::string(reason.substr(0, 300)));
    if (!file_exists(dir_ + "/queue/" + task_id + "/meta.json")) return;  // dropped already (refused / expired)
    write_atomic(dir_ + "/queue/" + task_id + "/end.json", e.dump());
    cv_.notify_all();
}

int64_t Controller::queue_bytes() const {
    int64_t n = 0;
    for (auto &t : platform::list_dir(dir_ + "/queue"))
        for (auto &f : platform::list_dir(dir_ + "/queue/" + t))
            if (f.size() > 5 && f.compare(f.size() - 5, 5, ".brpc") == 0) n += platform::file_size(dir_ + "/queue/" + t + "/" + f);
    return n;
}

void Controller::upload_loop() {
    double backoff = 2.0;
    while (!stop_) {
        bool ok;
        {
            std::lock_guard<std::mutex> g(mu_);
            ok = connected_ && !token_dead_ && !session_id_.empty();
        }
        if (!ok) {
            std::unique_lock<std::mutex> lk(wait_mu_);
            cv_.wait_for(lk, std::chrono::milliseconds(500), [&] { return stop_.load(); });
            continue;
        }
        auto tasks = platform::list_dir(dir_ + "/queue");
        std::sort(tasks.begin(), tasks.end());
        bool progress = false, down = false;
        for (auto &t : tasks) {
            if (stop_) break;
            try {
                if (upload_task(t)) progress = true;
                else down = true;
            } catch (const std::exception &e) {  // retried next round (the chunks stay until the task's deadline)
                log_.line("上传队列：" + t.substr(0, 12) + "：" + e.what());
            }
            if (down) break;
        }
        std::unique_lock<std::mutex> lk(wait_mu_);
        if (down) {
            cv_.wait_for(lk, std::chrono::duration<double>(backoff), [&] { return stop_.load(); });
            backoff = std::min(backoff * 2, 30.0);
        } else {
            backoff = 2.0;
            if (!progress) cv_.wait_for(lk, std::chrono::milliseconds(500), [&] { return stop_.load(); });
        }
    }
}

// true: progress or nothing to do yet; false: the gateway is out of reach (back off)
bool Controller::upload_task(const std::string &task_id) {
    std::string d = dir_ + "/queue/" + task_id;
    Bytes mb;
    // no meta.json yet: queue_open is between creating the directory and writing it (a directory a crash left
    // that way is cleaned up at the next start, in the constructor)
    if (!read_file(d + "/meta.json", mb)) return true;
    Json meta;
    try {
        meta = Json::parse(mb, 4);
    } catch (const Error &) {  // corrupt (write_atomic cannot leave this): drop it
        remove_tree(d);
        return true;
    }
    if (platform::unix_time() >= meta.get_int("deadline", 0)) {
        say("任务 " + task_id.substr(0, 8) + " 已过期，丢弃还没上传的数据段。");
        remove_tree(d);
        return true;
    }
    std::vector<std::string> chunks;
    bool has_end = false;
    for (auto &f : platform::list_dir(d)) {
        if (f.size() > 5 && f.compare(f.size() - 5, 5, ".brpc") == 0) chunks.push_back(f);
        if (f == "end.json") has_end = true;
    }
    std::sort(chunks.begin(), chunks.end());
    GatewayClient g = api();
    auto credit = [&](int64_t steps) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = task_slot_.find(task_id);
        if (it != task_slot_.end())
            for (auto &s : slots_)
                if (s->index == it->second) s->accepted += steps;
    };
    auto drop = [&](const std::string &why) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            refused_[task_id] = why;  // the instance playing it stops
            task_slot_.erase(task_id);
        }
        remove_tree(d);
    };
    if (!chunks.empty()) {
        const std::string &f = chunks.front();
        bool final = f.find(".final.") != std::string::npos;
        int64_t seq = std::atoll(f.substr(0, 6).c_str());
        Bytes data;
        if (!read_file(d + "/" + f, data)) {  // briefly locked (e.g. a virus scanner): next round
            if (++read_fail_[d + "/" + f] < 60) return true;
            log_.line("上传队列：数据段 " + task_id.substr(0, 8) + "/" + f + " 一直读不出来，丢弃这个任务。");
            read_fail_.erase(d + "/" + f);
            remove_tree(d);
            return true;
        }
        read_fail_.erase(d + "/" + f);
        Json r;
        try {
            r = g.chunk(data);
        } catch (const GatewayUnavailable &e) {
            log_.line(std::string("上传失败，稍后重试：") + e.what());
            return false;
        } catch (const GatewayRefusal &e) {
            if (e.code == "chunk_duplicate") {  // an earlier attempt arrived
                platform::remove_file(d + "/" + f);
                if (final) remove_tree(d);
                return true;
            }
            if (e.code == "session_expired" || e.code == "rate_limited" || e.code == "server_busy" ||
                e.code == "internal_error") {  // transient: retried (the chunk stays queued)
                if (e.code == "session_expired") refused(e, "chunk");
                return false;
            }
            Json ff = Json::object();
            ff.set("task_id", Json::string(task_id));
            ff.set("seq", Json::integer(seq));
            ff.set("reason", e.reason);
            emit("chunk_refused", ff);
            refused(e, "chunk");
            drop(e.message);
            return true;
        }
        Json ff = Json::object();
        ff.set("task_id", Json::string(task_id));
        ff.set("seq", r.at("seq"));
        ff.set("status", r.at("status"));
        ff.set("verify", r.at("verify"));
        ff.set("steps", r.at("steps"));
        if (const Json *rs = r.find("reason")) ff.set("reason", *rs);
        emit("chunk", ff);
        if (r.at("status").as_string() == "rejected" || r.get_bool("token_revoked", false)) {
            const Json *rs = r.find("reason");
            bool fatal = r.get_bool("token_revoked", false);
            set_banner(fatal ? "revoked" : "rejected", rs, rs ? "" : "数据段被网关拒收。");
            if (fatal) {
                std::lock_guard<std::mutex> lk(mu_);
                token_dead_ = true, want_running_ = false, connected_ = false;
            }
            drop(rs ? rs->get_string("message", "") : "rejected");
            return true;
        }
        credit(r.at("steps").as_int());
        platform::remove_file(d + "/" + f);
        if (final) {
            remove_tree(d);
            std::lock_guard<std::mutex> lk(mu_);
            task_slot_.erase(task_id);
        }
        return true;
    }
    if (!has_end) return true;  // still being played
    Bytes eb;
    read_file(d + "/end.json", eb);
    Json e = Json::parse(eb, 4);
    int64_t n = e.get_int("chunks", 0);
    for (int attempt = 0; attempt < 2; ++attempt) {
        try {
            g.task_end(task_id, std::max<int64_t>(n, 0), e.get_string("reason", ""));
            Json ff = Json::object();
            ff.set("task_id", Json::string(task_id));
            ff.set("chunks", Json::integer(std::max<int64_t>(n, 0)));
            emit("task_closed", ff);
            break;
        } catch (const GatewayUnavailable &) {
            return false;
        } catch (const GatewayRefusal &ex) {
            // an orphan of an earlier run (chunks -1): the gateway's own count once (it only frees the slot early)
            const Json *pa = ex.reason.find("params");
            if (n < 0 && attempt == 0 && ex.code == "chunk_out_of_order" && pa && pa->find("expected")) {
                n = pa->at("expected").as_int();
                continue;
            }
            log_.line("task_end " + task_id.substr(0, 8) + "：" + ex.message);  // unknown / closed: the deadline closes it
            if (ex.fatal) refused(ex, "task_end");
            break;
        }
    }
    remove_tree(d);
    std::lock_guard<std::mutex> lk(mu_);
    task_slot_.erase(task_id);
    return true;
}

// ---------------------------------------------------------------------------------------------- view

bool Controller::finished() const {
    if (max_tasks <= 0 || tasks_done_ < max_tasks) return false;
    if (!platform::list_dir(dir_ + "/queue").empty()) return false;
    std::lock_guard<std::mutex> g(mu_);
    for (auto &s : slots_)
        if (!s->task_id.empty()) return false;
    return true;
}

ControllerView Controller::view() const {
    ControllerView v;
    int64_t queued = 0;
    for (auto &t : platform::list_dir(dir_ + "/queue"))
        for (auto &f : platform::list_dir(dir_ + "/queue/" + t))
            if (f.size() > 5 && f.compare(f.size() - 5, 5, ".brpc") == 0) ++queued;
    v.log = log_.tail(200);
    std::lock_guard<std::mutex> g(mu_);
    v.has_token = !token_.empty();
    v.connected = connected_, v.want_running = want_running_, v.running = is_running(), v.token_dead = token_dead_;
    v.auto_pause = auto_pause_, v.banner = banner_, v.user_name = user_name_, v.notice = notice_;
    v.leaderboard = leaderboard_, v.instances = cfg_.instances, v.max_instances = max_instances_, v.stats = stats_;
    v.queued_chunks = queued, v.tasks_done = tasks_done_;
    v.update_version = update_version_, v.update_url = update_url_, v.update_notes = update_notes_;
    double now = platform::monotonic_s();
    for (auto &s : slots_) {
        InstanceView i;
        i.slot = s->index, i.state = s->state, i.state_text = state_text(s->state), i.task_id = s->task_id;
        i.detail = s->detail, i.task_steps = s->task_steps, i.total_steps = s->total_steps + s->task_steps;
        i.accepted_steps = s->accepted;
        double busy = s->busy_s + (s->task_id.empty() ? 0.0 : now - s->task_t0);
        i.steps_per_s = busy > 0 ? double(i.total_steps) / busy : 0.0;
        v.inst.push_back(i);
    }
    return v;
}

// ---------------------------------------------------------------------------------------------- diagnostics, updates

DiagFiles Controller::diagnostics() const {
    std::vector<std::pair<std::string, std::string>> rep;
    Json info = Json::object();
    Json saved_user;
    Bytes ub;
    if (read_file(dir_ + "/user.json", ub)) {
        try {
            saved_user = Json::parse(ub, 4);
        } catch (const Error &) {
        }
    }
    {
        std::lock_guard<std::mutex> g(mu_);
        Json user = user_.is_object() ? user_ : saved_user;
        if (!token_.empty()) rep.push_back({token_, "<令牌>"});
        for (const char *k : {"username", "nickname"}) {
            std::string v = user.is_object() ? user.get_string(k, "") : "";
            if (!v.empty()) rep.push_back({v, std::string("<论坛") + (k[0] == 'u' ? "用户名" : "昵称") + ">"});
        }
        if (user.is_object())
            if (const Json *id = user.find("id"); id && id->is_number()) {
                std::string sid = std::to_string(id->as_int());
                if (sid.size() >= 4) rep.push_back({"\"id\":" + sid, "\"id\":\"<论坛 id>\""});
            }
        info.set("client_version", Json::string(CLIENT_VERSION));
        info.set("platform", Json::string(platform_name() + " " + release_platform_key()));
        info.set("engine", engine_);
        info.set("game_files", game_files_);
        info.set("instances", Json::integer(cfg_.instances));
        info.set("max_instances", Json::integer(max_instances_));
        info.set("connected", Json::boolean(connected_));
        info.set("want_running", Json::boolean(want_running_));
        info.set("auto_pause", Json::string(auto_pause_));
        if (banner_) {
            Json b = Json::object();
            b.set("kind", Json::string(banner_->kind));
            b.set("code", Json::string(banner_->code));
            b.set("message", Json::string(banner_->message));
            info.set("banner", b);
        }
        info.set("stats", stats_);
        info.set("tasks_done", Json::integer(tasks_done_));
        Json sl = Json::array();
        for (auto &s : slots_) {
            Json j = Json::object();
            j.set("slot", Json::integer(s->index));
            j.set("state", Json::string(s->state));
            j.set("fails", Json::integer(s->fails));
            j.set("total_steps", Json::integer(s->total_steps));
            j.set("accepted_steps", Json::integer(s->accepted));
            sl.push(j);
        }
        info.set("instances_state", sl);
        info.set("update_version", Json::string(update_version_));
    }
    rep.push_back({dir_, "<数据目录>"});
    std::string home = platform::home_dir(), user = platform::user_name();
    if (!home.empty()) rep.push_back({home, "~"});
    if (user.size() >= 2) rep.push_back({user, "<用户>"});
    info.set("queued_chunks", Json::integer(view().queued_chunks));
    DiagFiles out;
    out.push_back({"info.json", scrub_text(info.dump(true), rep)});
    Bytes b;
    if (read_file(dir_ + "/config.json", b)) out.push_back({"config.json", scrub_text(b, rep)});
    std::vector<std::string> logs = platform::list_dir(log_.dir());
    std::sort(logs.begin(), logs.end());
    for (auto &n : logs) {
        if (n.rfind("client.log", 0) != 0 && !(n.rfind("engine-", 0) == 0 && n.find(".log") != std::string::npos)) continue;
        Bytes lb;
        if (!read_file(log_.dir() + "/" + n, lb)) continue;
        if (lb.size() > (4 << 20)) lb = lb.substr(lb.size() - (4 << 20));
        out.push_back({"logs/" + n, scrub_text(lb, rep)});
    }
    return out;
}

std::string Controller::export_diagnostics(const std::string &zip_path) {
    std::string p = zip_path;
    if (p.empty()) {
        std::time_t t = std::time(nullptr);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char ts[32];
        std::strftime(ts, sizeof ts, "%Y%m%d-%H%M%S", &tm);
        p = dir_ + "/diagnostics-" + ts + ".zip";
    }
    write_zip(diagnostics(), p);
    say("诊断信息已导出：" + p);
    return p;
}

void Controller::check_update(const std::string &url, const std::string &pubkey) {
    std::string err;
    auto r = check_for_update(url, pubkey, CLIENT_VERSION, release_platform_key(), &err);
    if (r) {
        {
            std::lock_guard<std::mutex> g(mu_);
            update_version_ = r->version, update_url_ = r->url, update_notes_ = r->notes;
        }
        say("有新版本 " + r->version + "：" + r->url);
        Json f = Json::object();
        f.set("version", Json::string(r->version));
        f.set("url", Json::string(r->url));
        emit("update_available", f);
    } else if (!err.empty()) {
        log_.line(err);
    }
}

}  // namespace ballrain
