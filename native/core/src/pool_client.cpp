#include "ballrain/pool_client.h"

#include <algorithm>
#include <cctype>
#include <cmath>

#include "ballrain/npz.h"
#include "ballrain/sha256.h"
#include "platform.h"

namespace ballrain {

#ifndef BALLRAIN_CLIENT_VERSION
#define BALLRAIN_CLIENT_VERSION "0.0.0-dev"
#endif
// native/CMakeLists.txt project(VERSION): 0.3 = chunk format 3 (engine-state chain v3, task format 3)
const char *const CLIENT_VERSION = BALLRAIN_CLIENT_VERSION;
const std::vector<const char *> CLIENT_CAPABILITIES = {"task_format3", "branch_v1"};
static const size_t MAX_RESPONSE = 64 * 1024, MAX_MANIFEST = 256 * 1024, MAX_DOWNLOAD = 64 * 1024 * 1024;

GatewayRefusal::GatewayRefusal(Json r, int st)
    : Error(r.get_string("code", "?") + ": " + r.get_string("message", "")), reason(std::move(r)), http_status(st) {
    code = reason.get_string("code", "");
    message = reason.get_string("message", "");
    fatal = code == "token_missing" || code == "token_invalid" || code == "token_revoked" || code == "token_revoked_replay";
}

std::string check_gateway_url(const std::string &url) {
    std::string u = url;
    while (!u.empty() && (u.back() == '/' || u.back() == ' ')) u.pop_back();
    size_t se = u.find("://");
    if (se == std::string::npos) throw Error("网关地址必须以 https:// 开头");
    std::string scheme = u.substr(0, se);
    for (auto &c : scheme) c = char(std::tolower(static_cast<unsigned char>(c)));
    std::string rest = u.substr(se + 3);
    std::string hostport = rest.substr(0, rest.find('/'));
    if (hostport.empty() || u.find('?') != std::string::npos || u.find('#') != std::string::npos ||
        hostport.find('@') != std::string::npos)
        throw Error("网关地址不能带查询参数或账号信息");
    std::string host = hostport[0] == '[' ? hostport.substr(1, hostport.find(']') - 1) : hostport.substr(0, hostport.rfind(':'));
    if (hostport[0] != '[' && hostport.find(':') == std::string::npos) host = hostport;
    for (auto &c : host) c = char(std::tolower(static_cast<unsigned char>(c)));
    if (scheme == "https") return u;
    if (scheme == "http" && (host == "localhost" || host == "127.0.0.1" || host == "::1")) return u;
    throw Error("网关地址必须使用 https（只有本机测试网关可以用 http）");
}

Json make_message(const std::string &type, Json fields) {
    Json m = Json::object();
    m.set("protocol", Json::integer(1));
    m.set("type", Json::string(type));
    for (auto &kv : fields.members()) m.set(kv.first, kv.second);
    return m;
}

GatewayClient::GatewayClient(const std::string &base_url, std::string tok, double timeout_s)
    : token(std::move(tok)), base_(check_gateway_url(base_url)), timeout_s_(timeout_s) {}

Json GatewayClient::call(const std::string &method, const std::string &path, const std::vector<std::string> &expect,
                         const Bytes &body, const std::string &ctype, bool auth, bool session) {
    std::vector<std::pair<std::string, std::string>> h{{"User-Agent", std::string("ballrain-native-client/") + CLIENT_VERSION},
                                                       {"Accept", "application/json"}};
    if (!ctype.empty()) h.push_back({"Content-Type", ctype});
    if (auth && !token.empty()) h.push_back({"Authorization", "Bearer " + token});
    if (session && !session_id.empty()) h.push_back({"X-Pool-Session", session_id});
    platform::HttpResponse r;
    try {
        r = platform::http_request(method, base_ + path, h, body, timeout_s_, MAX_RESPONSE);
    } catch (const Error &e) {
        throw GatewayUnavailable(std::string("网关连接失败：") + e.what());
    }
    Json obj;
    try {
        obj = Json::parse(r.body, 13);
    } catch (const Error &) {
        throw GatewayUnavailable("网关的响应不是协议消息（HTTP " + std::to_string(r.status) + "）");
    }
    std::string kind = obj.is_object() ? obj.get_string("type", "") : "";
    if (obj.get_int("protocol", 0) != 1) throw GatewayUnavailable("网关的响应协议版本不对");
    if (kind == "error") {
        const Json *rs = obj.find("reason");
        if (!rs || !rs->is_object() || !rs->has("code") || !rs->has("message")) throw GatewayUnavailable("网关的出错响应格式不对");
        throw GatewayRefusal(*rs, r.status);
    }
    if (std::find(expect.begin(), expect.end(), kind) == expect.end())
        throw GatewayUnavailable("网关返回了意外的消息 " + kind.substr(0, 40) + "（HTTP " + std::to_string(r.status) + "）");
    return obj;
}

static Bytes json_body(const Json &m) { return m.dump(true); }
static const char *JSON_CT = "application/json; charset=utf-8";

Json GatewayClient::login_start() {
    Json f = Json::object();
    f.set("client_version", Json::string(CLIENT_VERSION));
    return call("POST", "/pool/login", {"login_started"}, json_body(make_message("login_start", f)), JSON_CT, false, false);
}

Json GatewayClient::login_poll(const std::string &id, const std::string &secret) {
    Json f = Json::object();
    f.set("login_id", Json::string(id));
    f.set("poll_secret", Json::string(secret));
    return call("POST", "/pool/login", {"login_status"}, json_body(make_message("login_poll", f)), JSON_CT, false, false);
}

Json GatewayClient::hello(const std::string &plat, const Json &engine, const Json &game_files, int instances) {
    Json f = Json::object();
    f.set("client_version", Json::string(CLIENT_VERSION));
    f.set("platform", Json::string(plat));
    f.set("engine", engine);
    f.set("game_files", game_files);
    f.set("instances", Json::integer(instances));
    Json caps = Json::array();  // what this client can run (ballrain.pool.capabilities; the gateway gates tasks by them)
    for (const char *c : CLIENT_CAPABILITIES) caps.push(Json::string(c));
    f.set("capabilities", caps);
    Json r = call("POST", "/pool/hello", {"hello_ok"}, json_body(make_message("hello", f)), JSON_CT, true, false);
    session_id = r.at("session_id").as_string();
    return r;
}

Json GatewayClient::task() { return call("GET", "/pool/task", {"task", "no_task"}, Bytes(), "", true, true); }

Json GatewayClient::chunk(const Bytes &data) {
    return call("POST", "/pool/chunk", {"chunk_result"}, data, "application/octet-stream", true, true);
}

Json GatewayClient::heartbeat(bool running, const Json &instances, double sps) {
    Json f = Json::object();
    f.set("session_id", Json::string(session_id));
    f.set("running", Json::boolean(running));
    f.set("instances", instances);
    f.set("steps_per_s", Json::number(sps));
    return call("POST", "/pool/heartbeat", {"heartbeat_ok"}, json_body(make_message("heartbeat", f)), JSON_CT, true, true);
}

Json GatewayClient::task_end(const std::string &task_id, int64_t chunks, const std::string &reason) {
    Json f = Json::object();
    f.set("task_id", Json::string(task_id));
    f.set("chunks", Json::integer(chunks));
    f.set("reason", Json::string(reason.substr(0, 300)));
    return call("POST", "/pool/task", {"task_end_ok"}, json_body(make_message("task_end", f)), JSON_CT, true, true);
}

Json GatewayClient::me() { return call("GET", "/pool/me", {"me"}, Bytes(), "", true, false); }

Json GatewayClient::set_leaderboard(bool on) {
    Json f = Json::object();
    f.set("leaderboard", Json::boolean(on));
    return call("POST", "/pool/me", {"me"}, json_body(make_message("me_update", f)), JSON_CT, true, false);
}

Bytes GatewayClient::download(const std::string &path, size_t max_bytes, const std::string &expect_sha) {
    platform::HttpResponse r;
    try {
        r = platform::http_request("GET", base_ + path, {{"User-Agent", std::string("ballrain-native-client/") + CLIENT_VERSION}},
                                   Bytes(), timeout_s_ * 6, max_bytes);
    } catch (const Error &e) {
        throw GatewayUnavailable(std::string("下载 ") + path + " 失败：" + e.what());
    }
    if (r.status != 200) {
        try {
            Json obj = Json::parse(r.body, 13);
            if (obj.get_string("type", "") == "error") throw GatewayRefusal(obj.at("reason"), r.status);
        } catch (const GatewayRefusal &) {
            throw;
        } catch (const Error &) {
        }
        throw GatewayUnavailable("下载 " + path + " 失败（HTTP " + std::to_string(r.status) + "）");
    }
    if (!expect_sha.empty() && Sha256::hex(r.body) != expect_sha) throw GatewayUnavailable("下载的 " + path + " 的 sha256 不对");
    return r.body;
}

// ---------------------------------------------------------------------------------------------- cache

static bool is_hex64(const std::string &s) {
    if (s.size() != 64) return false;
    for (char c : s)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}

static void put_file(const std::string &path, const Bytes &data) {
    std::string tmp = path + ".part";
    write_file(tmp, data);
    platform::remove_file(path);
    if (std::rename(tmp.c_str(), path.c_str()) != 0) throw Error("cannot write " + path);
}

Policy PoolCache::policy(const std::string &sha) {
    if (!is_hex64(sha)) throw Error("policy sha");
    std::string d = root_ + "/weights";
    platform::make_dirs(d);
    std::string mp = d + "/" + sha + ".json", npz = d + "/" + sha + ".npz";
    try {
        if (!file_exists(mp)) put_file(mp, api_.download("/pool/weights/" + sha + ".json", MAX_MANIFEST));
        Json m = Policy::read_manifest(mp);
        if (m.get_string("policy_sha", "") != sha || !is_hex64(m.get_string("npz_sha256", "")))
            throw Error("the cached manifest is not this policy version's");
        Bytes cur;
        if (!read_file(npz, cur) || int64_t(cur.size()) != m.at("npz_bytes").as_int())
            put_file(npz, api_.download("/pool/weights/" + sha + ".npz", MAX_DOWNLOAD, m.at("npz_sha256").as_string()));
        return Policy::load(npz, m, sha);
    } catch (...) {
        platform::remove_file(mp);  // a bad or stale cached copy: fetch both again next time
        platform::remove_file(npz);
        throw;
    }
}

std::string PoolCache::asset(const std::string &sha) {
    if (!is_hex64(sha)) throw Error("asset sha");
    std::string d = root_ + "/assets";
    platform::make_dirs(d);
    std::string p = d + "/" + sha + ".npz";
    if (!file_exists(p) || file_sha256(p) != sha) put_file(p, api_.download("/pool/assets/" + sha + ".npz", MAX_DOWNLOAD, sha));
    return p;
}

TasStarts PoolCache::starts(const std::string &sha) { return TasStarts::load_pack(asset(sha), sha); }

std::string PoolCache::path_dir(const std::string &sha, int level) {
    std::string src = asset(sha);
    // task.install_path_asset: points <f4 (n,3), sector |u1 (n), meta a JSON string naming the level; optional
    // t_ms <f8 (n) (without it the ref_speed features are zero, e.g. L02.npz)
    auto z = load_npz(src);
    for (auto &kv : z)
        if (kv.first != "points" && kv.first != "sector" && kv.first != "t_ms" && kv.first != "meta")
            throw Error("reference line: unexpected array " + kv.first);
    if (!z.count("points") || !z.count("sector") || !z.count("meta")) throw Error("reference line: arrays");
    const NpyArray &pts = z.at("points"), &sec = z.at("sector"), &meta = z.at("meta");
    const NpyArray *t = z.count("t_ms") ? &z.at("t_ms") : nullptr;
    if (pts.descr != "<f4" || sec.descr != "|u1" || (t && t->descr != "<f8") || !meta.shape.empty())
        throw Error("reference line: dtypes");
    int64_t n = pts.shape.empty() ? 0 : pts.shape[0];
    if (pts.shape != std::vector<int64_t>{n, 3} || sec.shape != std::vector<int64_t>{n} ||
        (t && t->shape != std::vector<int64_t>{n}))
        throw Error("reference line: shapes");
    for (float v : pts.f32())
        if (!std::isfinite(v)) throw Error("reference line: non-finite points");
    if (t)
        for (double v : t->f64())
            if (!std::isfinite(v)) throw Error("reference line: non-finite t_ms");
    Json m = Json::parse(meta.unicode(), 12);
    if (!m.is_object() || m.get_int("level", -1) != level) throw Error("reference line is not level " + std::to_string(level));
    std::string d = root_ + "/refpaths/" + sha;
    platform::make_dirs(d);
    char name[16];
    std::snprintf(name, sizeof name, "/L%02d.npz", level);
    std::string dst = d + name;
    if (!file_exists(dst) || file_sha256(dst) != sha) {
        Bytes b;
        read_file(src, b);
        put_file(dst, b);
    }
    return d;
}

std::string PoolCache::branch_dir(const std::string &sha, int level, const FeatureConfig &fc) {
    return install_branch_pack(asset(sha), root_, level, sha, fc);
}

static std::string level_name(int level, const char *ext) {
    char n[24];
    std::snprintf(n, sizeof n, "L%02d%s", level, ext);
    return n;
}

std::string install_branch_pack(const std::string &pack, const std::string &root, int level,
                                const std::string &expect_sha, const FeatureConfig &fc) {
    if (!is_hex64(expect_sha) || !fc.has_branch()) throw Error("branch pack: no pins");
    if (file_sha256(pack) != expect_sha) throw Error("branch pack: sha256 differs from " + expect_sha.substr(0, 12));
    auto z = load_npz(pack);
    const char *names[3] = {"line_npz", "desc_json", "drive_npz"};
    const std::string *pins[3] = {&fc.branch_line_sha256, &fc.branch_desc_sha256, &fc.branch_drive_sha256};
    if (z.size() != 3) throw Error("branch pack: arrays");
    Bytes files[3];
    for (int i = 0; i < 3; ++i) {
        auto it = z.find(names[i]);
        if (it == z.end() || it->second.descr != "|u1" || it->second.shape.size() != 1 || it->second.data.empty())
            throw Error(std::string("branch pack: ") + names[i]);
        files[i] = it->second.data;
        if (Sha256::hex(files[i]) != *pins[i]) throw Error(std::string("branch pack: ") + names[i] + " is not the policy's");
    }
    // task.check_branch_files (what load_branch / the branch rest start rely on; both check again when they load)
    Bytes jb = files[1];
    if (jb.compare(0, 3, "\xEF\xBB\xBF") == 0) jb = jb.substr(3);
    Json desc = Json::parse(jb, 16);
    if (!desc.is_object() || desc.get_int("level", -1) != level) throw Error("branch description is not for this level");
    if (desc.get_string("branch_sha256", "") != fc.branch_line_sha256)
        throw Error("branch description names another branch line");
    auto line = load_npz_bytes(files[0]);
    if (!line.count("points") || !line.count("sector") || !line.count("t_ms") || !line.count("meta") || line.size() != 4)
        throw Error("branch line: arrays");
    Json lm = Json::parse(line.at("meta").unicode(), 12);
    if (!lm.is_object() || lm.get_int("level", -1) != level) throw Error("branch line is not for this level");
    auto drive = load_npz_bytes(files[2]);
    if (!drive.count("branch_sha256") || drive.at("branch_sha256").unicode() != fc.branch_line_sha256)
        throw Error("branch drive does not belong to the branch line");
    std::string d = root + "/branch/" + expect_sha;
    platform::make_dirs(d);
    std::string out[3] = {d + "/" + level_name(level, ".npz"), d + "/" + level_name(level, ".json"), d + "/drive.npz"};
    for (int i = 0; i < 3; ++i)
        if (!file_exists(out[i]) || file_sha256(out[i]) != *pins[i]) put_file(out[i], files[i]);
    return d;
}

void check_branch_dir(const std::string &dir, int level, const FeatureConfig &fc) {
    std::string f[3] = {level_name(level, ".npz"), level_name(level, ".json"), "drive.npz"};
    const std::string *pins[3] = {&fc.branch_line_sha256, &fc.branch_desc_sha256, &fc.branch_drive_sha256};
    for (int i = 0; i < 3; ++i) {
        std::string p = dir + "/" + f[i];
        if (!file_exists(p) || file_sha256(p) != *pins[i]) throw Error("the branch directory's " + f[i] + " is not the policy's");
    }
}

// ---------------------------------------------------------------------------------------------- game files

std::string platform_name() {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "macos";
#else
    return "linux";
#endif
}

static std::string lower_ascii(std::string s) {
    for (auto &c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

Json hash_game_files(const std::string &root, const std::string &engine) {
    // protocol.GAME_FILE_LIST(_2002): the x87-soft edition reads only these four; the 2002 engine also the retail
    // Bin/VxMath.dll and BuildingBlocks/physics_RT.dll
    std::vector<std::string> names{"base.cmo", "Database.tdb", "3D Entities/", "Text/"};
    if (engine == "2002") names.insert(names.end(), {"Bin/VxMath.dll", "BuildingBlocks/physics_RT.dll"});
    else if (engine != "x87soft") throw Error("engine " + engine);
    Json out = Json::object();
    std::string zeros(64, '0');
    for (auto &n : names) {
        std::string p = root + "/" + (n.back() == '/' ? n.substr(0, n.size() - 1) : n);
        if (n.back() == '/') {
            std::vector<platform::DirEntry> files;
            if (!platform::list_files(p, files)) {
                out.set(n, Json::string(zeros));
                continue;
            }
            std::vector<std::pair<std::string, std::string>> lines;
            for (auto &f : files) lines.push_back({lower_ascii(f.rel), f.link ? zeros : file_sha256(p + "/" + f.rel)});
            std::sort(lines.begin(), lines.end());
            std::string text;
            for (auto &l : lines) text += l.second + " " + l.first + "\n";
            out.set(n, Json::string(Sha256::hex(text)));
        } else {
            out.set(n, Json::string(file_exists(p) && !platform::is_dir(p) ? file_sha256(p) : zeros));
        }
    }
    return out;
}

Json engine_info(const std::string &binary, const std::string &kind) {
    std::string sha = file_sha256(binary);
    std::string ver;
    std::string dir = binary.substr(0, binary.find_last_of("/\\") + 1);
    Bytes v;
    if (read_file(dir + "ENGINE_VERSION", v)) {
        while (!v.empty() && (v.back() == '\n' || v.back() == '\r' || v.back() == ' ')) v.pop_back();
        bool ok = !v.empty() && v.size() <= 32;
        for (char c : v) ok = ok && (std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '+' || c == '_' || c == '-');
        if (ok) ver = v;
    }
    if (ver.empty()) ver = "dev-" + sha.substr(0, 12);
    Json e = Json::object();
    e.set("kind", Json::string(kind));
    e.set("sha256", Json::string(sha));
    e.set("version", Json::string(ver));
    return e;
}

}  // namespace ballrain
