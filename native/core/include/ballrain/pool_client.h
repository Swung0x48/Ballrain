// The volunteer side of the pool protocol (a port of ballrain/pool/client_api.py and the parts of task.py /
// client_game.py a client needs): JSON messages over HTTP(S), the sha-addressed download cache, the game-file hashes.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ballrain/features.h"
#include "ballrain/json.h"
#include "ballrain/policy.h"
#include "ballrain/starts.h"
#include "ballrain/task_runner.h"

namespace ballrain {

extern const char *const CLIENT_VERSION;
// hello.capabilities (ballrain.pool.capabilities): task format 3, the route branch and its rules, the f7 reach block
// of this client's version (weights.reach_capability: "reach_" + reach_version with "_" for ".")
std::vector<std::string> client_capabilities();

struct GatewayUnavailable : Error {  // no usable answer: network, timeout, not the protocol
    explicit GatewayUnavailable(const std::string &m) : Error(m) {}
};
struct GatewayRefusal : Error {  // the gateway's reason; .message is shown to the player verbatim
    GatewayRefusal(Json reason, int http_status);
    Json reason;
    std::string code, message;
    int http_status;
    bool fatal;  // reasons.TOKEN_FATAL: stop everything, log in again
};

// "https://host[/path]", or http only for this machine (a local test gateway); no query / credentials
std::string check_gateway_url(const std::string &url);
Json make_message(const std::string &type, Json fields = Json::object());

class GatewayClient {
public:
    explicit GatewayClient(const std::string &base_url, std::string token = "", double timeout_s = 20.0);
    std::string token, session_id;

    Json login_start();
    Json login_poll(const std::string &login_id, const std::string &poll_secret);
    Json hello(const std::string &platform, const Json &engine, const Json &game_files, int instances);
    Json task();  // a "task" or "no_task" message
    Json chunk(const Bytes &data);  // chunk_result (a "rejected" result is returned, an error message raised)
    // task_end: close a task that ended without a final chunk once its `chunks` chunks were all posted
    Json task_end(const std::string &task_id, int64_t chunks, const std::string &reason);
    Json heartbeat(bool running, const Json &instances, double steps_per_s);
    Json me();
    Json set_leaderboard(bool on);
    Bytes download(const std::string &path, size_t max_bytes, const std::string &expect_sha = "");

private:
    Json call(const std::string &method, const std::string &path, const std::vector<std::string> &expect,
              const Bytes &body, const std::string &ctype, bool auth, bool session);
    std::string base_;
    double timeout_s_;
};

// <data dir>/cache: weights/<sha>.{json,npz}, assets/<sha>.npz, refpaths/<sha>/L<nn>.npz
class PoolCache {
public:
    PoolCache(std::string root, GatewayClient &api) : root_(std::move(root)), api_(api) {}
    Policy policy(const std::string &policy_sha);  // downloads, checks (Policy::load), drops a bad cached copy
    std::string asset(const std::string &sha);     // the asset file, downloaded and sha-checked
    TasStarts starts(const std::string &sha);      // the starts pack, checked (install_starts_pack's checks)
    std::string path_dir(const std::string &sha, int level);  // the reference line installed as L<nn>.npz
    // the route-branch pack installed as the FeatureConfig.branch_dir (install_branch_pack)
    std::string branch_dir(const std::string &sha, int level, const FeatureConfig &fc);
    // f7: the task's reach fields (assets.reach) installed as <root>/fastreach/L<nn>-<key_sha256[:16]>.npz (the
    // ReachBlock's fields directory); each file is downloaded once (zstd, size-capped, sha-checked) and kept
    std::string reach_dir(const std::vector<Task::ReachAsset> &assets, int level);

private:
    std::string root_;
    GatewayClient &api_;
};

// task.install_branch_pack: a branch pack (uint8 arrays line_npz, desc_json, drive_npz) whose file sha256 is
// expect_sha and whose files are the manifest's pins, laid out as <root>/branch/<sha>/ L<nn>.npz, L<nn>.json,
// drive.npz; returns that directory. task.check_branch_dir: the directory holds exactly the pinned files.
std::string install_branch_pack(const std::string &pack, const std::string &root, int level,
                                const std::string &expect_sha, const FeatureConfig &fc);
void check_branch_dir(const std::string &dir, int level, const FeatureConfig &fc);

// hello.game_files (task.hash_game_files): GAME_FILE_LIST (+ BuildingBlocks/physics_RT.dll for the 2002 engine);
// a directory entry is the tree digest of its files; a missing entry is 64 zeros
Json hash_game_files(const std::string &root, const std::string &engine_kind);
// hello.engine: kind, the binary's sha256, version (ENGINE_VERSION next to the binary, else dev-<sha12>)
Json engine_info(const std::string &binary, const std::string &kind);
std::string platform_name();

}  // namespace ballrain
