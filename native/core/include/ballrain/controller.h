// The volunteer client's controller, without any UI (a port of ballrain/pool/client_core.py; docs/native-client.md
// section 7): configuration and token, the game-folder check, the browser login, the gateway session (hello,
// heartbeat, "me", the leaderboard switch), up to MAX_INSTANCES sampling instances (threads, one engine child each),
// the on-disk upload queue, the automatic pause, the log. The TUI (native/apps/ballrain_pool.cpp), the command-line
// driver (nc_client run) and a later Android app are thin layers over view() and the actions.
//
// Rules (docs/volunteer-compute.md 11): start / pause is the player's; closing the controller ends every instance and
// engine; a refusal or revocation shows the gateway's reason.message exactly as received (banner); a token-fatal
// reason stops every instance until the player logs in again; the leaderboard switch: the gateway's state is the
// truth; the controller pauses itself while Player.exe runs or the machine is on battery, and resumes after.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "ballrain/diagnostics.h"
#include "ballrain/json.h"
#include "ballrain/pool_client.h"
#include "ballrain/pool_task.h"

namespace ballrain {

constexpr int MAX_INSTANCES = 32;  // the client's own cap (the gateway's max_instances and the player's choice decide)

struct ClientConfig {  // <data dir>/config.json
    std::string gateway_url, ballance_dir;
    std::string engine_binary;  // "" = the engine shipped next to the client (bundled_engine)
    std::string engine_kind = "x87soft";
    int instances = 1;
    bool auto_pause_player = true, auto_pause_battery = true;
    bool setup_done = false;  // the first-run setup was completed once
    // a stop switch for shared machines: while this file or directory exists every instance pauses (checked every
    // 5 s, like the other automatic pauses; "" = none), e.g. C:\ballrain-quiet
    std::string quiet_file;
    static int default_instances();  // min(8, cores / 2), at least 1
    static ClientConfig load(const std::string &data_dir);
    void save(const std::string &data_dir) const;
};

struct GameProblem {
    std::string name;     // the game_files entry, or "engine" / ""
    std::string kind;     // "missing" | "empty" | "wrong" | "engine"
    std::string message;  // Chinese, for the player
};
// The player's Ballance folder for an engine kind (empty = fine here; the gateway checks the rest at hello)
std::vector<GameProblem> check_game_dir(const std::string &root, const std::string &engine_kind);
std::vector<GameProblem> check_engine(const std::string &binary);
// <the client's directory>/engine/<kind>/<platform>/BallanceMMOSimTool[.exe], or "" when the bundle has none
std::string bundled_engine(const std::string &kind);
// Folders that often hold a Ballance installation (setup's "search common locations"); only existing ones with a
// base.cmo are returned
std::vector<std::string> find_ballance_dirs();

// logs/client.log (rotated at 2 MB, 3 old files kept) plus the last lines in memory for the UI
class ClientLog {
public:
    explicit ClientLog(std::string dir);
    void line(const std::string &msg);
    std::vector<std::string> tail(size_t n) const;
    const std::string &dir() const { return dir_; }

private:
    std::string dir_;
    mutable std::mutex mu_;
    std::deque<std::string> mem_;
};

struct InstanceView {
    int slot = 0;
    std::string state = "idle";  // idle | starting | running | paused | error
    std::string state_text;      // Chinese
    std::string task_id, detail;
    int64_t task_steps = 0, total_steps = 0, accepted_steps = 0;
    double steps_per_s = 0;
};

struct Banner {
    std::string kind;  // "revoked" | "rejected" | "error"
    std::string title;  // Chinese title of the kind
    std::string code, message;  // message: the gateway's reason.message verbatim (or a local message)
};

struct ControllerView {
    bool has_token = false, connected = false, want_running = false, running = false, token_dead = false;
    std::string auto_pause;  // "" or the Chinese reason
    std::optional<Banner> banner;
    std::string user_name, notice;
    bool leaderboard = true;
    int instances = 1, max_instances = MAX_INSTANCES;
    Json stats;  // /pool/me stats (steps_accepted, steps_verified, ...) or null
    std::vector<InstanceView> inst;
    int64_t queued_chunks = 0, tasks_done = 0;
    std::vector<std::string> log;
    std::string update_version, update_url, update_notes;  // a newer signed release (empty: none known)
};

class Controller {
public:
    // on_event(kind, fields): every event as a JSON-able record (the command-line driver prints them); may be null
    using EventFn = std::function<void(const std::string &, const Json &)>;
    Controller(std::string data_dir, EventFn on_event = nullptr);
    ~Controller();  // shutdown()

    const std::string &data_dir() const { return dir_; }
    ClientConfig config() const;
    void set_config(const ClientConfig &c);  // saves; a changed folder / engine / gateway reconnects
    std::string engine_binary() const;
    std::vector<GameProblem> problems() const;  // game folder + engine of the configuration
    ClientLog &log() { return log_; }

    // ---- login (the gateway's flow in the system browser; the forum's client secret never reaches the client)
    Json login_start();                    // login_started (user_code, browser_url, poll_s, expires_in_s)
    std::string login_poll(const Json &started);  // "pending" | "ok" (token saved) | "refused" (banner) | "expired"
    bool has_token() const;
    void logout();  // forgets the token (the instances stop)

    // ---- session and player actions
    bool connect();  // hello (engine sha256, game files); false with a banner
    void start();
    void pause();
    int set_instances(int n);
    bool set_leaderboard(bool on);  // POST /pool/me; throws GatewayRefusal / GatewayUnavailable (shown by the caller)
    void start_service();           // service / heartbeat / upload threads (once)
    void shutdown();                // every instance and engine ends, the threads join
    ControllerView view() const;

    // ---- diagnostics and updates
    DiagFiles diagnostics() const;                             // the scrubbed files (shown before saving)
    std::string export_diagnostics(const std::string &zip_path = "");  // default <data dir>/diagnostics-<time>.zip
    // the signed release manifest (update.h); a newer version shows in view(); runs in the calling thread
    void check_update(const std::string &manifest_url, const std::string &pubkey_hex);

    // the command-line driver: stop fetching after this many tasks (0 = no limit); extra engine environment
    int64_t max_tasks = 0;
    std::vector<std::pair<std::string, std::string>> engine_env;
    bool finished() const;  // max_tasks reached, every instance idle, the upload queue empty

private:
    struct Slot;
    struct QueueTask;
    void emit(const std::string &kind, Json f = Json::object());
    void say(const std::string &msg);
    void set_banner(const std::string &kind, const Json *reason, const std::string &message = "");
    void refused(const GatewayRefusal &e, const char *where);
    GatewayClient api() const;  // a copy with the current token / session (thread-safe use)
    bool is_running() const;    // under mu_
    void service_loop();
    void heartbeat_once();
    void probe();
    void reconcile();
    void slot_loop(Slot *s);
    bool run_one_task(Slot *s);  // false: no task / not run (waited)
    void upload_loop();
    bool upload_task(const std::string &task_id);  // false: stop this round (network down / waiting)
    void queue_chunk(const std::string &task_id, int64_t seq, const Bytes &data, bool final);
    void queue_end(const std::string &task_id, int64_t chunks, const std::string &reason);
    void queue_open(const Task &t, int slot);
    int64_t queue_bytes() const;
    bool wait_for(double seconds, const std::function<bool()> &until = nullptr);  // false when shutting down

    std::string dir_;
    EventFn on_event_;
    ClientLog log_;
    mutable std::mutex mu_;      // the state below
    std::mutex wait_mu_;         // wait_for's sleeps (woken by cv_)
    std::condition_variable cv_;
    ClientConfig cfg_;
    std::string token_, gateway_, session_id_;
    Json engine_, game_files_;
    std::optional<Banner> banner_;
    std::string user_name_, notice_, auto_pause_;
    Json user_, stats_;
    bool leaderboard_ = true, connected_ = false, token_dead_ = false, want_running_ = false;
    bool reconnect_wanted_ = false;
    double next_reconnect_ = 0, last_me_ = 0, last_probe_ = -1e9;
    int max_instances_ = MAX_INSTANCES, heartbeat_s_ = 30;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::atomic<bool> closing_{false};  // shutdown: instances and the service end
    std::atomic<bool> stop_{false};     // then the uploader
    std::atomic<int64_t> tasks_fetched_{0}, tasks_done_{0};
    std::vector<std::thread> threads_;
    std::mutex cache_mu_;
    std::unique_ptr<GatewayClient> cache_api_;
    std::unique_ptr<PoolCache> cache_;
    std::map<std::string, int> task_slot_;          // task_id -> slot (accepted steps shown per instance)
    std::map<std::string, std::string> refused_;    // task_id -> why its chunks are refused (the instance stops)
    std::map<std::string, int> read_fail_;          // the uploaders' unreadable chunk files (under upload_mu_)
    std::set<std::string> uploading_;               // the tasks an uploader thread holds (under upload_mu_)
    // where the instances' wall time goes (summed over tasks, logged every 5 minutes): waiting for the previous
    // task's upload, the task fetch, assets, engine start, steps, close; tasks and steps
    struct Phases {
        double wait = 0, fetch = 0, assets = 0, start = 0, play = 0, close = 0;
        int64_t tasks = 0, steps = 0;
    } phases_;
    double phases_t0_ = -1;
    void add_phases(double wait, double fetch, const TaskOutcome &out);
    std::mutex upload_mu_;
    double t_start_ = 0;
    std::string update_version_, update_url_, update_notes_;
};

}  // namespace ballrain
