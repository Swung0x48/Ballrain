// One pool task on this machine (the part of a volunteer client between "a task arrived" and "its chunks are out"):
// the policy, starts pack and reference line from the sha-addressed cache, the checks that they belong to the task,
// the env / runner on the local engine, sample_task. Used by the controller's instances and by the command-line tools.
#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "ballrain/pool_client.h"
#include "ballrain/task_runner.h"

namespace ballrain {

struct LocalEngine {  // the player's side: the engine binary and the Ballance folder
    std::string binary, root, kind = "x87soft";
    std::vector<std::pair<std::string, std::string>> extra_env;  // BALLRAIN_* only (tests / diagnostics)
    bool background = true;  // below-normal priority, inside the client's job object
};

struct TaskOutcome {
    std::string end;     // the task's end ("error": the engine failed)
    std::string error;   // what failed (end "error")
    int64_t steps = 0, frames = 0, chunks = 0;
    int episodes = 0;
    bool final_chunk = false;  // the last chunk handed out was the final one (else the caller sends task_end)
    double wall_s = 0;
};

// Where the engine's copies and run directories go when the client's own paths are not plain ASCII (the engine
// resolves its files with narrow code-page paths): BALLRAIN_WORK_DIR if set, else <data_dir>/engine-work, in its 8.3
// form when that path is not plain ASCII itself (a Chinese user name). Throws a message for the player when no
// plain-ASCII form exists.
std::string engine_work_dir(const std::string &data_dir);
// The engine's view of le (Windows): a binary or Ballance folder whose path is not plain ASCII is replaced by its 8.3
// form, else by a copy under work_dir (the engine's directory; the four game entries the x87-soft edition reads,
// re-copied when the folder's game-file hashes change). note: what was done (for the log), or "".
LocalEngine engine_view(const LocalEngine &le, const Json &game_files, const std::string &data_dir,
                        std::string *note = nullptr);
bool plain_ascii(const std::string &s);

// Plays the task. on_chunk(seq, data, final) receives every encoded chunk in order; stop() is sample_task's;
// on_step(steps) after every step (progress). cache_mu serialises the cache (several instances share it). Throws
// before the first chunk when the task does not fit this machine or the engine cannot start (the caller backs off);
// an engine failure after that returns end "error". run_dir is removed afterwards (headless.log of a failed run is
// kept as <run_dir>.headless.log for the caller's logs).
TaskOutcome play_task(const Task &task, PoolCache &cache, std::mutex &cache_mu, const LocalEngine &eng,
                      const std::string &run_dir, const std::function<void(int64_t, const Bytes &, bool)> &on_chunk,
                      const std::function<std::string()> &stop, const std::function<void(int64_t)> &on_step = nullptr);

}  // namespace ballrain
