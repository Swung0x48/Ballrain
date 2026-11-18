// Thin OS layer of the native client: local TCP, child processes, a monotonic clock, directories, the environment.
// Everything above it (native/core) is portable C++17. Implementations: platform_win.cpp, platform_posix.cpp.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ballrain {
namespace platform {

double monotonic_s();
double thread_cpu_s();  // the calling thread's CPU time (user + kernel), seconds
int self_pid();
std::vector<std::pair<std::string, std::string>> environment();  // the current process environment
bool make_dirs(const std::string &path);
bool remove_file(const std::string &path);
std::string absolute_path(const std::string &path);

// Directory listing for game-file hashing: regular files under dir (recursively), paths relative to dir with '/'.
// link = a symbolic link / reparse point (not followed; the hash counts it as a mismatch). false: dir unreadable.
struct DirEntry {
    std::string rel;
    bool link = false;
};
bool list_files(const std::string &dir, std::vector<DirEntry> &out, size_t max_files = 20000);
bool is_dir(const std::string &path);
std::vector<std::string> list_dir(const std::string &dir);  // the names in dir (no "." / ".."), unsorted
bool rename_file(const std::string &from, const std::string &to);  // replaces an existing target
int64_t file_size(const std::string &path);                       // -1 when missing
int64_t unix_time();                                              // wall-clock seconds

// The client's one root: BALLRAIN_WORK_DIR when set, else %USERPROFILE%\.ballrain on Windows, ~/.ballrain elsewhere.
// Everything the client writes is a subdirectory of it (config, token, logs, cache, upload queue, run directories,
// engine-work copies); nothing goes to temp, LOCALAPPDATA, ProgramData or other guessed places
std::string data_dir();
std::string home_dir();
std::string executable_path();  // this program (absolute)
std::string user_name();        // the login name (diagnostics scrub it from paths)
std::string short_path(const std::string &path);  // Windows: the 8.3 form when the volume has one, else path
bool open_url(const std::string &url);  // the system browser (ShellExecute / open / xdg-open)
// One client per data directory: true when this process holds the lock (kept until it exits)
bool single_instance(const std::string &data_dir);

// Automatic-pause probes (best effort: false when unknown). process_running compares image names case-insensitively
// (Windows: the process list; Linux: /proc/<pid>/cmdline contains the name, e.g. Player.exe under Wine)
bool process_running(const std::string &image_name);
bool on_battery();
int cpu_count();

// One HTTP(S) request. Windows: WinHTTP (http and https, the system's TLS and proxy settings). POSIX: https through
// libcurl when the build found it (macOS always; Linux the distro's libcurl), plain http by the built-in client
// (a loopback test gateway); Android: http only so far. Throws ballrain::Error on a connection failure, a timeout or a
// body larger than max_body.
struct HttpResponse {
    int status = 0;
    std::string body;
};
HttpResponse http_request(const std::string &method, const std::string &url,
                          const std::vector<std::pair<std::string, std::string>> &headers, const std::string &body,
                          double timeout_s, size_t max_body);

// A connected byte stream (TCP). recv_exact returns false on EOF / error; throws ballrain::Error("timeout") when
// the deadline (monotonic_s()) passes first.
class Stream {
public:
    virtual ~Stream() = default;
    virtual bool send_all(const void *p, size_t n) = 0;
    virtual bool recv_exact(void *p, size_t n, double deadline) = 0;
    virtual void close() = 0;
};

class Listener {
public:
    virtual ~Listener() = default;
    virtual int port() const = 0;
    virtual std::unique_ptr<Stream> accept(double timeout_s) = 0;  // nullptr on timeout
    static std::unique_ptr<Listener> listen_loopback(int port = 0);
    static std::unique_ptr<Stream> connect_loopback(int port, double timeout_s);  // tests / tools
};

// A child process. The destructor kills it if it is still running.
class Process {
public:
    struct Spec {
        std::vector<std::string> argv;
        std::vector<std::pair<std::string, std::string>> env;  // the complete environment of the child
        std::string cwd;          // "" = inherit
        std::string stderr_path;  // "" = discard
        // the volunteer client's engines: below-normal priority, and (Windows) inside a Job Object of this process
        // with KILL_ON_JOB_CLOSE, so no engine outlives the client however it ends (POSIX: the engine exits when its
        // socket closes; nice 10)
        bool background = false;
    };
    virtual ~Process() = default;
    virtual int pid() const = 0;
    virtual bool wait(double timeout_s) = 0;  // true when it exited
    virtual bool running() = 0;
    virtual int exit_code() = 0;              // -1 while running
    virtual void kill() = 0;
    static std::unique_ptr<Process> spawn(const Spec &spec);
};

}  // namespace platform
}  // namespace ballrain
