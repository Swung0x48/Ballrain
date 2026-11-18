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
int self_pid();
std::vector<std::pair<std::string, std::string>> environment();  // the current process environment
bool make_dirs(const std::string &path);
bool remove_file(const std::string &path);
std::string absolute_path(const std::string &path);
std::string temp_dir();

// Directory listing for game-file hashing: regular files under dir (recursively), paths relative to dir with '/'.
// link = a symbolic link / reparse point (not followed; the hash counts it as a mismatch). false: dir unreadable.
struct DirEntry {
    std::string rel;
    bool link = false;
};
bool list_files(const std::string &dir, std::vector<DirEntry> &out, size_t max_files = 20000);
bool is_dir(const std::string &path);

// One HTTP(S) request. Windows: WinHTTP (http and https, the system's TLS and proxy settings). POSIX: http only for
// now (a loopback test gateway); https on macOS / Linux / Android is not implemented yet. Throws ballrain::Error on a
// connection failure, a timeout or a body larger than max_body.
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
