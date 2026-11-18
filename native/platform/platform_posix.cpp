// POSIX implementation of native/platform/platform.h (macOS, Linux, Android).
#ifndef _WIN32
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <spawn.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "ballrain/util.h"
#include "platform.h"

extern char **environ;

namespace ballrain {
namespace platform {

double monotonic_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int self_pid() { return int(getpid()); }

std::vector<std::pair<std::string, std::string>> environment() {
    std::vector<std::pair<std::string, std::string>> out;
    for (char **e = environ; e && *e; ++e) {
        const char *eq = std::strchr(*e, '=');
        if (!eq || eq == *e) continue;
        out.emplace_back(std::string(*e, size_t(eq - *e)), std::string(eq + 1));
    }
    return out;
}

bool make_dirs(const std::string &path) {
    std::string p;
    for (size_t i = 0; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '/') {
            p = path.substr(0, i);
            if (!p.empty() && mkdir(p.c_str(), 0755) != 0 && errno != EEXIST) {
                struct stat st;
                if (stat(p.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return false;
            }
        }
    }
    return true;
}

bool remove_file(const std::string &path) { return unlink(path.c_str()) == 0; }

std::string absolute_path(const std::string &path) {
    if (!path.empty() && path[0] == '/') return path;
    char buf[PATH_MAX];
    if (!getcwd(buf, sizeof buf)) return path;
    return std::string(buf) + "/" + path;
}

std::string temp_dir() {
    const char *t = std::getenv("TMPDIR");
    std::string s = t && *t ? t : "/tmp";
    while (s.size() > 1 && s.back() == '/') s.pop_back();
    return s;
}

// ---------------------------------------------------------------------------------------------- sockets

// close-on-exec: posix_spawn passes every other open descriptor to the child (an engine started later must not hold
// this listener or another engine's connection)
static int cloexec(int fd) {
    if (fd >= 0) fcntl(fd, F_SETFD, fcntl(fd, F_GETFD) | FD_CLOEXEC);
    return fd;
}

class PosixStream : public Stream {
public:
    explicit PosixStream(int s) : s_(s) {
        int one = 1;
        setsockopt(s_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
#ifdef SO_NOSIGPIPE
        setsockopt(s_, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
    }
    ~PosixStream() override { close(); }
    bool send_all(const void *p, size_t n) override {
        auto b = static_cast<const char *>(p);
        while (n) {
#ifdef MSG_NOSIGNAL
            ssize_t k = ::send(s_, b, n, MSG_NOSIGNAL);
#else
            ssize_t k = ::send(s_, b, n, 0);
#endif
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) return false;
            b += k, n -= size_t(k);
        }
        return true;
    }
    bool recv_exact(void *p, size_t n, double deadline) override {
        auto b = static_cast<char *>(p);
        while (n) {
            if (deadline > 0) {
                double left = deadline - monotonic_s();
                if (left <= 0) throw Error("timeout");
                fd_set r;
                FD_ZERO(&r);
                FD_SET(s_, &r);
                timeval tv;
                tv.tv_sec = long(left);
                tv.tv_usec = long((left - double(tv.tv_sec)) * 1e6);
                int k = select(s_ + 1, &r, nullptr, nullptr, &tv);
                if (k < 0 && errno == EINTR) continue;
                if (k == 0) throw Error("timeout");
                if (k < 0) return false;
            }
            ssize_t k = ::recv(s_, b, n, 0);
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) return false;
            b += k, n -= size_t(k);
        }
        return true;
    }
    void close() override {
        if (s_ >= 0) ::close(s_), s_ = -1;
    }

private:
    int s_;
};

class PosixListener : public Listener {
public:
    explicit PosixListener(int port) {
        signal(SIGPIPE, SIG_IGN);
        s_ = cloexec(socket(AF_INET, SOCK_STREAM, 0));
        if (s_ < 0) throw Error("socket() failed");
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = htons(uint16_t(port));
        if (bind(s_, reinterpret_cast<sockaddr *>(&a), sizeof a) != 0) throw Error("bind() failed");
        if (listen(s_, 1) != 0) throw Error("listen() failed");
        socklen_t len = sizeof a;
        getsockname(s_, reinterpret_cast<sockaddr *>(&a), &len);
        port_ = ntohs(a.sin_port);
    }
    ~PosixListener() override {
        if (s_ >= 0) ::close(s_);
    }
    int port() const override { return port_; }
    std::unique_ptr<Stream> accept(double timeout_s) override {
        fd_set r;
        FD_ZERO(&r);
        FD_SET(s_, &r);
        timeval tv;
        tv.tv_sec = long(timeout_s);
        tv.tv_usec = long((timeout_s - double(tv.tv_sec)) * 1e6);
        int k = select(s_ + 1, &r, nullptr, nullptr, &tv);
        if (k <= 0) return nullptr;
        int c = cloexec(::accept(s_, nullptr, nullptr));
        if (c < 0) return nullptr;
        return std::make_unique<PosixStream>(c);
    }

private:
    int s_ = -1;
    int port_ = 0;
};

std::unique_ptr<Listener> Listener::listen_loopback(int port) { return std::make_unique<PosixListener>(port); }

std::unique_ptr<Stream> Listener::connect_loopback(int port, double timeout_s) {
    double end = monotonic_s() + timeout_s;
    for (;;) {
        int s = cloexec(socket(AF_INET, SOCK_STREAM, 0));
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = htons(uint16_t(port));
        if (connect(s, reinterpret_cast<sockaddr *>(&a), sizeof a) == 0) return std::make_unique<PosixStream>(s);
        ::close(s);
        if (monotonic_s() > end) return nullptr;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

// ---------------------------------------------------------------------------------------------- processes

class PosixProcess : public Process {
public:
    explicit PosixProcess(const Spec &spec) {
        std::vector<char *> argv, envp;
        for (auto &a : spec.argv) argv.push_back(const_cast<char *>(a.c_str()));
        argv.push_back(nullptr);
        std::vector<std::string> env;
        for (auto &kv : spec.env) env.push_back(kv.first + "=" + kv.second);
        for (auto &e : env) envp.push_back(const_cast<char *>(e.c_str()));
        envp.push_back(nullptr);
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
        posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
        if (spec.stderr_path.empty()) posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
        else posix_spawn_file_actions_addopen(&fa, 2, spec.stderr_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        // cwd: posix_spawn_file_actions_addchdir_np is not portable; the engine takes absolute paths
        int rc = posix_spawn(&pid_, spec.argv[0].c_str(), &fa, nullptr, argv.data(), envp.data());
        posix_spawn_file_actions_destroy(&fa);
        if (rc != 0) throw Error("posix_spawn failed (" + std::to_string(rc) + ") for " + spec.argv[0]);
    }
    ~PosixProcess() override {
        if (running()) kill();
    }
    int pid() const override { return int(pid_); }
    bool wait(double timeout_s) override {
        double end = monotonic_s() + timeout_s;
        for (;;) {
            if (!running()) return true;
            if (monotonic_s() >= end) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    bool running() override {
        if (done_) return false;
        int st;
        pid_t r = waitpid(pid_, &st, WNOHANG);
        if (r == pid_) {
            done_ = true;
            code_ = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
            return false;
        }
        return r == 0;
    }
    int exit_code() override { return running() ? -1 : code_; }
    void kill() override {
        if (running()) {
            ::kill(pid_, SIGKILL);
            wait(5.0);
        }
    }

private:
    pid_t pid_ = 0;
    bool done_ = false;
    int code_ = -1;
};

std::unique_ptr<Process> Process::spawn(const Spec &spec) { return std::make_unique<PosixProcess>(spec); }

}  // namespace platform
}  // namespace ballrain
#endif
