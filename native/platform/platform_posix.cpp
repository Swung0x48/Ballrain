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
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <dirent.h>
#include <pwd.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#include <chrono>
#include <climits>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <thread>

#include "ballrain/util.h"
#include "platform.h"

#ifdef BALLRAIN_HAVE_CURL
#include <curl/curl.h>
#endif
#include <algorithm>

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

std::vector<std::string> list_dir(const std::string &dir) {
    std::vector<std::string> out;
    DIR *d = opendir(dir.c_str());
    if (!d) return out;
    while (dirent *e = readdir(d)) {
        std::string n = e->d_name;
        if (n != "." && n != "..") out.push_back(n);
    }
    closedir(d);
    return out;
}

bool rename_file(const std::string &from, const std::string &to) { return std::rename(from.c_str(), to.c_str()) == 0; }

int64_t file_size(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 ? int64_t(st.st_size) : -1;
}

int64_t unix_time() { return int64_t(std::time(nullptr)); }

std::string home_dir() {
    const char *h = std::getenv("HOME");
    if (h && *h) return h;
    passwd *pw = getpwuid(getuid());
    return pw && pw->pw_dir ? pw->pw_dir : "/tmp";
}

std::string data_dir() { return home_dir() + "/.ballrain"; }

std::string executable_path() {
#ifdef __APPLE__
    char buf[PATH_MAX];
    uint32_t n = sizeof buf;
    if (_NSGetExecutablePath(buf, &n) == 0) {
        char real[PATH_MAX];
        if (realpath(buf, real)) return real;
        return buf;
    }
    return "";
#else
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
    return n > 0 ? std::string(buf, size_t(n)) : std::string();
#endif
}

std::string short_path(const std::string &path) { return path; }

std::string user_name() {
    passwd *pw = getpwuid(getuid());
    return pw && pw->pw_name ? pw->pw_name : "";
}

bool open_url(const std::string &url) {
    if (url.rfind("https://", 0) != 0 && url.rfind("http://", 0) != 0) return false;
#ifdef __APPLE__
    const char *tool = "/usr/bin/open";
#else
    const char *tool = "xdg-open";
#endif
    pid_t pid;
    char *argv[] = {const_cast<char *>(tool), const_cast<char *>(url.c_str()), nullptr};
    int rc = posix_spawnp(&pid, tool, nullptr, nullptr, argv, environ);
    if (rc != 0) return false;
    int st;
    waitpid(pid, &st, 0);
    return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}

bool single_instance(const std::string &data_dir) {
    make_dirs(data_dir);
    int fd = open((data_dir + "/.lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);  // kept open until exit
    if (fd < 0) return true;
    struct flock fl {};
    fl.l_type = F_WRLCK, fl.l_whence = SEEK_SET;
    return fcntl(fd, F_SETLK, &fl) == 0;
}

bool process_running(const std::string &image_name) {
#ifdef __linux__
    std::string want;
    for (char c : image_name) want += char(std::tolower(static_cast<unsigned char>(c)));
    for (auto &d : list_dir("/proc")) {
        if (d.empty() || !std::isdigit(static_cast<unsigned char>(d[0]))) continue;
        std::ifstream f("/proc/" + d + "/cmdline", std::ios::binary);
        std::string cmd((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        for (auto &c : cmd) c = char(std::tolower(static_cast<unsigned char>(c)));
        if (cmd.find(want) != std::string::npos) return true;
    }
#else
    (void)image_name;  // macOS: not probed yet (Ballance does not run there natively)
#endif
    return false;
}

bool on_battery() {
#ifdef __linux__
    for (auto &d : list_dir("/sys/class/power_supply")) {
        std::ifstream t("/sys/class/power_supply/" + d + "/type"), s("/sys/class/power_supply/" + d + "/status");
        std::string type, status;
        std::getline(t, type), std::getline(s, status);
        if (type == "Battery" && status == "Discharging") return true;
    }
#endif
    return false;
}

int cpu_count() {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? int(n) : 1;
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
        if (spec.background) setpriority(PRIO_PROCESS, pid_, 10);  // below normal (best effort)
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

// ---------------------------------------------------------------------------------------------- files, HTTP (POSIX)
#ifndef _WIN32
#include <dirent.h>
#include <netdb.h>

namespace ballrain {
namespace platform {

bool is_dir(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static bool walk(const std::string &root, const std::string &rel, std::vector<DirEntry> &out, size_t max_files) {
    std::string d = root + (rel.empty() ? "" : "/" + rel);
    DIR *dir = opendir(d.c_str());
    if (!dir) return false;
    bool ok = true;
    while (struct dirent *e = readdir(dir)) {
        std::string n = e->d_name;
        if (n == "." || n == "..") continue;
        std::string r = rel.empty() ? n : rel + "/" + n;
        struct stat st;
        if (lstat((root + "/" + r).c_str(), &st) != 0) continue;
        if (S_ISLNK(st.st_mode)) {
            struct stat t;
            if (stat((root + "/" + r).c_str(), &t) == 0 && S_ISDIR(t.st_mode)) continue;  // os.walk skips linked dirs
            out.push_back({r, true});
        } else if (S_ISDIR(st.st_mode)) {
            ok = walk(root, r, out, max_files) && ok;
        } else {
            out.push_back({r, !S_ISREG(st.st_mode)});
        }
        if (out.size() > max_files) {
            ok = false;
            break;
        }
    }
    closedir(dir);
    return ok;
}

bool list_files(const std::string &dir, std::vector<DirEntry> &out, size_t max_files) {
    out.clear();
    if (!is_dir(dir)) return false;
    return walk(dir, "", out, max_files);
}

#ifdef BALLRAIN_HAVE_CURL
// https (and http) through libcurl: the system's TLS and CA store (macOS: always present; Linux: the distro's)
struct CurlBody {
    std::string data;
    size_t max = 0;
    bool over = false;
};
static size_t curl_write(char *p, size_t sz, size_t n, void *ud) {
    auto *b = static_cast<CurlBody *>(ud);
    if (b->data.size() + sz * n > b->max) {
        b->over = true;
        return 0;
    }
    b->data.append(p, sz * n);
    return sz * n;
}

static HttpResponse curl_request(const std::string &method, const std::string &url,
                                 const std::vector<std::pair<std::string, std::string>> &headers,
                                 const std::string &body, double timeout_s, size_t max_body) {
    static const bool inited = [] { return curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK; }();
    if (!inited) throw Error("https: libcurl did not initialise");
    CURL *c = curl_easy_init();
    if (!c) throw Error("https: no curl handle");
    curl_slist *hl = nullptr;
    for (auto &h : headers) hl = curl_slist_append(hl, (h.first + ": " + h.second).c_str());
    hl = curl_slist_append(hl, "Expect:");
    CurlBody out;
    out.max = max_body;
    curl_easy_setopt(c, CURLOPT_URL, url.c_str());
    curl_easy_setopt(c, CURLOPT_CUSTOMREQUEST, method.c_str());
    curl_easy_setopt(c, CURLOPT_HTTPHEADER, hl);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, curl_write);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &out);
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_MAXREDIRS, 5L);
    curl_easy_setopt(c, CURLOPT_REDIR_PROTOCOLS_STR, "https");
    curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(c, CURLOPT_TIMEOUT_MS, long(timeout_s * 1000));
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT_MS, long(std::min(timeout_s, 20.0) * 1000));
    if (!body.empty() || method == "POST") {
        curl_easy_setopt(c, CURLOPT_POSTFIELDS, body.data());
        curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE_LARGE, curl_off_t(body.size()));
    }
    CURLcode rc = curl_easy_perform(c);
    long status = 0;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &status);
    curl_slist_free_all(hl);
    curl_easy_cleanup(c);
    if (out.over) throw Error("http: response larger than " + std::to_string(max_body) + " bytes");
    if (rc != CURLE_OK) throw Error(std::string("https: ") + curl_easy_strerror(rc));
    HttpResponse r;
    r.status = int(status);
    r.body = std::move(out.data);
    return r;
}
#endif

HttpResponse http_request(const std::string &method, const std::string &url,
                          const std::vector<std::pair<std::string, std::string>> &headers, const std::string &body,
                          double timeout_s, size_t max_body) {
#ifdef BALLRAIN_HAVE_CURL
    if (url.compare(0, 8, "https://") == 0) return curl_request(method, url, headers, body, timeout_s, max_body);
#endif
    if (url.compare(0, 7, "http://") != 0)
        throw Error("http: https needs libcurl on this platform (this build has none)");
    std::string rest = url.substr(7);
    size_t slash = rest.find('/');
    std::string hostport = rest.substr(0, slash), path = slash == std::string::npos ? "/" : rest.substr(slash);
    std::string host = hostport, port = "80";
    size_t colon = hostport.rfind(':');
    if (colon != std::string::npos) host = hostport.substr(0, colon), port = hostport.substr(colon + 1);
    addrinfo hints{}, *res = nullptr;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || !res) throw Error("http: cannot resolve " + host);
    int s = -1;
    for (addrinfo *a = res; a; a = a->ai_next) {
        s = cloexec(socket(a->ai_family, a->ai_socktype, a->ai_protocol));
        if (s >= 0 && connect(s, a->ai_addr, a->ai_addrlen) == 0) break;
        if (s >= 0) ::close(s), s = -1;
    }
    freeaddrinfo(res);
    if (s < 0) throw Error("http: connect failed");
    PosixStream st(s);
    std::string req = method + " " + path + " HTTP/1.1\r\nHost: " + hostport +
                      "\r\nConnection: close\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
    for (auto &kv : headers) req += kv.first + ": " + kv.second + "\r\n";
    req += "\r\n";
    req += body;
    if (!st.send_all(req.data(), req.size())) throw Error("http: send failed");
    double deadline = monotonic_s() + timeout_s;
    std::string resp;
    char buf[65536];
    for (;;) {  // read to EOF (Connection: close)
        double left = deadline - monotonic_s();
        if (left <= 0) throw Error("http: timeout");
        fd_set r;
        FD_ZERO(&r);
        FD_SET(s, &r);
        timeval tv;
        tv.tv_sec = long(left);
        tv.tv_usec = long((left - double(tv.tv_sec)) * 1e6);
        int k = select(s + 1, &r, nullptr, nullptr, &tv);
        if (k < 0 && errno == EINTR) continue;
        if (k == 0) throw Error("http: timeout");
        if (k < 0) throw Error("http: read failed");
        ssize_t n = ::recv(s, buf, sizeof buf, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        resp.append(buf, size_t(n));
        if (resp.size() > max_body + 65536) throw Error("http: response too large");
    }
    size_t he = resp.find("\r\n\r\n");
    if (he == std::string::npos || resp.compare(0, 5, "HTTP/") != 0) throw Error("http: bad response");
    HttpResponse out;
    out.status = std::atoi(resp.c_str() + resp.find(' ') + 1);
    std::string head = resp.substr(0, he), b = resp.substr(he + 4);
    for (auto &c : head) c = char(std::tolower(static_cast<unsigned char>(c)));
    if (head.find("transfer-encoding: chunked") != std::string::npos) {
        std::string dec;
        size_t p = 0;
        for (;;) {
            size_t e = b.find("\r\n", p);
            if (e == std::string::npos) throw Error("http: bad chunked body");
            size_t n = std::strtoul(b.substr(p, e - p).c_str(), nullptr, 16);
            p = e + 2;
            if (!n) break;
            if (p + n > b.size()) throw Error("http: bad chunked body");
            dec += b.substr(p, n);
            p += n + 2;
        }
        b = dec;
    }
    if (b.size() > max_body) throw Error("http: response too large");
    out.body = b;
    return out;
}

}  // namespace platform
}  // namespace ballrain
#endif
