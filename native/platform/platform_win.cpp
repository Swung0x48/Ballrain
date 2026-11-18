// Windows implementation of native/platform/platform.h (Winsock2, CreateProcessW).
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cwchar>
#include <map>
#include <vector>

#include "ballrain/util.h"
#include "platform.h"

namespace ballrain {
namespace platform {

static std::wstring widen(const std::string &s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
    std::wstring w(size_t(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), &w[0], n);
    return w;
}
static std::string narrow(const std::wstring &w) {
    if (w.empty()) return std::string();
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(size_t(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), &s[0], n, nullptr, nullptr);
    return s;
}

double monotonic_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int self_pid() { return int(GetCurrentProcessId()); }

std::vector<std::pair<std::string, std::string>> environment() {
    std::vector<std::pair<std::string, std::string>> out;
    wchar_t *blk = GetEnvironmentStringsW();
    if (!blk) return out;
    for (wchar_t *p = blk; *p; p += std::wcslen(p) + 1) {
        std::wstring e(p);
        size_t eq = e.find(L'=', 1);  // entries like "=C:=C:\..." start with '='
        if (eq == std::wstring::npos || e[0] == L'=') continue;
        out.emplace_back(narrow(e.substr(0, eq)), narrow(e.substr(eq + 1)));
    }
    FreeEnvironmentStringsW(blk);
    return out;
}

bool make_dirs(const std::string &path) {
    std::wstring w = widen(path);
    for (auto &c : w)
        if (c == L'/') c = L'\\';
    for (size_t i = 1; i <= w.size(); ++i) {
        if (i == w.size() || w[i] == L'\\') {
            std::wstring part = w.substr(0, i);
            if (part.size() == 2 && part[1] == L':') continue;
            if (!CreateDirectoryW(part.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS) {
                DWORD a = GetFileAttributesW(part.c_str());
                if (a == INVALID_FILE_ATTRIBUTES || !(a & FILE_ATTRIBUTE_DIRECTORY)) return false;
            }
        }
    }
    return true;
}

bool remove_file(const std::string &path) { return DeleteFileW(widen(path).c_str()) != 0; }

std::string absolute_path(const std::string &path) {
    std::wstring w = widen(path);
    DWORD n = GetFullPathNameW(w.c_str(), 0, nullptr, nullptr);
    if (!n) return path;
    std::wstring out(n, L'\0');
    n = GetFullPathNameW(w.c_str(), n, &out[0], nullptr);
    out.resize(n);
    return narrow(out);
}

std::string temp_dir() {
    wchar_t buf[MAX_PATH + 1];
    DWORD n = GetTempPathW(MAX_PATH + 1, buf);
    std::string s = narrow(std::wstring(buf, n));
    while (!s.empty() && (s.back() == '\\' || s.back() == '/')) s.pop_back();
    return s;
}

// ---------------------------------------------------------------------------------------------- sockets

static void wsa_init() {
    static bool done = false;
    if (!done) {
        WSADATA d;
        if (WSAStartup(MAKEWORD(2, 2), &d) != 0) throw Error("WSAStartup failed");
        done = true;
    }
}

class WinStream : public Stream {
public:
    explicit WinStream(SOCKET s) : s_(s) {
        BOOL one = TRUE;
        setsockopt(s_, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char *>(&one), sizeof one);
    }
    ~WinStream() override { close(); }
    bool send_all(const void *p, size_t n) override {
        auto b = static_cast<const char *>(p);
        while (n) {
            int k = ::send(s_, b, int(std::min<size_t>(n, 1 << 30)), 0);
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
                int k = select(0, &r, nullptr, nullptr, &tv);
                if (k == 0) throw Error("timeout");
                if (k < 0) return false;
            }
            int k = ::recv(s_, b, int(std::min<size_t>(n, 1 << 30)), 0);
            if (k <= 0) return false;
            b += k, n -= size_t(k);
        }
        return true;
    }
    void close() override {
        if (s_ != INVALID_SOCKET) closesocket(s_), s_ = INVALID_SOCKET;
    }

private:
    SOCKET s_;
};

class WinListener : public Listener {
public:
    explicit WinListener(int port) {
        wsa_init();
        // not inheritable: an engine started later must not hold this listener (or another engine's connection)
        s_ = WSASocketW(AF_INET, SOCK_STREAM, IPPROTO_TCP, nullptr, 0, WSA_FLAG_OVERLAPPED | WSA_FLAG_NO_HANDLE_INHERIT);
        if (s_ == INVALID_SOCKET) throw Error("socket() failed");
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = htons(uint16_t(port));
        if (bind(s_, reinterpret_cast<sockaddr *>(&a), sizeof a) != 0) throw Error("bind() failed");
        if (listen(s_, 1) != 0) throw Error("listen() failed");
        int len = sizeof a;
        getsockname(s_, reinterpret_cast<sockaddr *>(&a), &len);
        port_ = ntohs(a.sin_port);
    }
    ~WinListener() override {
        if (s_ != INVALID_SOCKET) closesocket(s_);
    }
    int port() const override { return port_; }
    std::unique_ptr<Stream> accept(double timeout_s) override {
        fd_set r;
        FD_ZERO(&r);
        FD_SET(s_, &r);
        timeval tv;
        tv.tv_sec = long(timeout_s);
        tv.tv_usec = long((timeout_s - double(tv.tv_sec)) * 1e6);
        int k = select(0, &r, nullptr, nullptr, &tv);
        if (k <= 0) return nullptr;
        SOCKET c = ::accept(s_, nullptr, nullptr);
        if (c == INVALID_SOCKET) return nullptr;
        SetHandleInformation(reinterpret_cast<HANDLE>(c), HANDLE_FLAG_INHERIT, 0);
        return std::make_unique<WinStream>(c);
    }

private:
    SOCKET s_ = INVALID_SOCKET;
    int port_ = 0;
};

std::unique_ptr<Listener> Listener::listen_loopback(int port) { return std::make_unique<WinListener>(port); }

std::unique_ptr<Stream> Listener::connect_loopback(int port, double timeout_s) {
    wsa_init();
    double end = monotonic_s() + timeout_s;
    for (;;) {
        SOCKET s = WSASocketW(AF_INET, SOCK_STREAM, IPPROTO_TCP, nullptr, 0, WSA_FLAG_OVERLAPPED | WSA_FLAG_NO_HANDLE_INHERIT);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = htons(uint16_t(port));
        if (connect(s, reinterpret_cast<sockaddr *>(&a), sizeof a) == 0) return std::make_unique<WinStream>(s);
        closesocket(s);
        if (monotonic_s() > end) return nullptr;
        Sleep(20);
    }
}

// ---------------------------------------------------------------------------------------------- processes

static std::wstring quote_arg(const std::wstring &a) {
    if (!a.empty() && a.find_first_of(L" \t\n\v\"") == std::wstring::npos) return a;
    std::wstring out = L"\"";
    for (size_t i = 0;; ++i) {
        size_t bs = 0;
        while (i < a.size() && a[i] == L'\\') ++i, ++bs;
        if (i == a.size()) {
            out.append(bs * 2, L'\\');
            break;
        }
        if (a[i] == L'"') out.append(bs * 2 + 1, L'\\'), out += L'"';
        else out.append(bs, L'\\'), out += a[i];
    }
    return out + L"\"";
}

class WinProcess : public Process {
public:
    explicit WinProcess(const Spec &spec) {
        std::wstring cmd;
        for (size_t i = 0; i < spec.argv.size(); ++i) {
            if (i) cmd += L' ';
            cmd += quote_arg(widen(spec.argv[i]));
        }
        // environment block: sorted (case-insensitively) NAME=value\0 ... \0
        std::vector<std::wstring> vars;
        for (auto &kv : spec.env) vars.push_back(widen(kv.first + "=" + kv.second));
        std::sort(vars.begin(), vars.end(), [](const std::wstring &a, const std::wstring &b) {
            return _wcsicmp(a.c_str(), b.c_str()) < 0;
        });
        std::wstring blk;
        for (auto &v : vars) blk += v, blk += L'\0';
        blk += L'\0';
        SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
        HANDLE nul_in = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
        HANDLE nul_out = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
        HANDLE err = nul_out;
        if (!spec.stderr_path.empty())
            err = CreateFileW(widen(spec.stderr_path).c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
        // the child inherits exactly its three standard handles (PROC_THREAD_ATTRIBUTE_HANDLE_LIST), nothing else this
        // process has open: sockets of other engines, recording files (Python's Popen does the same)
        STARTUPINFOEXW si{};
        si.StartupInfo.cb = sizeof si;
        si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        si.StartupInfo.hStdInput = nul_in, si.StartupInfo.hStdOutput = nul_out, si.StartupInfo.hStdError = err;
        HANDLE inherit[3] = {nul_in, nul_out, err};
        DWORD n_inherit = err == nul_out ? 2 : 3;
        SIZE_T attr_size = 0;
        InitializeProcThreadAttributeList(nullptr, 1, 0, &attr_size);
        std::vector<char> attr_buf(attr_size);
        si.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attr_buf.data());
        bool attr_ok = InitializeProcThreadAttributeList(si.lpAttributeList, 1, 0, &attr_size) &&
                       UpdateProcThreadAttribute(si.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit,
                                                 n_inherit * sizeof(HANDLE), nullptr, nullptr);
        PROCESS_INFORMATION pi{};
        std::wstring cwd = widen(spec.cwd);
        BOOL ok = attr_ok && CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, TRUE,
                                            CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
                                            &blk[0], cwd.empty() ? nullptr : cwd.c_str(), &si.StartupInfo, &pi);
        DWORD e = GetLastError();
        if (attr_ok) DeleteProcThreadAttributeList(si.lpAttributeList);
        if (nul_in != INVALID_HANDLE_VALUE) CloseHandle(nul_in);
        if (err != nul_out && err != INVALID_HANDLE_VALUE) CloseHandle(err);
        if (nul_out != INVALID_HANDLE_VALUE) CloseHandle(nul_out);
        if (!ok) throw Error("CreateProcess failed (" + std::to_string(e) + ") for " + spec.argv[0]);
        CloseHandle(pi.hThread);
        h_ = pi.hProcess;
        pid_ = int(pi.dwProcessId);
    }
    ~WinProcess() override {
        if (h_) {
            if (running()) kill();
            CloseHandle(h_);
        }
    }
    int pid() const override { return pid_; }
    bool wait(double timeout_s) override {
        DWORD ms = timeout_s <= 0 ? 0 : DWORD(std::min(timeout_s * 1000.0, 4.0e9));
        return WaitForSingleObject(h_, ms) == WAIT_OBJECT_0;
    }
    bool running() override { return WaitForSingleObject(h_, 0) == WAIT_TIMEOUT; }
    int exit_code() override {
        DWORD c;
        if (!GetExitCodeProcess(h_, &c) || c == STILL_ACTIVE) return -1;
        return int(c);
    }
    void kill() override {
        if (running()) {
            TerminateProcess(h_, 1);
            WaitForSingleObject(h_, 5000);
        }
    }

private:
    HANDLE h_ = nullptr;
    int pid_ = 0;
};

std::unique_ptr<Process> Process::spawn(const Spec &spec) { return std::make_unique<WinProcess>(spec); }

}  // namespace platform
}  // namespace ballrain
#endif
