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

// One Job Object per client process: every background engine is assigned to it; KILL_ON_JOB_CLOSE ends them when
// the last handle (this process's) closes, also on a crash or a TerminateProcess of the client.
static HANDLE client_job() {
    static HANDLE job = [] {
        HANDLE j = CreateJobObjectW(nullptr, nullptr);
        if (j) {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
            li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if (!SetInformationJobObject(j, JobObjectExtendedLimitInformation, &li, sizeof li)) {
                CloseHandle(j);
                j = nullptr;
            }
        }
        return j;
    }();
    return job;
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
        DWORD flags = CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT;
        if (spec.background) flags |= CREATE_SUSPENDED | BELOW_NORMAL_PRIORITY_CLASS;
        BOOL ok = attr_ok && CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, TRUE, flags, &blk[0],
                                            cwd.empty() ? nullptr : cwd.c_str(), &si.StartupInfo, &pi);
        DWORD e = GetLastError();
        if (attr_ok) DeleteProcThreadAttributeList(si.lpAttributeList);
        if (nul_in != INVALID_HANDLE_VALUE) CloseHandle(nul_in);
        if (err != nul_out && err != INVALID_HANDLE_VALUE) CloseHandle(err);
        if (nul_out != INVALID_HANDLE_VALUE) CloseHandle(nul_out);
        if (!ok) throw Error("CreateProcess failed (" + std::to_string(e) + ") for " + spec.argv[0]);
        if (spec.background) {  // created suspended: in the job before it runs a single instruction
            HANDLE job = client_job();
            if (!job || !AssignProcessToJobObject(job, pi.hProcess)) {
                DWORD je = GetLastError();
                TerminateProcess(pi.hProcess, 1);
                CloseHandle(pi.hThread);
                CloseHandle(pi.hProcess);
                throw Error("cannot put the engine into the client's job object (" + std::to_string(je) + ")");
            }
            ResumeThread(pi.hThread);
        }
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

// ---------------------------------------------------------------------------------------------- files, HTTP (Windows)
#ifdef _WIN32
#include <winhttp.h>
#include <shellapi.h>
#include <shlobj.h>
#include <tlhelp32.h>
#include <cctype>
#include <cwctype>
#include <ctime>

namespace ballrain {
namespace platform {

bool is_dir(const std::string &path) {
    DWORD a = GetFileAttributesW(widen(path).c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}

std::vector<std::string> list_dir(const std::string &dir) {
    std::vector<std::string> out;
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(widen(dir + "\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return out;
    do {
        std::wstring n = fd.cFileName;
        if (n != L"." && n != L"..") out.push_back(narrow(n));
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    return out;
}

bool rename_file(const std::string &from, const std::string &to) {
    // a file just written is often held for a moment by a virus scanner / the indexer (sharing violation, access
    // denied): retry for up to ~2 s before giving up
    std::wstring f = widen(from), t = widen(to);
    for (int i = 0; i < 100; ++i) {
        if (MoveFileExW(f.c_str(), t.c_str(), MOVEFILE_REPLACE_EXISTING)) return true;
        DWORD e = GetLastError();
        if (e != ERROR_SHARING_VIOLATION && e != ERROR_ACCESS_DENIED && e != ERROR_LOCK_VIOLATION) return false;
        Sleep(20);
    }
    return false;
}

int64_t file_size(const std::string &path) {
    WIN32_FILE_ATTRIBUTE_DATA d;
    if (!GetFileAttributesExW(widen(path).c_str(), GetFileExInfoStandard, &d)) return -1;
    return (int64_t(d.nFileSizeHigh) << 32) | int64_t(d.nFileSizeLow);
}

int64_t unix_time() { return int64_t(std::time(nullptr)); }

static std::string known_folder(REFKNOWNFOLDERID id) {
    PWSTR p = nullptr;
    std::string out;
    if (SHGetKnownFolderPath(id, 0, nullptr, &p) == S_OK) out = narrow(p);
    if (p) CoTaskMemFree(p);
    return out;
}

std::string data_dir() {
    std::string base = known_folder(FOLDERID_LocalAppData);
    if (base.empty()) base = home_dir() + "\\AppData\\Local";
    return base + "\\BallrainPool";
}

std::string home_dir() { return known_folder(FOLDERID_Profile); }

std::string engine_work_dir() {
    wchar_t buf[1024];
    DWORD n = GetEnvironmentVariableW(L"BALLRAIN_WORK_DIR", buf, 1024);
    if (n > 0 && n < 1024) return narrow(std::wstring(buf, n));
    std::string base = known_folder(FOLDERID_ProgramData);
    if (base.empty()) base = "C:\\ProgramData";
    return base + "\\BallrainPool";
}

std::string short_path(const std::string &path) {
    std::wstring w = widen(path);
    DWORD n = GetShortPathNameW(w.c_str(), nullptr, 0);
    if (!n) return path;
    std::wstring out(n, L'\0');
    n = GetShortPathNameW(w.c_str(), &out[0], n);
    out.resize(n);
    return n ? narrow(out) : path;
}

std::string executable_path() {
    std::wstring buf(32768, L'\0');
    DWORD n = GetModuleFileNameW(nullptr, &buf[0], DWORD(buf.size()));
    buf.resize(n);
    return narrow(buf);
}

std::string user_name() {
    wchar_t buf[256];
    DWORD n = 256;
    return GetUserNameW(buf, &n) ? narrow(std::wstring(buf, n ? n - 1 : 0)) : std::string();
}

bool open_url(const std::string &url) {
    if (url.rfind("https://", 0) != 0 && url.rfind("http://", 0) != 0) return false;  // never a file or a program
    return reinterpret_cast<INT_PTR>(ShellExecuteW(nullptr, L"open", widen(url).c_str(), nullptr, nullptr,
                                                   SW_SHOWNORMAL)) > 32;
}

bool single_instance(const std::string &data_dir) {
    std::wstring name = L"Local\\BallrainPool-";
    uint32_t h = 2166136261u;  // FNV-1a of the folded path: one mutex per data directory
    for (char c : data_dir) h = (h ^ uint32_t(std::tolower(static_cast<unsigned char>(c == '\\' ? '/' : c)))) * 16777619u;
    name += std::to_wstring(h);
    HANDLE m = CreateMutexW(nullptr, TRUE, name.c_str());  // never closed: held until the process exits
    return m && GetLastError() != ERROR_ALREADY_EXISTS;
}

bool process_running(const std::string &image_name) {
    std::wstring want = widen(image_name);
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return false;
    PROCESSENTRY32W pe{};
    pe.dwSize = sizeof pe;
    bool found = false;
    for (BOOL ok = Process32FirstW(snap, &pe); ok && !found; ok = Process32NextW(snap, &pe))
        found = _wcsicmp(pe.szExeFile, want.c_str()) == 0;
    CloseHandle(snap);
    return found;
}

bool on_battery() {
    SYSTEM_POWER_STATUS s;
    if (!GetSystemPowerStatus(&s)) return false;
    return s.ACLineStatus == 0 && s.BatteryFlag != 128;  // 128: no system battery
}

int cpu_count() {
    DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    return n ? int(n) : 1;
}

static bool walk(const std::wstring &root, const std::wstring &rel, std::vector<DirEntry> &out, size_t max_files) {
    std::wstring pat = root + (rel.empty() ? L"" : L"\\" + rel) + L"\\*";
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(pat.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return GetLastError() == ERROR_FILE_NOT_FOUND;
    bool ok = true;
    do {
        std::wstring n = fd.cFileName;
        if (n == L"." || n == L"..") continue;
        std::wstring r = rel.empty() ? n : rel + L"\\" + n;
        bool link = (fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
        if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && !link) {
            ok = walk(root, r, out, max_files) && ok;
        } else if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            std::string s = narrow(r);
            for (auto &c : s)
                if (c == '\\') c = '/';
            out.push_back({s, link});
            if (out.size() > max_files) {
                ok = false;
                break;
            }
        }
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    return ok;
}

bool list_files(const std::string &dir, std::vector<DirEntry> &out, size_t max_files) {
    out.clear();
    if (!is_dir(dir)) return false;
    return walk(widen(dir), L"", out, max_files);
}

HttpResponse http_request(const std::string &method, const std::string &url,
                          const std::vector<std::pair<std::string, std::string>> &headers, const std::string &body,
                          double timeout_s, size_t max_body) {
    std::wstring wurl = widen(url);
    URL_COMPONENTS uc{};
    uc.dwStructSize = sizeof uc;
    uc.dwHostNameLength = DWORD(-1);
    uc.dwUrlPathLength = DWORD(-1);
    uc.dwExtraInfoLength = DWORD(-1);
    if (!WinHttpCrackUrl(wurl.c_str(), 0, 0, &uc)) throw Error("http: bad url");
    std::wstring whost(uc.lpszHostName, uc.dwHostNameLength);
    std::wstring wpath(uc.lpszUrlPath, uc.dwUrlPathLength);
    if (uc.lpszExtraInfo && uc.dwExtraInfoLength) wpath += std::wstring(uc.lpszExtraInfo, uc.dwExtraInfoLength);
    if (wpath.empty()) wpath = L"/";
    bool tls = uc.nScheme == INTERNET_SCHEME_HTTPS;
    struct H {
        HINTERNET h = nullptr;
        ~H() {
            if (h) WinHttpCloseHandle(h);
        }
    } ses, con, req;
    ses.h = WinHttpOpen(L"ballrain-native-client", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME,
                        WINHTTP_NO_PROXY_BYPASS, 0);
    if (!ses.h) throw Error("http: WinHttpOpen failed");
    int ms = int(timeout_s * 1000);
    WinHttpSetTimeouts(ses.h, ms, ms, ms, ms);
    con.h = WinHttpConnect(ses.h, whost.c_str(), uc.nPort, 0);
    if (!con.h) throw Error("http: connect failed");
    req.h = WinHttpOpenRequest(con.h, widen(method).c_str(), wpath.c_str(), nullptr, WINHTTP_NO_REFERER,
                               WINHTTP_DEFAULT_ACCEPT_TYPES, tls ? WINHTTP_FLAG_SECURE : 0);
    if (!req.h) throw Error("http: open request failed");
    DWORD no_redirect = WINHTTP_DISABLE_REDIRECTS;
    WinHttpSetOption(req.h, WINHTTP_OPTION_DISABLE_FEATURE, &no_redirect, sizeof no_redirect);
    std::wstring hs;
    for (auto &kv : headers) hs += widen(kv.first + ": " + kv.second) + L"\r\n";
    if (!WinHttpSendRequest(req.h, hs.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : hs.c_str(), DWORD(hs.size()),
                            body.empty() ? WINHTTP_NO_REQUEST_DATA : const_cast<char *>(body.data()), DWORD(body.size()),
                            DWORD(body.size()), 0))
        throw Error("http: send failed (" + std::to_string(GetLastError()) + ")");
    if (!WinHttpReceiveResponse(req.h, nullptr)) throw Error("http: no response (" + std::to_string(GetLastError()) + ")");
    DWORD status = 0, sz = sizeof status;
    WinHttpQueryHeaders(req.h, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status,
                        &sz, WINHTTP_NO_HEADER_INDEX);
    HttpResponse r;
    r.status = int(status);
    for (;;) {
        DWORD avail = 0;
        if (!WinHttpQueryDataAvailable(req.h, &avail)) throw Error("http: read failed");
        if (!avail) break;
        if (r.body.size() + avail > max_body) throw Error("http: response too large");
        size_t at = r.body.size();
        r.body.resize(at + avail);
        DWORD got = 0;
        if (!WinHttpReadData(req.h, &r.body[at], avail, &got)) throw Error("http: read failed");
        r.body.resize(at + got);
    }
    return r;
}

}  // namespace platform
}  // namespace ballrain
#endif
