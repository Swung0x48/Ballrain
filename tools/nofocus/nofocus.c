// nofocus.dll (x86): loaded into Player.exe before its first window (nofocus_launch.exe queues LoadLibraryW as an
// APC on the suspended main thread, so it runs inside the loader's initialisation, before Player's WinMain).
// It installs a thread WH_CBT hook on that thread that refuses every window activation (HCBT_ACTIVATE): the
// process's windows (splash, main window) can never become the active / foreground window, so the game never takes
// the foreground away from the user.  Nothing else in the process changes.
//
// NOFOCUS_LOG=<file>: one line per refused activation (and the install line).
#include <windows.h>
#include <stdio.h>

static HHOOK g_hook;
static HMODULE g_self;
static LONG g_refused;

static void log_line(const char *fmt, DWORD a, DWORD b) {
    char path[MAX_PATH];
    DWORD n = GetEnvironmentVariableA("NOFOCUS_LOG", path, sizeof(path));
    if (!n || n >= sizeof(path)) return;
    HANDLE f = CreateFileA(path, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_ALWAYS, 0, NULL);
    if (f == INVALID_HANDLE_VALUE) return;
    char line[256];
    int len = wsprintfA(line, fmt, a, b);
    DWORD written;
    WriteFile(f, line, (DWORD)len, &written, NULL);
    CloseHandle(f);
}

static LRESULT CALLBACK cbt(int code, WPARAM wparam, LPARAM lparam) {
    if (code == HCBT_ACTIVATE) {
        LONG n = InterlockedIncrement(&g_refused);
        log_line("nofocus: pid %lu refused activation #%ld\r\n", GetCurrentProcessId(), (DWORD)n);
        return 1;   // prevent the activation
    }
    return CallNextHookEx(g_hook, code, wparam, lparam);
}

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID reserved) {
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        g_self = inst;
        DisableThreadLibraryCalls(inst);
        // The loading thread is Player's main thread (APC); its windows are created on it.
        g_hook = SetWindowsHookExW(WH_CBT, cbt, inst, GetCurrentThreadId());
        log_line("nofocus: pid %lu CBT hook %lu\r\n", GetCurrentProcessId(), (DWORD)(g_hook ? 1 : 0));
    }
    return TRUE;
}
