// nofocus_launch.exe (x86): start a 32-bit program so that it never takes the foreground.
//
//   nofocus_launch.exe <nofocus.dll> <program.exe> [args...]
//
// The program is created suspended in the launcher's working directory with the launcher's environment, its
// STARTUPINFO show flags (so a caller's SW_HIDE reaches it) and no console; LoadLibraryW(<nofocus.dll>) is queued as
// an APC on its main thread (it runs during the loader's initialisation, before the program's own code), then the
// thread is resumed.  Prints "pid <n>" and exits 0, or prints an error and exits non-zero (the child is terminated).
#include <windows.h>
#include <stdio.h>
#include <wchar.h>

static int fail(const char *what, PROCESS_INFORMATION *pi) {
    fprintf(stdout, "error %s %lu\n", what, GetLastError());
    if (pi && pi->hProcess) { TerminateProcess(pi->hProcess, 1); CloseHandle(pi->hThread); CloseHandle(pi->hProcess); }
    return 2;
}

int wmain(int argc, wchar_t **argv) {
    if (argc < 3) { fprintf(stdout, "usage: nofocus_launch <nofocus.dll> <program.exe> [args...]\n"); return 1; }
    wchar_t dll[MAX_PATH];
    if (!GetFullPathNameW(argv[1], MAX_PATH, dll, NULL) || GetFileAttributesW(dll) == INVALID_FILE_ATTRIBUTES)
        return fail("dll", NULL);
    // command line: the program and its arguments, quoted
    static wchar_t cmd[32768];
    cmd[0] = 0;
    for (int i = 2; i < argc; ++i) {
        if (i > 2) wcscat_s(cmd, 32768, L" ");
        wcscat_s(cmd, 32768, L"\"");
        wcscat_s(cmd, 32768, argv[i]);
        wcscat_s(cmd, 32768, L"\"");
    }
    STARTUPINFOW mine;
    GetStartupInfoW(&mine);
    STARTUPINFOW si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    if (mine.dwFlags & STARTF_USESHOWWINDOW) { si.dwFlags |= STARTF_USESHOWWINDOW; si.wShowWindow = mine.wShowWindow; }
    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));
    if (!CreateProcessW(argv[2], cmd, NULL, NULL, FALSE, CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP, NULL, NULL, &si, &pi))
        return fail("CreateProcess", NULL);
    const SIZE_T bytes = (wcslen(dll) + 1) * sizeof(wchar_t);
    void *remote = VirtualAllocEx(pi.hProcess, NULL, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!remote || !WriteProcessMemory(pi.hProcess, remote, dll, bytes, NULL)) return fail("write", &pi);
    // kernel32 is mapped at the same address in every 32-bit process of the session.
    PAPCFUNC load = (PAPCFUNC)GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "LoadLibraryW");
    if (!load || !QueueUserAPC(load, pi.hThread, (ULONG_PTR)remote)) return fail("apc", &pi);
    if (ResumeThread(pi.hThread) == (DWORD)-1) return fail("resume", &pi);
    fprintf(stdout, "pid %lu\n", pi.dwProcessId);
    fflush(stdout);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return 0;
}
