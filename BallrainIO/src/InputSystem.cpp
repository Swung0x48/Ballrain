#include "InputSystem.h"

#define PIPENAME ("\\\\.\\pipe\\BallancePipe")
#define BUFSIZE 512

InputSystem::InputSystem(InputHook* input) : m_inputManager(input) {
    m_hPipe = CreateFile(
        PIPENAME,
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL);

    if (m_hPipe == INVALID_HANDLE_VALUE)
    {
        return;
    }
    auto bSuccess = SetNamedPipeHandleState(
        m_hPipe, 
        (LPDWORD)PIPE_READMODE_BYTE,
        NULL,
        NULL
    );

    if (!bSuccess)
    {
        CloseHandle(m_hPipe);
        return;
    }

    WriteFile(m_hPipe, "123456789", 10, NULL, NULL);
}

void InputSystem::Process()
{
	auto* kbd = m_inputManager->GetKeyboardState();
	kbd[CKKEY_UP] = KS_PRESSED;
}
