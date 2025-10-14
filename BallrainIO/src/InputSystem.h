#include <BML/IMod.h>
#include <CKInputManager.h>
#define _WIN32_LEAN_AND_MEAN
#include <Windows.h>

class InputSystem {
public:
	InputSystem(InputHook* input);
	void Process();
private:
	HANDLE m_hPipe = INVALID_HANDLE_VALUE;
	InputHook* m_inputManager = nullptr;
};