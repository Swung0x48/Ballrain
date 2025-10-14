#include <BML/IMod.h>
#include <CKInputManager.h>
#define _WIN32_LEAN_AND_MEAN
#include <Windows.h>

struct KeyState {
	enum Keys {
		BR_UP = 0,
		BR_DOWN,
		BR_LEFT,
		BR_RIGHT,
		BR_ESC,
		BR_SHIFT,
		BR_KEY_COUNT
	};
	char states[BR_KEY_COUNT];
};


class InputSystem {
public:
	InputSystem(InputHook* input);
	void Process(const KeyState& state);
	static void ApplyKeyState(unsigned char& state, char newState);
private:
	HANDLE m_hPipe = INVALID_HANDLE_VALUE;
	InputHook* m_inputManager = nullptr;
	int m_keyMap[KeyState::BR_KEY_COUNT] = {
		CKKEY_UP,
		CKKEY_DOWN,
		CKKEY_LEFT,
		CKKEY_RIGHT,
		CKKEY_ESCAPE,
		CKKEY_LSHIFT
	};
};