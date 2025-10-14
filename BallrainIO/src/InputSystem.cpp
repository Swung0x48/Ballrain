#include "InputSystem.h"

InputSystem::InputSystem(InputHook* input) : m_inputManager(input) {
}

void InputSystem::Process(const KeyState& state)
{
	auto* kbd = m_inputManager->GetKeyboardState();
	for (int i = 0; i < KeyState::BR_KEY_COUNT; ++i) {
		ApplyKeyState(kbd[m_keyMap[i]], state.states[i]);
	}
}

void InputSystem::ApplyKeyState(unsigned char& state, char newState)
{
	if (newState) {
		state = KS_PRESSED;
	}
	else {
		if (state == KS_PRESSED)
			state == KS_RELEASED;
		else
			state == KS_IDLE;
	}
}
