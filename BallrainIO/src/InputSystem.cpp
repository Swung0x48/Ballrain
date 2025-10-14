#include "InputSystem.h"

InputSystem::InputSystem(InputHook* input) : m_inputManager(input) {
}

void InputSystem::Process()
{
	auto* kbd = m_inputManager->GetKeyboardState();
	kbd[CKKEY_UP] = KS_PRESSED;
}
