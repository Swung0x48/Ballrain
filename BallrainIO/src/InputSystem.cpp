#include "InputSystem.h"

void InputSystem::Process()
{
	auto* kbd = m_inputManager->GetKeyboardState();
	kbd[CKKEY_UP] = KS_PRESSED;
}
