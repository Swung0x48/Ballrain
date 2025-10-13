#include <BML/IMod.h>
#include <CKInputManager.h>

class InputSystem {
public:
	InputSystem(InputHook* input): m_inputManager(input) {}
	void Process();
private:
	InputHook* m_inputManager = nullptr;
};