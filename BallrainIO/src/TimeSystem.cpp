#include "TimeSystem.h"

void TimeSystem::Process()
{
	m_timeManager->SetLastDeltaTime(DEFAULT_INPUT_DT);
}
