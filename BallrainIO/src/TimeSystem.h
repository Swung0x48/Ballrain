#include <BML/IMod.h>

class TimeSystem {
public:
	TimeSystem(CKTimeManager* timemgr): m_timeManager(timemgr) {}
	void Process();
private:
	const float PHYSICS_DT = 1000.0f / 132.0f;
	const float DEFAULT_INPUT_DT = PHYSICS_DT / 2.0f;
	CKTimeManager* m_timeManager = nullptr;
};