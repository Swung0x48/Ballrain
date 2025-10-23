#include <BML/IMod.h>

class TimeSystem {
public:
	TimeSystem(CKTimeManager* timemgr): m_timeManager(timemgr) {}
	void Process();
	CKTimeManager* m_timeManager = nullptr;
};