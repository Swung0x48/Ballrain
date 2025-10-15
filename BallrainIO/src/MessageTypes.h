#ifndef BALLRAIN_MESSAGETYPES_H
#define BALLRAIN_MESSAGETYPES_H

#include <BML/IMod.h>
#include "InputSystem.h"

enum class MessageType : int {
	BRM_BallNavActive,
	BRM_BallNavInactive,
	BRM_GameState,
	BRM_KbdInput,
	BRM_Tick, // Tell server to advance to next tick
	BRM_ResetInput,
	BRM_InvalidType
};

struct MsgGameState {
	int ballType = 0;
	VxVector position;
	VxQuaternion quaternion;
	int currentSector = 0;
	VxVector nextSectorPosition;
};

struct MsgKbdInput {
	KeyState keyState;
};

#endif // BALLRAIN_MESSAGETYPES_H