from enum import Enum

class MsgType(Enum):
    BallNavActive = 0
    BallNavInactive = 1
    BallState = 2,
    KbdInput = 3