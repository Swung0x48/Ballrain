from enum import Enum
import numpy as np

class MsgType(Enum):
    BallNavActive = 0
    BallNavInactive = 1
    BallState = 2
    KbdInput = 3

msg_body_len = {
    MsgType.BallNavActive.value: 0,
    MsgType.BallNavInactive.value: 0,
    MsgType.BallState.value: 32,
    MsgType.KbdInput.value: 4,
}

class MsgBallState:
    def __init__(self, bmsg_body):
        assert(len(bmsg_body) == msg_body_len[MsgType.BallState.value])
        self.ball_type = int.from_bytes(bmsg_body, byteorder='little')
        dt = np.dtype("float32")
        dt = dt.newbyteorder('<')
        self.position = np.frombuffer(bmsg_body, dtype=dt, count=3, offset=4)
        self.quaternion = np.frombuffer(bmsg_body, dtype=dt, count=4, offset=16)
