from enum import Enum
import numpy as np

class MsgType(Enum):
    BallNavActive = 0
    BallNavInactive = 1
    GameState = 2
    KbdInput = 3
    Tick = 4
    ResetInput = 5
    BallOff = 6
    SceneRep = 7
    DepthImage = 8

msg_body_len = {
    MsgType.BallNavActive.value: 0,
    MsgType.BallNavInactive.value: 0,
    MsgType.GameState.value: 60,
    MsgType.KbdInput.value: 4,
    MsgType.Tick.value: 0,
    MsgType.ResetInput.value: 0,
    MsgType.BallOff.value: 0,
    MsgType.SceneRep.value: -1, # dynamic
    MsgType.DepthImage.value: 320 * 240 * 2
}

class MsgGameState:
    def __init__(self, bmsg_body):
        assert(len(bmsg_body) == msg_body_len[MsgType.GameState.value])
        self.ball_type = int.from_bytes(bmsg_body[:3], byteorder='little')
        dt = np.dtype("float32")
        dt = dt.newbyteorder('<')
        self.position = np.frombuffer(bmsg_body, dtype=dt, count=3, offset=4)
        self.quaternion = np.frombuffer(bmsg_body, dtype=dt, count=4, offset=16)
        self.current_sector = int.from_bytes(bmsg_body[32:35], byteorder='little')
        self.next_sector_position = np.frombuffer(bmsg_body, dtype=dt, count=3, offset=36)
        self.last_sector_position = np.frombuffer(bmsg_body, dtype=dt, count=3, offset=48)

class MsgSceneRep:
    def __init__(self, count, bmsg_body):
        self.floor_count = count
        self.floor_boxes = []
        dt = np.dtype("float32")
        dt = dt.newbyteorder('<')
        for i in range(self.floor_count):
            box = np.frombuffer(bmsg_body, dtype=dt, count=6, offset=24*i).reshape(2, 3)
            self.floor_boxes.append(box)

class MsgDepthImage:
    def __init__(self, bmsg_body):
        self.image = np.frombuffer(bmsg_body, dtype=np.float16).reshape(240, 320)