import random
from typing import Optional
import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt

from TCPServer import TCPServer
from Message import MsgType


def _is_in_box(position, box: np.ndarray) -> bool:
    return np.all(position <= box[0]) and np.all(position >= box[1])


class BallanceEnv(gym.Env):
    def __init__(self):
        self._ball_id = 0
        self._position = np.zeros(shape=(3,), dtype=np.float32)
        self._last_position = np.zeros(shape=(3,), dtype=np.float32)
        self._quaternion = np.zeros(shape=(4,), dtype=np.float32)
        self._current_sector = 0
        self._last_sector_position = np.zeros(shape=(4,), dtype=np.float32)
        self._next_sector_position = np.zeros(shape=(4,), dtype=np.float32)
        self._step = 0
        self._advance_count = 0.
        self._up_hold_count = 0.
        self._should_truncate = False
        self._floor_boxes = []
        self._reward = 0.
        self._depth_image = np.zeros(shape=(1, 240, 320), dtype=np.uint8)

        print("Started server. Waiting for connection...")
        self.server = TCPServer(port=27787)
        self.server.start()
        # Wait for game to go online
        print("Connection established")
        self.server.send('Ping'.encode('utf-8'))
        pong_msg = self.server.recv(4)
        if pong_msg == b'Pong':
            print('Client ping-pong OK')
        else:
            raise Exception("Not a valid client!")

        print('Waiting for client to be controllable...')
        msg_type, _ = self.server.recv_msg()
        while msg_type != MsgType.BallNavActive.value:
            msg_type, _ = self.server.recv_msg()
        print('Client control taken. Env created.')

        # 4 keys that player can control, and multiple keys can be pressed at once
        # Up / Down / Left / Right
        # Not necessarily useful, but put those here just in case
        self.action_space = gym.spaces.Discrete(16)

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, 240, 320), dtype=np.uint8)

    def _get_obs(self):
        return self._depth_image

    def _is_on_floor(self):
        count = 0
        for box in self._floor_boxes:
            if _is_in_box(self._position, box):
                count += 1
        return count

    def _get_reward(self):
        # sector_dist = np.linalg.norm(self._last_sector_position - self._next_sector_position)
        # dist_past = np.linalg.norm(self._position - self._last_sector_position)
        # dist_ahead = np.linalg.norm(self._position - self._next_sector_position)
        # last_dist_past = np.linalg.norm(self._last_position - self._last_sector_position)

        lsp_2d = self._last_sector_position[0:3:2]
        nsp_2d = self._next_sector_position[0:3:2]
        pos_2d = self._position[0:3:2]
        lpos_2d = self._last_position[0:3:2]

        sector_dist = np.linalg.norm(lsp_2d - nsp_2d)

        sector_vec = (lsp_2d - nsp_2d) / sector_dist
        ball_vec = (lsp_2d - pos_2d) / sector_dist
        ball_progress = np.dot(ball_vec, sector_vec)

        pos_delta_vec = (pos_2d - lpos_2d) / np.linalg.norm(ball_vec) # dont normalize here?
        delta_progress = np.dot(pos_delta_vec, sector_vec)

        if self._step % 10 == 0:
            self._last_position = self._position

        ball_reward = ball_progress * 200.
        delta_reward = delta_progress * 100.

        if self._step % 200 == 0:
            print("Ball progress: {:.2f}%".format(ball_progress))
            print("Delta progress: {:.2f}%".format(delta_progress))

        return (
            ball_reward
            + delta_reward
            - (2000 * self._step if ball_progress < 0.001 else 0)
            - (2000. if self._should_truncate else 0.)
        )

    def _get_info(self):
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        self._step = 0
        self._should_truncate = False

        self.server.send_msg(MsgType.ResetInput)
        print('Game reset request sent...')
        msg_type, msg_body = self.server.recv_msg()
        lingering_msg_cnt = 0
        # This should eat up lingering states from last session
        while msg_type != MsgType.BallNavActive.value:
            if msg_type == MsgType.SceneRep.value:
                self._floor_boxes = msg_body.floor_boxes.copy()
            else:
                lingering_msg_cnt += 1
            msg_type, msg_body = self.server.recv_msg()
        print(f'After eating up {lingering_msg_cnt} lingering states, BallNav active regained.')

        # print(f'scene rep: {self._floor_boxes}')

        # Should get first game tick here
        # Game should send tick first then wait for input
        self.fetch_tick()
        self._last_position = self._position

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def fetch_tick(self):
        msgtype, msgbody = self.server.recv_msg()
        while msgtype != MsgType.Tick.value:
            if msgtype == MsgType.GameState.value:
                self._ball_id = msgbody.ball_type
                self._position = msgbody.position
                self._quaternion = msgbody.quaternion
                if msgbody.current_sector != self._current_sector:
                    self._reward += 200.
                self._current_sector = msgbody.current_sector
                self._next_sector_position = msgbody.next_sector_position
                self._last_sector_position = msgbody.last_sector_position
            elif msgtype == MsgType.BallOff.value:
                self._should_truncate = True
            elif msgtype == MsgType.DepthImage.value:
                self._depth_image = msgbody.image
            msgtype, msgbody = self.server.recv_msg()

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self._reward = 0.

        # TODO: Apply input to game
        self._naction = np.array([
            (action & 0b0001) >> 0,
            (action & 0b0010) >> 1,
            (action & 0b0100) >> 2,
            (action & 0b1000) >> 3,
        ], dtype=np.uint8)
        # print(f"Action: {action} {self._naction}", end=' ')
        self.server.send_msg(MsgType.KbdInput, self._naction.tobytes())

        # TODO: Update observable state from game, check if still in valid shape
        self.fetch_tick()
        # print(f"position: {self._position}")
        # plt.imshow(self._depth_image.reshape(240, 320), cmap='gray', vmin=200, vmax=250)
        # plt.title("Depth Buffer")
        # plt.colorbar()
        # plt.show()

        # Check if agent reached the target or failed
        terminated = self._should_truncate

        # Decide if should truncate
        truncated = self._should_truncate

        reward = self._get_reward()
        # print(f"Reward: {reward}")

        observation = self._get_obs()
        info = self._get_info()

        self._step += 1

        return observation, reward, terminated, truncated, info


gym.register(
    id="ballance_env/Ballance-v0",
    entry_point=BallanceEnv,
    max_episode_steps=300,  # Prevent infinite episodes
)
