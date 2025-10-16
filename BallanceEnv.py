import random
from typing import Optional
import numpy as np
import gymnasium as gym

from TCPServer import TCPServer
from Message import MsgType


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
        self._should_truncate = False
        self._floor_boxes = []

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

        self.observation_space = gym.spaces.Dict(
            {
                "ball_id": gym.spaces.Discrete(3),
                "position": gym.spaces.Box(-9999., 9999., shape=(3,), dtype=np.float32),
                "quaternion": gym.spaces.Box(-1., 1., shape=(4,), dtype=np.float32),
                "current_sector": gym.spaces.Discrete(100),
                "last_sector_position": gym.spaces.Box(-9999., 9999., shape=(3,), dtype=np.float32),
                "next_sector_position": gym.spaces.Box(-9999., 9999., shape=(3,), dtype=np.float32),
            }
        )

    def _get_obs(self):
        return {
            "ball_id": self._ball_id,
            "position": self._position,
            "quaternion": self._quaternion,
            "current_sector": self._current_sector,
            "last_sector_position": self._last_sector_position,
            "next_sector_position": self._next_sector_position,
        }

    def _get_reward(self):
        sector_dist = np.linalg.norm(self._last_sector_position - self._next_sector_position)
        dist_done = np.linalg.norm(self._position - self._last_sector_position)
        dist_ahead = np.linalg.norm(self._position - self._next_sector_position)
        last_dist_done = np.linalg.norm(self._last_position - self._last_sector_position)
        return ((dist_done - sector_dist) / sector_dist * 100.
                - dist_ahead / sector_dist * 150.
                - (100. if (self._naction[0] == self._naction[1] and self._naction[2] == self._naction[3]) else 0)
                + (last_dist_done - dist_done) / sector_dist * self._step
                + self._naction[0] * 10.
                + ((self._position[1] + 10.) / 10.) * 20.
                - (20. if self._should_truncate else 0))

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
        print('Game reset request sent...', end='')
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

        print(f'scene rep: {self._floor_boxes}')

        # Should get first game tick here
        # Game should send tick first then wait for input
        self.fetch_tick()

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
                self._current_sector = msgbody.current_sector
                self._next_sector_position = msgbody.next_sector_position
                self._last_sector_position = msgbody.last_sector_position
            elif msgtype == MsgType.BallOff.value:
                self._should_truncate = True
            msgtype, msgbody = self.server.recv_msg()

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self._step += 1
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

        # Check if agent reached the target or failed
        terminated = self._should_truncate

        # Decide if should truncate
        truncated = self._should_truncate

        reward = self._get_reward()
        # print(f"Reward: {reward}")

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info


gym.register(
    id="ballance_env/Ballance-v0",
    entry_point=BallanceEnv,
    max_episode_steps=300,  # Prevent infinite episodes
)
