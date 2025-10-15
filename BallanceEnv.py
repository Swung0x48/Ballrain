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
        self._quaternion = np.zeros(shape=(4,), dtype=np.float32)
        self._current_sector = 0
        self._last_sector_position = np.zeros(shape=(4,), dtype=np.float32)
        self._next_sector_position = np.zeros(shape=(4,), dtype=np.float32)

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
        self.action_space = gym.spaces.Box(0, 1, shape=(4,), dtype=np.int8)

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
        return np.linalg.norm(self._position - self._next_sector_position)

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

        self.server.send_msg(MsgType.ResetInput)
        print('Game reset request sent...', end='')
        msg_type, _ = self.server.recv_msg()
        # This should eat up lingering states from last session
        while msg_type != MsgType.BallNavActive.value:
            msg_type, _ = self.server.recv_msg()
        print('BallNav active regained.')

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
            msgtype, msgbody = self.server.recv_msg()

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # TODO: Apply input to game
        # print("Action: ", action)
        self.server.send_msg(MsgType.KbdInput, action.tobytes())

        # Store previous state for reward calculation
        prev_position = self._position.copy()

        # TODO: Update observable state from game, check if still in valid shape
        self.fetch_tick()

        # Check if agent reached the target or failed
        # In a ball balancing game, we might consider falling off as terminated
        # For now, let's define terminated if the ball goes too far from origin
        terminated = False  # Terminate if too far from center

        # We don't use truncation now
        # (could add a step limit here if desired)
        truncated = False

        reward =self._get_reward()

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info


gym.register(
    id="ballance_env/Ballance-v0",
    entry_point=BallanceEnv,
    max_episode_steps=300,  # Prevent infinite episodes
)
