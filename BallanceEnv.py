from enum import Enum
import random
from typing import Optional
import numpy as np
import gymnasium as gym

from TCPServer import TCPServer

class MsgType(Enum):
    BallNavActive = 0
    BallNavInactive = 1
    BallState = 2

class BallanceEnv(gym.Env):

    def __init__(self):
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

        # 4 keys that player can control, and multiple keys can be pressed at once
        # Up / Down / Left / Right
        # Not necessarily useful, but put those here just in case
        self.action_space = gym.spaces.Box(0, 1, shape=(4,), dtype=np.int8)

        self.observation_space = gym.spaces.Dict(
            {
                "location": gym.spaces.Box(-9999., 9999., shape=(3,), dtype=np.float32)
            }
        )

    def _get_obs(self):
        return {
            "location": np.array([random.randint(10,20), 2., 3.], dtype=np.float32)
        }

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

        msg_type = -1
        while msg_type != MsgType.BallNavActive.value:
            msg_type, byte = self.server.recv_msg()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # TODO: Apply input to game

        # TODO: Update observable state from game, check if still in valid shape

        # Check if agent reached the target
        terminated = False

        # We don't use truncation now
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info


gym.register(
    id="ballance_env/Ballance-v0",
    entry_point=BallanceEnv,
    max_episode_steps=300,  # Prevent infinite episodes
)
