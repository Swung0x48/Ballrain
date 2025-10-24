import math
import random
from typing import Optional
import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt

from TCPServer import TCPServer
from Message import MsgType


def _is_in_box(position, box: np.ndarray) -> bool:
    return np.all(position <= box[0]) and np.all(position >= box[1])


def get_dist_to_edge(image: np.ndarray,  # in (C,H,W)
                     point: np.ndarray,
                     direction: np.ndarray,
                     ignore_dist: float,
                     threshold: float) -> float:
    direction = direction / np.linalg.norm(direction)
    ignore_vec = direction * ignore_dist
    shape = image.shape
    width = shape[2]
    height = shape[1]
    first_sample_loc = point + ignore_vec
    next_sample_loc = first_sample_loc
    while ((0 <= int(next_sample_loc[0]) < width) and
           (0 <= int(next_sample_loc[1]) < height)):
        sample = image[0, int(next_sample_loc[1]), int(next_sample_loc[0])]
        if sample >= threshold:
            break
        next_sample_loc = next_sample_loc + direction
    return float(np.linalg.norm(next_sample_loc - first_sample_loc))


class BallanceEnv(gym.Env):
    def __init__(self):
        self._action_lut = [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
        ]
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
        self._reward = 0.
        self._depth_image = np.zeros(shape=(1, 240, 320), dtype=np.uint8)
        self._dist_to_edge = np.zeros(shape=(8,), dtype=np.float32)

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
        self.action_space = gym.spaces.Discrete(len(self._action_lut))

        self.observation_space = gym.spaces.Dict(
            {
                'position': gym.spaces.Box(-1e6, 1e6, shape=(3,), dtype=np.float32),
                'quaternion': gym.spaces.Box(-1., 1., shape=(4,), dtype=np.float32),
                'last_sector_position': gym.spaces.Box(-1e6, 1e6, shape=(3,), dtype=np.float32),
                'next_sector_position': gym.spaces.Box(-1e6, 1e6, shape=(3,), dtype=np.float32),
                'dist_to_edge': gym.spaces.Box(0., 1., shape=(8,), dtype=np.float32),
            }
        )

    def _get_obs(self):
        return {
            'position': self._position,
            'quaternion': self._quaternion,
            'last_sector_position': self._last_sector_position,
            'next_sector_position': self._next_sector_position,
            'dist_to_edge': self._dist_to_edge,
        }

    def _is_in_any_box(self):
        count = 0
        for box in self._floor_boxes:
            if _is_in_box(self._position, box):
                count += 1
        return count

    def _get_reward(self):
        # Extract 2D positions (x, z coordinates)
        lsp_2d = self._last_sector_position[0:3:2]
        nsp_2d = self._next_sector_position[0:3:2]
        pos_2d = self._position[0:3:2]
        lpos_2d = self._last_position[0:3:2]

        # Calculate the distance of the current sector
        sector_dist = np.linalg.norm(nsp_2d - lsp_2d)

        # Handle case where sector positions are the same to avoid division by zero
        if sector_dist == 0:
            sector_dist = 1e-8

        # Calculate the unit vector pointing from last sector to next sector
        sector_vec = (nsp_2d - lsp_2d) / sector_dist

        # Calculate the vector from last sector position to current ball position
        ball_vec = pos_2d - lsp_2d

        # Calculate how much progress the ball has made along the sector direction
        # Project the ball vector onto the sector vector to get progress
        ball_progress = np.dot(ball_vec, sector_vec)

        # Normalize the progress to be in range [0, 1] based on the sector length
        # This represents the percentage of the sector completed
        if sector_dist > 0:
            normalized_progress = np.clip(ball_progress / sector_dist, 0.0, 1.0)
        else:
            normalized_progress = 0.0

        # Calculate movement since last position check
        pos_delta_vec = pos_2d - lpos_2d
        # Calculate progress made in the current step along the sector direction
        delta_progress = np.dot(pos_delta_vec, sector_vec)

        # Update last position every 10 steps to track movement
        if self._step % 10 == 0:
            self._last_position = self._position.copy()

        # Base reward based on progress made in the level
        progress_reward = normalized_progress * 1.0

        # Reward for moving forward in the sector
        movement_reward = np.clip(delta_progress * 10.0, -100.0, 10.0)  # Reduced weight and limits

        # Small reward for staying in the game
        # Reduced time penalty to encourage longer episodes
        time_reward = -1. * math.log2(self._step + 1)

        dist_to_edge_prod = np.float_power(abs(np.prod(self._dist_to_edge, where=self._dist_to_edge > 0)), 1./8.)
        is_on_track = np.count_nonzero(self._dist_to_edge) >= 5
        on_track_reward = (2.
            * dist_to_edge_prod
            * (10. if is_on_track else -250.)
        )

        # Large penalty for falling off or truncation
        failure_penalty = -500.0 if self._should_truncate else 0.0

        # Combine rewards
        reward = (
            # progress_reward      # Reward for how far along the sector we are, [0, 1]
            + movement_reward    # Reward for positive movement in the right direction, [-100, 10]
            + time_reward        # Small penalty for each step to encourage efficiency, -log
            + on_track_reward    # Reward for being on the sector path
            + failure_penalty    # Penalty for falling off
        )

        if self._step % 200 == 0:
            np.set_printoptions(precision=3)
            print("Normalized progress: {:.2f}%, progress reward: {:.2f}".format(normalized_progress * 100, progress_reward))
            print("Delta progress: {:.2f}, movement reward: {:.2f}".format(delta_progress, movement_reward))
            print("dist_to_edge_prod: {:.2f}".format(dist_to_edge_prod))
            print("On-track reward: {:.2f}".format(on_track_reward))
            print("Reward: {:.2f}".format(reward))
            print(f"Dist to edge: {self._dist_to_edge}")

        return reward

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
        self._naction = np.array(self._action_lut[action], dtype=np.uint8)
        # print(f"Action: {action} {self._naction}", end=' ')
        self.server.send_msg(MsgType.KbdInput, self._naction.tobytes())

        # TODO: Update observable state from game, check if still in valid shape
        self.fetch_tick()
        # print(f"position: {self._position}")
        # plt.imshow(self._depth_image.reshape(240, 320), cmap='gray', vmin=200, vmax=250)
        # plt.title("Depth Buffer")
        # plt.colorbar()
        # plt.show()

        img_shape = self._depth_image.shape[1:]
        diag_len = float(np.linalg.norm(img_shape))
        sample_pt = np.array([img_shape[1] / 2, img_shape[0] / 2])
        ignore_dist = 15 # ball diameter, in px
        self._dist_to_edge = np.array([
            get_dist_to_edge(self._depth_image, sample_pt, np.array([-1, 0]), ignore_dist, 254) / img_shape[0], # Left
            get_dist_to_edge(self._depth_image, sample_pt, np.array([ 1, 0]), ignore_dist, 254) / img_shape[0], # Right
            get_dist_to_edge(self._depth_image, sample_pt, np.array([ 0,-1]), ignore_dist, 254) / img_shape[1], # Up
            get_dist_to_edge(self._depth_image, sample_pt, np.array([ 0, 1]), ignore_dist, 254) / img_shape[1], # Down

            get_dist_to_edge(self._depth_image, sample_pt, np.array([-1,-1]), ignore_dist, 254) / diag_len,     # Up-left
            get_dist_to_edge(self._depth_image, sample_pt, np.array([-1, 1]), ignore_dist, 254) / diag_len,     # Up-right
            get_dist_to_edge(self._depth_image, sample_pt, np.array([ 1,-1]), ignore_dist, 254) / diag_len,     # Down-left
            get_dist_to_edge(self._depth_image, sample_pt, np.array([ 1, 1]), ignore_dist, 254) / diag_len,     # Down-right
        ], dtype=np.float32)

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
