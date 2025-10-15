import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import BallanceEnv

env = gym.make("ballance_env/Ballance-v0")
obs, info = env.reset()
print(f'reset obs: {obs}')

for i in range(1000000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 5000 == 0:
        print(obs)
