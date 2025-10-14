import gymnasium as gym

from stable_baselines3 import PPO

import BallanceEnv

env = gym.make("ballance_env/Ballance-v0")
obs, info = env.reset()

for i in range(10000000):
    # print(obs['location'])
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    # print(obs['location'])

