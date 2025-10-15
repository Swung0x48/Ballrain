import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import BallanceEnv

env = gym.make("ballance_env/Ballance-v0", max_episode_steps=-1)
obs, info = env.reset()
print(f'reset obs: {obs}')

# Define the PPO model with custom parameters
# model = PPO(
#     "MultiInputPolicy",  # Using MultiInputPolicy since observation space is a Dict
#     env,
#     policy_kwargs=dict(
#         net_arch=[dict(pi=[64, 64], vf=[64, 64])]  # Define policy and value networks
#     ),
#     n_steps=204800000,  # Number of steps to run for each environment per update
#     batch_size=64000000,  # Minibatch size
#     n_epochs=10000000,  # Number of epoch when optimizing the surrogate loss
#     gamma=0.99,  # Discount factor
#     gae_lambda=0.95,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
#     clip_range=0.2,  # Clipping parameter
#     verbose=1  # Print out information
# )

model = DQN(
    "MultiInputPolicy",
    env,
    learning_rate=1e-5,
    verbose=1)

# Train the model
print("Starting training...")
model.learn(total_timesteps=50000)

# Save the trained model
model.save("ballance_ppo_model")
print("Model saved as 'ballance_ppo_model'")

for i in range(1000000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 5000 == 0:
        print(obs)
