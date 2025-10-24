import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
import BallanceEnv

seq = 3

env = gym.make("ballance_env/Ballance-v0", max_episode_steps=-1)
obs, info = env.reset()
print(f'reset obs: {obs}')

# check_env(env)
# exit(0)

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

model = None
try:
    model = DQN.load('ballance_dqn_model' + str(seq), env=env)
    print('model loaded')
except FileNotFoundError:
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=10000,
        verbose=1)
    print('new model created')

# Train the model
print("Starting training...")
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ballance_dqn_model" + str(seq+1))
print(f"Model saved as 'ballance_model{seq+1}'")

# for i in range(1000000):
#     action, _ = model.predict(obs)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if i % 5000 == 0:
#         print(obs)
