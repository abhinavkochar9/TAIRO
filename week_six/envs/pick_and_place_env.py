import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make('FetchPickAndPlaceDense-v4', max_episode_steps=100, render_mode="human")

max_ep = 100
total_reward = 0
observation, info = env.reset()
for ep in range(max_ep):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
env.close()