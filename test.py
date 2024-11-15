import gymnasium as gym

# Initialise the environment
env = gym.make("CarRacing-v2", render_mode="human", continuous=False)

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
total_reward = 0
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()
        total_reward = 0

env.close()
