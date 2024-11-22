import gymnasium as gym
import torch
from envs.gridworld import make_gridworld_env
from gridworld.envs import Go2TargetEnv
from agents import QNetwork

device = "cuda:0"
model_path = "runs/dqn-gridworld-seed_1-bs_20000/dqn-gridworld.cleanrl_model"

dummyenv = gym.vector.SyncVectorEnv(
    [
        make_gridworld_env(
            env_id="gridworld",
            seed=0,
            idx=1,
            capture_video=False,
            run_name="gridworld",
        )
        for i in range(1)
    ]
)

# Initialise the environment
# env = gym.make("CarRacing-v2", render_mode="human", continuous=False)
env = Go2TargetEnv(render_mode="human", size=8)

model = QNetwork(dummyenv)
del dummyenv

model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

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
