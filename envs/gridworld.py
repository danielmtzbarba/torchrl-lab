import gymnasium as gym
from gymnasium.wrappers import (
    GrayScaleObservation,
    FrameStack,
    ResizeObservation,
)

from gridworld.envs import GridWorldEnv


def make_gridworld_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = GridWorldEnv(render_mode="rgb_array", size=5)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = GridWorldEnv(render_mode="rgb_array", size=5)

        env = GrayScaleObservation(env)
        env = ResizeObservation(env, (96, 96))
        env = FrameStack(env, num_stack=4)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk
