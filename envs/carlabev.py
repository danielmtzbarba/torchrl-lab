import gymnasium as gym
from gymnasium.wrappers import (
    GrayScaleObservation,
    ResizeObservation,
    FrameStack,
)
from CarlaBEV.envs import CarlaBEV

"""
HOME
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation
)
"""


def make_carlabev_env(env_id, seed, idx, capture_video, run_name, size: int = 64):
    def thunk():
        if capture_video and idx == 0:
            env = CarlaBEV(render_mode="rgb_array", size=size)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = CarlaBEV(render_mode="rgb_array", size=size)

        env = GrayScaleObservation(env)
        env = ResizeObservation(env, (96, 96))
        # env = FrameStackObservation(env, stack_size=4)
        env = FrameStack(env, num_stack=4)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk
