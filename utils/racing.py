# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
import torch.nn
import torch.optim
from tensordict.nn import TensorDictSequential
from torchrl.envs import RewardSum, StepCounter, RewardScaling
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import DuelingCnnDQNet, EGreedyModule, QValueActor
from torchrl.record import VideoRecorder

from torchrl.envs.transforms import (
    CatFrames,
    Compose,
    GrayScale,
    ObservationNorm,
    Resize,
    ToTensorImage,
    TransformedEnv,
)


# ====================================================================
# Environment utils
# --------------------------------------------------------------------
def get_norm_stats():
    test_env = make_env()
    test_env.transform[-1].init_stats(
        num_iter=1000, cat_dim=0, reduce_dim=[-1, -2, -4], keep_dims=(-1, -2)
    )
    obs_norm_sd = test_env.transform[-1].state_dict()
    # let's check that normalizing constants have a size of ``[C, 1, 1]`` where
    # ``C=4`` (because of :class:`~torchrl.envs.CatFrames`).
    test_env.close()
    del test_env
    return obs_norm_sd


def make_env(obs_norm_sd=None, env_name="CarRacing-v2", device="cpu", from_pixels=True):
    if obs_norm_sd is None:
        obs_norm_sd = {"standard_normal": True}

    base_env = GymEnv(
        env_name,
        device=device,
        from_pixels=from_pixels,
        pixels_only=True,
        continuous=False,
    )
    env = TransformedEnv(
        base_env,
        Compose(
            StepCounter(),  # to count the steps of each trajectory
            ToTensorImage(),
            RewardSum(),
            RewardScaling(loc=0.0, scale=0.1),
            GrayScale(),
            Resize(64, 64),
            CatFrames(4, in_keys=["pixels"], dim=-3),
            ObservationNorm(in_keys=["pixels"], **obs_norm_sd),
        ),
    )
    env.transform[7].init_stats(3)
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_dqn_modules(dummyenv, cfg, device="cpu", init_bias=2.0):
    cnn_kwargs = {
        "num_cells": [32, 64, 64],
        "kernel_sizes": [6, 4, 3],
        "strides": [2, 2, 1],
        "activation_class": nn.ELU,
        # This can be used to reduce the size of the last layer of the CNN
        # "squeeze_output": True,
        # "aggregator_class": nn.AdaptiveAvgPool2d,
        # "aggregator_kwargs": {"output_size": (1, 1)},
    }
    mlp_kwargs = {
        "depth": 2,
        "num_cells": [
            64,
            64,
        ],
        "activation_class": nn.ELU,
    }
    net = DuelingCnnDQNet(dummyenv.action_spec.shape[-1], 1, cnn_kwargs, mlp_kwargs).to(
        device
    )
    net.value[-1].bias.data.fill_(init_bias)

    actor = QValueActor(net, in_keys=["pixels"], spec=dummyenv.action_spec).to(device)
    # Initialize
    tensordict = dummyenv.fake_tensordict()
    actor(tensordict)

    greedy_module = EGreedyModule(
        annealing_num_steps=cfg.collector.annealing_frames,
        eps_init=cfg.collector.eps_start,
        eps_end=cfg.collector.eps_end,
        spec=actor.spec,
    )
    actor_explore = TensorDictSequential(
        actor,
        greedy_module,
    ).to(device)

    return actor, actor_explore, greedy_module


def make_dqn_model(cfg):
    dummyenv = make_env(env_name=cfg.env.env_name, device="cpu")
    actor, actor_explore, greedy_module = make_dqn_modules(dummyenv, cfg)
    del dummyenv
    return actor, actor_explore, greedy_module


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def eval_model(actor, test_env, num_episodes=3):
    test_rewards = torch.zeros(num_episodes, dtype=torch.float32)
    for i in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        test_env.apply(dump_video)
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards[i] = reward.sum()
    del td_test
    return test_rewards.mean()


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()
