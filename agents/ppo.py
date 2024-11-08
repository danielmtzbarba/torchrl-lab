import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from hyperparameters.ppo import num_cells, frames_per_batch, total_frames
from hyperparameters.ppo import lr, lmbda, clip_epsilon, gamma, entropy_eps

from utils import device


class PPO_Actor(nn.Module):
    def __init__(self, env) -> None:
        super().__init__()
        self._net = nn.Sequential(
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
            NormalParamExtractor(),
        )

    def forward(self, x):
        return self._net(x)


class PPO_Value(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._net = nn.Sequential(
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(1, device=device),
        )

    def forward(self, x):
        return self._net(x)


def build_actor_net(env):
    actor_net = PPO_Actor(env)
    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )
    policy_module(env.reset())
    return policy_module


def build_value_net(env):
    value_net = PPO_Value()
    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )
    value_module(env.reset())
    return value_module


def build_data_collector(env, policy_module):
    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
    )
    return collector


def build_ppo_elements(env):
    policy_module = build_actor_net(env)
    value_module = build_value_net(env)

    collector = build_data_collector(env, policy_module)

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )
    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        # these keys match by default but we set this for completeness
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )
    return (
        policy_module,
        value_module,
        collector,
        replay_buffer,
        advantage_module,
        loss_module,
        optim,
        scheduler,
    )
