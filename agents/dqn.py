from torchrl.modules import DuelingCnnDQNet, EGreedyModule, QValueActor
from tensordict.nn import TensorDictSequential
from torch import nn


def make_dqn_model(cfg, dummy_env, device):
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
    net = DuelingCnnDQNet(
        dummy_env.action_spec.shape[-1], 1, cnn_kwargs, mlp_kwargs
    ).to(device)
    net.value[-1].bias.data.fill_(cfg.agent.init_bias)

    actor = QValueActor(net, in_keys=["pixels"], spec=dummy_env.action_spec).to(device)
    # init actor: because the model is composed of lazy conv/linear layers,
    # we must pass a fake batch of data through it to instantiate them.
    tensordict = dummy_env.fake_tensordict()
    actor(tensordict)

    # we join our actor with an EGreedyModule for data collection
    exploration_module = EGreedyModule(
        spec=dummy_env.action_spec,
        annealing_num_steps=cfg.collector.total_frames,
        eps_init=cfg.env.eps_greedy_val,
        eps_end=cfg.env.eps_greedy_val_env,
    )
    actor_explore = TensorDictSequential(actor, exploration_module)

    return actor, actor_explore
