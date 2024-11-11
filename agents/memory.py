from torchrl.data import LazyMemmapStorage, MultiStep, TensorDictReplayBuffer
from torchrl.collectors import MultiaSyncDataCollector, SyncDataCollector
from torchrl.envs import ExplorationType

from utils import is_fork, DEVICE
from envs.cartpole import make_env


def get_replay_buffer(cfg):
    replay_buffer = TensorDictReplayBuffer(
        batch_size=cfg.buffer.batch_size,
        storage=LazyMemmapStorage(cfg.buffer.buffer_size),
        prefetch=cfg.optim.n_optim,
    )
    return replay_buffer


def get_collector(
    cfg,
    stats,
    actor_explore,
):
    # We can't use nested child processes with mp_start_method="fork"
    if is_fork:
        cls = SyncDataCollector
        env_arg = make_env(
            parallel=True, obs_norm_sd=stats, num_workers=cfg.collector.num_workers
        )
    else:
        cls = MultiaSyncDataCollector
        env_arg = [
            make_env(
                parallel=True, obs_norm_sd=stats, num_workers=cfg.collector.num_workers
            )
        ] * cfg.collector.num_collectors

    data_collector = cls(
        env_arg,
        policy=actor_explore,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        # this is the default behavior: the collector runs in ``"random"`` (or explorative) mode
        exploration_type=ExplorationType.RANDOM,
        # We set the all the devices to be identical. Below is an example of
        # heterogeneous devices
        device=DEVICE,
        storing_device=DEVICE,
        split_trajs=False,
        postproc=MultiStep(gamma=cfg.loss.gamma, n_steps=5),
    )
    return data_collector
