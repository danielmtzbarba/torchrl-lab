import hydra
import torch
import uuid
import tempfile
import warnings


from torchrl.record.loggers.csv import CSVLogger
from torchrl.envs import ExplorationType

from torchrl.trainers import (
    LogReward,
    Recorder,
    ReplayBufferTrainer,
    Trainer,
    UpdateWeights,
)

from agents.dqn import make_dqn_model
from envs.cartpole import make_env, get_norm_stats
from agents.loss import get_loss_module
from agents.memory import get_replay_buffer, get_collector

from utils import DEVICE

warnings.filterwarnings("ignore")


@hydra.main(
    config_path="configs/", config_name="config_cartpole.yaml", version_base="1.1"
)
def main(cfg: "DictConfig"):
    stats = get_norm_stats()
    test_env = make_env(
        parallel=False,
        obs_norm_sd=stats,
        check_env=True,
    )

    # Get model
    actor, actor_explore = make_dqn_model(cfg, test_env)
    loss_module, target_net_updater = get_loss_module(actor, cfg.loss.gamma)

    collector = get_collector(
        cfg=cfg,
        stats=stats,
        actor_explore=actor_explore,
    )
    for data in collector:
        print(data)
        break

    optimizer = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.wd,
        betas=cfg.optim.betas,
    )
    exp_name = f"dqn_exp_{uuid.uuid1()}"
    tmpdir = tempfile.TemporaryDirectory()
    logger = CSVLogger(exp_name=exp_name, log_dir=tmpdir.name)
    warnings.warn(f"log dir: {logger.experiment.log_dir}")

    trainer = Trainer(
        collector=collector,
        total_frames=cfg.collector.total_frames,
        frame_skip=cfg.env.frame_skip,
        loss_module=loss_module,
        optimizer=optimizer,
        logger=logger,
        optim_steps_per_batch=cfg.optim.n_optim,
        log_interval=cfg.logger.log_interval,
    )

    buffer_hook = ReplayBufferTrainer(
        get_replay_buffer(cfg),
        flatten_tensordicts=True,
    )
    buffer_hook.register(trainer)
    weight_updater = UpdateWeights(collector, update_weights_interval=1)
    weight_updater.register(trainer)
    recorder = Recorder(
        record_interval=100,  # log every 100 optimization steps
        record_frames=1000,  # maximum number of frames in the record
        frame_skip=cfg.env.frame_skip,
        policy_exploration=actor_explore,
        environment=test_env,
        exploration_type=ExplorationType.DETERMINISTIC,
        log_keys=[("next", "reward")],
        out_keys={("next", "reward"): "rewards"},
        log_pbar=True,
    )
    recorder.register(trainer)

    trainer.register_op(
        "post_steps", actor_explore[1].step, frames=cfg.collector.frames_per_batch
    )
    log_reward = LogReward(log_pbar=True)
    log_reward.register(trainer)
    trainer.register_op("post_optim", target_net_updater.step)
    #
    trainer.train()


if __name__ == "__main__":
    main()
