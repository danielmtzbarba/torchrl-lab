from torchrl.envs.utils import check_env_specs
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    CatFrames,
    Compose,
    GrayScale,
    ObservationNorm,
    Resize,
    ToTensorImage,
    TransformedEnv,
)

from torchrl.envs import (
    EnvCreator,
    ExplorationType,
    ParallelEnv,
    RewardScaling,
    StepCounter,
)


def make_env(
    parallel=False, obs_norm_sd=None, num_workers=1, mp_context=None, device=None
):
    if obs_norm_sd is None:
        obs_norm_sd = {"standard_normal": True}
    if parallel:

        def maker():
            return GymEnv(
                "CartPole-v1",
                from_pixels=True,
                pixels_only=True,
                device=device,
            )

        base_env = ParallelEnv(
            num_workers,
            EnvCreator(maker),
            # Don't create a sub-process if we have only one worker
            serial_for_single=True,
            mp_start_method=mp_context,
        )
    else:
        base_env = GymEnv(
            "CartPole-v1",
            from_pixels=True,
            pixels_only=True,
            device=device,
        )

    env = TransformedEnv(
        base_env,
        Compose(
            StepCounter(),  # to count the steps of each trajectory
            ToTensorImage(),
            RewardScaling(loc=0.0, scale=0.1),
            GrayScale(),
            Resize(64, 64),
            CatFrames(4, in_keys=["pixels"], dim=-3),
            ObservationNorm(in_keys=["pixels"], **obs_norm_sd),
        ),
    )

    # check_env_specs(env)

    return env


def get_norm_stats():
    dummyenv = make_env()
    dummyenv.transform[-1].init_stats(
        num_iter=1000, cat_dim=0, reduce_dim=[-1, -2, -4], keep_dims=(-1, -2)
    )
    obs_norm_sd = dummyenv.transform[-1].state_dict()
    # let's check that normalizing constants have a size of ``[C, 1, 1]`` where
    # ``C=4`` (because of :class:`~torchrl.envs.CatFrames`).
    print("state dict of the observation norm:", obs_norm_sd)
    dummyenv.close()
    del dummyenv
    return obs_norm_sd
