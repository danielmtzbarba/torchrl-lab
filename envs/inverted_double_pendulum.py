from torchrl.envs.utils import check_env_specs
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)

from utils import device


def build_env():
    base_env = GymEnv("InvertedDoublePendulum-v4", device=device)

    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )

    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    #    print("normalization constant shape:", env.transform[0].loc.shape)
    #    print("observation_spec:", env.observation_spec)
    #    print("reward_spec:", env.reward_spec)
    #    print("input_spec:", env.input_spec)
    #    print("action_spec (as defined by input_spec):", env.action_spec)

    check_env_specs(env)

    return env
