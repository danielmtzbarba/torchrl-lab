from torchrl.envs.utils import check_env_specs
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import (
    Compose,
    ToTensorImage,
    StepCounter,
    TransformedEnv,
    GrayScale,
    FrameSkipTransform,
    Resize,
    CatFrames,
)

from utils import device


def build_env(frame_skip, stacked_frames):
    base_env = GymEnv("CarRacing-v3", device=device)

    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ToTensorImage(),
            GrayScale(in_keys="pixels"),
            Resize(w=96, h=96),
            FrameSkipTransform(frame_skip=frame_skip),
            StepCounter(),
        ),
    )

    check_env_specs(env)

    return env
