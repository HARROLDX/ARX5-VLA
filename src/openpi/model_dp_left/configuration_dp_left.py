import dataclasses
from collections.abc import Sequence

import openpi.model_dp.configuration_dp as dp_config


@dataclasses.dataclass(frozen=True)
class DPLeftConfig(dp_config.DPConfig):
    """Single-left-arm DP config.

    This keeps the same LeRobot-like diffusion implementation as model_dp while
    changing only the observation/action surface to 7-D left arm and two cameras.
    """

    action_dim: int = 7
    state_dim: int = 7
    image_keys: Sequence[str] = (
        "base_0_rgb",
        "left_wrist_0_rgb",
    )

    def create_pytorch(self) -> "DiffusionPolicyLeftModel":
        from openpi.model_dp_left.diffusion_policy_left import DiffusionPolicyLeftModel

        return DiffusionPolicyLeftModel(self)
