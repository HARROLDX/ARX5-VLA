import dataclasses
import logging
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import safetensors.torch
import torch

import openpi.models.model as _model
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class DPConfig(_model.BaseModelConfig):
    """Configuration for a conventional ResNet + conditional 1D U-Net diffusion policy.

    The model is intentionally shaped to reuse OpenPi's existing LeRobot ARX data transforms,
    normalization assets, websocket policy server, and ARX deployment client. Actions are
    absolute 14-D joint targets by convention when used with LerobotARXDataConfig.
    """

    action_dim: int = 14
    action_horizon: int = 32
    max_token_len: int = 1
    state_dim: int = 14
    image_keys: Sequence[str] = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    image_resolution: tuple[int, int] = (224, 224)
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: Any | None = None
    use_group_norm: bool = True
    use_separate_rgb_encoder_per_camera: bool = True
    crop_shape: tuple[int, int] | None = (224, 224)
    crop_is_random: bool = True
    imagenet_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    diffusion_step_embed_dim: int = 128
    down_dims: Sequence[int] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    use_film_scale_modulation: bool = True
    n_obs_steps: int = 1
    n_action_steps: int = 16
    num_train_timesteps: int = 100
    num_inference_steps: int | None = 20
    # 推理去噪步数
    noise_scheduler_type: str = "DDPM"
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    clip_sample_range: float = 1.0
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    # 之前在此处出错，训练端和部署端超参未同步
    do_mask_loss_for_padding: bool = False

    def __post_init__(self):
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"`vision_backbone` must be a torchvision ResNet, got {self.vision_backbone!r}.")
        if self.prediction_type not in ("epsilon", "sample"):
            raise ValueError("`prediction_type` must be either 'epsilon' or 'sample'.")
        if self.noise_scheduler_type not in ("DDPM", "DDIM"):
            raise ValueError("`noise_scheduler_type` must be either 'DDPM' or 'DDIM'.")
        if self.n_action_steps > self.action_horizon - self.n_obs_steps + 1:
            raise ValueError("n_action_steps must be <= action_horizon - n_obs_steps + 1")
        if self.n_obs_steps < 1:
            raise ValueError("n_obs_steps must be >= 1")
        if self.crop_shape is not None and (len(self.crop_shape) != 2 or any(dim <= 0 for dim in self.crop_shape)):
            raise ValueError(f"`crop_shape` must be a pair of positive integers. Got {self.crop_shape}.")
        if len(self.down_dims) < 1:
            raise ValueError("down_dims must contain at least one U-Net stage.")
        downsampling_factor = 2 ** len(self.down_dims)
        if self.action_horizon % downsampling_factor != 0:
            raise ValueError(
                "action_horizon must be an integer multiple of the U-Net downsampling factor "
                f"2 ** len(down_dims). Got action_horizon={self.action_horizon}, down_dims={tuple(self.down_dims)}."
            )

    @property
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.DP

    @property
    def horizon(self) -> int:
        return self.action_horizon

    def create(self, rng: at.KeyArrayLike) -> "DiffusionPolicyModel":
        del rng
        return self.create_pytorch()

    def create_pytorch(self) -> "DiffusionPolicyModel":
        from openpi.model_dp.diffusion_policy import DiffusionPolicyModel

        return DiffusionPolicyModel(self)

    def load_pytorch(self, train_config, weight_path: str):
        del train_config
        model = self.create_pytorch()
        safetensors.torch.load_model(model, weight_path)
        return model

    def load(self, params: at.Params, *, remove_extra_params: bool = True):
        raise NotImplementedError("DPConfig is a PyTorch-only model. Use load_pytorch().")

    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        h, w = self.image_resolution
        images = {
            key: jax.ShapeDtypeStruct((batch_size, h, w, 3), jnp.float32)
            for key in self.image_keys
        }
        image_masks = {
            key: jax.ShapeDtypeStruct((batch_size,), jnp.bool_)
            for key in self.image_keys
        }
        observation = _model.Observation(
            images=images,
            image_masks=image_masks,
            state=jax.ShapeDtypeStruct((batch_size, self.state_dim), jnp.float32),
        )
        actions = jax.ShapeDtypeStruct((batch_size, self.action_horizon, self.action_dim), jnp.float32)
        return observation, actions
