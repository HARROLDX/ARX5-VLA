import torch
import torch.nn.functional as F
from torch import Tensor, nn

import openpi.models.model as _model
from openpi.model_dp.modules import ConditionalUnet1d, RgbEncoder

try:
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
except Exception as exc:  # pragma: no cover - validated at runtime in offline envs
    DDIMScheduler = None
    DDPMScheduler = None
    _DIFFUSERS_IMPORT_ERROR = exc
else:
    _DIFFUSERS_IMPORT_ERROR = None


def _make_noise_scheduler(config):
    if DDPMScheduler is None or DDIMScheduler is None:
        raise ImportError("diffusers is required for LeRobot-like DP scheduler") from _DIFFUSERS_IMPORT_ERROR
    kwargs = dict(
        num_train_timesteps=config.num_train_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        clip_sample=config.clip_sample,
        clip_sample_range=config.clip_sample_range,
        prediction_type=config.prediction_type,
    )
    if config.noise_scheduler_type == "DDPM":
        return DDPMScheduler(**kwargs)
    if config.noise_scheduler_type == "DDIM":
        return DDIMScheduler(**kwargs)
    raise ValueError(f"Unsupported noise_scheduler_type={config.noise_scheduler_type!r}")


class DiffusionPolicyModel(nn.Module):
    """OpenPi-compatible adapter of LeRobot-style Diffusion Policy.

    It keeps the OpenPi policy boundary intact: inputs are
    ``openpi.models.model.Observation`` and outputs are normalized action chunks.
    Internally it follows LeRobot's DP structure more closely: diffusers scheduler,
    n_obs_steps/n_action_steps semantics, and action slicing after generation.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.n_obs_steps != 1:
            raise ValueError("OpenPi ARX DP adapter currently supports n_obs_steps=1. Add history stacking before >1.")
        if config.n_action_steps > config.action_horizon - config.n_obs_steps + 1:
            raise ValueError("n_action_steps must be <= action_horizon - n_obs_steps + 1")
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.max_token_len = config.max_token_len
        self.image_keys = tuple(config.image_keys)
        if config.use_separate_rgb_encoder_per_camera:
            self.rgb_encoder = nn.ModuleList([RgbEncoder(config) for _ in self.image_keys])
            image_feature_dim = self.rgb_encoder[0].feature_dim * len(self.image_keys)
        else:
            self.rgb_encoder = RgbEncoder(config)
            image_feature_dim = self.rgb_encoder.feature_dim * len(self.image_keys)
        global_cond_dim = (config.state_dim + image_feature_dim) * config.n_obs_steps
        self.unet = ConditionalUnet1d(config, global_cond_dim=global_cond_dim)
        self.noise_scheduler = _make_noise_scheduler(config)
        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = int(config.num_inference_steps)

    def _image_to_nchw(self, image: Tensor) -> Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor [B,C,H,W] or [B,H,W,C], got {tuple(image.shape)}")
        if image.shape[-1] == 3 and image.shape[1] != 3:
            image = image.permute(0, 3, 1, 2)
        return image

    def _prepare_global_conditioning(self, observation: _model.Observation) -> Tensor:
        state = observation.state.to(dtype=torch.float32)
        image_features = []
        for idx, key in enumerate(self.image_keys):
            if key not in observation.images:
                raise KeyError(f"Missing image key {key}; got {tuple(observation.images)}")
            image = self._image_to_nchw(observation.images[key])
            if isinstance(self.rgb_encoder, nn.ModuleList):
                image_features.append(self.rgb_encoder[idx](image))
            else:
                image_features.append(self.rgb_encoder(image))
        # LeRobot flattens n_obs_steps into global conditioning. OpenPi provides one
        # current observation here, so this is the n_obs_steps=1 equivalent.
        return torch.cat([state, *image_features], dim=-1).flatten(start_dim=1)

    def forward(self, observation: _model.Observation, actions: Tensor) -> Tensor:
        return self.compute_loss(None, observation, actions, train=True)

    def compute_loss(self, rng, observation: _model.Observation, actions: Tensor, *, train: bool = False) -> Tensor:
        del rng, train
        trajectory = actions.to(dtype=torch.float32)
        if trajectory.shape[-2:] != (self.config.action_horizon, self.config.action_dim):
            raise ValueError(
                f"Expected actions [B,{self.config.action_horizon},{self.config.action_dim}], got {tuple(trajectory.shape)}"
            )
        global_cond = self._prepare_global_conditioning(observation)
        eps = torch.randn_like(trajectory)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
            dtype=torch.long,
        )
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
        pred = self.unet(noisy_trajectory, timesteps, global_cond)
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction_type={self.config.prediction_type!r}")
        loss = F.mse_loss(pred, target, reduction="none")
        return loss.mean(dim=(1, 2))

    @torch.no_grad()
    def conditional_sample(self, batch_size: int, global_cond: Tensor, noise: Tensor | None = None) -> Tensor:
        device = global_cond.device
        dtype = global_cond.dtype
        if noise is None:
            sample = torch.randn(
                batch_size,
                self.config.action_horizon,
                self.config.action_dim,
                device=device,
                dtype=dtype,
            )
        else:
            sample = noise.to(device=device, dtype=dtype)
            if sample.ndim == 2:
                sample = sample.unsqueeze(0)
        self.noise_scheduler.set_timesteps(self.num_inference_steps, device=device)
        for timestep in self.noise_scheduler.timesteps:
            model_output = self.unet(sample, timestep.expand(batch_size), global_cond)
            sample = self.noise_scheduler.step(model_output, timestep, sample).prev_sample
        return sample

    @torch.no_grad()
    def sample_actions(self, rng_or_device, observation: _model.Observation, **kwargs) -> Tensor:
        del rng_or_device
        global_cond = self._prepare_global_conditioning(observation)
        actions = self.conditional_sample(global_cond.shape[0], global_cond, noise=kwargs.get("noise"))
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]
