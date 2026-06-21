import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn


def _valid_num_groups(channels: int, requested: int) -> int:
    groups = min(int(requested), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


def replace_submodules(root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    matches = [name.split(".") for name, module in root_module.named_modules(remove_duplicate=True) if predicate(module)]
    for *parents, key in matches:
        parent = root_module.get_submodule(".".join(parents)) if parents else root_module
        src = parent[int(key)] if isinstance(parent, nn.Sequential) else getattr(parent, key)
        dst = func(src)
        if isinstance(parent, nn.Sequential):
            parent[int(key)] = dst
        else:
            setattr(parent, key, dst)
    return root_module


class SpatialSoftmax(nn.Module):
    def __init__(self, input_shape: Sequence[int], num_keypoints: int):
        super().__init__()
        channels, height, width = [int(x) for x in input_shape]
        self.height = height
        self.width = width
        self.num_keypoints = int(num_keypoints)
        self.proj = nn.Conv2d(channels, self.num_keypoints, kernel_size=1)
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        self.register_buffer("pos_grid", torch.stack((pos_x, pos_y), dim=-1).reshape(height * width, 2))

    def forward(self, features: Tensor) -> Tensor:
        features = self.proj(features)
        attention = F.softmax(features.reshape(features.shape[0] * self.num_keypoints, -1), dim=-1)
        expected_xy = attention @ self.pos_grid
        return expected_xy.reshape(features.shape[0], self.num_keypoints * 2)


class RgbEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        backbone_model.fc = nn.Identity()
        self.backbone = backbone_model
        if config.use_group_norm:
            if config.pretrained_backbone_weights is not None:
                raise ValueError("BatchNorm cannot be replaced safely when pretrained weights are loaded.")
            self.backbone = replace_submodules(
                self.backbone,
                predicate=lambda module: isinstance(module, nn.BatchNorm2d),
                func=lambda module: nn.GroupNorm(
                    num_groups=_valid_num_groups(module.num_features, module.num_features // 16),
                    num_channels=module.num_features,
                ),
            )
        self.crop_shape = tuple(config.crop_shape) if config.crop_shape is not None else None
        if self.crop_shape is None:
            self.random_crop = nn.Identity()
            self.center_crop = nn.Identity()
        else:
            self.random_crop = torchvision.transforms.RandomCrop(self.crop_shape)
            self.center_crop = torchvision.transforms.CenterCrop(self.crop_shape)
        self.crop_is_random = bool(config.crop_is_random)
        if config.imagenet_norm:
            self.image_norm = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        else:
            self.image_norm = nn.Identity()
        h, w = config.image_resolution
        if self.crop_shape is not None:
            h, w = self.crop_shape
        with torch.no_grad():
            dummy = torch.zeros(1, 3, h, w)
            feature_shape = self.backbone(dummy).shape[1:]
        self.feature_dim = math.prod(feature_shape)

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor [B,C,H,W] or [B,H,W,C], got {tuple(image.shape)}")
        if image.shape[-1] == 3 and image.shape[1] != 3:
            image = image.permute(0, 3, 1, 2)
        image = image.to(dtype=torch.float32)
        if image.numel() > 0 and image.detach().amax() > 2.5:
            image = image / 255.0
        elif image.numel() > 0 and image.detach().amin() < 0.0:
            image = (image + 1.0) / 2.0
        image = image.clamp(0.0, 1.0)
        if self.training and self.crop_is_random:
            image = self.random_crop(image)
        else:
            image = self.center_crop(image)
        image = self.image_norm(image)
        return torch.flatten(self.backbone(image), start_dim=1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float32) * -scale)
        emb = x.to(torch.float32).unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(_valid_num_groups(out_channels, n_groups), out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConditionalResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int,
        n_groups: int,
        use_film_scale_modulation: bool,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.use_film_scale_modulation = bool(use_film_scale_modulation)
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups)
        cond_channels = out_channels * 2 if self.use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale, bias = cond_embed[:, : self.out_channels], cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            out = out + cond_embed
        out = self.conv2(out)
        return out + self.residual_conv(x)


def _match_length(x: Tensor, target_length: int) -> Tensor:
    diff = target_length - x.shape[-1]
    if diff == 0:
        return x
    if diff > 0:
        return F.pad(x, (0, diff))
    return x[..., :target_length]


class ConditionalUnet1d(nn.Module):
    def __init__(self, config, global_cond_dim: int):
        super().__init__()
        self.config = config
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim
        dims = [config.action_dim, *config.down_dims]
        block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList()
        for idx, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            is_last = idx == len(dims) - 2
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(dim_in, dim_out, **block_kwargs),
                        ConditionalResidualBlock1d(dim_out, dim_out, **block_kwargs),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1d(dims[-1], dims[-1], **block_kwargs),
                ConditionalResidualBlock1d(dims[-1], dims[-1], **block_kwargs),
            ]
        )
        self.up_modules = nn.ModuleList()
        reversed_pairs = list(zip(reversed(dims[2:]), reversed(dims[1:-1]), strict=True))
        for dim_in, dim_out in reversed_pairs:
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(dim_in * 2, dim_out, **block_kwargs),
                        ConditionalResidualBlock1d(dim_out, dim_out, **block_kwargs),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1),
                    ]
                )
            )
        self.final_conv = nn.Sequential(
            Conv1dBlock(config.down_dims[0], config.down_dims[0], config.kernel_size, config.n_groups),
            nn.Conv1d(config.down_dims[0], config.action_dim, 1),
        )

    def forward(self, sample: Tensor, timestep: Tensor | int, global_cond: Tensor) -> Tensor:
        input_length = sample.shape[1]
        x = sample.transpose(1, 2)
        if not torch.is_tensor(timestep):
            timestep = torch.full((sample.shape[0],), int(timestep), device=sample.device, dtype=torch.long)
        elif timestep.ndim == 0:
            timestep = timestep.expand(sample.shape[0]).to(device=sample.device, dtype=torch.long)
        step_embed = self.diffusion_step_encoder(timestep)
        cond = torch.cat((step_embed, global_cond), dim=-1)

        skips = []
        for res1, res2, downsample in self.down_modules:
            x = res1(x, cond)
            x = res2(x, cond)
            skips.append(x)
            x = downsample(x)
        for mid in self.mid_modules:
            x = mid(x, cond)
        for res1, res2, upsample in self.up_modules:
            skip = skips.pop()
            x = _match_length(x, skip.shape[-1])
            x = torch.cat((x, skip), dim=1)
            x = res1(x, cond)
            x = res2(x, cond)
            x = upsample(x)
        if x.shape[-1] != input_length:
            raise RuntimeError(
                "ConditionalUnet1d decoder did not recover the input horizon. "
                f"Got {x.shape[-1]} steps for input length {input_length}."
            )
        return self.final_conv(x).transpose(1, 2)


def _betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta: float = 0.999) -> Tensor:
    """Cosine beta schedule used by diffusers' squaredcos_cap_v2."""

    def alpha_bar(time_step: float) -> float:
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class DDPMScheduler(nn.Module):
    def __init__(self, config):
        super().__init__()
        beta_schedule = getattr(config, "beta_schedule", "linear")
        if beta_schedule == "linear":
            betas = torch.linspace(config.beta_start, config.beta_end, config.num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "squaredcos_cap_v2":
            betas = _betas_for_alpha_bar(config.num_train_timesteps)
        else:
            raise ValueError(f"Unsupported beta_schedule={beta_schedule!r}")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat((torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]))
        self.num_train_timesteps = int(config.num_train_timesteps)
        self.clip_sample_range = float(getattr(config, "clip_sample_range", 1.0))
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))

    @staticmethod
    def _extract(values: Tensor, timesteps: Tensor, target: Tensor) -> Tensor:
        out = values.gather(0, timesteps.to(values.device))
        return out.reshape(timesteps.shape[0], *((1,) * (target.ndim - 1))).to(target.device)

    def add_noise(self, sample: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        return (
            self._extract(self.sqrt_alphas_cumprod, timesteps, sample) * sample
            + self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, sample) * noise
        )

    def predict_x0(self, sample: Tensor, model_output: Tensor, timesteps: Tensor) -> Tensor:
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, timesteps, sample) * sample
            - self._extract(self.sqrt_recipm1_alphas_cumprod, timesteps, sample) * model_output
        )

    def step(self, model_output: Tensor, timesteps: Tensor, sample: Tensor, *, clip_sample: bool = True) -> Tensor:
        pred_x0 = self.predict_x0(sample, model_output, timesteps)
        if clip_sample:
            sample_range = float(getattr(self, "clip_sample_range", 1.0))
            pred_x0 = pred_x0.clamp(-sample_range, sample_range)
        alpha_t = self._extract(self.alphas, timesteps, sample)
        alpha_cum_t = self._extract(self.alphas_cumprod, timesteps, sample)
        alpha_cum_prev = self._extract(self.alphas_cumprod_prev, timesteps, sample)
        beta_t = self._extract(self.betas, timesteps, sample)
        coef_x0 = beta_t * torch.sqrt(alpha_cum_prev) / (1.0 - alpha_cum_t)
        coef_xt = (1.0 - alpha_cum_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_cum_t)
        mean = coef_x0 * pred_x0 + coef_xt * sample
        variance = self._extract(self.posterior_variance, timesteps, sample)
        noise = torch.randn_like(sample)
        nonzero_mask = (timesteps != 0).float().reshape(timesteps.shape[0], *((1,) * (sample.ndim - 1)))
        return mean + nonzero_mask * torch.sqrt(variance) * noise
