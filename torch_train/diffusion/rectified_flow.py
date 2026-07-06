from __future__ import annotations

import dataclasses
from typing import Optional

import torch


@dataclasses.dataclass
class RectifiedFlowConfig:
    prediction: str = "velocity"
    use_lognorm: bool = True
    lognorm_mu: float = 0.0
    lognorm_sigma: float = 1.0
    train_timestep_shift: float = 0.0
    cfg_interval_start: float = 0.0
    inference_timestep_shift: float = 0.3
    sampling_method: str = "euler"

    @classmethod
    def from_config(cls, config) -> "RectifiedFlowConfig":
        cfg = dict(config)
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


def _broadcast_t(t: torch.Tensor, ndim: int) -> torch.Tensor:
    return t.reshape((t.shape[0],) + (1,) * (ndim - 1))


def sample_times(
    batch_size: int,
    cfg: RectifiedFlowConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if cfg.use_lognorm:
        normal_samples = cfg.lognorm_mu + cfg.lognorm_sigma * torch.randn(
            batch_size, device=device, dtype=dtype, generator=generator
        )
        t = torch.sigmoid(normal_samples)
    else:
        t = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)

    shift = cfg.train_timestep_shift or 0.0
    if shift != 0.0 and shift != 1.0:
        t = (shift * t) / (1.0 + (shift - 1.0) * t)
    return t


def prepare_rectified_flow_inputs(
    latents: torch.Tensor,
    cfg: RectifiedFlowConfig,
    noise_generator: Optional[torch.Generator] = None,
    time_generator: Optional[torch.Generator] = None,
    noise: Optional[torch.Tensor] = None,
    t: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x1 = latents
    x0 = noise if noise is not None else torch.randn(
        latents.shape, device=latents.device, dtype=latents.dtype, generator=noise_generator
    )
    if t is None:
        t = sample_times(latents.shape[0], cfg, latents.device, latents.dtype, time_generator)
    t_expanded = _broadcast_t(t, latents.ndim)
    xt = (1.0 - t_expanded) * x0 + t_expanded * x1
    ut = x1 - x0
    return xt, ut, t.to(latents.dtype)
