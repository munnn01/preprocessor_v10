"""Frozen differentiable CompressAI codec used in place of the paper's virtual codec."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _likelihood_tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _likelihood_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _likelihood_tensors(child)


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = x.shape[-2:]
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    return F.pad(x, (0, pad_width, 0, pad_height), mode="replicate"), (height, width)


class CompressAIVideoCodec(nn.Module):
    """Pretrained SSF2020 as a differentiable, frozen video codec.

    SSF2020 supplies inter prediction, motion/residual transforms, quantization,
    reconstruction, and entropy likelihoods. During preprocessor training the codec
    remains in training mode so CompressAI uses its differentiable quantization path,
    while every codec parameter is frozen. Call ``eval()`` for hard quantization.
    """

    def __init__(self, quality: int = 3, metric: str = "mse", pad_multiple: int = 128) -> None:
        super().__init__()
        self.metric = metric
        self.pad_multiple = pad_multiple
        self.quality = -1
        self.model: nn.Module
        self.set_quality(quality)

    def set_quality(self, quality: int, device: torch.device | str | None = None) -> None:
        if quality not in range(1, 10):
            raise ValueError("SSF2020 quality must be from 1 through 9")
        if quality == self.quality:
            if device is not None:
                self.to(device)
            return

        from compressai.zoo import ssf2020

        was_training = self.training
        model = ssf2020(quality=quality, metric=self.metric, pretrained=True, progress=True)
        model.requires_grad_(False)
        model.train(was_training)
        if device is not None:
            model.to(device)
        self.model = model
        self.quality = quality

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if clip.ndim != 5 or clip.shape[2] != 3:
            raise ValueError(f"expected [B,T,3,H,W], got {tuple(clip.shape)}")
        batch, time, _, height, width = clip.shape
        padded, _ = _pad_to_multiple(
            clip.reshape(batch * time, 3, height, width), self.pad_multiple
        )
        padded = padded.reshape(batch, time, 3, *padded.shape[-2:])
        output = self.model([padded[:, index] for index in range(time)])

        reconstruction = torch.stack(output["x_hat"], dim=1)[..., :height, :width]
        bits = torch.zeros(batch, device=clip.device, dtype=torch.float32)
        for likelihood in _likelihood_tensors(output["likelihoods"]):
            if likelihood.shape[0] != batch:
                raise RuntimeError("CompressAI likelihood batch dimension changed unexpectedly")
            bits = bits - torch.log2(likelihood.float().clamp_min(1e-9)).reshape(batch, -1).sum(1)
        bpp = bits / float(time * height * width)
        return reconstruction.clamp(0.0, 1.0), bpp


def compression_loss(
    source: torch.Tensor,
    reconstruction: torch.Tensor,
    bpp: torch.Tensor,
    alpha: float = 10.0,
    rate_lambda: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return paper loss component ``alpha * (L_D + lambda * L_R)``."""

    distortion = F.mse_loss(reconstruction, source)
    rate = bpp.mean()
    return alpha * (distortion + rate_lambda * rate), distortion, rate
