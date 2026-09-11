"""Lightweight QP-conditioned post-processors for neural codec sandwiches."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _condition(
    qp: int | float | torch.Tensor,
    batch: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.as_tensor(qp, device=device, dtype=dtype).flatten()
    if value.numel() == 1:
        value = value.expand(batch)
    if value.numel() != batch:
        raise ValueError(f"QP must be scalar or have {batch} values")
    return ((value - 25.5) / 25.5).unsqueeze(1)


class IdentityPostprocessor(nn.Module):
    """Ablation that leaves the decoded codec signal unchanged."""

    def forward(
        self, clip: torch.Tensor, qp: int | float | torch.Tensor | None = None
    ) -> torch.Tensor:
        del qp
        return clip

    @property
    def config(self) -> dict[str, Any]:
        return {"architecture": "identity"}


class _FiLM3D(nn.Module):
    def __init__(self, condition_channels: int, channels: int) -> None:
        super().__init__()
        self.affine = nn.Linear(condition_channels, 2 * channels)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(condition).chunk(2, dim=1)
        gamma = 0.25 * torch.tanh(gamma)[:, :, None, None, None]
        beta = 0.10 * torch.tanh(beta)[:, :, None, None, None]
        return (1.0 + gamma) * value + beta


class _SeparableResidual3D(nn.Module):
    """Compact depthwise-separable 3-D residual block."""

    def __init__(self, channels: int, condition_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.pointwise = nn.Conv3d(channels, channels, 1, bias=False)
        self.activation = nn.GELU()
        self.film = _FiLM3D(condition_channels, channels)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = self.pointwise(self.depthwise(value))
        residual = self.film(self.activation(residual), condition)
        return self.activation(value + residual)


class FiLM3DPostprocessor(nn.Module):
    """Compact residual 3-D U-Net with codec-QP FiLM conditioning.

    Inputs and outputs use ``[B,T,3,H,W]`` in ``[0,1]``.  The RGB head is
    zero-initialized, so a new model is exactly the identity mapping.  Temporal
    resolution is never reduced; the postprocessor can therefore operate on
    short streaming windows without changing frame rate.
    """

    ARCHITECTURE = "film3d_post_v1"

    def __init__(
        self,
        channels: int = 32,
        bottleneck_channels: int = 48,
        blocks: int = 2,
        condition_channels: int = 48,
        max_residual: float = 0.25,
    ) -> None:
        super().__init__()
        if min(channels, bottleneck_channels, blocks, condition_channels) < 1:
            raise ValueError("postprocessor dimensions must be positive")
        if not 0.0 < max_residual <= 1.0:
            raise ValueError("max_residual must lie in (0,1]")
        self.channels = channels
        self.bottleneck_channels = bottleneck_channels
        self.blocks = blocks
        self.condition_channels = condition_channels
        self.max_residual = max_residual

        self.qp_embedding = nn.Sequential(
            nn.Linear(1, condition_channels),
            nn.GELU(),
            nn.Linear(condition_channels, condition_channels),
            nn.GELU(),
        )
        self.stem = nn.Conv3d(3, channels, (3, 5, 5), padding=(1, 2, 2))
        self.high = nn.ModuleList(
            _SeparableResidual3D(channels, condition_channels) for _ in range(blocks)
        )
        self.down = nn.Conv3d(
            channels,
            bottleneck_channels,
            (3, 4, 4),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
        )
        self.low = nn.ModuleList(
            _SeparableResidual3D(bottleneck_channels, condition_channels)
            for _ in range(blocks)
        )
        self.fuse = nn.Conv3d(channels + bottleneck_channels, channels, 1)
        self.to_rgb = nn.Conv3d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.to_rgb.weight)
        nn.init.zeros_(self.to_rgb.bias)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "architecture": self.ARCHITECTURE,
            "channels": self.channels,
            "bottleneck_channels": self.bottleneck_channels,
            "blocks": self.blocks,
            "condition_channels": self.condition_channels,
            "max_residual": self.max_residual,
        }

    def forward(
        self, clip: torch.Tensor, qp: int | float | torch.Tensor
    ) -> torch.Tensor:
        if clip.ndim != 5 or clip.shape[2] != 3:
            raise ValueError(f"expected [B,T,3,H,W], got {tuple(clip.shape)}")
        batch = clip.shape[0]
        condition = self.qp_embedding(
            _condition(qp, batch, device=clip.device, dtype=clip.dtype)
        )
        source = clip.permute(0, 2, 1, 3, 4)
        high = self.stem(source)
        for block in self.high:
            high = block(high, condition)
        low = self.down(high)
        for block in self.low:
            low = block(low, condition)
        low = F.interpolate(
            low, size=high.shape[-3:], mode="trilinear", align_corners=False
        )
        fused = F.gelu(self.fuse(torch.cat((high, low), dim=1)))
        residual = self.max_residual * torch.tanh(self.to_rgb(fused))
        restored = (source + residual).clamp(0.0, 1.0)
        return restored.permute(0, 2, 1, 3, 4)


def build_postprocessor(
    kind: str = "film3d",
    *,
    channels: int = 32,
    bottleneck_channels: int = 48,
    blocks: int = 2,
    condition_channels: int = 48,
    max_residual: float = 0.25,
) -> nn.Module:
    if kind == "identity":
        return IdentityPostprocessor()
    if kind == "film3d":
        return FiLM3DPostprocessor(
            channels=channels,
            bottleneck_channels=bottleneck_channels,
            blocks=blocks,
            condition_channels=condition_channels,
            max_residual=max_residual,
        )
    raise ValueError(f"unknown postprocessor {kind!r}; choose 'film3d' or 'identity'")


def postprocessor_from_checkpoint(checkpoint: dict[str, Any]) -> nn.Module:
    args = checkpoint.get("args", {})
    config = checkpoint.get("postprocessor_config", {})
    architecture = config.get("architecture")
    if architecture is None:
        architecture = "identity" if "postprocessor" not in checkpoint else "film3d_post_v1"
    if architecture == "identity":
        return IdentityPostprocessor()
    if architecture != FiLM3DPostprocessor.ARCHITECTURE:
        raise ValueError(f"unsupported postprocessor architecture: {architecture!r}")
    model = FiLM3DPostprocessor(
        channels=int(config.get("channels", args.get("post_channels", 32))),
        bottleneck_channels=int(
            config.get("bottleneck_channels", args.get("post_bottleneck_channels", 48))
        ),
        blocks=int(config.get("blocks", args.get("post_blocks", 2))),
        condition_channels=int(
            config.get("condition_channels", args.get("post_condition_channels", 48))
        ),
        max_residual=float(
            config.get("max_residual", args.get("post_max_residual", 0.25))
        ),
    )
    model.load_state_dict(checkpoint["postprocessor"])
    return model
