"""Human-oriented and RPP-inspired differentiable video losses."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _frames(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 5 or video.shape[2] != 3:
        raise ValueError(f"expected [B,T,3,H,W], got {tuple(video.shape)}")
    return video.reshape(-1, 3, video.shape[-2], video.shape[-1])


def charbonnier_loss(
    prediction: torch.Tensor, reference: torch.Tensor, epsilon: float = 1e-3
) -> torch.Tensor:
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference shapes must match")
    return torch.sqrt((prediction - reference).square() + epsilon * epsilon).mean()


def _ssim_components(
    prediction: torch.Tensor, reference: torch.Tensor, window: int = 11
) -> tuple[torch.Tensor, torch.Tensor]:
    padding = window // 2
    mu_x = F.avg_pool2d(prediction, window, stride=1, padding=padding)
    mu_y = F.avg_pool2d(reference, window, stride=1, padding=padding)
    var_x = F.avg_pool2d(prediction.square(), window, 1, padding) - mu_x.square()
    var_y = F.avg_pool2d(reference.square(), window, 1, padding) - mu_y.square()
    cov = F.avg_pool2d(prediction * reference, window, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    luminance = (2.0 * mu_x * mu_y + c1) / (mu_x.square() + mu_y.square() + c1)
    contrast_structure = (2.0 * cov + c2) / (var_x + var_y + c2)
    return (
        (luminance * contrast_structure).mean(dim=(1, 2, 3)),
        contrast_structure.mean(dim=(1, 2, 3)),
    )


def multiscale_ssim_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    weights: tuple[float, ...] = (0.07105472, 0.45297383, 0.47597145),
) -> torch.Tensor:
    """Return a compact three-scale MS-SSIM loss averaged over video frames."""

    x, y = _frames(prediction), _frames(reference)
    if x.shape != y.shape:
        raise ValueError("prediction and reference shapes must match")
    active_weights = list(weights)
    scores: list[torch.Tensor] = []
    contrasts: list[torch.Tensor] = []
    for scale in range(len(weights)):
        ssim, contrast = _ssim_components(x, y, window=min(11, x.shape[-1] | 1))
        scores.append(ssim.clamp_min(1e-6))
        contrasts.append(contrast.clamp_min(1e-6))
        if scale + 1 == len(weights) or min(x.shape[-2:]) < 4:
            active_weights = active_weights[: scale + 1]
            break
        x = F.avg_pool2d(x, 2, 2)
        y = F.avg_pool2d(y, 2, 2)
    weight = torch.tensor(
        active_weights, device=prediction.device, dtype=prediction.dtype
    )
    weight = weight / weight.sum()
    factors = contrasts[:-1] + [scores[-1]]
    score = torch.ones_like(factors[0])
    for exponent, factor in zip(weight, factors, strict=True):
        score = score * factor.pow(exponent)
    return 1.0 - score.mean()


def _dct_matrix(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.arange(size, device=device, dtype=dtype) + 0.5
    frequency = torch.arange(size, device=device, dtype=dtype).unsqueeze(1)
    matrix = torch.cos(math.pi * frequency * position / size)
    matrix[0] *= math.sqrt(1.0 / size)
    if size > 1:
        matrix[1:] *= math.sqrt(2.0 / size)
    return matrix


def block_dct(video: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """Return orthonormal block DCT coefficients as ``[BT,C,N,U,V]``."""

    frames = _frames(video)
    height, width = frames.shape[-2:]
    pad_h = (block_size - height % block_size) % block_size
    pad_w = (block_size - width % block_size) % block_size
    if pad_h or pad_w:
        frames = F.pad(frames, (0, pad_w, 0, pad_h), mode="replicate")
    patches = F.unfold(frames, kernel_size=block_size, stride=block_size)
    patches = patches.transpose(1, 2).reshape(
        frames.shape[0], -1, 3, block_size, block_size
    ).permute(0, 2, 1, 3, 4)
    matrix = _dct_matrix(block_size, device=video.device, dtype=video.dtype)
    return torch.einsum("ui,bcnij,vj->bcnuv", matrix, patches, matrix)


def adaptive_dct_loss(
    neural_code: torch.Tensor, *, block_size: int = 8, high_frequency_start: int = 6
) -> torch.Tensor:
    """RPP-style penalty that removes weak high-frequency DCT coefficients.

    Coefficients above the zig-zag diagonal are compared with their detached
    per-block mean magnitude.  Only below-mean high-frequency coefficients are
    pulled toward zero; strong edge/texture coefficients are preserved.
    """

    if not 1 <= high_frequency_start <= 2 * block_size - 2:
        raise ValueError("high_frequency_start is outside the DCT block")
    coefficients = block_dct(neural_code, block_size)
    axis = torch.arange(block_size, device=neural_code.device)
    high = (axis[:, None] + axis[None, :]) >= high_frequency_start
    high = high[None, None, None]
    magnitude = coefficients.abs()
    count = high.sum().to(magnitude.dtype).clamp_min(1.0)
    threshold = (magnitude * high).sum(dim=(-2, -1), keepdim=True) / count
    weak = high & (magnitude < threshold.detach())
    denominator = weak.sum().to(magnitude.dtype).clamp_min(1.0)
    return (magnitude * weak).sum() / denominator


def temporal_gradient_loss(
    prediction: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference shapes must match")
    if prediction.shape[1] < 2:
        return prediction.new_zeros(())
    pred_delta = prediction[:, 1:] - prediction[:, :-1]
    ref_delta = reference[:, 1:] - reference[:, :-1]
    return F.l1_loss(pred_delta, ref_delta)


class LPIPSLoss(nn.Module):
    """Optional frozen LPIPS metric used by Sandwiched Compression."""

    def __init__(self, backbone: str = "alex") -> None:
        super().__init__()
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError(
                "LPIPS requested; install the research extra with `pip install -e .[research]`"
            ) from exc
        self.network = lpips.LPIPS(net=backbone).requires_grad_(False).eval()

    def train(self, mode: bool = True) -> LPIPSLoss:
        super().train(False)
        self.network.eval()
        return self

    def forward(self, prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return self.network(2.0 * _frames(prediction) - 1.0, 2.0 * _frames(reference) - 1.0).mean()


class HumanPerceptualObjective(nn.Module):
    """Auditable human-view objective with optional LPIPS and RPP terms."""

    def __init__(
        self,
        *,
        charbonnier_weight: float = 0.25,
        ms_ssim_weight: float = 1.0,
        lpips_weight: float = 0.0,
        temporal_weight: float = 0.1,
        dct_weight: float = 0.05,
        dct_block_size: int = 8,
        dct_high_frequency_start: int = 6,
        lpips_backbone: str = "alex",
    ) -> None:
        super().__init__()
        weights = (
            charbonnier_weight,
            ms_ssim_weight,
            lpips_weight,
            temporal_weight,
            dct_weight,
        )
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("perceptual weights must be finite and non-negative")
        self.charbonnier_weight = charbonnier_weight
        self.ms_ssim_weight = ms_ssim_weight
        self.lpips_weight = lpips_weight
        self.temporal_weight = temporal_weight
        self.dct_weight = dct_weight
        self.dct_block_size = dct_block_size
        self.dct_high_frequency_start = dct_high_frequency_start
        self.lpips = LPIPSLoss(lpips_backbone) if lpips_weight > 0 else None

    def forward(
        self,
        restored: torch.Tensor,
        reference: torch.Tensor,
        neural_code: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        charb = charbonnier_loss(restored.float(), reference.float())
        ms_ssim = multiscale_ssim_loss(restored.float(), reference.float())
        temporal = temporal_gradient_loss(restored.float(), reference.float())
        dct = adaptive_dct_loss(
            neural_code.float(),
            block_size=self.dct_block_size,
            high_frequency_start=self.dct_high_frequency_start,
        )
        lpips_value = restored.new_zeros((), dtype=torch.float32)
        if self.lpips is not None:
            lpips_value = self.lpips(restored.float(), reference.float())
        total = (
            self.charbonnier_weight * charb
            + self.ms_ssim_weight * ms_ssim
            + self.lpips_weight * lpips_value
            + self.temporal_weight * temporal
            + self.dct_weight * dct
        )
        return {
            "human": total,
            "charbonnier": charb,
            "ms_ssim": ms_ssim,
            "lpips": lpips_value,
            "temporal": temporal,
            "adaptive_dct": dct,
        }
