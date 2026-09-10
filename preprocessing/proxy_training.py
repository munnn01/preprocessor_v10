"""Rate fitting, paired-delta supervision, and real-codec gradient audits."""

import torch
from torch.nn import functional as F


def log_rate(rate: torch.Tensor) -> torch.Tensor:
    return rate.float().clamp_min(1e-6).log()


def rate_fit_loss(
    predicted: torch.Tensor,
    measured: torch.Tensor,
    mode: str = "log",
) -> torch.Tensor:
    """Fit rate in log space by default, with the legacy absolute mode available."""

    if mode == "log":
        return F.smooth_l1_loss(log_rate(predicted), log_rate(measured))
    if mode == "absolute":
        return F.smooth_l1_loss(predicted.float(), measured.float())
    raise ValueError(f"unknown rate loss: {mode}")


def rate_delta_loss(
    base_predicted: torch.Tensor,
    variant_predicted: torch.Tensor,
    base_measured: torch.Tensor,
    variant_measured: torch.Tensor,
) -> torch.Tensor:
    """Match within-clip log-rate changes rather than unrelated samples."""
    return F.smooth_l1_loss(
        log_rate(variant_predicted) - log_rate(base_predicted),
        log_rate(variant_measured) - log_rate(base_measured),
    )


def rate_direction_loss(
    base_predicted: torch.Tensor,
    variant_predicted: torch.Tensor,
    base_measured: torch.Tensor,
    variant_measured: torch.Tensor,
    margin: float = 0.01,
) -> torch.Tensor:
    """Penalize a proxy log-rate delta whose sign contradicts real H.264."""
    measured_delta = log_rate(variant_measured) - log_rate(base_measured)
    predicted_delta = log_rate(variant_predicted) - log_rate(base_predicted)
    informative = measured_delta.abs() >= margin
    if not bool(informative.any()):
        return predicted_delta.sum() * 0.0
    target = measured_delta[informative].sign()
    return F.softplus(-target * predicted_delta[informative] / margin).mean()


@torch.no_grad()
def mixed_qp_roundtrip(codec, clips: torch.Tensor, qps: torch.Tensor):
    qps = torch.as_tensor(qps, device=clips.device).flatten()
    if qps.numel() == 1:
        qps = qps.expand(clips.shape[0])
    reconstruction = torch.empty_like(clips)
    rates = torch.empty(clips.shape[0], device=clips.device, dtype=torch.float32)
    previous_qp = codec.qp
    try:
        for qp in qps.unique(sorted=True).tolist():
            indices = (qps == qp).nonzero(as_tuple=True)[0]
            codec.set_qp(int(qp))
            decoded, bpp = codec(clips[indices])
            reconstruction[indices] = decoded
            rates[indices] = bpp.float()
    finally:
        codec.set_qp(previous_qp)
    return reconstruction, rates


def probe_rate_descent(proxy, codec, clips, qps, real_bpp, step_size):
    """Measure whether one proxy-rate descent step also reduces real BPP."""
    with torch.enable_grad():
        source = clips.detach().float().requires_grad_(True)
        _, before = proxy(source, qps)
        gradient, = torch.autograd.grad(log_rate(before).mean(), source)
        proposal = (source - step_size * gradient.sign()).clamp(0, 1).detach()
    with torch.no_grad():
        _, after = proxy(proposal, qps)
        _, measured_after = mixed_qp_roundtrip(codec, proposal, qps)
    return {
        "probe_real_delta_percent": 100.0 * float((measured_after / real_bpp - 1).mean()),
        "probe_proxy_down_fraction": float((after < before.detach()).float().mean()),
        "probe_real_down_fraction": float((measured_after < real_bpp).float().mean()),
    }
