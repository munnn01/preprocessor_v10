"""Composite VCM objective with explicit, named loss terms."""

from __future__ import annotations

import math

import torch


def adaptive_vcm_objective(
    *,
    bpp: torch.Tensor,
    task_loss: torch.Tensor,
    dino_loss: torch.Tensor,
    human_terms: dict[str, torch.Tensor],
    anchor_bpp: float | None,
    rate_weight: float,
    task_weight: float,
    dino_weight: float,
    human_weight: float,
) -> dict[str, torch.Tensor]:
    weights = (rate_weight, task_weight, dino_weight, human_weight)
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("objective weights must be finite and non-negative")
    rate = bpp.float().mean()
    if anchor_bpp is not None:
        if not math.isfinite(anchor_bpp) or anchor_bpp <= 0:
            raise ValueError("anchor_bpp must be finite and positive")
        rate_ratio = rate / anchor_bpp
    else:
        rate_ratio = rate
    human = human_terms["human"].float()
    total = (
        rate_weight * rate_ratio
        + task_weight * task_loss.float()
        + dino_weight * dino_loss.float()
        + human_weight * human
    )
    return {
        "total": total,
        "rate": rate,
        "rate_ratio": rate_ratio,
        "task": task_loss.float(),
        "dino": dino_loss.float(),
        **{key: value.float() for key, value in human_terms.items()},
    }
