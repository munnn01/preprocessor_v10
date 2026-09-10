from __future__ import annotations

import pytest
import torch
from torch import nn

from preprocessing import (
    AdaptiveVideoSandwich,
    FiLM3DPostprocessor,
    FrozenDinoV2,
    HumanPerceptualObjective,
    adaptive_dct_loss,
    adaptive_vcm_objective,
    dino_video_loss,
    multiscale_ssim_loss,
)


class _ScalePreprocessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.8))

    def forward(self, clip: torch.Tensor, qp) -> torch.Tensor:
        del qp
        return clip * self.gain


class _BiasPostprocessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, clip: torch.Tensor, qp) -> torch.Tensor:
        del qp
        return clip + self.bias


class _Proxy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.5))

    def forward(self, clip: torch.Tensor, qp):
        del qp
        return clip * self.gain, clip.mean((1, 2, 3, 4))


class _RealForwardProxyBackwardCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qp = 35
        self.proxy = _Proxy()

    def set_qp(self, qp: int) -> None:
        self.qp = qp

    def forward(self, clip, *, codec_source="real", use_proxy_gradient=None):
        proxy_recon, proxy_bpp = self.proxy(clip, self.qp)
        if codec_source == "proxy":
            return proxy_recon, proxy_bpp
        real_recon = torch.full_like(clip, 0.375)
        real_bpp = torch.full((clip.shape[0],), 1.25, device=clip.device)
        if not use_proxy_gradient:
            return real_recon, real_bpp
        return (
            proxy_recon + (real_recon - proxy_recon).detach(),
            proxy_bpp + (real_bpp - proxy_bpp).detach(),
        )


def test_film3d_postprocessor_starts_as_exact_identity_and_is_qp_aware():
    model = FiLM3DPostprocessor(
        channels=8, bottleneck_channels=12, blocks=1, condition_channels=8
    )
    clip = torch.rand(2, 3, 3, 15, 17)
    torch.testing.assert_close(model(clip, 35), clip)
    assert model.config["architecture"] == "film3d_post_v1"


def test_sandwich_uses_real_forward_and_proxy_backward_with_frozen_codec():
    model = AdaptiveVideoSandwich(
        _ScalePreprocessor(),
        _RealForwardProxyBackwardCodec(),
        _BiasPostprocessor(),
    ).train()
    clip = torch.rand(2, 2, 3, 8, 8)
    output = model(clip, 35, use_proxy_gradient=True)
    torch.testing.assert_close(output.decoded, torch.full_like(clip, 0.375))
    torch.testing.assert_close(output.bpp, torch.full((2,), 1.25))
    (output.restored.mean() + output.bpp.mean()).backward()
    assert model.preprocessor.gain.grad is not None
    assert model.postprocessor.bias.grad is not None
    assert all(parameter.grad is None for parameter in model.codec.parameters())
    assert all(not parameter.requires_grad for parameter in model.codec.parameters())


def test_human_metrics_are_zero_or_small_for_identical_constant_video():
    clip = torch.full((1, 2, 3, 16, 16), 0.5)
    assert multiscale_ssim_loss(clip, clip).item() == pytest.approx(0.0, abs=1e-6)
    assert adaptive_dct_loss(clip).item() == pytest.approx(0.0, abs=1e-5)
    changed = clip.clone()
    changed[..., ::2, ::2] += 0.1
    assert multiscale_ssim_loss(changed, clip).item() > 0


def test_adaptive_dct_is_differentiable_for_weak_texture():
    clip = (0.5 + 0.01 * torch.rand(1, 2, 3, 16, 16)).requires_grad_()
    loss = adaptive_dct_loss(clip)
    loss.backward()
    assert torch.isfinite(loss)
    assert clip.grad is not None and clip.grad.abs().sum() > 0


class _DummyDino(nn.Module):
    def forward_features(self, frames: torch.Tensor):
        pooled = frames.mean((2, 3))
        return {
            "x_norm_clstoken": pooled,
            "x_norm_patchtokens": pooled[:, None, :].expand(-1, 4, -1),
        }


def test_dino_teacher_is_frozen_but_propagates_to_prediction():
    teacher = FrozenDinoV2(model=_DummyDino(), image_size=28, frame_stride=1)
    reference = torch.rand(1, 2, 3, 28, 28)
    prediction = (reference * 0.8).requires_grad_()
    loss = dino_video_loss(teacher, prediction, reference)
    loss.backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0
    assert all(not parameter.requires_grad for parameter in teacher.parameters())


def test_composite_objective_reports_named_terms_and_backpropagates():
    restored = torch.rand(1, 2, 3, 16, 16, requires_grad=True)
    source = torch.rand_like(restored)
    code = restored * 0.9
    human = HumanPerceptualObjective(
        charbonnier_weight=0.25,
        ms_ssim_weight=1.0,
        temporal_weight=0.1,
        dct_weight=0.05,
    )(restored, source, code)
    objective = adaptive_vcm_objective(
        bpp=torch.tensor([0.5], requires_grad=True),
        task_loss=restored.mean(),
        dino_loss=(restored - source).square().mean(),
        human_terms=human,
        anchor_bpp=1.0,
        rate_weight=0.1,
        task_weight=1.0,
        dino_weight=0.2,
        human_weight=1.0,
    )
    objective["total"].backward()
    assert restored.grad is not None and torch.isfinite(restored.grad).all()
    assert {"total", "rate_ratio", "task", "dino", "human", "adaptive_dct"} <= objective.keys()
