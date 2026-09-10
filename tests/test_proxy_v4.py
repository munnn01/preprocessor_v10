from types import SimpleNamespace

import pytest
import torch
from torch import nn

import train
import train_proxy
from preprocessing.feature_distillation import feature_distillation
from preprocessing.model import preprocessor_from_checkpoint
from preprocessing.proxy_training import rate_delta_loss, rate_direction_loss
from preprocessing.proxy_audit import audit_proxy_metrics
from preprocessing.standard_codec import StandardCodecProxy
from preprocessing.swin import VideoSwinLitePreprocessor


def tiny_proxy():
    return StandardCodecProxy(
        hidden_channels=8,
        latent_channels=12,
        bottleneck_channels=16,
        blocks_per_stage=1,
        film_channels=8,
    )


def test_prediction_residual_is_explicit_previous_frame_difference():
    source = torch.tensor([[[[[0.2]], [[0.4]], [[0.1]]]]])
    residual = StandardCodecProxy._prediction_residual(source)
    torch.testing.assert_close(
        residual, torch.tensor([[[[[-0.3]], [[0.2]], [[-0.3]]]]])
    )


def test_entropy_rate_backpropagates_to_input_of_frozen_proxy():
    torch.manual_seed(4)
    proxy = tiny_proxy().requires_grad_(False).eval()
    clip = torch.rand(2, 4, 3, 16, 16, requires_grad=True)
    reconstruction, rate = proxy(clip, torch.tensor([30, 45]))
    assert reconstruction.shape == clip.shape
    assert rate.shape == (2,)
    assert torch.isfinite(rate).all() and bool((rate > 0).all())
    rate.mean().backward()
    assert clip.grad is not None and torch.isfinite(clip.grad).all()
    assert clip.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in proxy.parameters())


def test_entropy_calibration_is_positive_and_architecture_is_versioned():
    proxy = tiny_proxy()
    assert proxy.config["architecture"] == "predictive_entropy_v1"
    assert proxy.config["entropy_floor"] == pytest.approx(1e-9)
    with pytest.raises(ValueError, match="entropy_floor"):
        StandardCodecProxy(entropy_floor=0)


def test_paired_rate_losses_reward_the_real_direction():
    base_predicted = torch.tensor([0.20], requires_grad=True)
    variant_predicted = torch.tensor([0.25], requires_grad=True)
    base_measured, variant_measured = torch.tensor([0.20]), torch.tensor([0.30])
    correct = rate_direction_loss(
        base_predicted, variant_predicted, base_measured, variant_measured
    )
    wrong = rate_direction_loss(
        variant_predicted, base_predicted, base_measured, variant_measured
    )
    assert correct < wrong
    delta = rate_delta_loss(
        base_predicted, variant_predicted, base_measured, variant_measured
    )
    (correct + delta).backward()
    assert base_predicted.grad is not None and variant_predicted.grad is not None


def test_optional_feature_loss_detaches_teacher_and_updates_student():
    teacher = {"layer4": torch.rand(2, 4, 2, 3, 3, requires_grad=True)}
    student = {"layer4": torch.rand(2, 4, 2, 3, 3, requires_grad=True)}
    loss, terms = feature_distillation(student, teacher, ["layer4"])
    loss.backward()
    torch.testing.assert_close(terms["layer4"], loss)
    assert student["layer4"].grad is not None
    assert teacher["layer4"].grad is None


def test_optional_feature_objective_backpropagates_to_preprocessor():
    class Preprocessor(nn.Module):
        def __init__(self):
            super().__init__()
            self.channel_scale = nn.Parameter(torch.tensor([0.8, 1.0, 1.0]))

        def forward(self, clip, qp):
            del qp
            return clip * self.channel_scale.view(1, 1, 3, 1, 1)

    class Codec(nn.Module):
        def forward(self, clip, codec_source=None):
            del codec_source
            return clip, clip.square().mean(dim=(1, 2, 3, 4))

    class Analyzer(nn.Module):
        def forward(self, clip):
            score = clip.mean(dim=(1, 2, 3, 4))
            return torch.stack((score, -score), dim=1)

        def forward_with_features(self, clip, layers):
            return self.forward(clip), {name: clip.permute(0, 2, 1, 3, 4) for name in layers}

    args = type("Args", (), {
        "alpha": 0.0, "qp_to_rate_lambda": {35: 0.0},
        "feature_weight": 0.2, "feature_layers": ["layer4"],
    })()
    clip = torch.rand(2, 3, 3, 4, 4)
    analyzer, preprocessor = Analyzer(), Preprocessor()
    _, clean = analyzer.forward_with_features(clip, ["layer4"])
    losses = train.forward_losses(
        clip, torch.zeros(2, dtype=torch.long), preprocessor, Codec(),
        analyzer, args, False, 35, clean_features=clean,
    )
    assert losses["feature_loss"].item() > 0
    losses["total"].backward()
    assert preprocessor.channel_scale.grad is not None
    assert preprocessor.channel_scale.grad.abs().sum() > 0


def test_v6_gated_swin_checkpoint_can_initialize_v4():
    model = VideoSwinLitePreprocessor(
        patch_size=2, embed_dim=12, depth=1, num_heads=3,
        window_size=(2, 2, 2), qp_embed_dim=8,
        gated_smoothing=True, smoothing_max_strength=0.5, max_residual=0.1,
    )
    checkpoint = {
        "args": {
            "preprocessor": "swin", "swin_patch_size": 2,
            "swin_embed_dim": 12, "swin_depth": 1, "swin_heads": 3,
            "swin_window_temporal": 2, "swin_window_spatial": 2,
            "swin_qp_conditioning": True, "swin_qp_embed_dim": 8,
            "swin_gated_smoothing": True, "swin_smoothing_max_strength": 0.5,
            "max_residual": 0.1,
        },
        "preprocessor": model.state_dict(),
    }
    restored = preprocessor_from_checkpoint(checkpoint)
    clip = torch.rand(1, 2, 3, 8, 8)
    torch.testing.assert_close(restored(clip, 40), model(clip, 40))


def test_proxy_validation_reports_pair_and_gradient_audit_per_qp():
    class RealCodec(nn.Module):
        def __init__(self):
            super().__init__()
            self.qp = 30

        def set_qp(self, qp):
            self.qp = int(qp)

        def forward(self, clip):
            rate = 2.0 * clip.var(dim=(1, 2, 3, 4), unbiased=False) + self.qp / 100.0
            return clip, rate

    torch.manual_seed(3)
    clips = torch.rand(2, 3, 3, 16, 16)
    qps = torch.tensor([30, 45])
    codec = RealCodec()
    _, real_bpp = train_proxy.mixed_qp_roundtrip(codec, clips, qps)
    args = SimpleNamespace(
        precomputed_root="cache", amp=False, qps=[30, 45],
        pair_strengths=[0.5], rate_direction_margin=0.001,
        rate_weight=0.1, rate_delta_weight=0.5, rate_direction_weight=0.1,
        gradient_probe_batches=1, gradient_probe_step=2 / 255,
    )
    metrics = train_proxy.run_epoch(
        [(clips, clips, real_bpp, qps)], tiny_proxy(), codec, args,
        torch.device("cpu"),
    )
    for qp in qps.tolist():
        assert f"qp{qp}_rate_mape_percent" in metrics
        assert f"qp{qp}_pair_direction_accuracy" in metrics
        assert f"qp{qp}_probe_real_delta_percent" in metrics
        assert f"qp{qp}_probe_real_down_fraction" in metrics


def test_proxy_audit_requires_every_qp_to_pass():
    metrics = {}
    for qp in (30, 35):
        metrics.update({
            f"qp{qp}_rate_mape_percent": 12.0,
            f"qp{qp}_pair_direction_accuracy": 0.72,
            f"qp{qp}_probe_real_delta_percent": -1.0,
            f"qp{qp}_probe_real_down_fraction": 0.65,
        })
    passed = audit_proxy_metrics(metrics, [30, 35])
    assert passed["feasible"]
    metrics["qp35_probe_real_delta_percent"] = 0.2
    failed = audit_proxy_metrics(metrics, [30, 35])
    assert not failed["feasible"]
    assert failed["per_qp"][1]["reasons"] == ["real_bpp_did_not_decrease"]
