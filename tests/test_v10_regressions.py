from __future__ import annotations

import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate_sandwich import aggregate_operating_points
from preprocessing import AdaptiveVideoSandwich
from preprocessing.evaluation import bootstrap_bd_rate, calculate_bd_rate_details
from preprocessing.proxy_audit import audit_proxy_metrics
from preprocessing.standard_codec import StandardCodecProxy, StandardVideoCodec
from train_proxy import validate_cache, validate_pair_audit_configuration
from train_sandwich import (
    _capture_rng_state,
    _restore_rng_state,
    _validate_proxy_checkpoint,
    _validate_resume_configuration,
    build_optimizer,
    build_parser,
    forward_losses,
    initial_selection_state,
    run_epoch,
    sandwich_rate_weights_by_qp,
    update_selection_state,
)


def _validation(loss: float, bd_rate: float | None, *, top1: float = 0.5) -> dict:
    return {
        "loss": loss,
        "task": loss + 0.25,
        "top1": top1,
        "task_bd_rate_percent": bd_rate,
    }


def test_best_checkpoint_follows_loss_while_bd_rate_is_undefined():
    state = initial_selection_state("task_bd_rate")
    state, first = update_selection_state(state, _validation(2.0, None), 1)
    state, second = update_selection_state(state, _validation(1.0, None), 2)

    assert "best.pt" in first
    assert "best.pt" in second
    assert state["primary_epoch"] == 2
    assert state["primary_metric_used"] == "loss_fallback"
    assert state["primary_value"] == 1.0


def test_first_valid_bd_rate_replaces_loss_fallback_and_undefined_cannot_replace_it():
    state = initial_selection_state("task_bd_rate")
    state, _ = update_selection_state(state, _validation(1.0, None), 1)
    state, selected = update_selection_state(state, _validation(1.5, -3.0), 2)
    assert {"best.pt", "best_task_bd_rate.pt"} <= selected
    assert state["primary_epoch"] == 2
    assert state["primary_metric_used"] == "task_bd_rate"

    state, selected = update_selection_state(state, _validation(0.5, None), 3)
    assert "best_loss.pt" in selected
    assert "best.pt" not in selected
    assert state["primary_epoch"] == 2


@pytest.mark.parametrize(
    ("metric", "first", "second", "should_replace"),
    [
        ("loss", _validation(1.0, None), _validation(2.0, None), False),
        ("ce", _validation(1.0, None), _validation(0.5, None), True),
        (
            "top1",
            _validation(1.0, None, top1=0.5),
            _validation(2.0, None, top1=0.6),
            True,
        ),
    ],
)
def test_explicit_checkpoint_metrics_are_honored(metric, first, second, should_replace):
    state = initial_selection_state(metric)
    state, _ = update_selection_state(state, first, 1)
    state, selected = update_selection_state(state, second, 2)
    assert ("best.pt" in selected) is should_replace


def test_sandwich_parser_does_not_expose_legacy_noop_rate_flags():
    destinations = {action.dest for action in build_parser()._actions}
    assert "mask_rate_outside_weight" not in destinations
    assert "rate_dual_control" not in destinations
    assert "refresh_proxy_on_resume" not in destinations


def test_sandwich_parser_accepts_per_qp_rate_weights():
    args = build_parser().parse_args(
        [
            "--proxy-checkpoint",
            "proxy.pt",
            "--codec-qps",
            "30",
            "35",
            "40",
            "45",
            "--sandwich-rate-weight",
            "0.05",
            "--sandwich-rate-weights",
            "0.12",
            "0.10",
            "0.06",
            "0.05",
        ]
    )
    assert sandwich_rate_weights_by_qp(args) == pytest.approx(
        {30: 0.12, 35: 0.10, 40: 0.06, 45: 0.05}
    )


def test_sandwich_scalar_rate_weight_remains_the_backward_compatible_default():
    args = SimpleNamespace(
        codec_qps=[30, 45],
        sandwich_rate_weight=0.07,
        sandwich_rate_weights=None,
    )
    assert sandwich_rate_weights_by_qp(args) == {30: 0.07, 45: 0.07}


@pytest.mark.parametrize(
    "weights",
    ([0.1], [0.1, float("nan")], [0.1, -0.2]),
)
def test_sandwich_rejects_invalid_per_qp_rate_weights(weights):
    args = SimpleNamespace(
        codec_qps=[30, 45],
        sandwich_rate_weight=0.05,
        sandwich_rate_weights=weights,
    )
    with pytest.raises(ValueError, match=r"rate.?weights"):
        sandwich_rate_weights_by_qp(args)


def test_optimizer_flag_selects_the_real_optimizer():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    adam = build_optimizer(
        SimpleNamespace(optimizer="adam", lr=1e-3, weight_decay=0.0), [parameter]
    )
    adamw = build_optimizer(
        SimpleNamespace(optimizer="adamw", lr=1e-3, weight_decay=0.0), [parameter]
    )
    assert type(adam) is torch.optim.Adam
    assert type(adamw) is torch.optim.AdamW


def test_rng_state_restores_python_numpy_torch_and_qp_streams():
    outer_qp = random.Random()
    outer = _capture_rng_state(outer_qp)
    try:
        random.seed(3)
        np.random.seed(3)
        torch.manual_seed(3)
        qp_rng = random.Random(19)
        state = _capture_rng_state(qp_rng)
        expected = (random.random(), float(np.random.rand()), float(torch.rand(())), qp_rng.random())
        _restore_rng_state(state, qp_rng)
        observed = (random.random(), float(np.random.rand()), float(torch.rand(())), qp_rng.random())
        assert observed == pytest.approx(expected)
    finally:
        _restore_rng_state(outer, outer_qp)


def test_exact_resume_rejects_changed_runtime_knobs():
    checkpoint = {
        "format_version": 10,
        "args": {"amp": True, "ffmpeg_version": "ffmpeg version A"},
        "proxy_sha256": "same-proxy",
    }
    args = SimpleNamespace(amp=False, ffmpeg_version="ffmpeg version A")
    with pytest.raises(ValueError, match="amp"):
        _validate_resume_configuration(checkpoint, args, "same-proxy")


def test_reportable_sandwich_rejects_proxy_without_codec_provenance(tmp_path):
    proxy_path = tmp_path / "proxy.pt"
    torch.save({"codec_config": {}, "args": {}, "proxy_audit": {"feasible": True}}, proxy_path)
    args = SimpleNamespace(
        proxy_checkpoint=str(proxy_path),
        codec="h264",
        codec_fps=30.0,
        codec_preset="medium",
        ffmpeg_threads=1,
        ffmpeg_version="ffmpeg version test",
        codec_qps=[30, 35],
        frames=16,
        frame_stride=2,
        frame_size=128,
        allow_unaudited_proxy=False,
    )
    with pytest.raises(ValueError, match="codec provenance"):
        _validate_proxy_checkpoint(args)


def _paired_rows() -> list[dict]:
    rows = []
    operating_points = ((30, 0.8, 0.9), (35, 0.4, 0.7), (40, 0.2, 0.5))
    for sample_index in range(4):
        for qp, anchor_rate, top1 in operating_points:
            for method, scale in (("anchor", 1.0), ("sandwich", 0.8)):
                rows.append(
                    {
                        "sample_id": f"{sample_index:08d}",
                        "sample_index": sample_index,
                        "codec": "h264",
                        "method": method,
                        "qp": qp,
                        "bpp": anchor_rate * scale,
                        "mse": 0.1 / (1.0 + top1),
                        "psnr_db": -10.0 * math.log10(0.1 / (1.0 + top1)),
                        "ms_ssim": top1,
                        "top1": top1,
                        "top5": 1.0,
                    }
                )
    return rows


def test_bootstrap_accepts_real_method_names_and_matches_point_estimate():
    result = bootstrap_bd_rate(
        _paired_rows(),
        "top1_percent",
        anchor_method="anchor",
        proposed_method="sandwich",
        samples=100,
        seed=7,
    )
    assert result["point_estimate_percent"] == pytest.approx(-20.0, abs=1e-6)
    assert result["samples_valid"] == 100
    assert result["samples_invalid"] == 0
    assert result["valid_fraction"] == 1.0


def test_summary_psnr_uses_the_predeclared_aggregation():
    rows = []
    for index, mse in enumerate((0.01, 0.09)):
        rows.append(
            {
                "sample_index": index,
                "codec": "h264",
                "qp": 35,
                "method": "anchor",
                "bpp": 0.5,
                "mse": mse,
                "psnr_db": -10.0 * math.log10(mse),
                "ms_ssim": 0.8,
                "top1": 1.0,
                "top5": 1.0,
            }
        )
    aggregate = aggregate_operating_points(
        rows, psnr_aggregation="psnr_from_mean_video_mse", include_lpips=False
    )[0]
    mean_video = aggregate_operating_points(
        rows, psnr_aggregation="mean_video_psnr", include_lpips=False
    )[0]
    assert aggregate["psnr_db"] == pytest.approx(-10.0 * math.log10(0.05))
    assert aggregate["psnr_db"] != pytest.approx(mean_video["psnr_db"])


def test_raw_curve_sensitivity_reports_nonmonotonic_quality():
    rows = []
    for method, scale in (("anchor", 1.0), ("sandwich", 0.8)):
        for rate, quality in ((0.1, 10.0), (0.2, 30.0), (0.3, 20.0)):
            rows.append({"method": method, "bpp": rate * scale, "quality": quality})
    result = calculate_bd_rate_details(
        rows,
        "quality",
        anchor_method="anchor",
        proposed_method="sandwich",
        apply_monotone_envelope=False,
    )
    assert result["bd_rate_percent"] is None
    assert result["anchor_curve"]["invalid_reason"] == "raw_quality_not_strictly_increasing"


def test_proxy_reports_hard_clamp_saturation():
    proxy = StandardCodecProxy(
        hidden_channels=8,
        latent_channels=8,
        bottleneck_channels=8,
        blocks_per_stage=1,
        film_channels=8,
    )
    torch.nn.init.zeros_(proxy.to_rgb.weight)
    torch.nn.init.constant_(proxy.to_rgb.bias, 10.0)
    proxy(torch.full((2, 4, 3, 16, 16), 0.5), 35)
    assert float(proxy.last_diagnostics["proxy_clamp_fraction"]) > 0.5
    assert proxy.last_diagnostics["proxy_clamp_fraction_per_sample"].shape == (2,)


def test_proxy_audit_rejects_excessive_clamp_for_each_qp():
    metrics = {}
    for qp in (30, 35):
        metrics.update(
            {
                f"qp{qp}_rate_mape_percent": 10.0,
                f"qp{qp}_pair_direction_accuracy": 0.8,
                f"qp{qp}_probe_real_delta_percent": -1.0,
                f"qp{qp}_probe_real_down_fraction": 0.8,
                f"qp{qp}_proxy_clamp_fraction": 0.01 if qp == 30 else 0.20,
            }
        )
    result = audit_proxy_metrics(metrics, (30, 35), max_proxy_clamp_fraction=0.05)
    assert not result["feasible"]
    assert result["per_qp"][1]["reasons"] == ["proxy_clamp_fraction_too_high"]


def test_codec_command_manifest_matches_runtime_configuration():
    codec = StandardVideoCodec("h264", 35, fps=25.0, preset="fast", ffmpeg_threads=3)
    commands = codec.pipe_command_spec(16, 128, 128)
    encoded = " ".join(commands["encode"])
    assert "-framerate 25.0" in encoded
    assert "-preset fast" in encoded
    assert "-qp 35" in encoded
    assert "keyint=16:min-keyint=16:scenecut=0" in encoded


def test_proxy_training_rejects_legacy_precompute_without_ffmpeg_provenance():
    args = SimpleNamespace(
        codec="h264",
        qps=[30, 35],
        fps=30.0,
        preset="medium",
        codec_io="pipe",
        ffmpeg_threads=1,
        frames=16,
        frame_stride=2,
        frame_size=128,
    )
    manifest = {
        "version": 2,
        "codec": {
            "name": "h264",
            "qps": [30, 35],
            "fps": 30.0,
            "preset": "medium",
            "io_backend": "pipe",
            "ffmpeg_threads": 1,
            "ffmpeg_version": "ffmpeg version test",
        },
        "video": {"frames": 16, "frame_stride": 2, "frame_size": 128},
    }
    validate_cache(args, SimpleNamespace(manifest=manifest))
    manifest["version"] = 1
    with pytest.raises(ValueError, match="cache_version"):
        validate_cache(args, SimpleNamespace(manifest=manifest))


def test_proxy_requires_pairs_before_promising_a_feasible_checkpoint():
    args = SimpleNamespace(
        init_checkpoint=None,
        resume=None,
        preprocessor_checkpoint=None,
        pair_strengths=None,
        allow_incomplete_audit=False,
    )
    with pytest.raises(ValueError, match="best_feasible.pt"):
        validate_pair_audit_configuration(args)

    args.allow_incomplete_audit = True
    validate_pair_audit_configuration(args)


def _epoch_smoke_args() -> SimpleNamespace:
    return SimpleNamespace(
        amp=False,
        codec_qps=[30, 45],
        qp_sampling_weights=None,
        gradient_audit_interval=0,
        accumulation_steps=1,
        clip_grad=1.0,
        train_codec_source="real",
        sandwich_rate_weight=0.05,
        sandwich_task_weight=1.0,
        dino_weight=0.0,
        human_weight=0.0,
    )


class _EpochPreprocessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, clips, qp):
        del qp
        return clips * self.gain


class _EpochPostprocessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, clips, qp):
        del qp
        return clips + self.bias


class _EpochCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qp = 30
        self.last_proxy_bpp = None
        self.last_proxy_diagnostics = {}

    def set_qp(self, qp):
        self.qp = int(qp)

    def forward(self, clips, *, codec_source="real", use_proxy_gradient=None):
        del use_proxy_gradient
        proxy_bpp = clips.mean((1, 2, 3, 4)) * 0.0 + (self.qp / 100.0 - 0.05)
        self.last_proxy_bpp = proxy_bpp.detach()
        zero = clips.new_zeros(())
        self.last_proxy_diagnostics = {
            "proxy_clamp_fraction": zero,
            "proxy_below_zero_fraction": zero,
            "proxy_above_one_fraction": zero,
        }
        if codec_source == "proxy":
            return clips, proxy_bpp
        real_bpp = proxy_bpp + 0.05
        return clips, proxy_bpp + (real_bpp - proxy_bpp).detach()


class _ContractViolatingCodec(_EpochCodec):
    """Return real BPP without refreshing the proxy-drift side-channel."""

    def forward(self, clips, *, codec_source="real", use_proxy_gradient=None):
        del codec_source, use_proxy_gradient
        real_bpp = clips.new_full((clips.shape[0],), self.qp / 100.0)
        return clips, real_bpp


class _InvalidProxyBppCodec(_EpochCodec):
    def __init__(self, invalid_proxy_bpp) -> None:
        super().__init__()
        self.invalid_proxy_bpp = invalid_proxy_bpp

    def forward(self, clips, *, codec_source="real", use_proxy_gradient=None):
        reconstruction, real_bpp = super().forward(
            clips,
            codec_source=codec_source,
            use_proxy_gradient=use_proxy_gradient,
        )
        self.last_proxy_bpp = torch.as_tensor(
            self.invalid_proxy_bpp, device=clips.device, dtype=clips.dtype
        )
        return reconstruction, real_bpp


class _EpochAnalyzer(nn.Module):
    def forward(self, clips):
        score = clips.mean((1, 2, 3, 4))
        zero = score * 0.0
        return torch.stack((score, -score, zero, zero, zero), dim=1)


class _EpochHuman(nn.Module):
    def forward(self, restored, reference, neural_code):
        zero = (restored.mean() + reference.mean() + neural_code.mean()) * 0.0
        return {
            "human": zero,
            "charbonnier": zero,
            "ms_ssim": zero,
            "lpips": zero,
            "temporal": zero,
            "adaptive_dct": zero,
        }


def test_sandwich_forward_applies_the_rate_weight_for_the_current_qp():
    clips = torch.rand(1, 2, 3, 8, 8)
    labels = torch.zeros(1, dtype=torch.long)
    args = _epoch_smoke_args()
    args.sandwich_rate_weights = [0.12, 0.05]
    model = AdaptiveVideoSandwich(
        _EpochPreprocessor(), _EpochCodec(), _EpochPostprocessor()
    )

    qp30 = forward_losses(
        clips,
        labels,
        30,
        model,
        _EpochAnalyzer(),
        None,
        _EpochHuman(),
        args,
        training=False,
        anchor_bpp=0.30,
    )
    qp45 = forward_losses(
        clips,
        labels,
        45,
        model,
        _EpochAnalyzer(),
        None,
        _EpochHuman(),
        args,
        training=False,
        anchor_bpp=0.45,
    )

    assert float(qp30["rate_ratio"].detach()) == pytest.approx(1.0)
    assert float(qp45["rate_ratio"].detach()) == pytest.approx(1.0)
    assert float((qp30["total"] - qp45["total"]).detach()) == pytest.approx(0.07)


def test_sandwich_validation_epoch_reports_real_and_proxy_bpp():
    clips = torch.rand(2, 2, 3, 8, 8)
    labels = torch.zeros(2, dtype=torch.long)
    loader = DataLoader(TensorDataset(clips, labels), batch_size=1, shuffle=False)
    model = AdaptiveVideoSandwich(
        _EpochPreprocessor(), _EpochCodec(), _EpochPostprocessor()
    )
    metrics = run_epoch(
        loader,
        model,
        _EpochAnalyzer(),
        None,
        _EpochHuman(),
        _epoch_smoke_args(),
        torch.device("cpu"),
        {"qp30_bpp": 0.30, "qp45_bpp": 0.45},
    )

    assert metrics["rate"] == pytest.approx(0.375)
    assert metrics["bpp"] == pytest.approx(metrics["rate"])
    assert metrics["proxy_bpp"] == pytest.approx(0.325)
    assert metrics["qp30_bpp"] == pytest.approx(0.30)
    assert metrics["qp45_bpp"] == pytest.approx(0.45)
    assert metrics["qp30_proxy_bpp"] == pytest.approx(0.25)
    assert metrics["qp45_proxy_bpp"] == pytest.approx(0.40)
    assert all(
        math.isfinite(value) for value in metrics.values() if isinstance(value, float)
    )


def test_sandwich_training_epoch_runs_backward_and_optimizer_step():
    clips = torch.rand(1, 2, 3, 8, 8)
    labels = torch.zeros(1, dtype=torch.long)
    loader = DataLoader(TensorDataset(clips, labels), batch_size=1, shuffle=False)
    model = AdaptiveVideoSandwich(
        _EpochPreprocessor(), _EpochCodec(), _EpochPostprocessor()
    )
    optimizer = torch.optim.SGD(model.trainable_parameters(), lr=0.1)
    before = [parameter.detach().clone() for parameter in model.trainable_parameters()]

    metrics = run_epoch(
        loader,
        model,
        _EpochAnalyzer(),
        None,
        _EpochHuman(),
        _epoch_smoke_args(),
        torch.device("cpu"),
        {"qp30_bpp": 0.30, "qp45_bpp": 0.45},
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        qp_rng=random.Random(7),
    )

    after = list(model.trainable_parameters())
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, after))
    assert metrics["bpp"] == pytest.approx(metrics["rate"])
    assert math.isfinite(metrics["gradient_total"])


def test_sandwich_epoch_rejects_missing_or_stale_proxy_bpp():
    clips = torch.rand(1, 2, 3, 8, 8)
    loader = DataLoader(
        TensorDataset(clips, torch.zeros(1, dtype=torch.long)),
        batch_size=1,
        shuffle=False,
    )
    codec = _ContractViolatingCodec()
    codec.last_proxy_bpp = torch.tensor([0.25])  # Plausible but stale and same-sized.
    model = AdaptiveVideoSandwich(
        _EpochPreprocessor(), codec, _EpochPostprocessor()
    )

    with pytest.raises(RuntimeError, match="did not update last_proxy_bpp at qp=30"):
        run_epoch(
            loader,
            model,
            _EpochAnalyzer(),
            None,
            _EpochHuman(),
            _epoch_smoke_args(),
            torch.device("cpu"),
            {"qp30_bpp": 0.30, "qp45_bpp": 0.45},
        )


def test_sandwich_epoch_rejects_misshaped_or_nonfinite_proxy_bpp():
    clips = torch.rand(1, 2, 3, 8, 8)
    loader = DataLoader(
        TensorDataset(clips, torch.zeros(1, dtype=torch.long)),
        batch_size=1,
        shuffle=False,
    )
    invalid_cases = (
        (torch.ones(99), "shape"),
        (torch.tensor([float("nan")]), "non-finite"),
        (torch.tensor([float("inf")]), "non-finite"),
    )

    for invalid_proxy_bpp, message in invalid_cases:
        model = AdaptiveVideoSandwich(
            _EpochPreprocessor(),
            _InvalidProxyBppCodec(invalid_proxy_bpp),
            _EpochPostprocessor(),
        )
        with pytest.raises(RuntimeError, match=message):
            run_epoch(
                loader,
                model,
                _EpochAnalyzer(),
                None,
                _EpochHuman(),
                _epoch_smoke_args(),
                torch.device("cpu"),
                {"qp30_bpp": 0.30, "qp45_bpp": 0.45},
            )
