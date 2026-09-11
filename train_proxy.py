"""Distill a differentiable proxy from FFmpeg H.264/H.265 reconstructions."""

from __future__ import annotations

import argparse
import math
import random
from copy import copy
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision.models.video import R3D_18_Weights
from tqdm import tqdm

from preprocessing import StandardCodecProxy, StandardVideoCodec
from preprocessing.data import (
    MixedQPBatchSampler,
    PrecomputedCodecDataset,
    VideoFolderDataset,
    stratified_split_indices,
)
from preprocessing.filters import spatial_lowpass
from preprocessing.model import preprocessor_from_checkpoint
from preprocessing.proxy_audit import audit_proxy_metrics
from preprocessing.proxy_training import (
    log_rate,
    mixed_qp_roundtrip,
    probe_rate_descent,
    rate_delta_loss,
    rate_direction_loss,
    rate_fit_loss,
)
from preprocessing.standard_codec import ffmpeg_version, require_ffmpeg
from preprocessing.utils import AverageMeter, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group("data")
    data.add_argument(
        "--precomputed-root",
        help="cache made by precompute_codec.py; avoids FFmpeg during training",
    )
    data.add_argument("--data-root", help="root containing train/ and optionally val/")
    data.add_argument("--train-dir")
    data.add_argument("--val-dir")
    data.add_argument("--val-ratio", type=float, default=0.1)
    data.add_argument("--frames", type=int, default=16)
    data.add_argument("--frame-stride", type=int, default=2)
    data.add_argument("--frame-size", type=int, default=128)
    data.add_argument("--limit-train", type=int)
    data.add_argument("--limit-val", type=int)
    data.add_argument("--workers", type=int, default=2)

    codec = parser.add_argument_group("codec")
    codec.add_argument("--codec", choices=("h264", "h265"), default="h264")
    codec.add_argument("--qps", type=int, nargs="+", default=[30, 35, 40, 45])
    codec.add_argument(
        "--qp-sampling-weights", type=float, nargs="+",
        help="raw-video training probabilities aligned with --qps; validation stays balanced",
    )
    codec.add_argument("--fps", type=float, default=30.0)
    codec.add_argument("--preset", default="medium")
    codec.add_argument("--ffmpeg", default="ffmpeg")
    codec.add_argument("--codec-io", choices=("pipe", "png"), default="pipe")
    codec.add_argument("--codec-workers", type=int, default=2)
    codec.add_argument("--ffmpeg-threads", type=int, default=1)

    model = parser.add_argument_group("proxy")
    model.add_argument("--hidden-channels", type=int, default=48)
    model.add_argument("--latent-channels", type=int, default=64)
    model.add_argument("--bottleneck-channels", type=int, default=96)
    model.add_argument("--blocks-per-stage", type=int, default=2)
    model.add_argument("--film-channels", type=int, default=64)
    model.add_argument("--qp-step-divisor", type=float, default=12.0)
    model.add_argument("--max-delta", type=float, default=1.0)
    model.add_argument("--entropy-floor", type=float, default=1e-9)

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--epochs", type=int, default=20)
    optimization.add_argument("--batch-size", type=int, default=8)
    optimization.add_argument("--lr", type=float, default=2e-4)
    optimization.add_argument("--rate-weight", type=float, default=0.1)
    optimization.add_argument("--rate-delta-weight", type=float, default=0.5)
    optimization.add_argument("--rate-direction-weight", type=float, default=0.1)
    optimization.add_argument("--rate-direction-margin", type=float, default=0.01)
    optimization.add_argument(
        "--pair-strengths", type=float, nargs="+",
        help="blur strengths for real-codec paired variants; enables online FFmpeg",
    )
    optimization.add_argument(
        "--preprocessor-checkpoint",
        help="frozen Swin used for on-policy paired variants; requires --pair-strengths",
    )
    optimization.add_argument("--gradient-probe-batches", type=int, default=0)
    optimization.add_argument("--gradient-probe-step", type=float, default=2.0 / 255.0)
    optimization.add_argument("--audit-max-rate-mape-percent", type=float, default=20.0)
    optimization.add_argument("--audit-min-pair-direction-accuracy", type=float, default=0.60)
    optimization.add_argument("--audit-max-real-delta-percent", type=float, default=0.0)
    optimization.add_argument("--audit-min-real-down-fraction", type=float, default=0.55)
    optimization.add_argument("--audit-max-proxy-clamp-fraction", type=float, default=0.05)
    optimization.add_argument("--init-checkpoint", help="load V4 proxy weights into a fresh run")
    optimization.add_argument("--weight-decay", type=float, default=1e-4)
    optimization.add_argument("--clip-grad", type=float, default=1.0)
    optimization.add_argument("--scheduler-factor", type=float, default=0.5)
    optimization.add_argument("--scheduler-patience", type=int, default=3)
    optimization.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    optimization.add_argument("--device", default="cuda")
    optimization.add_argument("--seed", type=int, default=42)
    optimization.add_argument("--resume")
    optimization.add_argument("--output-dir", default="checkpoints/proxy")
    optimization.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def resolve_directories(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.train_dir:
        return Path(args.train_dir), Path(args.val_dir) if args.val_dir else None
    if not args.data_root:
        raise ValueError("provide --data-root or --train-dir")
    root = Path(args.data_root)
    train = root / "train" if (root / "train").is_dir() else root
    validation = root / "val"
    return train, validation if validation.is_dir() else None


def validate_cache(args: argparse.Namespace, dataset: PrecomputedCodecDataset) -> None:
    manifest = dataset.manifest
    codec = manifest["codec"]
    video = manifest["video"]
    expected = {
        "cache_version": (int(manifest.get("version", 0)), 2),
        "codec": (codec["name"], args.codec),
        "qps": (list(codec["qps"]), list(args.qps)),
        "fps": (float(codec["fps"]), float(args.fps)),
        "preset": (codec["preset"], args.preset),
        "io_backend": (codec.get("io_backend"), args.codec_io),
        "ffmpeg_threads": (int(codec.get("ffmpeg_threads", 0)), args.ffmpeg_threads),
        "frames": (int(video["frames"]), args.frames),
        "frame_stride": (int(video["frame_stride"]), args.frame_stride),
        "frame_size": (int(video["frame_size"]), args.frame_size),
    }
    mismatches = [
        f"{name}: cache={cached!r}, CLI={requested!r}"
        for name, (cached, requested) in expected.items()
        if cached != requested
    ]
    if mismatches:
        raise ValueError("precomputed cache configuration mismatch: " + "; ".join(mismatches))
    if not codec.get("ffmpeg_version"):
        raise ValueError("precomputed cache has no FFmpeg version; rebuild it with V10")


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    common = {
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    if args.precomputed_root:
        train_set = PrecomputedCodecDataset(args.precomputed_root, "train", args.qps)
        val_set = PrecomputedCodecDataset(args.precomputed_root, "val", args.qps)
        validate_cache(args, train_set)
        batch_sampler = MixedQPBatchSampler(
            train_set, args.batch_size, seed=args.seed
        )
        train_loader = DataLoader(train_set, batch_sampler=batch_sampler, **common)
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        )
        return train_loader, val_loader

    train_dir, val_dir = resolve_directories(args)
    categories = list(R3D_18_Weights.DEFAULT.meta["categories"])
    options = {
        "frames": args.frames,
        "stride": args.frame_stride,
        "size": args.frame_size,
    }
    train_limit = 8 if args.smoke_test else args.limit_train
    val_limit = 4 if args.smoke_test else args.limit_val
    if val_dir is not None:
        train_set = VideoFolderDataset(
            train_dir, categories, train=True, limit=train_limit, **options
        )
        val_set = VideoFolderDataset(
            val_dir, categories, train=False, limit=val_limit, **options
        )
    else:
        source = VideoFolderDataset(train_dir, categories, train=False, **options)
        train_indices, val_indices = stratified_split_indices(
            source.samples, args.val_ratio, args.seed
        )
        if train_limit is not None:
            train_indices = train_indices[:train_limit]
        if val_limit is not None:
            val_indices = val_indices[:val_limit]
        augmented = copy(source)
        augmented.train = True
        train_set = Subset(augmented, train_indices)
        val_set = Subset(source, val_indices)
    common["batch_size"] = args.batch_size
    return (
        DataLoader(train_set, shuffle=True, drop_last=False, **common),
        DataLoader(val_set, shuffle=False, drop_last=False, **common),
    )


def _update_proxy_diagnostics(
    meters: dict[str, AverageMeter],
    diagnostics: dict[str, torch.Tensor],
    qp_values: torch.Tensor,
    qps: list[int],
) -> None:
    for name in (
        "proxy_clamp_fraction",
        "proxy_below_zero_fraction",
        "proxy_above_one_fraction",
    ):
        per_sample = diagnostics.get(f"{name}_per_sample")
        if per_sample is None:
            continue
        values = per_sample.detach().float().flatten()
        meters[name].update(float(values.mean()), values.numel())
        for qp in qps:
            selected = values[qp_values == qp]
            if selected.numel():
                meters[f"qp{qp}_{name}"].update(float(selected.mean()), selected.numel())


def run_epoch(
    loader: DataLoader,
    proxy: StandardCodecProxy,
    real_codec: StandardVideoCodec | None,
    args: argparse.Namespace,
    device: torch.device,
    *,
    optimizer: AdamW | None = None,
    scaler: torch.amp.GradScaler | None = None,
    epoch: int = 0,
    preprocessor: torch.nn.Module | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    proxy.train(training)
    if training and hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(epoch)
    names = (
        "loss", "reconstruction", "rate", "rate_delta", "rate_direction",
        "rate_mape_percent", "pair_direction_accuracy", "probe_real_delta_percent",
        "probe_proxy_down_fraction", "probe_real_down_fraction",
        "proxy_clamp_fraction", "proxy_below_zero_fraction", "proxy_above_one_fraction",
    )
    names += tuple(
        f"qp{qp}_{metric}"
        for qp in args.qps
        for metric in (
            "rate_mape_percent", "variant_rate_mape_percent",
            "pair_direction_accuracy",
            "probe_real_delta_percent", "probe_proxy_down_fraction",
            "probe_real_down_fraction",
            "proxy_clamp_fraction", "proxy_below_zero_fraction",
            "proxy_above_one_fraction",
        )
    )
    meters = {name: AverageMeter() for name in names}
    use_amp = bool(args.amp and device.type == "cuda")
    iterator = tqdm(loader, desc="proxy train" if training else "proxy valid", leave=False)
    context = torch.enable_grad if training else torch.no_grad
    if training:
        optimizer.zero_grad(set_to_none=True)
    with context():
        for step, batch_data in enumerate(iterator):
            if args.precomputed_root:
                clips, real_reconstruction, real_bpp, qp = batch_data
                clips = clips.to(device, non_blocking=True)
                real_reconstruction = real_reconstruction.to(device, non_blocking=True)
                real_bpp = real_bpp.to(device, non_blocking=True)
                qp = qp.to(device, non_blocking=True)
                qp_display = "mixed"
            else:
                clips, _ = batch_data
                if training:
                    qp = (
                        random.choices(args.qps, weights=args.qp_sampling_weights, k=1)[0]
                        if args.qp_sampling_weights is not None
                        else random.choice(args.qps)
                    )
                else:
                    qp = args.qps[step % len(args.qps)]
                real_codec.set_qp(qp)
                real_reconstruction, real_bpp = real_codec(clips)
                clips = clips.to(device, non_blocking=True)
                real_reconstruction = real_reconstruction.to(device, non_blocking=True)
                real_bpp = real_bpp.to(device, non_blocking=True)
                qp_display = str(qp)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                proxy_reconstruction, proxy_bpp = proxy(clips, qp)
            base_diagnostics = dict(getattr(proxy, "last_diagnostics", {}))
            reconstruction_loss = F.l1_loss(
                proxy_reconstruction.float(), real_reconstruction.float()
            )
            rate_loss = rate_fit_loss(proxy_bpp, real_bpp)
            delta_loss = rate_loss.new_zeros(())
            direction_loss = rate_loss.new_zeros(())
            predicted_rates, measured_rates = proxy_bpp, real_bpp
            metric_qps = qp_values = torch.as_tensor(qp, device=device).flatten()
            probe_clips, probe_bpp = clips, real_bpp
            if qp_values.numel() == 1:
                qp_values = qp_values.expand(clips.shape[0])
            metric_qps = qp_values
            _update_proxy_diagnostics(meters, base_diagnostics, qp_values, args.qps)
            if args.pair_strengths is not None:
                if real_codec is None:
                    raise ValueError("paired proxy training requires a real codec")
                # Validation cycles every perturbation at every QP instead of coupling
                # one fixed strength to one fixed QP through the step index.
                strength_index = (
                    step + epoch if training else step // len(args.qps)
                )
                strength = args.pair_strengths[strength_index % len(args.pair_strengths)]
                with torch.no_grad():
                    variant = clips if preprocessor is None else preprocessor(clips, qp_values)
                    variant = ((1.0 - strength) * variant + strength * spatial_lowpass(variant)).detach()
                    variant_reconstruction, variant_bpp = mixed_qp_roundtrip(
                        real_codec, variant, qp_values
                    )
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    predicted_reconstruction, predicted_bpp = proxy(variant, qp_values)
                _update_proxy_diagnostics(
                    meters,
                    dict(getattr(proxy, "last_diagnostics", {})),
                    qp_values,
                    args.qps,
                )
                reconstruction_loss = 0.5 * (
                    reconstruction_loss
                    + F.l1_loss(predicted_reconstruction.float(), variant_reconstruction.float())
                )
                rate_loss = 0.5 * (rate_loss + rate_fit_loss(predicted_bpp, variant_bpp))
                delta_loss = rate_delta_loss(proxy_bpp, predicted_bpp, real_bpp, variant_bpp)
                direction_margin = float(getattr(args, "rate_direction_margin", 0.01))
                direction_loss = rate_direction_loss(
                    proxy_bpp, predicted_bpp, real_bpp, variant_bpp,
                    margin=direction_margin,
                )
                predicted_delta = log_rate(predicted_bpp.detach()) - log_rate(proxy_bpp.detach())
                measured_delta = log_rate(variant_bpp) - log_rate(real_bpp)
                informative = measured_delta.abs() >= direction_margin
                if bool(informative.any()):
                    agreement = predicted_delta[informative].sign() == measured_delta[informative].sign()
                    meters["pair_direction_accuracy"].update(
                        float(agreement.float().mean()), int(informative.sum())
                    )
                    for current_qp in args.qps:
                        selected = informative & (qp_values == current_qp)
                        if bool(selected.any()):
                            current_agreement = (
                                predicted_delta[selected].sign() == measured_delta[selected].sign()
                            )
                            meters[f"qp{current_qp}_pair_direction_accuracy"].update(
                                float(current_agreement.float().mean()), int(selected.sum())
                            )
                predicted_rates = torch.cat((proxy_bpp, predicted_bpp))
                measured_rates = torch.cat((real_bpp, variant_bpp))
                metric_qps = qp_values.repeat(2)
                probe_clips, probe_bpp = variant, variant_bpp
                variant_error = 100.0 * (
                    predicted_bpp.detach().float() / variant_bpp.float() - 1.0
                ).abs()
                for current_qp in args.qps:
                    selected_error = variant_error[qp_values == current_qp]
                    if selected_error.numel():
                        meters[f"qp{current_qp}_variant_rate_mape_percent"].update(
                            float(selected_error.mean()), selected_error.numel()
                        )
            loss = (
                reconstruction_loss
                + args.rate_weight * rate_loss
                + args.rate_delta_weight * delta_loss
                + float(getattr(args, "rate_direction_weight", 0.0)) * direction_loss
            )
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(proxy.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch = clips.shape[0]
            values = torch.stack(
                (loss.detach(), reconstruction_loss.detach(), rate_loss.detach())
            ).float().cpu()
            meters["loss"].update(values[0].item(), batch)
            meters["reconstruction"].update(values[1].item(), batch)
            meters["rate"].update(values[2].item(), batch)
            meters["rate_delta"].update(float(delta_loss.detach()), batch)
            meters["rate_direction"].update(float(direction_loss.detach()), batch)
            relative_error = 100.0 * (
                predicted_rates.detach().float() / measured_rates.float() - 1.0
            ).abs()
            meters["rate_mape_percent"].update(float(relative_error.mean()), len(relative_error))
            for current_qp in args.qps:
                selected_error = relative_error[metric_qps == current_qp]
                if selected_error.numel():
                    meters[f"qp{current_qp}_rate_mape_percent"].update(
                        float(selected_error.mean()), selected_error.numel()
                    )
            if not training and step < args.gradient_probe_batches:
                if real_codec is None:
                    raise ValueError("gradient probes require a real codec")
                for current_qp in args.qps:
                    selected = (qp_values == current_qp).nonzero(as_tuple=True)[0]
                    if selected.numel():
                        per_qp = probe_rate_descent(
                            proxy, real_codec, probe_clips[selected], qp_values[selected],
                            probe_bpp[selected], args.gradient_probe_step,
                        )
                        for name, value in per_qp.items():
                            meters[name].update(value, selected.numel())
                            meters[f"qp{current_qp}_{name}"].update(value, selected.numel())
            iterator.set_postfix(loss=f"{meters['loss'].average:.4f}", qp=qp_display)
    return {name: meter.average for name, meter in meters.items() if meter.count > 0}


def main() -> None:
    args = parse_args()
    if args.init_checkpoint and args.resume:
        raise ValueError("choose --init-checkpoint or --resume, not both")
    if args.preprocessor_checkpoint and args.pair_strengths is None:
        raise ValueError("--preprocessor-checkpoint requires --pair-strengths")
    if args.pair_strengths is not None and any(
        not math.isfinite(value) or not 0 <= value <= 1 for value in args.pair_strengths
    ):
        raise ValueError("--pair-strengths must be finite values in [0, 1]")
    rate_weights = (
        args.rate_weight, args.rate_delta_weight, args.rate_direction_weight
    )
    if any(not math.isfinite(value) or value < 0 for value in rate_weights):
        raise ValueError("rate-loss weights must be finite and non-negative")
    if not math.isfinite(args.rate_direction_margin) or args.rate_direction_margin <= 0:
        raise ValueError("--rate-direction-margin must be positive")
    if args.gradient_probe_batches < 0 or not 0 < args.gradient_probe_step <= 1:
        raise ValueError("gradient probes require nonnegative batches and a step in (0, 1]")
    if args.audit_max_rate_mape_percent < 0:
        raise ValueError("--audit-max-rate-mape-percent must be nonnegative")
    if not 0 <= args.audit_min_pair_direction_accuracy <= 1:
        raise ValueError("--audit-min-pair-direction-accuracy must be in [0, 1]")
    if not 0 <= args.audit_min_real_down_fraction <= 1:
        raise ValueError("--audit-min-real-down-fraction must be in [0, 1]")
    if not 0 <= args.audit_max_proxy_clamp_fraction <= 1:
        raise ValueError("--audit-max-proxy-clamp-fraction must be in [0, 1]")
    if args.smoke_test:
        args.epochs = 1
    if args.precomputed_root and (args.data_root or args.train_dir or args.val_dir):
        raise ValueError("use --precomputed-root or raw video paths, not both")
    if not args.qps or any(qp < 0 or qp > 51 for qp in args.qps):
        raise ValueError("--qps must contain values in [0, 51]")
    if args.qp_sampling_weights is not None:
        if args.precomputed_root:
            raise ValueError("--qp-sampling-weights is only supported for raw-video training")
        if len(args.qp_sampling_weights) != len(args.qps):
            raise ValueError("--qp-sampling-weights must have one value per --qps entry")
        if any(not math.isfinite(value) or value < 0 for value in args.qp_sampling_weights):
            raise ValueError("--qp-sampling-weights must be finite and nonnegative")
        if sum(args.qp_sampling_weights) <= 0:
            raise ValueError("--qp-sampling-weights must contain a positive value")
    if args.clip_grad <= 0:
        raise ValueError("--clip-grad must be positive")
    if args.frame_size % 2:
        raise ValueError("--frame-size must be even for yuv420p H.264/H.265")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    needs_real_codec = (
        not args.precomputed_root
        or args.pair_strengths is not None
        or args.gradient_probe_batches > 0
    )
    if needs_real_codec:
        require_ffmpeg(args.ffmpeg)
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_loader, val_loader = make_loaders(args)
    if args.precomputed_root:
        cached_codec = train_loader.dataset.manifest["codec"]
        proxy_codec_config = {
            "codec": cached_codec["name"],
            "qps": list(cached_codec["qps"]),
            "fps": float(cached_codec["fps"]),
            "preset": cached_codec["preset"],
            "io_backend": cached_codec["io_backend"],
            "codec_workers": int(cached_codec.get("codec_workers", 1)),
            "ffmpeg_threads": int(cached_codec["ffmpeg_threads"]),
            "ffmpeg": cached_codec.get("ffmpeg", args.ffmpeg),
            "ffmpeg_version": cached_codec["ffmpeg_version"],
            "cache_version": int(train_loader.dataset.manifest["version"]),
        }
    else:
        proxy_codec_config = {
            "codec": args.codec,
            "qps": list(args.qps),
            "fps": args.fps,
            "preset": args.preset,
            "io_backend": args.codec_io,
            "codec_workers": args.codec_workers,
            "ffmpeg_threads": args.ffmpeg_threads,
            "ffmpeg": args.ffmpeg,
            "ffmpeg_version": ffmpeg_version(args.ffmpeg),
            "cache_version": None,
        }
    proxy = StandardCodecProxy(
        hidden_channels=args.hidden_channels,
        latent_channels=args.latent_channels,
        bottleneck_channels=args.bottleneck_channels,
        blocks_per_stage=args.blocks_per_stage,
        film_channels=args.film_channels,
        qp_step_divisor=args.qp_step_divisor,
        max_delta=args.max_delta,
        entropy_floor=args.entropy_floor,
    ).to(device)
    real_codec = None
    if needs_real_codec:
        real_codec = StandardVideoCodec(
            args.codec,
            args.qps[0],
            fps=args.fps,
            preset=args.preset,
            ffmpeg=args.ffmpeg,
            io_backend=args.codec_io,
            codec_workers=args.codec_workers,
            ffmpeg_threads=args.ffmpeg_threads,
        )
    preprocessor = None
    if args.preprocessor_checkpoint:
        payload = torch.load(args.preprocessor_checkpoint, map_location="cpu", weights_only=False)
        preprocessor = preprocessor_from_checkpoint(payload).to(device).requires_grad_(False).eval()
    optimizer = AdamW(proxy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 1
    best_loss = float("inf")
    best_audit_score = float("inf")
    best_feasible_audit_score = float("inf")
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("proxy_config") != proxy.config:
            raise ValueError("--init-checkpoint architecture differs from the requested V4 proxy")
        proxy.load_state_dict(checkpoint["proxy"])
        print(f"[init] loaded V4 proxy weights from {args.init_checkpoint}")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint.get("proxy_config", {})
        checkpoint_architecture = checkpoint_config.get("architecture")
        if checkpoint_architecture != StandardCodecProxy.ARCHITECTURE:
            raise ValueError(
                "--resume does not point to a predictive-entropy V4 checkpoint. "
                "The architecture changes tensor shapes, so start from epoch 1 "
                "with a new --output-dir. The precomputed codec cache can be reused."
            )
        if checkpoint_config != proxy.config:
            differences = [
                f"{name}: checkpoint={checkpoint_config.get(name)!r}, "
                f"CLI={proxy.config.get(name)!r}"
                for name in sorted(set(checkpoint_config) | set(proxy.config))
                if checkpoint_config.get(name) != proxy.config.get(name)
            ]
            raise ValueError(
                "proxy architecture arguments differ from the resume checkpoint: "
                + "; ".join(differences)
            )
        proxy.load_state_dict(checkpoint["proxy"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_val_loss", best_loss))
        best_audit_score = float(checkpoint.get("best_audit_score", best_audit_score))
        best_feasible_audit_score = float(
            checkpoint.get("best_feasible_audit_score", best_feasible_audit_score)
        )

    output_dir = Path(args.output_dir)
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            train_loader,
            proxy,
            real_codec,
            args,
            device,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            preprocessor=preprocessor,
        )
        val_metrics = run_epoch(
            val_loader, proxy, real_codec, args, device, epoch=epoch,
            preprocessor=preprocessor,
        )
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[epoch {epoch}/{args.epochs}] train={train_metrics} "
            f"valid={val_metrics} lr={current_lr:.3e}"
        )
        proxy_audit = audit_proxy_metrics(
            val_metrics,
            args.qps,
            max_rate_mape_percent=args.audit_max_rate_mape_percent,
            min_pair_direction_accuracy=args.audit_min_pair_direction_accuracy,
            max_real_delta_percent=args.audit_max_real_delta_percent,
            min_real_down_fraction=args.audit_min_real_down_fraction,
            max_proxy_clamp_fraction=args.audit_max_proxy_clamp_fraction,
        )
        print(
            f"[audit] feasible={proxy_audit['feasible']} "
            f"score={proxy_audit['score']:.4f}"
        )
        payload = {
            "epoch": epoch,
            "proxy": proxy.state_dict(),
            "proxy_config": proxy.config,
            "codec_config": proxy_codec_config,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_loss": min(best_loss, val_metrics["loss"]),
            "best_audit_score": min(best_audit_score, proxy_audit["score"]),
            "best_feasible_audit_score": min(
                best_feasible_audit_score,
                proxy_audit["score"] if proxy_audit["feasible"] else float("inf"),
            ),
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "proxy_audit": proxy_audit,
        }
        save_checkpoint(output_dir / "last.pt", payload)
        if proxy_audit["score"] < best_audit_score:
            best_audit_score = proxy_audit["score"]
            payload["best_audit_score"] = best_audit_score
            save_checkpoint(output_dir / "best_audit.pt", payload)
            print(f"[checkpoint] new best proxy audit score: {best_audit_score:.4f}")
        if proxy_audit["feasible"] and proxy_audit["score"] < best_feasible_audit_score:
            best_feasible_audit_score = proxy_audit["score"]
            payload["best_feasible_audit_score"] = best_feasible_audit_score
            save_checkpoint(output_dir / "best_feasible.pt", payload)
            print(
                "[checkpoint] new feasible proxy: "
                f"audit score={best_feasible_audit_score:.4f}"
            )
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            payload["best_val_loss"] = best_loss
            save_checkpoint(output_dir / "best.pt", payload)
            print(f"[checkpoint] new best proxy loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
