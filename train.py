"""Train a video preprocessor through a standard codec and frozen proxy/analyzer."""

from __future__ import annotations

import argparse
import random
from copy import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from preprocessing import (
    FrozenVideoAnalyzer,
    ParallelStandardVideoCodec,
    StandardCodecProxy,
    StandardVideoCodec,
    build_preprocessor,
)
from preprocessing.codec import compression_loss
from preprocessing.data import VideoFolderDataset, stratified_split_indices
from preprocessing.standard_codec import require_ffmpeg
from preprocessing.utils import AverageMeter, save_checkpoint, seed_everything, topk_correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group("data")
    data.add_argument("--data-root", help="root containing train/ and optionally val/ folders")
    data.add_argument("--train-dir", help="explicit class-folder training directory")
    data.add_argument("--val-dir", help="explicit class-folder validation directory")
    data.add_argument("--train-split", default="train")
    data.add_argument("--val-split", default="val")
    data.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="stratified validation fraction when no validation directory exists",
    )
    data.add_argument("--frames", type=int, default=16)
    data.add_argument("--frame-stride", type=int, default=2)
    data.add_argument("--frame-size", type=int, default=128)
    data.add_argument("--limit-train", type=int)
    data.add_argument("--limit-val", type=int)
    data.add_argument("--workers", type=int, default=4)

    model = parser.add_argument_group("model")
    model.add_argument("--preprocessor", choices=("vit", "cnn"), default="vit")
    model.add_argument("--temporal-frames", type=int, default=8)
    model.add_argument("--vit-patch-size", type=int, default=8)
    model.add_argument("--vit-embed-dim", type=int, default=96)
    model.add_argument("--vit-depth", type=int, default=4)
    model.add_argument("--vit-heads", type=int, default=4)
    model.add_argument("--max-residual", type=float, default=0.25)
    model.add_argument("--analyzer", default="r3d_18")
    model.add_argument(
        "--codec-qps",
        type=int,
        nargs="+",
        default=[30, 35, 40, 45, 50],
        help="standard-codec QPs sampled once per epoch",
    )
    model.add_argument("--codec", choices=("h264", "h265"), default="h264")
    model.add_argument("--proxy-checkpoint", required=True)
    model.add_argument("--codec-fps", type=float, default=30.0)
    model.add_argument("--codec-preset", default="medium")
    model.add_argument("--ffmpeg", default="ffmpeg")

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--epochs", type=int, default=30)
    optimization.add_argument("--batch-size", type=int, default=2)
    optimization.add_argument("--accumulation-steps", type=int, default=1)
    optimization.add_argument("--lr", type=float, default=1e-4)
    optimization.add_argument("--alpha", type=float, default=10.0)
    optimization.add_argument("--rate-lambda", type=float, default=0.001)
    optimization.add_argument("--weight-decay", type=float, default=0.0)
    optimization.add_argument("--clip-grad", type=float, default=1.0)
    optimization.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--output-dir", default="checkpoints")
    runtime.add_argument("--resume")
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--device", default="cuda")
    runtime.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def resolve_data_directories(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.train_dir:
        train_dir = Path(args.train_dir)
        return train_dir, Path(args.val_dir) if args.val_dir else None
    if args.data_root:
        root = Path(args.data_root)
        train_candidate = root / args.train_split
        train_dir = train_candidate if train_candidate.is_dir() else root
        val_candidate = root / args.val_split
        val_dir = val_candidate if val_candidate.is_dir() else None
        if not train_dir.is_dir():
            raise FileNotFoundError(f"training directory does not exist: {train_dir}")
        return train_dir, val_dir
    raise ValueError("provide --data-root or --train-dir")


def make_loaders(args: argparse.Namespace, categories: list[str]) -> tuple[DataLoader, DataLoader]:
    train_dir, val_dir = resolve_data_directories(args)
    train_limit = 8 if args.smoke_test else args.limit_train
    val_limit = 4 if args.smoke_test else args.limit_val
    dataset_options = {
        "frames": args.frames,
        "stride": args.frame_stride,
        "size": args.frame_size,
    }
    if val_dir is not None:
        train_set = VideoFolderDataset(
            train_dir, categories, train=True, limit=train_limit, **dataset_options
        )
        val_set = VideoFolderDataset(
            val_dir, categories, train=False, limit=val_limit, **dataset_options
        )
    else:
        train_source = VideoFolderDataset(train_dir, categories, train=True, **dataset_options)
        val_source = copy(train_source)
        val_source.train = False
        train_indices, val_indices = stratified_split_indices(
            train_source.samples, args.val_ratio, args.seed
        )
        if train_limit is not None:
            train_indices = train_indices[:train_limit]
        if val_limit is not None:
            val_indices = val_indices[:val_limit]
        train_set = Subset(train_source, train_indices)
        val_set = Subset(val_source, val_indices)
        print(
            f"[data] no validation directory found; stratified split "
            f"train={len(train_set)} val={len(val_set)} ratio={args.val_ratio:.3f}"
        )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    return (
        DataLoader(train_set, shuffle=True, drop_last=len(train_set) >= args.batch_size, **common),
        DataLoader(val_set, shuffle=False, drop_last=False, **common),
    )


def forward_losses(
    clips: torch.Tensor,
    labels: torch.Tensor,
    preprocessor: nn.Module,
    codec: ParallelStandardVideoCodec,
    analyzer: FrozenVideoAnalyzer,
    args: argparse.Namespace,
    use_amp: bool,
) -> dict[str, torch.Tensor]:
    device_type = clips.device.type
    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
        processed = preprocessor(clips)
        # Forward values come from FFmpeg; the frozen proxy supplies only the
        # reconstruction/rate Jacobian needed to update the preprocessor.
        reconstructed, bpp = codec(processed)
        logits = analyzer(reconstructed)

    # Accumulate probability-derived rate and both losses in FP32. This avoids
    # precision loss without forcing the expensive codec/analyzer convolutions to FP32.
    with torch.autocast(device_type=device_type, enabled=False):
        rd_loss, distortion, rate = compression_loss(
            clips.float(), reconstructed.float(), bpp.float(), args.alpha, args.rate_lambda
        )
        accuracy_loss = F.cross_entropy(logits.float(), labels)
        total = rd_loss + accuracy_loss
    return {
        "total": total,
        "distortion": distortion,
        "rate": rate,
        "accuracy_loss": accuracy_loss,
        "logits": logits,
    }


def run_epoch(
    loader: DataLoader,
    preprocessor: nn.Module,
    codec: ParallelStandardVideoCodec,
    analyzer: FrozenVideoAnalyzer,
    args: argparse.Namespace,
    device: torch.device,
    *,
    optimizer: Adam | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    preprocessor.train(training)
    codec.train(training)
    analyzer.eval()
    meters = {name: AverageMeter() for name in ("loss", "distortion", "bpp", "task_loss")}
    correct1 = correct5 = examples = 0
    use_amp = bool(args.amp and device.type == "cuda")
    if training:
        optimizer.zero_grad(set_to_none=True)

    iterator = tqdm(loader, desc="train" if training else "valid", leave=False)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for step, (clips, labels) in enumerate(iterator, start=1):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            losses = forward_losses(clips, labels, preprocessor, codec, analyzer, args, use_amp)
            if training:
                scaled_loss = losses["total"] / args.accumulation_steps
                assert scaler is not None
                scaler.scale(scaled_loss).backward()
                should_step = step % args.accumulation_steps == 0 or step == len(loader)
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(preprocessor.parameters(), args.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            batch = labels.numel()
            meters["loss"].update(float(losses["total"].detach()), batch)
            meters["distortion"].update(float(losses["distortion"].detach()), batch)
            meters["bpp"].update(float(losses["rate"].detach()), batch)
            meters["task_loss"].update(float(losses["accuracy_loss"].detach()), batch)
            correct1 += topk_correct(losses["logits"].detach(), labels, 1)
            correct5 += topk_correct(losses["logits"].detach(), labels, 5)
            examples += batch
            iterator.set_postfix(
                loss=f"{meters['loss'].average:.4f}", bpp=f"{meters['bpp'].average:.3f}"
            )

    return {
        "loss": meters["loss"].average,
        "distortion": meters["distortion"].average,
        "bpp": meters["bpp"].average,
        "task_loss": meters["task_loss"].average,
        "top1": correct1 / max(examples, 1),
        "top5": correct5 / max(examples, 1),
    }


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu for debugging")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if not args.codec_qps or any(qp < 0 or qp > 51 for qp in args.codec_qps):
        raise ValueError("--codec-qps must contain values in [0, 51]")
    if args.frame_size % 2:
        raise ValueError("--frame-size must be even for yuv420p H.264/H.265")
    require_ffmpeg(args.ffmpeg)
    print(f"[setup] device={device} autocast={bool(args.amp and device.type == 'cuda')}")
    analyzer = FrozenVideoAnalyzer(args.analyzer).to(device)
    train_loader, val_loader = make_loaders(args, analyzer.categories)
    preprocessor = build_preprocessor(
        args.preprocessor,
        temporal_frames=args.temporal_frames,
        patch_size=args.vit_patch_size,
        embed_dim=args.vit_embed_dim,
        depth=args.vit_depth,
        num_heads=args.vit_heads,
        max_residual=args.max_residual,
    ).to(device)
    proxy_checkpoint = torch.load(
        args.proxy_checkpoint, map_location="cpu", weights_only=False
    )
    proxy_codec_config = proxy_checkpoint.get("codec_config", {})
    trained_codec = proxy_codec_config.get("codec")
    if trained_codec and trained_codec != args.codec:
        raise ValueError(
            f"proxy was distilled for {trained_codec}, but --codec is {args.codec}"
        )
    trained_qps = set(proxy_codec_config.get("qps", []))
    missing_qps = set(args.codec_qps) - trained_qps if trained_qps else set()
    if missing_qps:
        raise ValueError(f"proxy checkpoint was not trained for QPs {sorted(missing_qps)}")
    trained_fps = proxy_codec_config.get("fps")
    if trained_fps is not None and abs(float(trained_fps) - args.codec_fps) > 1e-6:
        raise ValueError(
            f"proxy was distilled at {trained_fps} fps, but --codec-fps is {args.codec_fps}"
        )
    trained_preset = proxy_codec_config.get("preset")
    if trained_preset and trained_preset != args.codec_preset:
        raise ValueError(
            f"proxy was distilled with preset {trained_preset}, "
            f"but --codec-preset is {args.codec_preset}"
        )
    proxy_args = proxy_checkpoint.get("args", {})
    for name in ("frames", "frame_stride", "frame_size"):
        trained_value = proxy_args.get(name)
        current_value = getattr(args, name)
        if trained_value is not None and int(trained_value) != int(current_value):
            raise ValueError(
                f"proxy was distilled with {name}={trained_value}, "
                f"but preprocessor training uses {current_value}"
            )
    proxy = StandardCodecProxy.from_checkpoint(args.proxy_checkpoint).to(device)
    standard_codec = StandardVideoCodec(
        args.codec,
        args.codec_qps[0],
        fps=args.codec_fps,
        preset=args.codec_preset,
        ffmpeg=args.ffmpeg,
    )
    codec = ParallelStandardVideoCodec(standard_codec, proxy).to(device)
    optimizer = Adam(preprocessor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_loss = 1, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        preprocessor.load_state_dict(checkpoint["preprocessor"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_val_loss", best_loss))
        print(f"[resume] epoch={start_epoch} best_val_loss={best_loss:.6f}")

    output_dir = Path(args.output_dir)
    qp_rng = random.Random(args.seed + 17)
    for epoch in range(start_epoch, args.epochs + 1):
        qp = qp_rng.choice(args.codec_qps)
        codec.set_qp(qp)
        print(f"\n[epoch {epoch}/{args.epochs}] {args.codec.upper()} QP={qp}")
        train_metrics = run_epoch(
            train_loader,
            preprocessor,
            codec,
            analyzer,
            args,
            device,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_metrics = run_epoch(val_loader, preprocessor, codec, analyzer, args, device)
        scheduler.step(val_metrics["loss"])
        print(f"train={train_metrics}")
        print(f"valid={val_metrics}")

        payload = {
            "epoch": epoch,
            "preprocessor": preprocessor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": min(best_loss, val_metrics["loss"]),
            "codec": args.codec,
            "codec_qp": qp,
            "proxy_checkpoint": str(args.proxy_checkpoint),
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        save_checkpoint(output_dir / "last.pt", payload)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            payload["best_val_loss"] = best_loss
            save_checkpoint(output_dir / "best.pt", payload)
            print(f"[checkpoint] new best validation loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
