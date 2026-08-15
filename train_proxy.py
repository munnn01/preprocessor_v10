"""Distill a differentiable proxy from FFmpeg H.264/H.265 reconstructions."""

from __future__ import annotations

import argparse
import random
from copy import copy
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision.models.video import R3D_18_Weights

from preprocessing import StandardCodecProxy, StandardVideoCodec
from preprocessing.data import VideoFolderDataset, stratified_split_indices
from preprocessing.standard_codec import require_ffmpeg
from preprocessing.utils import AverageMeter, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group("data")
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
    codec.add_argument("--qps", type=int, nargs="+", default=[30, 35, 40, 45, 50])
    codec.add_argument("--fps", type=float, default=30.0)
    codec.add_argument("--preset", default="medium")
    codec.add_argument("--ffmpeg", default="ffmpeg")

    model = parser.add_argument_group("proxy")
    model.add_argument("--hidden-channels", type=int, default=48)
    model.add_argument("--latent-channels", type=int, default=64)
    model.add_argument("--max-delta", type=float, default=0.5)

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--epochs", type=int, default=20)
    optimization.add_argument("--batch-size", type=int, default=2)
    optimization.add_argument("--lr", type=float, default=2e-4)
    optimization.add_argument("--rate-weight", type=float, default=0.1)
    optimization.add_argument("--weight-decay", type=float, default=1e-4)
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


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
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
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    return (
        DataLoader(train_set, shuffle=True, drop_last=False, **common),
        DataLoader(val_set, shuffle=False, drop_last=False, **common),
    )


def run_epoch(
    loader: DataLoader,
    proxy: StandardCodecProxy,
    real_codec: StandardVideoCodec,
    args: argparse.Namespace,
    device: torch.device,
    *,
    optimizer: AdamW | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    proxy.train(training)
    meters = {name: AverageMeter() for name in ("loss", "reconstruction", "rate")}
    use_amp = bool(args.amp and device.type == "cuda")
    iterator = tqdm(loader, desc="proxy train" if training else "proxy valid", leave=False)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for step, (clips, _) in enumerate(iterator):
            clips = clips.to(device, non_blocking=True)
            qp = random.choice(args.qps) if training else args.qps[step % len(args.qps)]
            real_codec.set_qp(qp)
            real_reconstruction, real_bpp = real_codec(clips)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                proxy_reconstruction, proxy_bpp = proxy(clips, qp)
            reconstruction_loss = F.l1_loss(
                proxy_reconstruction.float(), real_reconstruction.float()
            )
            rate_loss = F.smooth_l1_loss(proxy_bpp.float(), real_bpp.float())
            loss = reconstruction_loss + args.rate_weight * rate_loss
            if training:
                assert scaler is not None
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch = clips.shape[0]
            meters["loss"].update(float(loss.detach()), batch)
            meters["reconstruction"].update(float(reconstruction_loss.detach()), batch)
            meters["rate"].update(float(rate_loss.detach()), batch)
            iterator.set_postfix(loss=f"{meters['loss'].average:.4f}", qp=qp)
    return {name: meter.average for name, meter in meters.items()}


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
    if not args.qps or any(qp < 0 or qp > 51 for qp in args.qps):
        raise ValueError("--qps must contain values in [0, 51]")
    if args.frame_size % 2:
        raise ValueError("--frame-size must be even for yuv420p H.264/H.265")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    require_ffmpeg(args.ffmpeg)
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_loader, val_loader = make_loaders(args)
    proxy = StandardCodecProxy(
        hidden_channels=args.hidden_channels,
        latent_channels=args.latent_channels,
        max_delta=args.max_delta,
    ).to(device)
    real_codec = StandardVideoCodec(
        args.codec,
        args.qps[0],
        fps=args.fps,
        preset=args.preset,
        ffmpeg=args.ffmpeg,
    )
    optimizer = AdamW(proxy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 1
    best_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        proxy.load_state_dict(checkpoint["proxy"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_val_loss", best_loss))

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
        )
        val_metrics = run_epoch(val_loader, proxy, real_codec, args, device)
        print(f"[epoch {epoch}/{args.epochs}] train={train_metrics} valid={val_metrics}")
        payload = {
            "epoch": epoch,
            "proxy": proxy.state_dict(),
            "proxy_config": proxy.config,
            "codec_config": {
                "codec": args.codec,
                "qps": list(args.qps),
                "fps": args.fps,
                "preset": args.preset,
            },
            "optimizer": optimizer.state_dict(),
            "best_val_loss": min(best_loss, val_metrics["loss"]),
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        save_checkpoint(output_dir / "last.pt", payload)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            payload["best_val_loss"] = best_loss
            save_checkpoint(output_dir / "best.pt", payload)
            print(f"[checkpoint] new best proxy loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
