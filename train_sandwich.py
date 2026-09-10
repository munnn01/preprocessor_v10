"""Jointly train Video Swin pre- and FiLM-3D post-processing around a frozen codec."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm import tqdm

from preprocessing import (
    AdaptiveVideoSandwich,
    FrozenDinoV2,
    FrozenVideoAnalyzer,
    HumanPerceptualObjective,
    ParallelStandardVideoCodec,
    StandardCodecProxy,
    StandardVideoCodec,
    adaptive_vcm_objective,
    build_postprocessor,
    build_preprocessor,
    dino_video_loss,
)
from preprocessing.utils import AverageMeter, save_checkpoint, seed_everything, topk_correct, write_json
from train import (
    build_parser as build_preprocessor_parser,
    load_or_evaluate_anchor_validation,
    make_loaders,
    validation_task_bd_rate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_preprocessor_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir="checkpoints/adaptive_sandwich", checkpoint_metric="task_bd_rate")
    post = parser.add_argument_group("sandwich postprocessor")
    post.add_argument("--postprocessor", choices=("film3d", "identity"), default="film3d")
    post.add_argument("--post-channels", type=int, default=32)
    post.add_argument("--post-bottleneck-channels", type=int, default=48)
    post.add_argument("--post-blocks", type=int, default=2)
    post.add_argument("--post-condition-channels", type=int, default=48)
    post.add_argument("--post-max-residual", type=float, default=0.25)
    post.add_argument(
        "--task-input",
        choices=("codec", "postprocessed"),
        default="postprocessed",
        help="route the decoded neural code or the restored video to the task analyzer",
    )

    objective = parser.add_argument_group("adaptive VCM objective")
    objective.add_argument("--sandwich-rate-weight", type=float, default=0.05)
    objective.add_argument("--sandwich-task-weight", type=float, default=1.0)
    objective.add_argument("--dino-weight", type=float, default=0.25)
    objective.add_argument("--human-weight", type=float, default=1.0)
    objective.add_argument("--charbonnier-weight", type=float, default=0.25)
    objective.add_argument("--ms-ssim-weight", type=float, default=1.0)
    objective.add_argument("--lpips-weight", type=float, default=0.0)
    objective.add_argument("--lpips-backbone", choices=("alex", "vgg", "squeeze"), default="alex")
    objective.add_argument("--temporal-human-weight", type=float, default=0.1)
    objective.add_argument("--adaptive-dct-weight", type=float, default=0.05)
    objective.add_argument("--dct-block-size", type=int, default=8)
    objective.add_argument("--dct-high-frequency-start", type=int, default=6)
    objective.add_argument("--dino-model", default="dinov2_vits14")
    objective.add_argument("--dino-repo", default="facebookresearch/dinov2")
    objective.add_argument("--dino-image-size", type=int, default=224)
    objective.add_argument("--dino-frame-stride", type=int, default=2)
    return parser


def build_models(args: argparse.Namespace, device: torch.device):
    analyzer = FrozenVideoAnalyzer(args.analyzer).to(device).eval()
    preprocessor = build_preprocessor(
        args.preprocessor,
        temporal_frames=args.temporal_frames,
        patch_size=args.vit_patch_size,
        embed_dim=args.vit_embed_dim,
        depth=args.vit_depth,
        num_heads=args.vit_heads,
        swin_patch_size=args.swin_patch_size,
        swin_embed_dim=args.swin_embed_dim,
        swin_depth=args.swin_depth,
        swin_num_heads=args.swin_heads,
        swin_window_size=(
            args.swin_window_temporal,
            args.swin_window_spatial,
            args.swin_window_spatial,
        ),
        swin_qp_conditioning=args.swin_qp_conditioning,
        swin_qp_embed_dim=args.swin_qp_embed_dim,
        swin_gated_smoothing=args.swin_gated_smoothing,
        swin_smoothing_max_strength=args.swin_smoothing_max_strength,
        max_residual=args.max_residual,
    )
    postprocessor = build_postprocessor(
        args.postprocessor,
        channels=args.post_channels,
        bottleneck_channels=args.post_bottleneck_channels,
        blocks=args.post_blocks,
        condition_channels=args.post_condition_channels,
        max_residual=args.post_max_residual,
    )
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        preprocessor.load_state_dict(initial["preprocessor"])
        if "postprocessor" in initial:
            postprocessor.load_state_dict(initial["postprocessor"])

    proxy = StandardCodecProxy.from_checkpoint(args.proxy_checkpoint)
    real_codec = StandardVideoCodec(
        args.codec,
        args.codec_qps[0],
        fps=args.codec_fps,
        preset=args.codec_preset,
        ffmpeg=args.ffmpeg,
    )
    codec = ParallelStandardVideoCodec(real_codec, proxy)
    sandwich = AdaptiveVideoSandwich(
        preprocessor, codec, postprocessor, task_input=args.task_input
    ).to(device)
    dino = None
    if args.dino_weight > 0:
        dino = FrozenDinoV2(
            args.dino_model,
            repo_or_dir=args.dino_repo,
            image_size=args.dino_image_size,
            frame_stride=args.dino_frame_stride,
        ).to(device)
    human = HumanPerceptualObjective(
        charbonnier_weight=args.charbonnier_weight,
        ms_ssim_weight=args.ms_ssim_weight,
        lpips_weight=args.lpips_weight,
        temporal_weight=args.temporal_human_weight,
        dct_weight=args.adaptive_dct_weight,
        dct_block_size=args.dct_block_size,
        dct_high_frequency_start=args.dct_high_frequency_start,
        lpips_backbone=args.lpips_backbone,
    ).to(device)
    return sandwich, analyzer, dino, human


def forward_losses(
    clips: torch.Tensor,
    labels: torch.Tensor,
    qp: int,
    model: AdaptiveVideoSandwich,
    analyzer: FrozenVideoAnalyzer,
    dino: FrozenDinoV2 | None,
    human: HumanPerceptualObjective,
    args: argparse.Namespace,
    *,
    training: bool,
    anchor_bpp: float,
) -> dict[str, torch.Tensor]:
    use_amp = bool(args.amp and clips.device.type == "cuda")
    device_type = clips.device.type
    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
        output = model(
            clips,
            qp,
            codec_source=args.train_codec_source if training else "real",
            use_proxy_gradient=training,
        )
        logits = analyzer(output.machine_view)
        task_loss = F.cross_entropy(logits.float(), labels)
        dino_loss = clips.new_zeros((), dtype=torch.float32)
        if dino is not None:
            dino_loss = dino_video_loss(dino, output.machine_view, clips)
    with torch.autocast(device_type=device_type, enabled=False):
        human_terms = human(output.restored.float(), clips.float(), output.neural_code.float())
        losses = adaptive_vcm_objective(
            bpp=output.bpp,
            task_loss=task_loss,
            dino_loss=dino_loss,
            human_terms=human_terms,
            anchor_bpp=anchor_bpp,
            rate_weight=args.sandwich_rate_weight,
            task_weight=args.sandwich_task_weight,
            dino_weight=args.dino_weight,
            human_weight=args.human_weight,
        )
    losses["logits"] = logits
    return losses


def run_epoch(
    loader,
    model: AdaptiveVideoSandwich,
    analyzer: FrozenVideoAnalyzer,
    dino: FrozenDinoV2 | None,
    human: HumanPerceptualObjective,
    args: argparse.Namespace,
    device: torch.device,
    anchor_metrics: dict[str, float],
    *,
    optimizer: AdamW | None = None,
    scaler: torch.amp.GradScaler | None = None,
    qp_rng: random.Random | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    analyzer.eval()
    if dino is not None:
        dino.eval()
    human.eval()
    meters = {
        name: AverageMeter()
        for name in (
            "loss",
            "bpp",
            "rate_ratio",
            "task",
            "dino",
            "human",
            "charbonnier",
            "ms_ssim",
            "lpips",
            "temporal",
            "adaptive_dct",
        )
    }
    per_qp = {
        qp: {"bpp": AverageMeter(), "top1": 0, "top5": 0, "examples": 0}
        for qp in args.codec_qps
    }
    correct1 = correct5 = examples = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    iterator = tqdm(loader, desc="sandwich train" if training else "sandwich valid", leave=False)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for step, (clips, labels) in enumerate(iterator, start=1):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                if qp_rng is None or scaler is None:
                    raise ValueError("training requires QP RNG and gradient scaler")
                qps = [qp_rng.choice(args.codec_qps)]
            else:
                qps = args.codec_qps
            for qp in qps:
                losses = forward_losses(
                    clips,
                    labels,
                    qp,
                    model,
                    analyzer,
                    dino,
                    human,
                    args,
                    training=training,
                    anchor_bpp=float(anchor_metrics[f"qp{qp}_bpp"]),
                )
                if training:
                    scaler.scale(losses["total"] / args.accumulation_steps).backward()
                    update = step % args.accumulation_steps == 0 or step == len(loader)
                    if update:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(model.trainable_parameters()), args.clip_grad
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                batch = labels.numel()
                meters["loss"].update(float(losses["total"].detach()), batch)
                for key in meters.keys() - {"loss"}:
                    meters[key].update(float(losses[key].detach()), batch)
                logits = losses["logits"].detach()
                top1 = topk_correct(logits, labels, 1)
                top5 = topk_correct(logits, labels, 5)
                correct1 += top1
                correct5 += top5
                examples += batch
                if not training:
                    per_qp[qp]["bpp"].update(float(losses["rate"].detach()), batch)
                    per_qp[qp]["top1"] += top1
                    per_qp[qp]["top5"] += top5
                    per_qp[qp]["examples"] += batch
            iterator.set_postfix(loss=f"{meters['loss'].average:.4f}")
    metrics = {name: meter.average for name, meter in meters.items()}
    metrics.update(
        {"top1": correct1 / max(examples, 1), "top5": correct5 / max(examples, 1)}
    )
    if not training:
        for qp, values in per_qp.items():
            count = max(int(values["examples"]), 1)
            metrics[f"qp{qp}_bpp"] = values["bpp"].average
            metrics[f"qp{qp}_top1"] = float(values["top1"]) / count
            metrics[f"qp{qp}_top5"] = float(values["top5"]) / count
    return metrics


def _checkpoint_payload(
    model: AdaptiveVideoSandwich,
    optimizer: AdamW,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> dict:
    post_config = getattr(model.postprocessor, "config", {"architecture": "identity"})
    return {
        "format_version": 9,
        "epoch": epoch,
        "preprocessor": model.preprocessor.state_dict(),
        "postprocessor": model.postprocessor.state_dict(),
        "postprocessor_config": post_config,
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "validation": metrics,
        "scientific_status": "measured_by_this_run",
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not args.codec_qps or len(set(args.codec_qps)) != len(args.codec_qps):
        raise ValueError("--codec-qps must contain unique values")
    weights = (
        args.sandwich_rate_weight,
        args.sandwich_task_weight,
        args.dino_weight,
        args.human_weight,
    )
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("sandwich objective weights must be finite and non-negative")
    seed_everything(args.seed)
    device = torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model, analyzer, dino, human = build_models(args, device)
    train_loader, val_loader = make_loaders(args, analyzer.categories)
    anchor_metrics = load_or_evaluate_anchor_validation(
        output, val_loader, model.codec, analyzer, args, device
    )
    optimizer = AdamW(
        model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch = 1
    history: list[dict] = []
    best_bd_rate = math.inf
    best_loss = math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.preprocessor.load_state_dict(checkpoint["preprocessor"])
        model.postprocessor.load_state_dict(checkpoint["postprocessor"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history_path = output / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
    scaler = torch.amp.GradScaler(
        device.type, enabled=bool(args.amp and device.type == "cuda")
    )
    qp_rng = random.Random(args.seed + 909)
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            train_loader,
            model,
            analyzer,
            dino,
            human,
            args,
            device,
            anchor_metrics,
            optimizer=optimizer,
            scaler=scaler,
            qp_rng=qp_rng,
        )
        val_metrics = run_epoch(
            val_loader, model, analyzer, dino, human, args, device, anchor_metrics
        )
        bd_rate = validation_task_bd_rate(anchor_metrics, val_metrics, args.codec_qps)
        val_metrics["task_bd_rate_percent"] = bd_rate
        record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(record)
        payload = _checkpoint_payload(model, optimizer, epoch, args, val_metrics)
        save_checkpoint(output / "last.pt", payload)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            save_checkpoint(output / "best_loss.pt", payload)
        if bd_rate is not None and bd_rate < best_bd_rate:
            best_bd_rate = bd_rate
            save_checkpoint(output / "best_task_bd_rate.pt", payload)
            save_checkpoint(output / "best.pt", payload)
        elif not (output / "best.pt").is_file():
            save_checkpoint(output / "best.pt", payload)
        write_json(output / "history.json", history)
        print(
            f"epoch={epoch} val_loss={val_metrics['loss']:.5f} "
            f"task_bd_rate={bd_rate if bd_rate is not None else 'undefined'}"
        )


if __name__ == "__main__":
    main()
