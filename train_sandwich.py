"""Jointly train Video Swin pre- and FiLM-3D post-processing around a frozen codec."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam, AdamW, Optimizer
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
from preprocessing.provenance import environment_report, file_sha256, git_state, json_safe
from preprocessing.standard_codec import ffmpeg_version, require_ffmpeg
from preprocessing.utils import (
    AverageMeter,
    save_checkpoint,
    seed_everything,
    topk_correct,
    validate_run_directory,
    write_json,
)
from train import (
    PresetArgumentParser,
    load_or_evaluate_anchor_validation,
    make_loaders,
    validation_task_bd_rate,
)


def build_parser() -> argparse.ArgumentParser:
    # Keep this parser explicit.  Reusing train.py's parser previously exposed
    # rate controllers and masking flags that the sandwich trainer silently ignored.
    parser = PresetArgumentParser(description=__doc__, fromfile_prefix_chars="@")
    data = parser.add_argument_group("data")
    data.add_argument("--data-root", help="root containing train/ and optionally val/")
    data.add_argument("--train-dir")
    data.add_argument("--val-dir")
    data.add_argument("--train-split", default="train")
    data.add_argument("--val-split", default="val")
    data.add_argument("--val-ratio", type=float, default=0.2)
    data.add_argument("--frames", type=int, default=16)
    data.add_argument("--frame-stride", type=int, default=2)
    data.add_argument("--frame-size", type=int, default=128)
    data.add_argument("--limit-train", type=int)
    data.add_argument("--limit-val", type=int)
    data.add_argument("--workers", type=int, default=4)

    model = parser.add_argument_group("model")
    model.add_argument("--preprocessor", choices=("swin", "vit", "cnn"), default="swin")
    model.add_argument("--temporal-frames", type=int, default=8)
    model.add_argument("--vit-patch-size", type=int, default=8)
    model.add_argument("--vit-embed-dim", type=int, default=96)
    model.add_argument("--vit-depth", type=int, default=4)
    model.add_argument("--vit-heads", type=int, default=4)
    model.add_argument("--swin-patch-size", type=int, default=4)
    model.add_argument("--swin-embed-dim", type=int, default=48)
    model.add_argument("--swin-depth", type=int, default=4)
    model.add_argument("--swin-heads", type=int, default=4)
    model.add_argument("--swin-window-temporal", type=int, default=4)
    model.add_argument("--swin-window-spatial", type=int, default=8)
    model.add_argument(
        "--swin-qp-conditioning", action=argparse.BooleanOptionalAction, default=True
    )
    model.add_argument("--swin-qp-embed-dim", type=int, default=64)
    model.add_argument(
        "--swin-gated-smoothing", action=argparse.BooleanOptionalAction, default=False
    )
    model.add_argument("--swin-smoothing-max-strength", type=float, default=0.5)
    model.add_argument("--init-checkpoint", help="initialize weights for a new run")
    model.add_argument("--max-residual", type=float, default=0.25)
    model.add_argument("--analyzer", default="r3d_18")
    model.add_argument("--codec-qps", type=int, nargs="+", default=[30, 35, 40, 45])
    model.add_argument("--codec", choices=("h264", "h265"), default="h264")
    model.add_argument("--proxy-checkpoint", required=True)
    model.add_argument("--codec-fps", type=float, default=30.0)
    model.add_argument("--codec-preset", default="medium")
    model.add_argument("--ffmpeg", default="ffmpeg")
    model.add_argument("--codec-workers", type=int, default=2)
    model.add_argument("--ffmpeg-threads", type=int, default=1)
    model.add_argument(
        "--train-codec-source",
        choices=("real", "proxy"),
        default="real",
        help="real-forward/proxy-backward, or proxy-only pilot training",
    )
    model.add_argument(
        "--allow-unaudited-proxy",
        action="store_true",
        help="permit a proxy checkpoint without a passing proxy_audit (non-reportable)",
    )

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--epochs", type=int, default=30)
    optimization.add_argument("--batch-size", type=int, default=2)
    optimization.add_argument("--accumulation-steps", type=int, default=1)
    optimization.add_argument("--lr", type=float, default=1e-4)
    optimization.add_argument("--qp-sampling-weights", type=float, nargs="+")
    optimization.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    optimization.add_argument("--weight-decay", type=float, default=0.0)
    optimization.add_argument("--clip-grad", type=float, default=1.0)
    optimization.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    optimization.add_argument(
        "--gradient-audit-interval",
        type=int,
        default=0,
        help="measure weighted per-objective gradient norms every N training batches; 0 disables",
    )
    optimization.add_argument(
        "--max-proxy-clamp-fraction",
        type=float,
        default=0.05,
        help="abort before best-checkpoint selection if the epoch mean exceeds this fraction",
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--output-dir", default="checkpoints/adaptive_sandwich")
    runtime.add_argument("--resume")
    runtime.add_argument(
        "--checkpoint-metric",
        choices=("task_bd_rate", "loss", "top1", "ce"),
        default="task_bd_rate",
        help="primary best.pt metric; task BD-rate falls back to loss until first defined value",
    )
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--device", default="cuda")
    runtime.add_argument("--smoke-test", action="store_true")

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
    # train.make_loaders reads these two legacy controller fields.  They are
    # deliberately fixed and hidden because that controller is not implemented here.
    parser.set_defaults(rate_dual_control=False, controller_limit_val=None)
    return parser


RESUME_LOCKED_ARGUMENTS = (
    "data_root",
    "train_dir",
    "val_dir",
    "train_split",
    "val_split",
    "val_ratio",
    "frames",
    "frame_stride",
    "frame_size",
    "limit_train",
    "limit_val",
    "workers",
    "preprocessor",
    "temporal_frames",
    "vit_patch_size",
    "vit_embed_dim",
    "vit_depth",
    "vit_heads",
    "swin_patch_size",
    "swin_embed_dim",
    "swin_depth",
    "swin_heads",
    "swin_window_temporal",
    "swin_window_spatial",
    "swin_qp_conditioning",
    "swin_qp_embed_dim",
    "swin_gated_smoothing",
    "swin_smoothing_max_strength",
    "max_residual",
    "postprocessor",
    "post_channels",
    "post_bottleneck_channels",
    "post_blocks",
    "post_condition_channels",
    "post_max_residual",
    "task_input",
    "analyzer",
    "codec_qps",
    "codec",
    "codec_fps",
    "codec_preset",
    "ffmpeg",
    "ffmpeg_version",
    "codec_workers",
    "ffmpeg_threads",
    "train_codec_source",
    "allow_unaudited_proxy",
    "batch_size",
    "accumulation_steps",
    "lr",
    "qp_sampling_weights",
    "optimizer",
    "weight_decay",
    "clip_grad",
    "amp",
    "gradient_audit_interval",
    "max_proxy_clamp_fraction",
    "sandwich_rate_weight",
    "sandwich_task_weight",
    "dino_weight",
    "human_weight",
    "charbonnier_weight",
    "ms_ssim_weight",
    "lpips_weight",
    "lpips_backbone",
    "temporal_human_weight",
    "adaptive_dct_weight",
    "dct_block_size",
    "dct_high_frequency_start",
    "dino_model",
    "dino_repo",
    "dino_image_size",
    "dino_frame_stride",
    "checkpoint_metric",
    "seed",
    "smoke_test",
)


def _validate_proxy_checkpoint(args: argparse.Namespace) -> tuple[dict, str]:
    path = Path(args.proxy_checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"proxy checkpoint does not exist: {path}")
    digest = file_sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("proxy checkpoint must contain a dictionary payload")
    codec = checkpoint.get("codec_config", {})
    expected = {
        "codec": args.codec,
        "fps": float(args.codec_fps),
        "preset": args.codec_preset,
        "ffmpeg_threads": int(args.ffmpeg_threads),
        "ffmpeg_version": args.ffmpeg_version,
    }
    for name, requested in expected.items():
        trained = codec.get(name)
        if trained is None:
            if args.allow_unaudited_proxy:
                continue
            raise ValueError(
                f"proxy checkpoint has no {name!r} codec provenance; retrain it from a "
                "V10 cache or use --allow-unaudited-proxy only for diagnostics"
            )
        if name == "fps":
            matches = abs(float(trained) - requested) <= 1e-6
        else:
            matches = trained == requested
        if not matches:
            raise ValueError(
                f"proxy checkpoint {name}={trained!r}, but sandwich run requests {requested!r}"
            )
    trained_qps = {int(value) for value in codec.get("qps", [])}
    if not trained_qps and not args.allow_unaudited_proxy:
        raise ValueError("proxy checkpoint has no audited codec QPs")
    missing_qps = set(args.codec_qps) - trained_qps if trained_qps else set()
    if missing_qps:
        raise ValueError(f"proxy checkpoint was not audited/trained for QPs {sorted(missing_qps)}")
    trained_args = checkpoint.get("args", {})
    for name in ("frames", "frame_stride", "frame_size"):
        trained = trained_args.get(name)
        if trained is None and not args.allow_unaudited_proxy:
            raise ValueError(
                f"proxy checkpoint has no {name!r} training provenance; retrain it from a "
                "V10 cache or use --allow-unaudited-proxy only for diagnostics"
            )
        if trained is not None and int(trained) != int(getattr(args, name)):
            raise ValueError(
                f"proxy checkpoint {name}={trained}, but sandwich run requests {getattr(args, name)}"
            )
    audit = checkpoint.get("proxy_audit_reaudit", checkpoint.get("proxy_audit"))
    if not args.allow_unaudited_proxy and not (
        isinstance(audit, dict) and audit.get("feasible") is True
    ):
        raise ValueError(
            "proxy checkpoint has no passing proxy audit; use best_feasible.pt or pass "
            "--allow-unaudited-proxy only for a clearly non-reportable diagnostic run"
        )
    return checkpoint, digest


def _validate_resume_configuration(
    checkpoint: dict, args: argparse.Namespace, proxy_sha256: str
) -> None:
    if int(checkpoint.get("format_version", 0)) < 10:
        raise ValueError(
            "exact resume requires a V10 checkpoint; use the V9 checkpoint with "
            "--init-checkpoint in a new output directory"
        )
    saved = checkpoint.get("args", {})
    mismatches = []
    for name in RESUME_LOCKED_ARGUMENTS:
        if name in saved and saved[name] != getattr(args, name):
            mismatches.append(f"{name}: checkpoint={saved[name]!r}, CLI={getattr(args, name)!r}")
    if checkpoint.get("proxy_sha256") != proxy_sha256:
        mismatches.append("proxy checkpoint SHA-256 changed")
    if mismatches:
        raise ValueError(
            "resume changed locked run configuration; use --init-checkpoint for a new run: "
            + "; ".join(mismatches)
        )


def _capture_rng_state(qp_rng: random.Random) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "qp": qp_rng.getstate(),
    }


def _restore_rng_state(state: dict, qp_rng: random.Random) -> None:
    if not state:
        raise ValueError("V10 resume checkpoint is missing RNG state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    qp_rng.setstate(state["qp"])


def initial_selection_state(requested_metric: str) -> dict:
    return {
        "requested_metric": requested_metric,
        "best_loss": math.inf,
        "best_ce": math.inf,
        "best_top1": -math.inf,
        "best_task_bd_rate": math.inf,
        "has_valid_task_bd_rate": False,
        "primary_metric_used": None,
        "primary_value": None,
        "primary_epoch": None,
        "fallback_used": False,
    }


def update_selection_state(
    state: dict, validation: dict[str, float], epoch: int
) -> tuple[dict, set[str]]:
    """Update metric-specific and primary checkpoint state without filesystem effects."""

    updated = dict(state)
    saves: set[str] = set()
    loss = float(validation["loss"])
    ce = float(validation["task"])
    top1 = float(validation["top1"])
    raw_bd_rate = validation.get("task_bd_rate_percent")
    bd_rate = (
        float(raw_bd_rate)
        if raw_bd_rate is not None and math.isfinite(float(raw_bd_rate))
        else None
    )
    old_has_valid_bd = bool(state["has_valid_task_bd_rate"])

    if loss < float(state["best_loss"]):
        updated["best_loss"] = loss
        saves.add("best_loss.pt")
    if ce < float(state["best_ce"]):
        updated["best_ce"] = ce
        saves.add("best_ce.pt")
    if top1 > float(state["best_top1"]):
        updated["best_top1"] = top1
        saves.add("best_top1.pt")
    if bd_rate is not None:
        updated["has_valid_task_bd_rate"] = True
        if bd_rate < float(state["best_task_bd_rate"]):
            updated["best_task_bd_rate"] = bd_rate
            saves.add("best_task_bd_rate.pt")

    requested = str(state["requested_metric"])
    primary_improved = False
    metric_used = requested
    value: float
    fallback = False
    if requested == "task_bd_rate":
        if bd_rate is not None:
            value = bd_rate
            primary_improved = (
                not old_has_valid_bd or bd_rate < float(state["best_task_bd_rate"])
            )
        elif not old_has_valid_bd:
            metric_used = "loss_fallback"
            value = loss
            fallback = True
            previous = state["primary_value"]
            primary_improved = previous is None or loss < float(previous)
        else:
            value = math.inf
    elif requested == "loss":
        value = loss
        primary_improved = loss < float(state["best_loss"])
    elif requested == "ce":
        value = ce
        primary_improved = ce < float(state["best_ce"])
    elif requested == "top1":
        value = top1
        primary_improved = top1 > float(state["best_top1"])
    else:  # Defensive guard for callers that bypass argparse.
        raise ValueError(f"unsupported checkpoint metric: {requested}")

    if primary_improved:
        updated.update(
            {
                "primary_metric_used": metric_used,
                "primary_value": value,
                "primary_epoch": int(epoch),
                "fallback_used": fallback,
            }
        )
        saves.add("best.pt")
    return updated, saves


def build_optimizer(args: argparse.Namespace, parameters) -> Optimizer:
    optimizer_class = AdamW if args.optimizer == "adamw" else Adam
    return optimizer_class(parameters, lr=args.lr, weight_decay=args.weight_decay)


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
        codec_workers=args.codec_workers,
        ffmpeg_threads=args.ffmpeg_threads,
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
    # Invalidate the side-channel before every call so a bridge that forgets to
    # refresh it cannot silently reuse proxy BPP from an earlier batch or QP.
    model.codec.last_proxy_bpp = None
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
    zero = clips.new_zeros((), dtype=torch.float32)
    diagnostics = getattr(model.codec, "last_proxy_diagnostics", {}) if training else {}
    for name in (
        "proxy_clamp_fraction",
        "proxy_below_zero_fraction",
        "proxy_above_one_fraction",
    ):
        value = diagnostics.get(name, zero)
        losses[name] = value.float() if isinstance(value, torch.Tensor) else zero + float(value)
    proxy_bpp = getattr(model.codec, "last_proxy_bpp", None)
    codec_name = type(model.codec).__name__
    if not isinstance(proxy_bpp, torch.Tensor):
        # Missing per-forward state is a runtime bridge failure, not caller type misuse.
        raise RuntimeError(  # noqa: TRY004
            f"{codec_name} did not update last_proxy_bpp at qp={qp}; "
            "proxy-drift tracking requires proxy BPP from every forward"
        )
    if proxy_bpp.shape[:1] != clips.shape[:1]:
        raise RuntimeError(
            f"{codec_name} reported proxy BPP with shape {tuple(proxy_bpp.shape)} "
            f"for batch size {clips.shape[0]} at qp={qp}; "
            "last_proxy_bpp is stale or mis-shaped"
        )
    if not bool(torch.isfinite(proxy_bpp).all()):
        raise RuntimeError(f"{codec_name} produced non-finite proxy BPP at qp={qp}")
    losses["proxy_bpp"] = proxy_bpp.detach().float().mean()
    losses["preprocessor_boundary_fraction"] = (
        (output.neural_code <= 0.0) | (output.neural_code >= 1.0)
    ).float().mean()
    losses["postprocessor_boundary_fraction"] = (
        (output.restored <= 0.0) | (output.restored >= 1.0)
    ).float().mean()
    losses["logits"] = logits
    return losses


def _gradient_norm(value: torch.Tensor, parameters: list[torch.nn.Parameter]) -> float:
    if not value.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(
        value, parameters, retain_graph=True, allow_unused=True
    )
    squared = sum(
        float(gradient.detach().float().square().sum())
        for gradient in gradients
        if gradient is not None
    )
    return math.sqrt(squared)


def objective_gradient_norms(
    losses: dict[str, torch.Tensor],
    model: AdaptiveVideoSandwich,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Measure weighted objective gradients for sparse, opt-in pilot audits."""

    parameters = [parameter for parameter in model.trainable_parameters() if parameter.requires_grad]
    terms = {
        "gradient_rate": args.sandwich_rate_weight * losses["rate_ratio"],
        "gradient_task": args.sandwich_task_weight * losses["task"],
        "gradient_dino": args.dino_weight * losses["dino"],
        "gradient_human": args.human_weight * losses["human"],
    }
    return {name: _gradient_norm(value, parameters) for name, value in terms.items()}


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
    optimizer: Optimizer | None = None,
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
            "rate",
            "proxy_bpp",
            "rate_ratio",
            "task",
            "dino",
            "human",
            "charbonnier",
            "ms_ssim",
            "lpips",
            "temporal",
            "adaptive_dct",
            "proxy_clamp_fraction",
            "proxy_below_zero_fraction",
            "proxy_above_one_fraction",
            "preprocessor_boundary_fraction",
            "postprocessor_boundary_fraction",
            "gradient_rate",
            "gradient_task",
            "gradient_dino",
            "gradient_human",
            "gradient_total",
        )
    }
    per_qp = {
        qp: {
            "bpp": AverageMeter(),
            "proxy_bpp": AverageMeter(),
            "top1": 0,
            "top5": 0,
            "examples": 0,
        }
        for qp in args.codec_qps
    }
    correct1 = correct5 = examples = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    iterator = tqdm(loader, desc="sandwich train" if training else "sandwich valid", leave=False)
    context = torch.enable_grad if training else torch.no_grad
    epoch_started = time.perf_counter()
    batches = 0
    with context():
        for step, (clips, labels) in enumerate(iterator, start=1):
            batches += 1
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                if qp_rng is None or scaler is None:
                    raise ValueError("training requires QP RNG and gradient scaler")
                qps = [
                    qp_rng.choices(
                        args.codec_qps, weights=args.qp_sampling_weights, k=1
                    )[0]
                    if args.qp_sampling_weights is not None
                    else qp_rng.choice(args.codec_qps)
                ]
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
                    if (
                        args.gradient_audit_interval > 0
                        and (step - 1) % args.gradient_audit_interval == 0
                    ):
                        for name, value in objective_gradient_norms(
                            losses, model, args
                        ).items():
                            meters[name].update(value)
                    scaler.scale(losses["total"] / args.accumulation_steps).backward()
                    update = step % args.accumulation_steps == 0 or step == len(loader)
                    if update:
                        scaler.unscale_(optimizer)
                        total_gradient = torch.nn.utils.clip_grad_norm_(
                            list(model.trainable_parameters()), args.clip_grad
                        )
                        meters["gradient_total"].update(float(total_gradient))
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                batch = labels.numel()
                meters["loss"].update(float(losses["total"].detach()), batch)
                for key in meters.keys() - {
                    "loss",
                    "gradient_rate",
                    "gradient_task",
                    "gradient_dino",
                    "gradient_human",
                    "gradient_total",
                }:
                    meters[key].update(float(losses[key].detach()), batch)
                logits = losses["logits"].detach()
                top1 = topk_correct(logits, labels, 1)
                top5 = topk_correct(logits, labels, 5)
                correct1 += top1
                correct5 += top5
                examples += batch
                if not training:
                    per_qp[qp]["bpp"].update(float(losses["rate"].detach()), batch)
                    per_qp[qp]["proxy_bpp"].update(
                        float(losses["proxy_bpp"].detach()), batch
                    )
                    per_qp[qp]["top1"] += top1
                    per_qp[qp]["top5"] += top5
                    per_qp[qp]["examples"] += batch
            iterator.set_postfix(loss=f"{meters['loss'].average:.4f}")
    metrics = {name: meter.average for name, meter in meters.items()}
    metrics["bpp"] = metrics["rate"]
    metrics.update(
        {"top1": correct1 / max(examples, 1), "top5": correct5 / max(examples, 1)}
    )
    if not training:
        for qp, values in per_qp.items():
            count = max(int(values["examples"]), 1)
            metrics[f"qp{qp}_bpp"] = values["bpp"].average
            metrics[f"qp{qp}_proxy_bpp"] = values["proxy_bpp"].average
            metrics[f"qp{qp}_top1"] = float(values["top1"]) / count
            metrics[f"qp{qp}_top5"] = float(values["top5"]) / count
    elapsed = time.perf_counter() - epoch_started
    metrics["epoch_seconds"] = elapsed
    metrics["seconds_per_batch"] = elapsed / max(batches, 1)
    metrics["batches"] = batches
    return metrics


def _checkpoint_payload(
    model: AdaptiveVideoSandwich,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    qp_rng: random.Random,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    metrics: dict[str, float],
    selection_state: dict,
    proxy_sha256: str,
    scientific_status: str,
) -> dict:
    post_config = getattr(model.postprocessor, "config", {"architecture": "identity"})
    return {
        "format_version": 10,
        "epoch": epoch,
        "global_step": global_step,
        "preprocessor": model.preprocessor.state_dict(),
        "postprocessor": model.postprocessor.state_dict(),
        "postprocessor_config": post_config,
        "optimizer": optimizer.state_dict(),
        "optimizer_name": args.optimizer,
        "scaler": scaler.state_dict(),
        "rng_state": _capture_rng_state(qp_rng),
        "args": vars(args),
        "validation": metrics,
        "selection_state": selection_state,
        "proxy_sha256": proxy_sha256,
        "codec_config": {
            "codec": args.codec,
            "qps": list(args.codec_qps),
            "fps": args.codec_fps,
            "preset": args.codec_preset,
            "ffmpeg": args.ffmpeg,
            "ffmpeg_version": args.ffmpeg_version,
            "ffmpeg_threads": args.ffmpeg_threads,
        },
        "scientific_status": scientific_status,
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_run_directory(args.output_dir, args.resume)
    if args.init_checkpoint and args.resume:
        raise ValueError("choose --init-checkpoint for a new run or --resume for continuation")
    if args.smoke_test:
        args.epochs = 1
        args.limit_train = min(args.limit_train or 8, 8)
        args.limit_val = min(args.limit_val or 4, 4)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if (
        not args.codec_qps
        or len(set(args.codec_qps)) != len(args.codec_qps)
        or any(qp < 0 or qp > 51 for qp in args.codec_qps)
    ):
        raise ValueError("--codec-qps must contain unique values in [0, 51]")
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation-steps must be positive")
    if args.gradient_audit_interval < 0:
        raise ValueError("--gradient-audit-interval must be non-negative")
    if not 0.0 <= args.max_proxy_clamp_fraction <= 1.0:
        raise ValueError("--max-proxy-clamp-fraction must be in [0, 1]")
    if args.qp_sampling_weights is not None:
        if len(args.qp_sampling_weights) != len(args.codec_qps):
            raise ValueError("--qp-sampling-weights must match --codec-qps")
        if any(not math.isfinite(value) or value < 0 for value in args.qp_sampling_weights):
            raise ValueError("--qp-sampling-weights must be finite and non-negative")
        if not any(value > 0 for value in args.qp_sampling_weights):
            raise ValueError("at least one QP sampling weight must be positive")
    weights = (
        args.sandwich_rate_weight,
        args.sandwich_task_weight,
        args.dino_weight,
        args.human_weight,
    )
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("sandwich objective weights must be finite and non-negative")
    require_ffmpeg(args.ffmpeg)
    args.ffmpeg_version = ffmpeg_version(args.ffmpeg)
    seed_everything(args.seed)
    device = torch.device(args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _, proxy_sha256 = _validate_proxy_checkpoint(args)
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        _validate_resume_configuration(resume_checkpoint, args, proxy_sha256)
    repository = git_state(Path(__file__).resolve().parent)
    write_json(output / "args.json", json_safe(vars(args)))
    write_json(output / "git_state.json", repository)
    (output / "git_commit.txt").write_text(str(repository["commit"]) + "\n", encoding="utf-8")
    (output / "proxy_sha256.txt").write_text(proxy_sha256 + "\n", encoding="utf-8")
    (output / "environment.txt").write_text(
        environment_report(args.ffmpeg), encoding="utf-8"
    )
    model, analyzer, dino, human = build_models(args, device)
    train_loader, val_loader = make_loaders(args, analyzer.categories)
    anchor_metrics = load_or_evaluate_anchor_validation(
        output, val_loader, model.codec, analyzer, args, device
    )
    optimizer = build_optimizer(args, model.trainable_parameters())
    scaler = torch.amp.GradScaler(
        device.type, enabled=bool(args.amp and device.type == "cuda")
    )
    qp_rng = random.Random(args.seed + 909)
    start_epoch = 1
    global_step = 0
    history: list[dict] = []
    selection_state = initial_selection_state(args.checkpoint_metric)
    if args.resume:
        checkpoint = resume_checkpoint
        if checkpoint is None:  # pragma: no cover - guarded by args.resume
            raise RuntimeError("resume checkpoint was not loaded")
        model.preprocessor.load_state_dict(checkpoint["preprocessor"])
        model.postprocessor.load_state_dict(checkpoint["postprocessor"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" not in checkpoint or "selection_state" not in checkpoint:
            raise ValueError("V10 resume checkpoint is missing scaler or selection state")
        scaler.load_state_dict(checkpoint["scaler"])
        selection_state = dict(checkpoint["selection_state"])
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
        history_path = output / "history.json"
        if not history_path.is_file():
            raise ValueError("exact resume requires the original history.json")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        _restore_rng_state(checkpoint.get("rng_state", {}), qp_rng)
        print(
            f"[resume] epoch={start_epoch} primary={selection_state['primary_metric_used']} "
            f"value={selection_state['primary_value']} epoch={selection_state['primary_epoch']}"
        )
    if args.epochs < start_epoch:
        raise ValueError(f"--epochs must be at least {start_epoch} when resuming")

    non_reportable = bool(
        args.smoke_test
        or args.allow_unaudited_proxy
        or args.train_codec_source != "real"
        or args.limit_train is not None
        or args.limit_val is not None
    )
    scientific_status = "non_reportable_diagnostic" if non_reportable else "measured_by_this_run"
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
        global_step += int(train_metrics["batches"])
        val_metrics = run_epoch(
            val_loader, model, analyzer, dino, human, args, device, anchor_metrics
        )
        bd_rate = validation_task_bd_rate(anchor_metrics, val_metrics, args.codec_qps)
        val_metrics["task_bd_rate_percent"] = bd_rate
        clamp_fraction = float(train_metrics["proxy_clamp_fraction"])
        clamp_passed = clamp_fraction <= args.max_proxy_clamp_fraction
        if clamp_passed:
            selection_state, selected_files = update_selection_state(
                selection_state, val_metrics, epoch
            )
        else:
            selected_files = set()
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "selection": dict(selection_state),
            "proxy_clamp_gate": {
                "passed": clamp_passed,
                "value": clamp_fraction,
                "maximum": args.max_proxy_clamp_fraction,
            },
        }
        history.append(record)
        payload = _checkpoint_payload(
            model,
            optimizer,
            scaler,
            qp_rng,
            epoch,
            global_step,
            args,
            val_metrics,
            selection_state,
            proxy_sha256,
            "proxy_clamp_gate_failed" if not clamp_passed else scientific_status,
        )
        save_checkpoint(output / "last.pt", payload)
        for filename in selected_files:
            save_checkpoint(output / filename, payload)
        write_json(output / "history.json", json_safe(history))
        total_epoch_seconds = train_metrics["epoch_seconds"] + val_metrics["epoch_seconds"]
        remaining_hours = total_epoch_seconds * max(args.epochs - epoch, 0) / 3600.0
        write_json(
            output / "compute_profile.json",
            {
                "latest_epoch": epoch,
                "train_seconds_per_batch": train_metrics["seconds_per_batch"],
                "validation_seconds_per_batch": val_metrics["seconds_per_batch"],
                "latest_epoch_seconds": total_epoch_seconds,
                "estimated_remaining_hours": remaining_hours,
                "codec_workers": args.codec_workers,
                "ffmpeg_threads": args.ffmpeg_threads,
            },
        )
        print(
            f"epoch={epoch} val_loss={val_metrics['loss']:.5f} "
            f"task_bd_rate={bd_rate if bd_rate is not None else 'undefined'} "
            f"proxy_clamp={clamp_fraction:.2%} eta={remaining_hours:.2f}h"
        )
        if not clamp_passed:
            raise RuntimeError(
                f"proxy clamp gate failed: epoch mean {clamp_fraction:.2%} exceeds "
                f"{args.max_proxy_clamp_fraction:.2%}; last.pt was saved but no best "
                "checkpoint was selected"
            )


if __name__ == "__main__":
    main()
