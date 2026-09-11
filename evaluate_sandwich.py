"""Evaluate anchor, pre-only, and full sandwich paths through real H.264/H.265."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm

from preprocessing import (
    FrozenVideoAnalyzer,
    LPIPSLoss,
    StandardVideoCodec,
    multiscale_ssim_loss,
    postprocessor_from_checkpoint,
    preprocessor_from_checkpoint,
)
from preprocessing.evaluation import (
    bootstrap_bd_rate,
    build_evaluation_dataset,
    calculate_bd_rate_details,
    dataset_sample_path,
)
from preprocessing.provenance import git_state, write_run_provenance
from preprocessing.standard_codec import require_ffmpeg
from preprocessing.utils import topk_correct, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--test-dir")
    parser.add_argument("--split", default="val")
    parser.add_argument("--codecs", nargs="+", choices=("h264", "h265"), default=["h264", "h265"])
    parser.add_argument("--qps", nargs="+", type=int, default=[30, 32, 35, 37, 40, 42, 45])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--preset")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--codec-workers", type=int, default=2)
    parser.add_argument("--ffmpeg-threads", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--lpips",
        action="store_true",
        help="evaluate the frozen LPIPS distance (requires the research extra)",
    )
    parser.add_argument("--lpips-backbone", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument(
        "--psnr-aggregation",
        choices=("psnr_from_mean_video_mse", "mean_video_psnr"),
        default="psnr_from_mean_video_mse",
        help="one predeclared estimator used by summary, BD-rate, and bootstrap",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--output-dir", default="outputs/sandwich_real_codec")
    return parser.parse_args()


def quality_metrics(
    decoded: torch.Tensor,
    source: torch.Tensor,
    lpips_metric: LPIPSLoss | None,
) -> dict[str, float]:
    mse = float(F.mse_loss(decoded, source))
    metrics = {
        "mse": mse,
        "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
        "ms_ssim": 1.0 - float(multiscale_ssim_loss(decoded, source)),
    }
    if lpips_metric is not None:
        with torch.no_grad():
            distance = float(lpips_metric(decoded.float(), source.float()))
        metrics.update({"lpips_distance": distance, "lpips_quality": -distance})
    return metrics


def aggregate_operating_points(
    rows: list[dict], *, psnr_aggregation: str, include_lpips: bool
) -> list[dict]:
    if psnr_aggregation not in {"psnr_from_mean_video_mse", "mean_video_psnr"}:
        raise ValueError("unknown PSNR aggregation")
    metric_names = ["bpp", "mse", "psnr_db", "ms_ssim", "top1", "top5"]
    if include_lpips:
        metric_names.extend(("lpips_distance", "lpips_quality"))
    totals: dict[tuple[str, int, str], dict[str, float]] = defaultdict(
        lambda: {"videos": 0.0, **{metric: 0.0 for metric in metric_names}}
    )
    for row in rows:
        key = (str(row["codec"]), int(row["qp"]), str(row["method"]))
        total = totals[key]
        total["videos"] += 1
        for metric in metric_names:
            total[metric] += float(row[metric])

    summaries = []
    for (codec_name, qp, method), total in sorted(totals.items()):
        count = total["videos"]
        mean_mse = total["mse"] / count
        psnr = (
            -10.0 * math.log10(max(mean_mse, 1e-12))
            if psnr_aggregation == "psnr_from_mean_video_mse"
            else total["psnr_db"] / count
        )
        summary = {
            "codec": codec_name,
            "qp": qp,
            "method": method,
            "videos": int(count),
            "bpp": total["bpp"] / count,
            "mse": mean_mse,
            "psnr_db": psnr,
            "ms_ssim": total["ms_ssim"] / count,
            "top1_percent": 100.0 * total["top1"] / count,
            "top5_percent": 100.0 * total["top5"] / count,
        }
        if include_lpips:
            summary["lpips_distance"] = total["lpips_distance"] / count
            summary["lpips_quality"] = total["lpips_quality"] / count
        summaries.append(summary)
    return summaries


def calculate_all_bd_rates(
    summary_rows: list[dict], codecs: list[str], *, include_lpips: bool
) -> dict:
    metric_names = ["top1_percent", "psnr_db", "ms_ssim"]
    if include_lpips:
        metric_names.append("lpips_quality")
    output = {}
    for codec_name in codecs:
        codec_rows = [row for row in summary_rows if row["codec"] == codec_name]
        output[codec_name] = {}
        for method in ("pre_only", "sandwich"):
            output[codec_name][method] = {}
            for metric in metric_names:
                output[codec_name][method][metric] = {
                    "primary": calculate_bd_rate_details(
                        codec_rows,
                        metric,
                        anchor_method="anchor",
                        proposed_method=method,
                        apply_monotone_envelope=True,
                    ),
                    "raw_curve_sensitivity": calculate_bd_rate_details(
                        codec_rows,
                        metric,
                        anchor_method="anchor",
                        proposed_method=method,
                        apply_monotone_envelope=False,
                    ),
                }
    return output


def calculate_all_bootstraps(
    rows: list[dict],
    codecs: list[str],
    *,
    include_lpips: bool,
    samples: int,
    confidence_level: float,
    seed: int,
    psnr_aggregation: str,
) -> dict:
    metric_names = ["top1_percent", "psnr_db", "ms_ssim"]
    if include_lpips:
        metric_names.append("lpips_quality")
    output = {}
    for codec_offset, codec_name in enumerate(codecs):
        codec_rows = [row for row in rows if row["codec"] == codec_name]
        output[codec_name] = {}
        for method_offset, method in enumerate(("pre_only", "sandwich")):
            output[codec_name][method] = {}
            for metric_offset, metric in enumerate(metric_names):
                metric_seed = seed + 1000 * codec_offset + 100 * method_offset + metric_offset
                output[codec_name][method][metric] = bootstrap_bd_rate(
                    codec_rows,
                    metric,
                    anchor_method="anchor",
                    proposed_method=method,
                    samples=samples,
                    confidence_level=confidence_level,
                    seed=metric_seed,
                    psnr_aggregation=psnr_aggregation,
                    apply_monotone_envelope=True,
                )
    return output


def codec_command_manifest(args: argparse.Namespace, *, fps: float, preset: str) -> dict:
    commands = {}
    for codec_name in args.codecs:
        commands[codec_name] = {}
        for qp in args.qps:
            codec = StandardVideoCodec(
                codec_name,
                qp,
                fps=fps,
                preset=preset,
                ffmpeg=args.ffmpeg,
                codec_workers=args.codec_workers,
                ffmpeg_threads=args.ffmpeg_threads,
            )
            commands[codec_name][str(qp)] = codec.pipe_command_spec(
                args.frames, args.frame_size, args.frame_size
            )
    return commands


def _assert_point_estimates_match(bd_rate: dict, bootstrap: dict) -> None:
    for codec_name, methods in bd_rate.items():
        for method, metrics in methods.items():
            for metric, details in metrics.items():
                expected = details["primary"]["bd_rate_percent"]
                observed = bootstrap[codec_name][method][metric]["point_estimate_percent"]
                if expected is None or observed is None:
                    if expected is not observed:
                        raise RuntimeError(
                            f"point estimate mismatch for {codec_name}/{method}/{metric}"
                        )
                elif not math.isclose(float(expected), float(observed), abs_tol=1e-9):
                    raise RuntimeError(
                        f"point estimate mismatch for {codec_name}/{method}/{metric}: "
                        f"summary={expected}, bootstrap={observed}"
                    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not args.qps or len(set(args.qps)) != len(args.qps) or any(
        qp < 0 or qp > 51 for qp in args.qps
    ):
        raise ValueError("--qps must contain unique values in [0, 51]")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("--confidence-level must lie in (0, 1)")
    require_ffmpeg(args.ffmpeg)
    repository_root = Path(__file__).resolve().parent
    repository_before_outputs = git_state(repository_root)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint.get("args", {})
    preprocessor = preprocessor_from_checkpoint(checkpoint).to(device).eval()
    postprocessor = postprocessor_from_checkpoint(checkpoint).to(device).eval()
    analyzer = FrozenVideoAnalyzer(saved.get("analyzer", "r3d_18")).to(device).eval()
    lpips_metric = LPIPSLoss(args.lpips_backbone).to(device).eval() if args.lpips else None
    dataset = build_evaluation_dataset(
        data_root=args.data_root,
        test_dir=args.test_dir,
        split=args.split,
        categories=analyzer.categories,
        frames=args.frames,
        stride=args.frame_stride,
        size=args.frame_size,
        limit=args.limit,
        saved_args=saved,
    )
    fps = args.fps if args.fps is not None else float(saved.get("codec_fps", 30.0))
    preset = args.preset or saved.get("codec_preset", "medium")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    fields = [
        "sample_id",
        "sample_index",
        "source",
        "label",
        "codec",
        "qp",
        "method",
        "bitstream_bytes",
        "pixels",
        "bpp",
        "mse",
        "psnr_db",
        "ms_ssim",
        "top1",
        "top5",
    ]
    if args.lpips:
        fields.extend(("lpips_distance", "lpips_quality"))
    with (output / "per_video_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample_index, (clip, label) in enumerate(tqdm(dataset, desc="real-codec sandwich eval")):
            source = clip.unsqueeze(0).to(device)
            target = torch.tensor([label], device=device)
            source_path = str(dataset_sample_path(dataset, sample_index))
            sample_id = f"{sample_index:08d}"
            pixels = int(source.shape[1] * source.shape[-2] * source.shape[-1])
            for qp in args.qps:
                with torch.no_grad():
                    neural_code = preprocessor(source, qp)
                for codec_name in args.codecs:
                    codec = StandardVideoCodec(
                        codec_name,
                        qp,
                        fps=fps,
                        preset=preset,
                        ffmpeg=args.ffmpeg,
                        codec_workers=args.codec_workers,
                        ffmpeg_threads=args.ffmpeg_threads,
                    )
                    anchor_decoded, anchor_bpp = codec(source.detach())
                    decoded, bpp = codec(neural_code.detach())
                    with torch.no_grad():
                        restored = postprocessor(decoded.to(device), qp)
                    candidates = (
                        ("anchor", anchor_decoded.to(device), float(anchor_bpp[0])),
                        ("pre_only", decoded.to(device), float(bpp[0])),
                        ("sandwich", restored, float(bpp[0])),
                    )
                    for method, result, rate in candidates:
                        with torch.no_grad():
                            logits = analyzer(result)
                        metrics = quality_metrics(
                            result.float(), source.float(), lpips_metric
                        )
                        row = {
                            "sample_id": sample_id,
                            "sample_index": sample_index,
                            "source": source_path,
                            "label": int(label),
                            "codec": codec_name,
                            "qp": qp,
                            "method": method,
                            "bitstream_bytes": round(rate * pixels / 8.0),
                            "pixels": pixels,
                            "bpp": rate,
                            **metrics,
                            "top1": topk_correct(logits, target, 1),
                            "top5": topk_correct(logits, target, 5),
                        }
                        writer.writerow(row)
                        rows.append(row)

    summary_rows = aggregate_operating_points(
        rows, psnr_aggregation=args.psnr_aggregation, include_lpips=args.lpips
    )
    bd_rate = calculate_all_bd_rates(summary_rows, args.codecs, include_lpips=args.lpips)
    bootstrap = calculate_all_bootstraps(
        rows,
        args.codecs,
        include_lpips=args.lpips,
        samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.bootstrap_seed,
        psnr_aggregation=args.psnr_aggregation,
    )
    _assert_point_estimates_match(bd_rate, bootstrap)
    reportable = bool(
        args.limit is None
        and args.bootstrap_samples >= 10000
        and int(checkpoint.get("format_version", 0)) >= 10
        and not repository_before_outputs["dirty"]
    )
    summary = {
        "reportable": reportable,
        "psnr_aggregation": args.psnr_aggregation,
        "operating_points": summary_rows,
        "bd_rate": {
            codec: {
                method: {
                    metric: details["primary"]["bd_rate_percent"]
                    for metric, details in metrics.items()
                }
                for method, metrics in methods.items()
            }
            for codec, methods in bd_rate.items()
        },
    }
    write_json(output / "summary.json", summary)
    write_json(output / "bd_rate.json", bd_rate)
    write_json(output / "bootstrap.json", bootstrap)
    commands = codec_command_manifest(args, fps=fps, preset=preset)
    write_run_provenance(
        output,
        args=args,
        checkpoint_path=args.checkpoint,
        checkpoint=checkpoint,
        ffmpeg=args.ffmpeg,
        codec_commands=commands,
        repository_root=repository_root,
        repository_state=repository_before_outputs,
    )
    print(f"wrote complete evaluation artifacts to {output}")


if __name__ == "__main__":
    main()
