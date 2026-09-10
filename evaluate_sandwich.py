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
    build_evaluation_dataset,
    calculate_bd_rate_details,
    dataset_sample_path,
)
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--lpips",
        action="store_true",
        help="evaluate the frozen LPIPS distance (requires the research extra)",
    )
    parser.add_argument("--lpips-backbone", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--output-dir", default="outputs/sandwich_real_codec")
    return parser.parse_args()


def _quality(
    decoded: torch.Tensor,
    source: torch.Tensor,
    lpips_metric: LPIPSLoss | None,
) -> tuple[float, float, float | None]:
    mse = float(F.mse_loss(decoded, source))
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    ms_ssim = 1.0 - float(multiscale_ssim_loss(decoded, source))
    lpips_distance = None
    if lpips_metric is not None:
        with torch.no_grad():
            lpips_distance = float(lpips_metric(decoded.float(), source.float()))
    return psnr, ms_ssim, lpips_distance


def _bd_rows(rows: list[dict], method: str) -> list[dict]:
    selected = []
    for row in rows:
        if row["method"] not in {"anchor", method}:
            continue
        copied = dict(row)
        if copied["method"] == method:
            copied["method"] = "preprocessed"
        selected.append(copied)
    return selected


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
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
        "sample_index",
        "source",
        "label",
        "codec",
        "qp",
        "method",
        "bpp",
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
            for qp in args.qps:
                with torch.no_grad():
                    neural_code = preprocessor(source, qp)
                for codec_name in args.codecs:
                    codec = StandardVideoCodec(
                        codec_name, qp, fps=fps, preset=preset, ffmpeg=args.ffmpeg
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
                        psnr, ms_ssim, lpips_distance = _quality(
                            result.float(), source.float(), lpips_metric
                        )
                        row = {
                            "sample_index": sample_index,
                            "source": source_path,
                            "label": label,
                            "codec": codec_name,
                            "qp": qp,
                            "method": method,
                            "bpp": rate,
                            "psnr_db": psnr,
                            "ms_ssim": ms_ssim,
                            "top1": topk_correct(logits, target, 1),
                            "top5": topk_correct(logits, target, 5),
                        }
                        if lpips_distance is not None:
                            row["lpips_distance"] = lpips_distance
                            row["lpips_quality"] = -lpips_distance
                        writer.writerow(row)
                        rows.append(row)

    metric_names = ["bpp", "psnr_db", "ms_ssim", "top1", "top5"]
    if args.lpips:
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
    summary_rows = []
    for (codec_name, qp, method), total in sorted(totals.items()):
        count = total["videos"]
        summary = {
                "codec": codec_name,
                "qp": qp,
                "method": method,
                "videos": int(count),
                "bpp": total["bpp"] / count,
                "psnr_db": total["psnr_db"] / count,
                "ms_ssim": total["ms_ssim"] / count,
                "top1_percent": 100.0 * total["top1"] / count,
                "top5_percent": 100.0 * total["top5"] / count,
        }
        if args.lpips:
            summary["lpips_distance"] = total["lpips_distance"] / count
            summary["lpips_quality"] = total["lpips_quality"] / count
        summary_rows.append(summary)
    bd_rate = {}
    for codec_name in args.codecs:
        codec_rows = [row for row in summary_rows if row["codec"] == codec_name]
        bd_rate[codec_name] = {}
        for method in ("pre_only", "sandwich"):
            comparison = _bd_rows(codec_rows, method)
            quality_metrics = ["top1_percent", "psnr_db", "ms_ssim"]
            if args.lpips:
                quality_metrics.append("lpips_quality")
            bd_rate[codec_name][method] = {
                metric: calculate_bd_rate_details(comparison, metric)
                for metric in quality_metrics
            }
    write_json(output / "summary.json", {"operating_points": summary_rows, "bd_rate": bd_rate})
    print(f"wrote {output / 'summary.json'}")


if __name__ == "__main__":
    main()
