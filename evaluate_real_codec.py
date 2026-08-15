"""Evaluate anchor and preprocessed clips with real H.264/H.265 codecs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm

from preprocessing import FrozenVideoAnalyzer, StandardVideoCodec, build_preprocessor
from preprocessing.data import VideoFolderDataset, resolve_split
from preprocessing.utils import topk_correct, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--test-dir")
    parser.add_argument("--split", default="val")
    parser.add_argument("--codecs", nargs="+", choices=("h264", "h265"), default=["h264", "h265"])
    parser.add_argument("--qps", nargs="+", type=int, default=[30, 35, 40, 45, 50])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--frame-size", type=int, default=128)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--preset")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/real_codec")
    return parser.parse_args()


def real_codec_roundtrip(
    clip: torch.Tensor,
    codec: str,
    qp: int,
    fps: float,
    *,
    preset: str = "medium",
    ffmpeg: str = "ffmpeg",
) -> tuple[torch.Tensor, float]:
    module = StandardVideoCodec(
        codec, qp, fps=fps, preset=preset, ffmpeg=ffmpeg
    )
    reconstruction, bpp = module(clip.unsqueeze(0))
    return reconstruction[0], float(bpp[0])


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    codec_fps = args.fps if args.fps is not None else float(saved_args.get("codec_fps", 30.0))
    codec_preset = args.preset or saved_args.get("codec_preset", "medium")
    analyzer_name = saved_args.get("analyzer", "r3d_18")
    preprocessor_kind = saved_args.get("preprocessor", "cnn")
    analyzer = FrozenVideoAnalyzer(analyzer_name).to(device).eval()
    preprocessor = build_preprocessor(
        preprocessor_kind,
        temporal_frames=int(saved_args.get("temporal_frames", 8)),
        patch_size=int(saved_args.get("vit_patch_size", 8)),
        embed_dim=int(saved_args.get("vit_embed_dim", 96)),
        depth=int(saved_args.get("vit_depth", 4)),
        num_heads=int(saved_args.get("vit_heads", 4)),
        max_residual=float(saved_args.get("max_residual", 0.25)),
    ).to(device).eval()
    preprocessor.load_state_dict(checkpoint["preprocessor"])

    test_root = Path(args.test_dir) if args.test_dir else resolve_split(args.data_root, args.split)
    dataset = VideoFolderDataset(
        test_root,
        analyzer.categories,
        frames=args.frames,
        stride=args.frame_stride,
        size=args.frame_size,
        train=False,
        limit=args.limit,
    )
    totals: dict[tuple[str, int, str], dict[str, float]] = defaultdict(
        lambda: {"videos": 0, "bpp": 0.0, "mse": 0.0, "top1": 0, "top5": 0}
    )

    for clip, label in tqdm(dataset, desc="real codec evaluation"):
        source = clip.to(device).unsqueeze(0)
        with torch.no_grad():
            proposed = preprocessor(source)[0].cpu()
        for codec in args.codecs:
            for qp in args.qps:
                for method, input_clip in (("anchor", clip), ("preprocessed", proposed)):
                    decoded, bpp = real_codec_roundtrip(
                        input_clip,
                        codec,
                        qp,
                        codec_fps,
                        preset=codec_preset,
                        ffmpeg=args.ffmpeg,
                    )
                    decoded_device = decoded.to(device).unsqueeze(0)
                    label_tensor = torch.tensor([label], device=device)
                    with torch.no_grad():
                        logits = analyzer(decoded_device)
                    key = (codec, qp, method)
                    row = totals[key]
                    row["videos"] += 1
                    row["bpp"] += bpp
                    row["mse"] += float(F.mse_loss(decoded_device, source))
                    row["top1"] += topk_correct(logits, label_tensor, 1)
                    row["top5"] += topk_correct(logits, label_tensor, 5)

    rows = []
    for (codec, qp, method), values in sorted(totals.items()):
        count = int(values["videos"])
        rows.append(
            {
                "codec": codec,
                "qp": qp,
                "method": method,
                "videos": count,
                "bpp": values["bpp"] / count,
                "mse": values["mse"] / count,
                "top1": values["top1"] / count,
                "top5": values["top5"] / count,
            }
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", rows)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output / 'metrics.csv'}")


if __name__ == "__main__":
    main()
