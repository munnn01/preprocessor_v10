"""Measure pre/post wrapper latency on the target Jetson without inventing results."""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from preprocessing import postprocessor_from_checkpoint, preprocessor_from_checkpoint
from preprocessing.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--qp", type=float, default=35.0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/jetson_benchmark.json")
    return parser.parse_args()


class _WrapperPair(nn.Module):
    """Measure neural wrapper cost only; the hardware codec is intentionally absent."""

    def __init__(self, preprocessor: nn.Module, postprocessor: nn.Module) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def forward(self, sample: torch.Tensor, qp: torch.Tensor) -> torch.Tensor:
        return self.postprocessor(self.preprocessor(sample, qp), qp)


def _measure(module, sample: torch.Tensor, qp: torch.Tensor, warmup: int, iterations: int):
    with torch.inference_mode():
        for _ in range(warmup):
            module(sample, qp)
        if sample.device.type == "cuda":
            torch.cuda.synchronize()
        if sample.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(sample.device)
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            module(sample, qp)
            if sample.device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(1000.0 * (time.perf_counter() - start))
    ordered = sorted(timings)
    percentile = lambda p: ordered[min(round(p * (len(ordered) - 1)), len(ordered) - 1)]
    peak_memory = (
        int(torch.cuda.max_memory_allocated(sample.device))
        if sample.device.type == "cuda"
        else None
    )
    return {
        "mean_ms": statistics.fmean(timings),
        "median_ms": statistics.median(timings),
        "p95_ms": percentile(0.95),
        "frames_per_second": sample.shape[1] * 1000.0 / statistics.fmean(timings),
        "peak_cuda_memory_bytes": peak_memory,
    }


def main() -> None:
    args = parse_args()
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pre = preprocessor_from_checkpoint(checkpoint).to(device).eval()
    post = postprocessor_from_checkpoint(checkpoint).to(device).eval()
    sample = torch.rand(1, args.frames, 3, args.height, args.width, device=device)
    qp = torch.tensor([args.qp], device=device)
    report = {
        "scientific_status": "measured",
        "host": platform.node(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "shape_btchw": list(sample.shape),
        "qp": args.qp,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "parameters": {
            "preprocessor": sum(value.numel() for value in pre.parameters()),
            "postprocessor": sum(value.numel() for value in post.parameters()),
        },
        "latency": {
            "preprocessor": _measure(pre, sample, qp, args.warmup, args.iterations),
            "postprocessor": _measure(post, sample, qp, args.warmup, args.iterations),
            "wrappers_without_codec": _measure(
                _WrapperPair(pre, post), sample, qp, args.warmup, args.iterations
            ),
        },
        "scope_note": "wrappers_without_codec is not end-to-end latency; measure hardware encode/decode and the task analyzer separately.",
        "power_note": "Capture tegrastats separately while this command runs; do not infer watts from latency.",
    }
    write_json(Path(args.output), report)
    print(f"wrote {Path(args.output)}")


if __name__ == "__main__":
    main()
