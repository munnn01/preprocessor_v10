"""Export the trainable pre/post wrappers to ONNX for Jetson TensorRT conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from preprocessing import postprocessor_from_checkpoint, preprocessor_from_checkpoint


class _WithQP(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, video: torch.Tensor, qp: torch.Tensor) -> torch.Tensor:
        return self.module(video, qp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts/onnx")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def _export(module: nn.Module, path: Path, sample: torch.Tensor, qp: torch.Tensor, opset: int) -> None:
    torch.onnx.export(
        _WithQP(module).eval(),
        (sample, qp),
        path,
        input_names=("video", "qp"),
        output_names=("video_out",),
        dynamic_axes={
            "video": {0: "batch", 1: "frames", 3: "height", 4: "width"},
            "video_out": {0: "batch", 1: "frames", 3: "height", 4: "width"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    preprocessor = preprocessor_from_checkpoint(checkpoint).eval()
    postprocessor = postprocessor_from_checkpoint(checkpoint).eval()
    sample = torch.rand(1, args.frames, 3, args.height, args.width)
    qp = torch.tensor([35.0])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _export(preprocessor, output / "preprocessor.onnx", sample, qp, args.opset)
    _export(postprocessor, output / "postprocessor.onnx", sample, qp, args.opset)
    print(f"exported ONNX wrappers to {output}")


if __name__ == "__main__":
    main()
