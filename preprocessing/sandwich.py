"""Pre/codec/post composition with an exact real-codec forward path."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class SandwichOutput(NamedTuple):
    neural_code: torch.Tensor
    decoded: torch.Tensor
    restored: torch.Tensor
    machine_view: torch.Tensor
    bpp: torch.Tensor


class AdaptiveVideoSandwich(nn.Module):
    """Wrap a frozen standard-codec bridge with trainable neural processors.

    The codec bridge is expected to use real H.264/H.265 reconstruction and BPP
    in the forward pass and a frozen differentiable proxy only in backward.
    """

    def __init__(
        self,
        preprocessor: nn.Module,
        codec: nn.Module,
        postprocessor: nn.Module,
        *,
        task_input: str = "postprocessed",
    ) -> None:
        super().__init__()
        if task_input not in {"codec", "postprocessed"}:
            raise ValueError("task_input must be 'codec' or 'postprocessed'")
        self.preprocessor = preprocessor
        self.codec = codec
        self.postprocessor = postprocessor
        self.task_input = task_input
        self.codec.requires_grad_(False)

    @property
    def qp(self) -> int:
        return int(getattr(self.codec, "qp"))

    def set_qp(self, qp: int) -> None:
        setter = getattr(self.codec, "set_qp", None)
        if setter is None:
            raise TypeError("codec does not implement set_qp")
        setter(qp)

    def train(self, mode: bool = True) -> AdaptiveVideoSandwich:
        super().train(mode)
        # ParallelStandardVideoCodec keeps its frozen proxy in eval mode while
        # its own training flag selects proxy-backward behavior by default.
        self.codec.train(mode)
        return self

    def forward(
        self,
        clip: torch.Tensor,
        qp: int | None = None,
        *,
        codec_source: str = "real",
        use_proxy_gradient: bool | None = None,
    ) -> SandwichOutput:
        if qp is not None:
            self.set_qp(qp)
        qp_value = self.qp if qp is None else qp
        neural_code = self.preprocessor(clip, qp_value)
        decoded, bpp = self.codec(
            neural_code,
            codec_source=codec_source,
            use_proxy_gradient=use_proxy_gradient,
        )
        restored = self.postprocessor(decoded, qp_value)
        machine_view = decoded if self.task_input == "codec" else restored
        return SandwichOutput(neural_code, decoded, restored, machine_view, bpp)

    def trainable_parameters(self):
        yield from self.preprocessor.parameters()
        yield from self.postprocessor.parameters()
