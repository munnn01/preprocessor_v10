from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train import build_qp_lambda_map, run_epoch


def test_build_qp_lambda_map_expands_scalar():
    assert build_qp_lambda_map([30, 35, 40, 45], [0.05]) == {
        30: 0.05,
        35: 0.05,
        40: 0.05,
        45: 0.05,
    }


def test_build_qp_lambda_map_accepts_per_qp_values():
    assert build_qp_lambda_map(
        [30, 35, 40, 45], [0.048, 0.151, 0.386, 0.576]
    ) == {30: 0.048, 35: 0.151, 40: 0.386, 45: 0.576}


def test_build_qp_lambda_map_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="one value per"):
        build_qp_lambda_map([30, 35, 40, 45], [0.05, 0.1])


class _RecordingPreprocessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_qps: list[int] = []

    def forward(self, clips: torch.Tensor, qp: int) -> torch.Tensor:
        self.seen_qps.append(qp)
        return clips


class _ValidationCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qp = 30

    def set_qp(self, qp: int) -> None:
        self.qp = qp

    def forward(
        self, clips: torch.Tensor, codec_source=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del codec_source
        bpp = torch.full((clips.shape[0],), self.qp / 100.0, device=clips.device)
        return clips, bpp


class _Analyzer(nn.Module):
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return torch.zeros((clips.shape[0], 5), device=clips.device)


def test_validation_runs_every_clip_at_every_qp():
    clips = torch.rand(2, 2, 3, 8, 8)
    labels = torch.zeros(2, dtype=torch.long)
    loader = DataLoader(TensorDataset(clips, labels), batch_size=1, shuffle=False)
    preprocessor = _RecordingPreprocessor()
    args = SimpleNamespace(
        amp=False,
        codec_qps=[30, 45],
        alpha=10.0,
        qp_to_rate_lambda={30: 0.05, 45: 0.1},
        accumulation_steps=1,
        clip_grad=1.0,
    )

    metrics = run_epoch(
        loader,
        preprocessor,
        _ValidationCodec(),
        _Analyzer(),
        args,
        torch.device("cpu"),
    )

    assert preprocessor.seen_qps == [30, 45, 30, 45]
    assert metrics["bpp"] == pytest.approx(0.375)
    assert metrics["qp30_bpp"] == pytest.approx(0.30)
    assert metrics["qp45_bpp"] == pytest.approx(0.45)
    assert "qp30_top1" in metrics
    assert "qp45_top1" in metrics
