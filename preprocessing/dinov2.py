"""Frozen DINOv2 feature teacher for task-agnostic semantic preservation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class FrozenDinoV2(nn.Module):
    """Framewise DINOv2 adapter whose parameters never receive gradients.

    A model may be injected for offline tests.  Production runs can use either
    an already cloned local DINOv2 repository or the official torch-hub entry.
    DINOv2 is a training/evaluation teacher and is not exported to Jetson.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        *,
        repo_or_dir: str = "facebookresearch/dinov2",
        model: nn.Module | None = None,
        image_size: int = 224,
        frame_stride: int = 2,
    ) -> None:
        super().__init__()
        if image_size < 14 or frame_stride < 1:
            raise ValueError("DINO image size and frame stride must be positive")
        if model is None:
            source = "local" if Path(repo_or_dir).is_dir() else "github"
            model = torch.hub.load(
                repo_or_dir,
                model_name,
                source=source,
                pretrained=True,
                trust_repo=True,
            )
        self.model_name = model_name
        self.repo_or_dir = repo_or_dir
        self.image_size = image_size
        self.frame_stride = frame_stride
        self.network = model.requires_grad_(False).eval()
        self.register_buffer(
            "mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True) -> FrozenDinoV2:
        super().train(False)
        self.network.eval()
        return self

    def _prepare(self, video: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError(f"expected [B,T,3,H,W], got {tuple(video.shape)}")
        sampled = video[:, :: self.frame_stride]
        batch, time = sampled.shape[:2]
        frames = sampled.reshape(-1, 3, sampled.shape[-2], sampled.shape[-1])
        frames = F.interpolate(
            frames,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return (frames - self.mean) / self.std, batch, time

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        frames, batch, time = self._prepare(video)
        if hasattr(self.network, "forward_features"):
            raw = self.network.forward_features(frames)
        else:
            raw = self.network(frames)
        if isinstance(raw, dict):
            cls = raw.get("x_norm_clstoken")
            patches = raw.get("x_norm_patchtokens")
            if cls is None and patches is None:
                tensor = next((value for value in raw.values() if torch.is_tensor(value)), None)
                if tensor is None:
                    raise ValueError("DINO model returned no tensor features")
                cls = tensor.flatten(1)
        elif torch.is_tensor(raw):
            cls, patches = raw.flatten(1), None
        else:
            raise TypeError("DINO model must return a tensor or feature dictionary")
        result: dict[str, torch.Tensor] = {}
        if cls is not None:
            result["cls"] = cls.reshape(batch, time, -1)
        if patches is not None:
            result["patch"] = patches.reshape(batch, time, patches.shape[-2], patches.shape[-1])
        return result


def dino_feature_loss(
    prediction: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    *,
    cls_weight: float = 0.25,
    patch_weight: float = 0.75,
) -> torch.Tensor:
    if min(cls_weight, patch_weight) < 0 or cls_weight + patch_weight <= 0:
        raise ValueError("DINO feature weights must be non-negative with positive sum")
    terms: list[tuple[float, torch.Tensor]] = []
    for key, weight in (("cls", cls_weight), ("patch", patch_weight)):
        if weight == 0:
            continue
        if key not in prediction or key not in reference:
            raise ValueError(f"DINO features are missing {key!r}")
        student = prediction[key].float()
        teacher = reference[key].detach().float()
        if student.shape != teacher.shape:
            raise ValueError(f"DINO {key} feature shapes do not match")
        cosine = F.cosine_similarity(student, teacher, dim=-1)
        terms.append((weight, 1.0 - cosine.mean()))
    normalizer = sum(weight for weight, _ in terms)
    return sum(weight * value for weight, value in terms) / normalizer


def dino_video_loss(
    teacher: FrozenDinoV2,
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    reference_features: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if reference_features is None:
        with torch.no_grad():
            reference_features = teacher(reference)
    return dino_feature_loss(teacher(prediction), reference_features)
