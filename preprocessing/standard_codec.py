"""Real standard video codecs and a frozen differentiable proxy.

The standard codec owns the forward values used by the loss and analyzer.  Its
FFmpeg round trip is non-differentiable, so ``ParallelStandardVideoCodec`` uses
the proxy only for the backward Jacobian:

    y = y_proxy + stop_gradient(y_real - y_proxy)

Consequently ``y == y_real`` in the forward pass while gradients with respect
to the preprocessed clip follow ``y_proxy``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import torch
from torch import nn
from torch.nn import functional as F


def _run_ffmpeg(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg was not found. Install it or pass --ffmpeg /path/to/ffmpeg."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"FFmpeg failed: {detail}") from exc


class StandardVideoCodec(nn.Module):
    """Non-differentiable H.264/H.265 encode/decode through FFmpeg.

    Each batch item is encoded as one elementary video stream.  BPP therefore
    includes codec headers but excludes MP4/MKV container overhead.
    """

    def __init__(
        self,
        codec: str = "h264",
        qp: int = 35,
        *,
        fps: float = 30.0,
        preset: str = "medium",
        ffmpeg: str = "ffmpeg",
    ) -> None:
        super().__init__()
        if codec not in {"h264", "h265"}:
            raise ValueError("codec must be 'h264' or 'h265'")
        if not 0 <= qp <= 51:
            raise ValueError("QP must be in [0, 51]")
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.codec = codec
        self.qp = qp
        self.fps = fps
        self.preset = preset
        self.ffmpeg = ffmpeg

    @property
    def encoder(self) -> str:
        return "libx264" if self.codec == "h264" else "libx265"

    @property
    def extension(self) -> str:
        return ".h264" if self.codec == "h264" else ".h265"

    @property
    def format_name(self) -> str:
        return "h264" if self.codec == "h264" else "hevc"

    def set_qp(self, qp: int) -> None:
        if not 0 <= qp <= 51:
            raise ValueError("QP must be in [0, 51]")
        self.qp = qp

    @staticmethod
    def _write_frames(directory: Path, clip: torch.Tensor) -> None:
        for index, frame in enumerate(clip):
            rgb = (
                frame.detach()
                .float()
                .clamp(0.0, 1.0)
                .permute(1, 2, 0)
                .cpu()
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .numpy()
            )
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(directory / f"input_{index:05d}.png"), bgr):
                raise RuntimeError("could not write a temporary codec input frame")

    @staticmethod
    def _read_frames(directory: Path, count: int) -> torch.Tensor:
        frames: list[torch.Tensor] = []
        for index in range(count):
            image = cv2.imread(str(directory / f"decoded_{index:05d}.png"), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"codec produced only {len(frames)} of {count} frames")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255.0))
        return torch.stack(frames)

    def _roundtrip_one(self, clip: torch.Tensor) -> tuple[torch.Tensor, float]:
        time, channels, height, width = clip.shape
        if channels != 3 or time < 1:
            raise ValueError(f"expected non-empty [T,3,H,W], got {tuple(clip.shape)}")
        if height % 2 or width % 2:
            raise ValueError(
                f"yuv420p requires even height and width, got {height}x{width}"
            )

        with tempfile.TemporaryDirectory(prefix="standard_codec_") as temporary:
            directory = Path(temporary)
            self._write_frames(directory, clip)
            stream = directory / f"stream{self.extension}"
            input_pattern = directory / "input_%05d.png"
            decoded_pattern = directory / "decoded_%05d.png"
            keyint = max(time, 1)
            command = [
                self.ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(self.fps),
                "-start_number",
                "0",
                "-i",
                str(input_pattern),
                "-frames:v",
                str(time),
                "-an",
                "-c:v",
                self.encoder,
                "-preset",
                self.preset,
                "-qp",
                str(self.qp),
                "-pix_fmt",
                "yuv420p",
            ]
            if self.codec == "h264":
                command.extend(
                    ["-x264-params", f"keyint={keyint}:min-keyint={keyint}:scenecut=0"]
                )
            else:
                command.extend(
                    [
                        "-x265-params",
                        f"log-level=error:keyint={keyint}:min-keyint={keyint}:scenecut=0",
                    ]
                )
            command.extend(["-f", self.format_name, str(stream)])
            _run_ffmpeg(command)
            _run_ffmpeg(
                [
                    self.ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(stream),
                    "-frames:v",
                    str(time),
                    "-start_number",
                    "0",
                    str(decoded_pattern),
                ]
            )
            reconstructed = self._read_frames(directory, time)
            bpp = stream.stat().st_size * 8.0 / float(time * height * width)
            return reconstructed, bpp

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if clip.ndim != 5 or clip.shape[2] != 3:
            raise ValueError(f"expected [B,T,3,H,W], got {tuple(clip.shape)}")
        device = clip.device
        dtype = clip.dtype
        reconstructions: list[torch.Tensor] = []
        rates: list[float] = []
        # FFmpeg is outside autograd by construction.
        for sample in clip.detach().cpu():
            reconstructed, bpp = self._roundtrip_one(sample)
            reconstructions.append(reconstructed)
            rates.append(bpp)
        return (
            torch.stack(reconstructions).to(device=device, dtype=dtype),
            torch.tensor(rates, device=device, dtype=torch.float32),
        )


class StandardCodecProxy(nn.Module):
    """Compact differentiable video network distilled from a standard codec."""

    def __init__(
        self,
        hidden_channels: int = 48,
        latent_channels: int = 64,
        max_delta: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.max_delta = max_delta
        self.encoder = nn.Sequential(
            nn.Conv3d(
                3, hidden_channels, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)
            ),
            nn.GELU(),
            nn.Conv3d(
                hidden_channels,
                latent_channels,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            ),
            nn.GELU(),
        )
        self.qp_embedding = nn.Sequential(
            nn.Linear(1, latent_channels), nn.GELU(), nn.Linear(latent_channels, latent_channels)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                latent_channels,
                hidden_channels,
                kernel_size=(3, 4, 4),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
            ),
            nn.GELU(),
            nn.ConvTranspose3d(
                hidden_channels,
                3,
                kernel_size=(3, 4, 4),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
            ),
        )
        self.rate_head = nn.Sequential(
            nn.Linear(latent_channels + 1, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "hidden_channels": self.hidden_channels,
            "latent_channels": self.latent_channels,
            "max_delta": self.max_delta,
        }

    @staticmethod
    def _qp_tensor(
        qp: int | float | torch.Tensor,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = torch.as_tensor(qp, device=device, dtype=dtype).flatten()
        if value.numel() == 1:
            value = value.expand(batch)
        if value.numel() != batch:
            raise ValueError(f"QP must be scalar or have {batch} elements")
        return value

    @staticmethod
    def _ste_round(value: torch.Tensor) -> torch.Tensor:
        return value + (value.round() - value).detach()

    def forward(
        self, clip: torch.Tensor, qp: int | float | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if clip.ndim != 5 or clip.shape[2] != 3:
            raise ValueError(f"expected [B,T,3,H,W], got {tuple(clip.shape)}")
        batch, time, _, height, width = clip.shape
        pad_height = (4 - height % 4) % 4
        pad_width = (4 - width % 4) % 4
        channels_first = clip.permute(0, 2, 1, 3, 4)
        if pad_height or pad_width:
            channels_first = F.pad(
                channels_first, (0, pad_width, 0, pad_height, 0, 0), mode="replicate"
            )

        qp_value = self._qp_tensor(
            qp, batch, device=clip.device, dtype=clip.dtype
        )
        qp_normalized = ((qp_value - 25.5) / 25.5).unsqueeze(1)
        latent = self.encoder(channels_first)
        condition = self.qp_embedding(qp_normalized).view(batch, -1, 1, 1, 1)
        latent = latent + condition
        q_step = torch.pow(2.0, (qp_value - 32.0) / 6.0).view(batch, 1, 1, 1, 1)
        quantized = self._ste_round(latent / q_step) * q_step

        delta = self.max_delta * torch.tanh(self.decoder(quantized))
        delta = delta[..., :time, :height, :width]
        reconstruction = (channels_first[..., :height, :width] + delta).clamp(0.0, 1.0)
        reconstruction = reconstruction.permute(0, 2, 1, 3, 4)

        statistics = quantized.abs().mean(dim=(2, 3, 4))
        rate = F.softplus(self.rate_head(torch.cat((statistics, qp_normalized), dim=1))).squeeze(1)
        return reconstruction, rate

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, map_location: str | torch.device = "cpu"
    ) -> StandardCodecProxy:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        config = checkpoint.get("proxy_config", {}) if isinstance(checkpoint, dict) else {}
        model = cls(**config)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"unsupported proxy checkpoint: {path}")
        state = checkpoint.get("proxy", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state)
        return model


class ParallelStandardVideoCodec(nn.Module):
    """Real codec forward pass with a frozen proxy providing backward gradients."""

    def __init__(self, standard_codec: nn.Module, proxy: StandardCodecProxy) -> None:
        super().__init__()
        self.standard_codec = standard_codec
        self.proxy = proxy.requires_grad_(False)
        self.proxy.eval()

    @property
    def qp(self) -> int:
        return int(getattr(self.standard_codec, "qp"))

    def set_qp(self, qp: int) -> None:
        setter = getattr(self.standard_codec, "set_qp", None)
        if setter is None:
            raise TypeError("standard codec does not implement set_qp")
        setter(qp)

    def train(self, mode: bool = True) -> ParallelStandardVideoCodec:
        super().train(mode)
        self.proxy.eval()
        return self

    def forward(
        self, clip: torch.Tensor, *, use_proxy_gradient: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real_reconstruction, real_bpp = self.standard_codec(clip.detach())
        if use_proxy_gradient is None:
            use_proxy_gradient = self.training and torch.is_grad_enabled()
        if not use_proxy_gradient:
            return real_reconstruction, real_bpp

        proxy_reconstruction, proxy_bpp = self.proxy(clip, self.qp)
        reconstruction = proxy_reconstruction + (
            real_reconstruction - proxy_reconstruction
        ).detach()
        bpp = proxy_bpp + (real_bpp - proxy_bpp).detach()
        return reconstruction, bpp


def require_ffmpeg(executable: str = "ffmpeg") -> None:
    """Fail before a long run if the configured FFmpeg executable is missing."""

    if Path(executable).parent != Path("."):
        available = Path(executable).is_file()
    else:
        available = shutil.which(executable) is not None
    if not available:
        raise RuntimeError(f"FFmpeg executable not found: {executable!r}")
