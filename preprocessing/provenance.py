"""Small, deterministic helpers for research-run provenance artifacts."""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .utils import write_json


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    return value


def _command_output(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return f"unavailable: {exc}"
    output = completed.stdout.strip()
    if completed.returncode != 0 and not output:
        return f"command exited {completed.returncode} with no output"
    return output


def git_state(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    revision_output = _command_output(("git", "rev-parse", "HEAD"), cwd=root)
    revision = revision_output.splitlines()[0] if revision_output else "unavailable"
    status = _command_output(("git", "status", "--porcelain"), cwd=root)
    return {
        "commit": revision,
        "dirty": bool(status),
        "status_porcelain": status,
    }


def environment_report(ffmpeg: str = "ffmpeg") -> str:
    """Capture the software facts needed to interpret codec measurements."""

    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except Exception as exc:  # noqa: BLE001  # optional runtime can fail during import
        torchvision_version = f"unavailable: {exc}"
    cuda_name = "unavailable"
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(torch.cuda.current_device())
    lines = [
        f"platform={platform.platform()}",
        f"python={sys.version.replace(chr(10), ' ')}",
        f"torch={torch.__version__}",
        f"torchvision={torchvision_version}",
        f"cuda_runtime={torch.version.cuda}",
        f"cudnn={torch.backends.cudnn.version()}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_device={cuda_name}",
        "",
        "ffmpeg:",
        _command_output((ffmpeg, "-version")),
    ]
    return "\n".join(str(value) for value in lines) + "\n"


def write_run_provenance(
    output_dir: str | Path,
    *,
    args: argparse.Namespace | Mapping[str, Any],
    checkpoint_path: str | Path,
    checkpoint: Mapping[str, Any],
    ffmpeg: str,
    codec_commands: Mapping[str, Any],
    repository_root: str | Path,
    repository_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the non-result portion of the experiment artifact contract."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arguments = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    checkpoint_digest = file_sha256(checkpoint_path)
    repository = dict(repository_state) if repository_state is not None else git_state(repository_root)
    metadata = {
        "format_version": checkpoint.get("format_version"),
        "epoch": checkpoint.get("epoch"),
        "scientific_status": checkpoint.get("scientific_status"),
        "selection_state": checkpoint.get("selection_state"),
        "validation": checkpoint.get("validation"),
        "proxy_sha256": checkpoint.get("proxy_sha256"),
        "checkpoint_sha256": checkpoint_digest,
    }
    write_json(output / "args.json", json_safe(arguments))
    write_json(output / "checkpoint_metadata.json", json_safe(metadata))
    write_json(output / "codec_commands.json", json_safe(codec_commands))
    write_json(output / "git_state.json", json_safe(repository))
    (output / "git_commit.txt").write_text(
        str(repository["commit"]) + "\n", encoding="utf-8"
    )
    (output / "checkpoint_sha256.txt").write_text(
        checkpoint_digest + "\n", encoding="utf-8"
    )
    (output / "environment.txt").write_text(
        environment_report(ffmpeg), encoding="utf-8"
    )
    return {
        "checkpoint_sha256": checkpoint_digest,
        "git": repository,
    }
