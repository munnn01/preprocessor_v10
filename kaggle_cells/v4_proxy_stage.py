"""Prepare, train, and audit the predictive-entropy Proxy V4 on Kaggle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
import csv
import hashlib
import json
import random
import subprocess
import sys


V7_NAME = "v7_rate_recovery_20260907_163149_952110"
CLEAN_DEFAULT = Path(
    "/kaggle/input/datasets/qktttttttttt/kineticscleaned/cleaned_final/"
    "kinetics400_5per/kinetics400_5per/train"
)
QPS = [30, 35, 40, 45]
TRAIN_SIZE = 1000
VAL_SIZE = 400
SEED = 2408


@dataclass(frozen=True)
class ProxyStage:
    project: Path
    root: Path
    clean: Path
    source_v7: Path
    swin_checkpoint: Path
    train_dir: Path
    val_dir: Path
    proxy_dir: Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_source(clean: Path, member: str) -> tuple[PurePosixPath, Path]:
    relative = PurePosixPath(member)
    if (
        relative.is_absolute()
        or len(relative.parts) < 2
        or ".." in relative.parts
        or "\\" in member
        or ":" in member
    ):
        raise ValueError(f"Unsafe split member: {member}")
    clean_root = clean.resolve(strict=True)
    source = clean_root.joinpath(*relative.parts).resolve(strict=True)
    if not source.is_relative_to(clean_root) or not source.is_file():
        raise ValueError(f"Missing or unsafe clean video: {member}")
    return relative, source


def balanced_subset(members: list[str], size: int, seed: int) -> list[str]:
    """Choose a deterministic class-balanced subset without changing membership."""

    unique = sorted(set(members))
    if len(unique) != len(members):
        raise ValueError("source split contains duplicate videos")
    if size < 1 or size > len(unique):
        raise ValueError(f"subset size {size} is invalid for {len(unique)} videos")
    by_class: dict[str, list[str]] = {}
    for member in unique:
        relative = PurePosixPath(member)
        if len(relative.parts) < 2:
            raise ValueError(f"Expected class/video path, got {member}")
        by_class.setdefault(relative.parts[0], []).append(member)
    generator = random.Random(seed)
    labels = sorted(by_class)
    generator.shuffle(labels)
    for values in by_class.values():
        generator.shuffle(values)
    chosen: list[str] = []
    while len(chosen) < size:
        for label in labels:
            if by_class[label]:
                chosen.append(by_class[label].pop())
                if len(chosen) == size:
                    break
    generator.shuffle(chosen)
    return chosen


def link_members(clean: Path, members: list[str], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    resolved = [_safe_source(clean, member) for member in members]
    for relative, source in resolved:
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)


def find_v7_root(input_root: Path, working: Path) -> Path:
    candidates = [working / V7_NAME]
    candidates.extend(path.parent for path in input_root.glob("**/recovery_split.json"))
    candidates = [
        path for path in sorted(set(candidates))
        if (path / "recovery_split.json").is_file()
        and (path / "selection.json").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            "Add the V7 Kaggle Output containing recovery_split.json and selection.json"
        )
    signatures = {
        (file_sha256(path / "recovery_split.json"), file_sha256(path / "selection.json"))
        for path in candidates
    }
    if len(signatures) != 1:
        raise RuntimeError("Different V7 recovery outputs are attached; keep only the intended one")
    return candidates[0]


def find_swin_checkpoint(
    input_root: Path, working: Path, expected_sha256: str
) -> Path:
    candidates = list(input_root.glob("**/best_task_bd_rate.pt"))
    candidates.extend(working.glob("**/best_task_bd_rate.pt"))
    matches = [path for path in sorted(set(candidates)) if file_sha256(path) == expected_sha256]
    if not matches:
        raise FileNotFoundError(
            "Add the V6 calibration Output containing "
            "swin_ratio_090/best_task_bd_rate.pt; its SHA-256 must match V7 selection.json"
        )
    return matches[0]


def prepare(
    project: Path = Path("/kaggle/working/proxy_v4"),
    clean: Path = CLEAN_DEFAULT,
    *,
    train_size: int = TRAIN_SIZE,
    val_size: int = VAL_SIZE,
) -> ProxyStage:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Enable GPU in Kaggle Settings before Run All")
    subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL)
    project = project.resolve(strict=True)
    clean = clean.resolve(strict=True)
    working, input_root = Path("/kaggle/working"), Path("/kaggle/input")
    source_v7 = find_v7_root(input_root, working)
    split = _read_json(source_v7 / "recovery_split.json")
    selection = _read_json(source_v7 / "selection.json")
    train_source = list(split["train"])
    val_source = list(split["controller_800"])
    if set(train_source) & set(val_source):
        raise ValueError("V7 train and controller overlap")
    expected_checkpoint_hash = selection.get("start_checkpoint_sha256")
    if not isinstance(expected_checkpoint_hash, str) or len(expected_checkpoint_hash) != 64:
        raise ValueError("V7 selection.json has no valid start_checkpoint_sha256")
    swin_checkpoint = find_swin_checkpoint(input_root, working, expected_checkpoint_hash)
    checkpoint = torch.load(swin_checkpoint, map_location="cpu", weights_only=False)
    if "preprocessor" not in checkpoint or checkpoint.get("args", {}).get("preprocessor") != "swin":
        raise ValueError("The matched V6 checkpoint does not contain VideoSwinLite weights")

    train_members = balanced_subset(train_source, train_size, SEED)
    val_members = balanced_subset(val_source, val_size, SEED + 1)
    for member in train_members + val_members:
        _safe_source(clean, member)
    run_root = working / (
        "v4_proxy_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    train_dir = run_root / "split/proxy_train_1000"
    val_dir = run_root / "split/proxy_val_400"
    link_members(clean, train_members, train_dir)
    link_members(clean, val_members, val_dir)
    proxy_dir = run_root / "predictive_entropy_proxy"
    record = {
        "source_v7": str(source_v7),
        "source_recovery_split_sha256": file_sha256(source_v7 / "recovery_split.json"),
        "swin_checkpoint": str(swin_checkpoint),
        "swin_checkpoint_sha256": file_sha256(swin_checkpoint),
        "swin_epoch": checkpoint.get("epoch"),
        "swin_controller_task_bd_rate_percent": checkpoint.get("val_metrics", {}).get(
            "task_bd_rate_percent"
        ),
        "clean_root": str(clean),
        "seed": SEED,
        "qps": QPS,
        "proxy_train": train_members,
        "proxy_val": val_members,
        "feature_weight": 0.0,
        "old_proxy_checkpoint_used": False,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "selection.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    result = ProxyStage(
        project=project,
        root=run_root,
        clean=clean,
        source_v7=source_v7,
        swin_checkpoint=swin_checkpoint,
        train_dir=train_dir,
        val_dir=val_dir,
        proxy_dir=proxy_dir,
    )
    print(json.dumps({
        "run_root": str(result.root),
        "V6_Swin_input": str(result.swin_checkpoint),
        "V7_split_input": str(result.source_v7),
        "train_videos": len(train_members),
        "validation_videos": len(val_members),
        "old_proxy_used": False,
        "feature_loss": 0.0,
    }, indent=2), flush=True)
    return result


def train_proxy(
    stage: ProxyStage,
    *,
    epochs: int = 5,
    lr: float = 2e-4,
    rate_weight: float = 0.10,
    init_checkpoint: Path | None = None,
    qp_sampling_weights: list[float] | None = None,
) -> None:
    command = [
        sys.executable, "-u", str(stage.project / "train_proxy.py"),
        "--train-dir", str(stage.train_dir),
        "--val-dir", str(stage.val_dir),
        "--codec", "h264",
        "--qps", *map(str, QPS),
        "--frames", "16",
        "--frame-stride", "2",
        "--frame-size", "128",
        "--fps", "30",
        "--preset", "medium",
        "--codec-io", "pipe",
        "--codec-workers", "2",
        "--ffmpeg-threads", "1",
        "--hidden-channels", "48",
        "--latent-channels", "64",
        "--bottleneck-channels", "96",
        "--blocks-per-stage", "2",
        "--film-channels", "64",
        "--qp-step-divisor", "12",
        "--max-delta", "1.0",
        "--entropy-floor", "1e-9",
        "--epochs", str(epochs),
        "--batch-size", "2",
        "--lr", str(lr),
        "--rate-weight", str(rate_weight),
        "--rate-delta-weight", "0.50",
        "--rate-direction-weight", "0.10",
        "--rate-direction-margin", "0.01",
        "--pair-strengths", "0", "0.05", "0.10", "0.20",
        "--preprocessor-checkpoint", str(stage.swin_checkpoint),
        "--gradient-probe-batches", "32",
        "--gradient-probe-step", str(2.0 / 255.0),
        "--audit-max-rate-mape-percent", "20",
        "--audit-min-pair-direction-accuracy", "0.60",
        "--audit-max-real-delta-percent", "0",
        "--audit-min-real-down-fraction", "0.55",
        "--weight-decay", "1e-4",
        "--clip-grad", "1.0",
        "--scheduler-factor", "0.5",
        "--scheduler-patience", "2",
        "--workers", "2",
        "--amp",
        "--seed", str(SEED),
        "--output-dir", str(stage.proxy_dir),
    ]
    if init_checkpoint is not None:
        command.extend(("--init-checkpoint", str(init_checkpoint)))
    if qp_sampling_weights is not None:
        command.extend(("--qp-sampling-weights", *map(str, qp_sampling_weights)))
    (stage.root / "train_command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8"
    )
    subprocess.run(command, cwd=stage.project, check=True)


def report(stage: ProxyStage) -> dict:
    import torch

    diagnostic_path = stage.proxy_dir / "best_audit.pt"
    feasible_path = stage.proxy_dir / "best_feasible.pt"
    if not diagnostic_path.is_file():
        raise FileNotFoundError(f"Training did not produce {diagnostic_path}")
    selected_path = feasible_path if feasible_path.is_file() else diagnostic_path
    payload = torch.load(selected_path, map_location="cpu", weights_only=False)
    audit = payload["proxy_audit"]
    result = {
        "eligible_for_swin_finetune": bool(feasible_path.is_file()),
        "selected_checkpoint": str(selected_path),
        "selected_epoch": payload.get("epoch"),
        "audit": audit,
        "next_action": (
            "Use best_feasible.pt for VideoSwin rate-recovery fine-tuning."
            if feasible_path.is_file()
            else "Do not fine-tune VideoSwin yet; best_audit.pt is diagnostic only."
        ),
    }
    report_path = stage.root / "proxy_audit.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv_path = stage.root / "proxy_audit_by_qp.csv"
    columns = [
        "qp", "rate_mape_percent", "pair_direction_accuracy",
        "probe_real_delta_percent", "probe_real_down_fraction", "passed", "reasons",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in audit["per_qp"]:
            writer.writerow({**row, "reasons": ";".join(row["reasons"])})
    print("\nQP | MAPE% | Pair direction | Real BPP delta% | Real down | Pass")
    for row in audit["per_qp"]:
        print(
            f"{row['qp']:>2} | {row['rate_mape_percent']:>6.2f} | "
            f"{row['pair_direction_accuracy']:>14.3f} | "
            f"{row['probe_real_delta_percent']:>15.3f} | "
            f"{row['probe_real_down_fraction']:>9.3f} | {row['passed']}"
        )
    print("\nEligible for Swin fine-tune:", result["eligible_for_swin_finetune"])
    print("Selected checkpoint:", selected_path)
    print("Report:", report_path)
    return result


def run_all() -> tuple[ProxyStage, dict]:
    stage = prepare()
    train_proxy(stage)
    result = report(stage)
    return stage, result


def recalibrate_low_qp(stage: ProxyStage) -> tuple[ProxyStage, dict]:
    """Refit absolute low-QP rate after a directionally sound first-stage proxy."""

    from dataclasses import replace

    start = stage.proxy_dir / "best_audit.pt"
    if not start.is_file():
        raise FileNotFoundError(f"Missing first-stage audit checkpoint: {start}")
    calibrated = replace(
        stage,
        proxy_dir=stage.root / "predictive_entropy_proxy_low_qp_calibration",
    )
    train_proxy(
        calibrated,
        epochs=3,
        lr=1e-4,
        rate_weight=0.30,
        init_checkpoint=start,
        qp_sampling_weights=[3.0, 2.0, 1.0, 1.0],
    )
    return calibrated, report(calibrated)
