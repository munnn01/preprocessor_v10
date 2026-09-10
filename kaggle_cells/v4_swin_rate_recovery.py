"""Kaggle fine-tuning of VideoSwinLite with an audited predictive-entropy proxy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys


CLEAN_DEFAULT = Path(
    "/kaggle/input/datasets/qktttttttttt/kineticscleaned/cleaned_final/"
    "kinetics400_5per/kinetics400_5per/train"
)
QPS = [30, 35, 40, 45]
TARGET_BPP_RATIO = 0.95
EPOCHS = 5
LEARNING_RATE = 1e-5


@dataclass(frozen=True)
class SwinRecoveryStage:
    project: Path
    root: Path
    clean: Path
    source_v7: Path
    start_checkpoint: Path
    proxy_checkpoint: Path
    train_dir: Path
    controller_dir: Path
    swin_dir: Path
    saved_args: dict


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


def _link_members(clean: Path, members: list[str], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    resolved = [_safe_source(clean, member) for member in members]
    for relative, source in resolved:
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)


def _proxy_is_eligible(payload: dict) -> bool:
    config = payload.get("proxy_config", {})
    audit = payload.get("proxy_audit_reaudit")
    codec = payload.get("codec_config", {})
    if config.get("architecture") != "predictive_entropy_v1":
        return False
    if codec.get("codec") != "h264" or set(codec.get("qps", [])) != set(QPS):
        return False
    if not isinstance(audit, dict) or audit.get("feasible") is not True:
        return False
    rows = audit.get("per_qp", [])
    return (
        {int(row.get("qp", -1)) for row in rows} == set(QPS)
        and all(row.get("passed") is True for row in rows)
    )


def find_audited_proxy(input_root: Path, working: Path, torch_module) -> tuple[Path, dict]:
    candidates = list(input_root.glob("**/large_reaudit/best_feasible_reaudit.pt"))
    candidates.extend(working.glob("**/large_reaudit/best_feasible_reaudit.pt"))
    eligible: list[tuple[Path, dict]] = []
    for path in sorted(set(candidates)):
        payload = torch_module.load(path, map_location="cpu", weights_only=False)
        if _proxy_is_eligible(payload):
            eligible.append((path, payload))
    if not eligible:
        raise FileNotFoundError(
            "Add the Kaggle Output containing "
            "large_reaudit/best_feasible_reaudit.pt, or run in the session that created it"
        )
    hashes = {file_sha256(path) for path, _ in eligible}
    if len(hashes) != 1:
        raise RuntimeError(
            "Several different eligible re-audited proxies are attached; keep only the intended one"
        )
    return eligible[0]


def prepare(
    project: Path = Path("/kaggle/working/proxy_v4"),
    clean: Path = CLEAN_DEFAULT,
) -> SwinRecoveryStage:
    import torch
    from kaggle_cells.v4_proxy_stage import find_swin_checkpoint, find_v7_root

    if not torch.cuda.is_available():
        raise RuntimeError("Enable GPU in Kaggle Settings before Run All")
    subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL)
    project = project.resolve(strict=True)
    clean = clean.resolve(strict=True)
    working, input_root = Path("/kaggle/working"), Path("/kaggle/input")

    source_v7 = find_v7_root(input_root, working)
    split = _read_json(source_v7 / "recovery_split.json")
    selection = _read_json(source_v7 / "selection.json")
    train_members = list(split["train"])
    controller_members = list(split["controller_800"])
    if len(set(train_members)) != len(train_members):
        raise ValueError("V7 training split contains duplicate videos")
    if len(set(controller_members)) != len(controller_members):
        raise ValueError("V7 controller contains duplicate videos")
    if set(train_members) & set(controller_members):
        raise ValueError("V7 train and controller overlap")
    for member in train_members + controller_members:
        _safe_source(clean, member)

    expected_hash = selection.get("start_checkpoint_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("V7 selection.json has no valid start_checkpoint_sha256")
    start_checkpoint = find_swin_checkpoint(input_root, working, expected_hash)
    start_payload = torch.load(start_checkpoint, map_location="cpu", weights_only=False)
    saved_args = dict(start_payload.get("args", {}))
    if "preprocessor" not in start_payload or saved_args.get("preprocessor") != "swin":
        raise ValueError("The V6 checkpoint does not contain VideoSwinLite weights")

    proxy_checkpoint, proxy_payload = find_audited_proxy(input_root, working, torch)
    proxy_audit = proxy_payload["proxy_audit_reaudit"]
    run_root = working / (
        "v4_swin_rate_recovery_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    run_root.mkdir(parents=True, exist_ok=False)
    split_root = run_root / "split"
    train_dir = split_root / "train"
    controller_dir = split_root / "controller_800"
    _link_members(clean, train_members, train_dir)
    _link_members(clean, controller_members, controller_dir)

    record = {
        "source_v7": str(source_v7),
        "source_recovery_split_sha256": file_sha256(source_v7 / "recovery_split.json"),
        "start_checkpoint": str(start_checkpoint),
        "start_checkpoint_sha256": file_sha256(start_checkpoint),
        "start_epoch": start_payload.get("epoch"),
        "start_controller_task_bd_rate_percent": start_payload.get(
            "val_metrics", {}
        ).get("task_bd_rate_percent"),
        "proxy_checkpoint": str(proxy_checkpoint),
        "proxy_checkpoint_sha256": file_sha256(proxy_checkpoint),
        "proxy_architecture": proxy_payload["proxy_config"]["architecture"],
        "proxy_large_reaudit": proxy_audit,
        "clean_root": str(clean),
        "train_videos": len(train_members),
        "controller_videos": len(controller_members),
        "qps": QPS,
        "target_bpp_ratio": TARGET_BPP_RATIO,
        "feature_weights_by_qp": [0.0] * len(QPS),
        "mask_rate_weight": 0.0,
        "train_codec_source": "real",
    }
    (run_root / "selection.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    stage = SwinRecoveryStage(
        project=project,
        root=run_root,
        clean=clean,
        source_v7=source_v7,
        start_checkpoint=start_checkpoint,
        proxy_checkpoint=proxy_checkpoint,
        train_dir=train_dir,
        controller_dir=controller_dir,
        swin_dir=run_root / "swin_ratio_095",
        saved_args=saved_args,
    )
    print(
        json.dumps(
            {
                "run_root": str(stage.root),
                "start_VideoSwin": str(stage.start_checkpoint),
                "audited_proxy_V4": str(stage.proxy_checkpoint),
                "train_videos": len(train_members),
                "controller_videos": len(controller_members),
                "target_bpp_ratio": TARGET_BPP_RATIO,
                "feature_loss": 0.0,
                "mask_loss": 0.0,
                "gradient_path": "real H264 forward + Proxy V4 backward",
            },
            indent=2,
        ),
        flush=True,
    )
    return stage


def _saved_training_cli(saved: dict, overrides: dict, parser) -> list[str]:
    values = {**saved, **overrides}
    result: list[str] = []
    for action in parser._actions:
        value = values.get(action.dest)
        if action.dest == "help" or not action.option_strings or value is None:
            continue
        option = action.option_strings[0]
        if isinstance(action, argparse.BooleanOptionalAction):
            result.append(option if value else action.option_strings[1])
        elif isinstance(action, argparse._StoreTrueAction):
            if value:
                result.append(option)
        else:
            result.append(option)
            result.extend(
                map(str, value if isinstance(value, (list, tuple)) else [value])
            )
    return result


def training_arguments(stage: SwinRecoveryStage) -> list[str]:
    os.chdir(stage.project)
    if str(stage.project) not in sys.path:
        sys.path.insert(0, str(stage.project))
    import train as training

    parser = importlib.reload(training).build_parser()
    overrides = {
        "data_root": None,
        "train_dir": str(stage.train_dir),
        "val_dir": str(stage.controller_dir),
        "limit_train": None,
        "limit_val": None,
        "controller_limit_val": None,
        "init_checkpoint": str(stage.start_checkpoint),
        "resume": None,
        "refresh_proxy_on_resume": False,
        "proxy_checkpoint": str(stage.proxy_checkpoint),
        "preprocessor": "swin",
        "swin_qp_conditioning": True,
        "swin_gated_smoothing": True,
        "swin_smoothing_max_strength": 0.5,
        "max_residual": 0.10,
        "codec": "h264",
        "codec_qps": QPS,
        "train_codec_source": "real",
        "rate_lambda": [0.0],
        "alpha": 10.0,
        "distortion_reconstruction_weight": 1.0,
        "ce_weight": 1.0,
        "kd_weight": 0.5,
        "kd_temperature": 2.0,
        "feature_weight": 0.0,
        "feature_weights_by_qp": [0.0] * len(QPS),
        "max_top1_drop_pp": 0.0,
        "mask_rate_weight": 0.0,
        "mask_rate_outside_weight": 0.0,
        "mask_rate_temporal_weight": 0.0,
        "dual_rate_control": False,
        "rate_dual_control": True,
        "target_bpp_ratio": TARGET_BPP_RATIO,
        "rate_dual_parity_lambda": 0.05,
        "rate_dual_kappa": 3.0,
        "rate_dual_ema_beta": 0.8,
        "rate_dual_min": 0.0001,
        "rate_dual_max": 10.0,
        "rate_dual_max_proxy_underestimate_percent": 25.0,
        "rate_dual_proxy_guard_patience": 2,
        "dual_feasibility_tolerance": 0.005,
        "normalize_rate_by_anchor": False,
        "qp_sampling_weights": None,
        "epochs": EPOCHS,
        "batch_size": 1,
        "accumulation_steps": 4,
        "lr": LEARNING_RATE,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "clip_grad": 1.0,
        "workers": 2,
        "amp": True,
        "device": "cuda",
        "validate_initial": True,
        "initial_validation_only": False,
        "checkpoint_metric": "task_bd_rate",
        "output_dir": str(stage.swin_dir),
        "smoke_test": False,
    }
    return _saved_training_cli(stage.saved_args, overrides, parser)


def fine_tune(stage: SwinRecoveryStage) -> tuple[Path, bool]:
    import torch

    arguments = training_arguments(stage)
    command = [sys.executable, "-u", str(stage.project / "train.py"), *arguments]
    (stage.root / "train_command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8"
    )
    outcome = subprocess.run(command, cwd=stage.project, check=False)
    last = stage.swin_dir / "last.pt"
    if not last.is_file():
        raise RuntimeError(
            f"Training stopped before writing last.pt (exit code {outcome.returncode})"
        )
    last_payload = torch.load(last, map_location="cpu", weights_only=False)
    last_metrics = last_payload.get("val_metrics", {})
    if last_metrics.get("rate_dual_proxy_guard_abort"):
        raise RuntimeError(
            "The proxy drift guard stopped the run; inspect last.pt before continuing"
        )
    if int(last_payload.get("epoch", 0)) != EPOCHS:
        raise RuntimeError(
            f"Training stopped at epoch {last_payload.get('epoch')} of {EPOCHS}"
        )

    feasible_path = stage.swin_dir / "best_feasible.pt"
    diagnostic_path = stage.swin_dir / "best_task_bd_rate.pt"
    feasible = feasible_path.is_file()
    candidate = feasible_path if feasible else diagnostic_path
    if not candidate.is_file():
        raise FileNotFoundError("Training produced no Task BD-rate checkpoint")
    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    metrics = payload.get("val_metrics", {})
    rows = []
    for qp in QPS:
        rows.append(
            {
                "qp": qp,
                "bpp_ratio": metrics.get(f"qp{qp}_bpp_ratio"),
                "top1_drop_pp": metrics.get(f"qp{qp}_top1_drop_pp"),
                "rate_dual_weight": metrics.get(f"qp{qp}_rate_dual_weight"),
                "proxy_drift_percent": metrics.get(f"qp{qp}_proxy_drift_percent"),
            }
        )
    result = {
        "train_exit_code": outcome.returncode,
        "feasible": feasible,
        "candidate": str(candidate),
        "candidate_epoch": payload.get("epoch"),
        "controller_task_bd_rate_percent": metrics.get("task_bd_rate_percent"),
        "mean_bpp_ratio": metrics.get("mean_bpp_ratio"),
        "accuracy_feasible": metrics.get("accuracy_feasible"),
        "per_qp": rows,
        "next_action": (
            "Evaluate best_feasible.pt on the full seven-QP validation split."
            if feasible
            else "The best-task checkpoint is diagnostic; do not claim the final result."
        ),
    }
    result_path = stage.root / "swin_recovery_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nQP | BPP ratio | Top-1 drop pp | Dual weight | Proxy drift %")
    for row in rows:
        print(
            f"{row['qp']:>2} | {row['bpp_ratio']!s:>9} | "
            f"{row['top1_drop_pp']!s:>13} | {row['rate_dual_weight']!s:>11} | "
            f"{row['proxy_drift_percent']!s:>13}"
        )
    print("\n" + json.dumps(result, indent=2), flush=True)
    return candidate, feasible


def _find_validation_manifest(stage: SwinRecoveryStage) -> tuple[Path, dict]:
    recovery = _read_json(stage.source_v7 / "recovery_split.json")
    expected_hash = recovery.get("source_manifest_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("V7 recovery_split.json has no source manifest SHA-256")
    roots = [Path("/kaggle/input"), Path("/kaggle/working")]
    candidates: list[Path] = []
    for root in roots:
        if root.is_dir():
            candidates.extend(root.glob("**/v5_fixed_split/split_manifest.json"))
    matches = [
        path for path in sorted(set(candidates)) if file_sha256(path) == expected_hash
    ]
    if not matches:
        raise FileNotFoundError(
            "Add the original checking Output containing "
            "v5_fixed_split/split_manifest.json for full validation"
        )
    payload = _read_json(matches[0])
    if not payload.get("groups", {}).get("validation_full"):
        raise ValueError("The matched split manifest has no validation_full members")
    return matches[0], payload


def prepare_full_validation(stage: SwinRecoveryStage) -> Path:
    destination = stage.root / "split/validation_full"
    if destination.is_dir():
        return destination
    manifest_path, manifest = _find_validation_manifest(stage)
    members = list(manifest["groups"]["validation_full"])
    if set(members) & set(_read_json(stage.source_v7 / "recovery_split.json")["train"]):
        raise ValueError("Full validation overlaps the training split")
    _link_members(stage.clean, members, destination)
    (stage.root / "full_validation_selection.json").write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
                "videos": len(members),
                "members": members,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Full validation: {destination} ({len(members)} videos)", flush=True)
    return destination


def evaluate_full(stage: SwinRecoveryStage, candidate: Path) -> Path:
    import torch

    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    metrics = payload.get("val_metrics", {})
    if not metrics.get("dual_feasible") or not metrics.get("accuracy_feasible"):
        raise ValueError("Only a feasible controller checkpoint may enter full evaluation")
    validation_dir = prepare_full_validation(stage)
    output_dir = stage.root / "full_evaluation_7qp"
    saved = payload.get("args", {})
    command = [
        sys.executable,
        "-u",
        str(stage.project / "evaluate_real_codec.py"),
        "--checkpoint",
        str(candidate),
        "--test-dir",
        str(validation_dir),
        "--codecs",
        "h264",
        "--qps",
        "30",
        "32",
        "35",
        "37",
        "40",
        "42",
        "45",
        "--frames",
        str(saved.get("frames", 16)),
        "--frame-stride",
        str(saved.get("frame_stride", 2)),
        "--frame-size",
        str(saved.get("frame_size", 128)),
        "--fps",
        str(saved.get("codec_fps", 30)),
        "--preset",
        str(saved.get("codec_preset", "medium")),
        "--bootstrap-samples",
        "2000",
        "--bootstrap-seed",
        "2026",
        "--confidence-level",
        "0.95",
        "--include-clean-reference",
        "--device",
        "cuda",
        "--output-dir",
        str(output_dir),
    ]
    (stage.root / "evaluation_command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8"
    )
    subprocess.run(command, cwd=stage.project, check=True)

    bd_rate = _read_json(output_dir / "bd_rate.json")["h264"]
    rows = _read_json(output_dir / "metrics.json")
    by_key = {(row["method"], int(row["qp"])): row for row in rows}
    task = bd_rate.get("task_bd_rate_percent")
    interval = bd_rate.get("task_bootstrap", {})
    result = {
        "task_bd_rate_percent": task,
        "bootstrap_lower_percent": interval.get("lower_percent"),
        "bootstrap_upper_percent": interval.get("upper_percent"),
        "point_estimate_below_minus_10": (
            isinstance(task, (int, float)) and math.isfinite(task) and task < -10.0
        ),
        "interval_upper_below_minus_10": (
            isinstance(interval.get("upper_percent"), (int, float))
            and math.isfinite(interval["upper_percent"])
            and interval["upper_percent"] < -10.0
        ),
        "top1_at_least_anchor_every_qp": all(
            by_key[("preprocessed", qp)]["top1"]
            >= by_key[("anchor", qp)]["top1"]
            for qp in (30, 32, 35, 37, 40, 42, 45)
        ),
        "bpp_no_more_than_anchor_every_qp": all(
            by_key[("preprocessed", qp)]["bpp"]
            <= by_key[("anchor", qp)]["bpp"]
            for qp in (30, 32, 35, 37, 40, 42, 45)
        ),
        "output_dir": str(output_dir),
    }
    (output_dir / "goal_check.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("\nFINAL FULL-VALIDATION CHECK")
    print(json.dumps(result, indent=2), flush=True)
    return output_dir


def run_all() -> tuple[SwinRecoveryStage, Path, bool, Path | None]:
    stage = prepare()
    candidate, feasible = fine_tune(stage)
    evaluation_dir = evaluate_full(stage, candidate) if feasible else None
    return stage, candidate, feasible, evaluation_dir
