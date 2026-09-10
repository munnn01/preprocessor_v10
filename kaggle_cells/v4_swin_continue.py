"""Resume the audited V4 VideoSwin run from epoch 5 through epoch 10 on Kaggle."""

from __future__ import annotations

from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from kaggle_cells.v4_swin_rate_recovery import (
    CLEAN_DEFAULT,
    QPS,
    SwinRecoveryStage,
    _link_members,
    _read_json,
    _saved_training_cli,
    _safe_source,
    evaluate_full,
    file_sha256,
)


TOTAL_EPOCHS = 10


def _is_v4_recovery_checkpoint(payload: dict) -> bool:
    args = payload.get("args", {})
    state = payload.get("rate_dual_state")
    feature_weights = args.get("feature_weights_by_qp")
    return (
        "preprocessor" in payload
        and isinstance(state, dict)
        and args.get("preprocessor") == "swin"
        and args.get("codec") == "h264"
        and list(args.get("codec_qps", [])) == QPS
        and args.get("rate_dual_control") is True
        and abs(float(args.get("target_bpp_ratio", 0.0)) - 0.95) < 1e-12
        and float(args.get("feature_weight", -1.0)) == 0.0
        and (feature_weights is None or list(map(float, feature_weights)) == [0.0] * 4)
        and float(args.get("mask_rate_weight", -1.0)) == 0.0
        and float(args.get("rate_dual_max_proxy_underestimate_percent", 0.0)) == 25.0
        and int(args.get("rate_dual_proxy_guard_patience", 0)) == 2
    )


def find_resume_checkpoint(input_root: Path, working: Path, torch_module) -> tuple[Path, dict]:
    candidates = list(
        input_root.glob("**/v4_swin_rate_recovery_*/swin_ratio_095/last.pt")
    )
    candidates.extend(
        working.glob("v4_swin_rate_recovery_*/swin_ratio_095/last.pt")
    )
    valid: list[tuple[Path, dict]] = []
    for path in sorted(set(candidates)):
        payload = torch_module.load(path, map_location="cpu", weights_only=False)
        if _is_v4_recovery_checkpoint(payload):
            valid.append((path, payload))
    if not valid:
        raise FileNotFoundError(
            "Add the Output containing "
            "v4_swin_rate_recovery_*/swin_ratio_095/last.pt from the completed 5-epoch run"
        )
    latest_epoch = max(int(payload.get("epoch", 0)) for _, payload in valid)
    latest = [(path, payload) for path, payload in valid if int(payload["epoch"]) == latest_epoch]
    hashes = {file_sha256(path) for path, _ in latest}
    if len(hashes) != 1:
        raise RuntimeError(
            f"Several different V4 recovery checkpoints exist at epoch {latest_epoch}; "
            "keep only the intended Input"
        )
    if latest_epoch >= TOTAL_EPOCHS:
        raise ValueError(
            f"The selected checkpoint is already at epoch {latest_epoch}; "
            "do not rerun the epoch-6-to-10 continuation"
        )
    return latest[0]


def find_recorded_proxy(
    expected_sha256: str,
    input_root: Path,
    working: Path,
    torch_module,
) -> tuple[Path, dict]:
    candidates = list(input_root.glob("**/large_reaudit/best_feasible_reaudit.pt"))
    candidates.extend(working.glob("**/large_reaudit/best_feasible_reaudit.pt"))
    candidates.extend(working.glob("**/frozen_proxy/best_feasible_reaudit.pt"))
    matches: list[tuple[Path, dict]] = []
    for path in sorted(set(candidates)):
        if file_sha256(path) != expected_sha256:
            continue
        payload = torch_module.load(path, map_location="cpu", weights_only=False)
        audit = payload.get("proxy_audit_reaudit", {})
        if (
            payload.get("proxy_config", {}).get("architecture") == "predictive_entropy_v1"
            and audit.get("feasible") is True
        ):
            matches.append((path, payload))
    if not matches:
        raise FileNotFoundError(
            "Add the Output containing the exact "
            "large_reaudit/best_feasible_reaudit.pt recorded by the 5-epoch run"
        )
    return matches[0]


def prepare(
    project: Path = Path("/kaggle/working/proxy_v4"),
    clean: Path = CLEAN_DEFAULT,
) -> tuple[SwinRecoveryStage, Path]:
    import torch
    from kaggle_cells.v4_proxy_stage import find_v7_root

    if not torch.cuda.is_available():
        raise RuntimeError("Enable GPU in Kaggle Settings before Run All")
    subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL)
    project = project.resolve(strict=True)
    clean = clean.resolve(strict=True)
    input_root, working = Path("/kaggle/input"), Path("/kaggle/working")
    source_resume, resume_payload = find_resume_checkpoint(input_root, working, torch)
    completed_epoch = int(resume_payload["epoch"])
    expected_proxy_hash = resume_payload.get("proxy_sha256")
    if not isinstance(expected_proxy_hash, str) or len(expected_proxy_hash) != 64:
        raise ValueError("The resume checkpoint has no valid proxy SHA-256")
    source_proxy, proxy_payload = find_recorded_proxy(
        expected_proxy_hash, input_root, working, torch
    )

    source_v7 = find_v7_root(input_root, working)
    split = _read_json(source_v7 / "recovery_split.json")
    train_members = list(split["train"])
    controller_members = list(split["controller_800"])
    if set(train_members) & set(controller_members):
        raise ValueError("V7 train and controller overlap")
    for member in train_members + controller_members:
        _safe_source(clean, member)

    run_root = working / (
        "v4_swin_continue_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    run_root.mkdir(parents=True, exist_ok=False)
    train_dir = run_root / "split/train"
    controller_dir = run_root / "split/controller_800"
    _link_members(clean, train_members, train_dir)
    _link_members(clean, controller_members, controller_dir)

    swin_dir = run_root / "swin_ratio_095"
    swin_dir.mkdir(parents=True, exist_ok=False)
    resume_copy = swin_dir / "last.pt"
    shutil.copy2(source_resume, resume_copy)
    for name in ("best_task_bd_rate.pt", "best_loss.pt", "best_ce.pt", "best_top1.pt"):
        source = source_resume.parent / name
        if source.is_file():
            shutil.copy2(source, swin_dir / name)
    anchor_source = source_resume.parent / "anchor_validation.json"
    if anchor_source.is_file():
        shutil.copy2(anchor_source, swin_dir / anchor_source.name)

    frozen_proxy_dir = run_root / "frozen_proxy"
    frozen_proxy_dir.mkdir(parents=True, exist_ok=False)
    proxy_copy = frozen_proxy_dir / "best_feasible_reaudit.pt"
    shutil.copy2(source_proxy, proxy_copy)
    if file_sha256(proxy_copy) != expected_proxy_hash:
        raise RuntimeError("Copied proxy failed its SHA-256 check")

    record = {
        "source_resume": str(source_resume),
        "source_resume_sha256": file_sha256(source_resume),
        "completed_epoch": completed_epoch,
        "next_epoch": completed_epoch + 1,
        "total_epochs": TOTAL_EPOCHS,
        "source_proxy": str(source_proxy),
        "proxy_sha256": expected_proxy_hash,
        "proxy_architecture": proxy_payload["proxy_config"]["architecture"],
        "source_v7": str(source_v7),
        "recovery_split_sha256": file_sha256(source_v7 / "recovery_split.json"),
        "train_videos": len(train_members),
        "controller_videos": len(controller_members),
        "objective_changed": False,
        "optimizer_restored": True,
        "rate_dual_state_restored": True,
    }
    (run_root / "continuation_selection.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    stage = SwinRecoveryStage(
        project=project,
        root=run_root,
        clean=clean,
        source_v7=source_v7,
        start_checkpoint=resume_copy,
        proxy_checkpoint=proxy_copy,
        train_dir=train_dir,
        controller_dir=controller_dir,
        swin_dir=swin_dir,
        saved_args=dict(resume_payload["args"]),
    )
    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "resume_from": str(source_resume),
                "saved_epoch": completed_epoch,
                "continue_epochs": f"{completed_epoch + 1}-{TOTAL_EPOCHS}",
                "proxy": str(proxy_copy),
                "train_videos": len(train_members),
                "controller_videos": len(controller_members),
            },
            indent=2,
        ),
        flush=True,
    )
    del resume_payload, proxy_payload
    return stage, resume_copy


def continue_training(
    stage: SwinRecoveryStage,
    resume_checkpoint: Path,
) -> tuple[Path, bool]:
    import torch
    import train as training

    parser = importlib.reload(training).build_parser()
    overrides = {
        "data_root": None,
        "train_dir": str(stage.train_dir),
        "val_dir": str(stage.controller_dir),
        "limit_train": None,
        "limit_val": None,
        "controller_limit_val": None,
        "init_checkpoint": None,
        "resume": str(resume_checkpoint),
        "refresh_proxy_on_resume": False,
        "proxy_checkpoint": str(stage.proxy_checkpoint),
        "output_dir": str(stage.swin_dir),
        "epochs": TOTAL_EPOCHS,
        "workers": 2,
        "device": "cuda",
        "validate_initial": False,
        "initial_validation_only": False,
        "smoke_test": False,
    }
    arguments = _saved_training_cli(stage.saved_args, overrides, parser)
    command = [sys.executable, "-u", str(stage.project / "train.py"), *arguments]
    (stage.root / "continue_command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8"
    )
    outcome = subprocess.run(command, cwd=stage.project, check=False)
    last = stage.swin_dir / "last.pt"
    if not last.is_file():
        raise RuntimeError(
            f"Continuation stopped before writing last.pt (exit code {outcome.returncode})"
        )
    last_payload = torch.load(last, map_location="cpu", weights_only=False)
    metrics = last_payload.get("val_metrics", {})
    if metrics.get("rate_dual_proxy_guard_abort"):
        raise RuntimeError("Proxy guard stopped the continuation; do not evaluate it")
    if int(last_payload.get("epoch", 0)) != TOTAL_EPOCHS:
        raise RuntimeError(
            f"Continuation stopped at epoch {last_payload.get('epoch')} of {TOTAL_EPOCHS}"
        )

    feasible_path = stage.swin_dir / "best_feasible.pt"
    diagnostic_path = stage.swin_dir / "best_task_bd_rate.pt"
    feasible = feasible_path.is_file()
    candidate = feasible_path if feasible else diagnostic_path
    if not candidate.is_file():
        raise FileNotFoundError("Continuation produced no Task BD-rate checkpoint")
    candidate_payload = torch.load(candidate, map_location="cpu", weights_only=False)
    candidate_metrics = candidate_payload.get("val_metrics", {})
    result = {
        "train_exit_code": outcome.returncode,
        "feasible": feasible,
        "candidate": str(candidate),
        "candidate_epoch": candidate_payload.get("epoch"),
        "controller_task_bd_rate_percent": candidate_metrics.get(
            "task_bd_rate_percent"
        ),
        "mean_bpp_ratio": candidate_metrics.get("mean_bpp_ratio"),
        "accuracy_feasible": candidate_metrics.get("accuracy_feasible"),
        "last_epoch": last_payload.get("epoch"),
        "last_task_bd_rate_percent": metrics.get("task_bd_rate_percent"),
        "last_mean_bpp_ratio": metrics.get("mean_bpp_ratio"),
        "next_action": (
            "Run full seven-QP evaluation."
            if feasible
            else "Do not run full evaluation; recalibrate the proxy on the epoch-10 outputs."
        ),
    }
    (stage.root / "continuation_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("\n" + json.dumps(result, indent=2), flush=True)
    return candidate, feasible


def run_all() -> tuple[SwinRecoveryStage, Path, bool, Path | None]:
    stage, resume_checkpoint = prepare()
    candidate, feasible = continue_training(stage, resume_checkpoint)
    evaluation_dir = evaluate_full(stage, candidate) if feasible else None
    return stage, candidate, feasible, evaluation_dir
