"""Dataset and Bjontegaard helpers shared by final evaluation scripts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset, Subset

from .data import (
    VideoFolderDataset,
    stratified_limit_indices,
    stratified_split_indices,
)


def build_evaluation_dataset(
    *,
    categories: Sequence[str],
    data_root: str | Path | None = None,
    test_dir: str | Path | None = None,
    split: str = "val",
    frames: int = 16,
    stride: int = 2,
    size: int = 128,
    limit: int | None = None,
    val_ratio: float | None = None,
    seed: int | None = None,
    saved_args: Mapping[str, Any] | None = None,
) -> Dataset:
    """Build a physical test split or reproduce the training validation split.

    An explicit ``test_dir`` or an existing ``data_root/split`` directory is
    evaluated directly.  Otherwise ``data_root`` is treated as the training
    class-folder tree and the same deterministic stratified split used by
    ``train.py`` is recreated in memory.
    """

    if test_dir is not None:
        dataset = VideoFolderDataset(
            Path(test_dir),
            categories,
            frames=frames,
            stride=stride,
            size=size,
            train=False,
            limit=limit,
        )
        print(f"[eval data] explicit directory: {len(dataset)} videos")
        return dataset

    if data_root is None:
        raise ValueError("provide --data-root or --test-dir")

    root = Path(data_root)
    split_root = root / split
    if split_root.is_dir():
        dataset = VideoFolderDataset(
            split_root,
            categories,
            frames=frames,
            stride=stride,
            size=size,
            train=False,
            limit=limit,
        )
        print(f"[eval data] physical {split!r} split: {len(dataset)} videos")
        return dataset

    if split not in {"val", "validation"}:
        raise FileNotFoundError(
            f"requested split directory does not exist: {split_root}; "
            "automatic splitting is supported only for val/validation"
        )

    train_root = root / "train" if (root / "train").is_dir() else root
    source = VideoFolderDataset(
        train_root,
        categories,
        frames=frames,
        stride=stride,
        size=size,
        train=False,
    )
    checkpoint_args = saved_args or {}
    effective_ratio = (
        float(val_ratio)
        if val_ratio is not None
        else float(checkpoint_args.get("val_ratio", 0.2))
    )
    effective_seed = (
        int(seed) if seed is not None else int(checkpoint_args.get("seed", 42))
    )
    _, validation_indices = stratified_split_indices(
        source.samples, effective_ratio, effective_seed
    )
    validation_indices = stratified_limit_indices(
        source.samples,
        validation_indices,
        limit,
        effective_seed + 202,
    )
    dataset = Subset(source, validation_indices)
    print(
        "[eval data] no physical validation directory; "
        f"recreated stratified validation={len(dataset)} "
        f"ratio={effective_ratio:.3f} seed={effective_seed}"
    )
    return dataset


def dataset_sample_path(dataset: Dataset, index: int) -> Path:
    """Return the source path through any nested ``Subset`` wrappers."""

    current: Dataset = dataset
    current_index = index
    while isinstance(current, Subset):
        current_index = int(current.indices[current_index])
        current = current.dataset
    samples = getattr(current, "samples", None)
    if samples is None:
        raise TypeError("evaluation dataset does not expose source samples")
    return Path(samples[current_index][0])


def _prepare_rd_curve(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    quality_key: str,
    *,
    apply_monotone_envelope: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    points = sorted(
        (
            (float(row["bpp"]), float(row[quality_key]))
            for row in rows
            if row["method"] == method
        ),
        key=lambda point: point[0],
    )
    diagnostics: dict[str, Any] = {
        "method": method,
        "raw_points": [
            {"bpp": float(rate), "quality": float(quality)} for rate, quality in points
        ],
        "raw_point_count": len(points),
        "processed_points": [],
        "processed_point_count": 0,
        "dropped_point_count": 0,
        "invalid_reason": None,
    }
    if not points:
        diagnostics["invalid_reason"] = "no_points"
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), diagnostics

    rates = np.asarray([point[0] for point in points], dtype=np.float64)
    qualities = np.asarray([point[1] for point in points], dtype=np.float64)
    if np.any(rates <= 0) or not np.all(np.isfinite(rates)):
        raise ValueError("BD-rate requires finite positive BPP values")
    if not np.all(np.isfinite(qualities)):
        raise ValueError("BD-rate requires finite quality values")

    if apply_monotone_envelope:
        # Accuracy measured on a finite validation set can move down by one sample.
        # Preserve the raw points above, then remove dominated higher-rate points.
        qualities = np.maximum.accumulate(qualities)
        quality_to_rate: dict[float, float] = {}
        for quality, rate in zip(qualities, rates, strict=True):
            key = float(quality)
            quality_to_rate[key] = min(quality_to_rate.get(key, math.inf), float(rate))
        unique_qualities = np.asarray(sorted(quality_to_rate), dtype=np.float64)
        unique_rates = np.asarray(
            [quality_to_rate[quality] for quality in unique_qualities], dtype=np.float64
        )
    else:
        if len(qualities) > 1 and np.any(np.diff(qualities) <= 0):
            diagnostics["invalid_reason"] = "raw_quality_not_strictly_increasing"
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64), diagnostics
        unique_qualities, unique_rates = qualities, rates

    diagnostics["processed_points"] = [
        {"bpp": float(rate), "quality": float(quality)}
        for quality, rate in zip(unique_qualities, unique_rates, strict=True)
    ]
    diagnostics["processed_point_count"] = len(unique_qualities)
    diagnostics["dropped_point_count"] = len(points) - len(unique_qualities)
    if len(unique_qualities) < 2:
        diagnostics["invalid_reason"] = "fewer_than_two_distinct_quality_points"
    return unique_qualities, unique_rates, diagnostics


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return shape-preserving PCHIP derivatives for strictly increasing ``x``."""

    if len(x) != len(y) or len(x) < 2:
        raise ValueError("PCHIP requires at least two paired points")
    intervals = np.diff(x)
    if np.any(intervals <= 0):
        raise ValueError("PCHIP coordinates must be strictly increasing")
    secants = np.diff(y) / intervals
    if len(x) == 2:
        return np.asarray([secants[0], secants[0]], dtype=np.float64)

    slopes = np.zeros_like(x, dtype=np.float64)
    for index in range(1, len(x) - 1):
        previous, following = secants[index - 1], secants[index]
        if previous == 0.0 or following == 0.0 or np.sign(previous) != np.sign(following):
            slopes[index] = 0.0
            continue
        previous_interval, following_interval = intervals[index - 1], intervals[index]
        weight_left = 2.0 * following_interval + previous_interval
        weight_right = following_interval + 2.0 * previous_interval
        slopes[index] = (weight_left + weight_right) / (
            weight_left / previous + weight_right / following
        )

    def endpoint_slope(
        near_interval: float,
        far_interval: float,
        near_secant: float,
        far_secant: float,
    ) -> float:
        value = (
            (2.0 * near_interval + far_interval) * near_secant
            - near_interval * far_secant
        ) / (near_interval + far_interval)
        if np.sign(value) != np.sign(near_secant):
            return 0.0
        if np.sign(near_secant) != np.sign(far_secant) and abs(value) > 3.0 * abs(
            near_secant
        ):
            return 3.0 * near_secant
        return float(value)

    slopes[0] = endpoint_slope(intervals[0], intervals[1], secants[0], secants[1])
    slopes[-1] = endpoint_slope(
        intervals[-1], intervals[-2], secants[-1], secants[-2]
    )
    return slopes


def _integrate_pchip(
    x: np.ndarray, y: np.ndarray, lower: float, upper: float
) -> float:
    """Integrate a shape-preserving cubic Hermite interpolant on a closed interval."""

    if lower < float(x[0]) or upper > float(x[-1]) or upper < lower:
        raise ValueError("PCHIP integration bounds must lie within the data range")
    slopes = _pchip_slopes(x, y)

    def primitives(value: float) -> tuple[float, float, float, float]:
        square = value * value
        cube = square * value
        fourth = cube * value
        return (
            0.5 * fourth - cube + value,
            0.25 * fourth - (2.0 / 3.0) * cube + 0.5 * square,
            -0.5 * fourth + cube,
            0.25 * fourth - (1.0 / 3.0) * cube,
        )

    area = 0.0
    for index in range(len(x) - 1):
        segment_lower = max(lower, float(x[index]))
        segment_upper = min(upper, float(x[index + 1]))
        if segment_upper <= segment_lower:
            continue
        width = float(x[index + 1] - x[index])
        start = (segment_lower - float(x[index])) / width
        stop = (segment_upper - float(x[index])) / width
        start_basis = primitives(start)
        stop_basis = primitives(stop)
        integrated_basis = tuple(
            stop_value - start_value
            for start_value, stop_value in zip(start_basis, stop_basis, strict=True)
        )
        area += width * (
            float(y[index]) * integrated_basis[0]
            + width * float(slopes[index]) * integrated_basis[1]
            + float(y[index + 1]) * integrated_basis[2]
            + width * float(slopes[index + 1]) * integrated_basis[3]
        )
    return area


def calculate_bd_rate_details(
    rows: Sequence[Mapping[str, Any]],
    quality_key: str,
    *,
    anchor_method: str = "anchor",
    proposed_method: str = "preprocessed",
    apply_monotone_envelope: bool = True,
) -> dict[str, Any]:
    """Return PCHIP BD-rate and diagnostics over the shared quality interval."""

    anchor_quality, anchor_rate, anchor_diagnostics = _prepare_rd_curve(
        rows,
        anchor_method,
        quality_key,
        apply_monotone_envelope=apply_monotone_envelope,
    )
    proposed_quality, proposed_rate, proposed_diagnostics = _prepare_rd_curve(
        rows,
        proposed_method,
        quality_key,
        apply_monotone_envelope=apply_monotone_envelope,
    )
    details: dict[str, Any] = {
        "interpolation": "pchip",
        "curve_policy": "monotone_pareto_envelope" if apply_monotone_envelope else "raw",
        "anchor_method": anchor_method,
        "proposed_method": proposed_method,
        "anchor_points": len(anchor_quality),
        "preprocessed_points": len(proposed_quality),
        "anchor_curve": anchor_diagnostics,
        "proposed_curve": proposed_diagnostics,
        "quality_min": None,
        "quality_max": None,
        "overlap_span": 0.0,
        "bd_rate_percent": None,
    }
    if len(anchor_quality) < 2 or len(proposed_quality) < 2:
        return details

    quality_min = max(float(anchor_quality.min()), float(proposed_quality.min()))
    quality_max = min(float(anchor_quality.max()), float(proposed_quality.max()))
    details.update(
        {
            "quality_min": quality_min,
            "quality_max": quality_max,
            "overlap_span": max(0.0, quality_max - quality_min),
        }
    )
    if quality_max <= quality_min:
        return details

    anchor_area = _integrate_pchip(
        anchor_quality, np.log(anchor_rate), quality_min, quality_max
    )
    proposed_area = _integrate_pchip(
        proposed_quality, np.log(proposed_rate), quality_min, quality_max
    )
    interval = quality_max - quality_min
    bd_rate = (math.exp((proposed_area - anchor_area) / interval) - 1.0) * 100.0
    details["bd_rate_percent"] = float(bd_rate)
    return details


def calculate_bd_rate(
    rows: Sequence[Mapping[str, Any]],
    quality_key: str,
    *,
    anchor_method: str = "anchor",
    proposed_method: str = "preprocessed",
    apply_monotone_envelope: bool = True,
) -> float | None:
    """Return preprocessed-vs-anchor PCHIP BD-rate over shared quality."""

    value = calculate_bd_rate_details(
        rows,
        quality_key,
        anchor_method=anchor_method,
        proposed_method=proposed_method,
        apply_monotone_envelope=apply_monotone_envelope,
    )["bd_rate_percent"]
    return None if value is None else float(value)


def bootstrap_bd_rate(
    rows: Sequence[Mapping[str, Any]],
    quality_key: str,
    *,
    anchor_method: str = "anchor",
    proposed_method: str = "preprocessed",
    samples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 2026,
    psnr_aggregation: str = "psnr_from_mean_video_mse",
    apply_monotone_envelope: bool = True,
) -> dict[str, Any]:
    """Estimate a paired video-level confidence interval for aggregate BD-rate."""

    if samples < 0:
        raise ValueError("bootstrap samples must be non-negative")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must lie strictly between zero and one")
    if psnr_aggregation not in {"psnr_from_mean_video_mse", "mean_video_psnr"}:
        raise ValueError("unknown PSNR aggregation")
    selected = [
        row for row in rows if str(row["method"]) in {anchor_method, proposed_method}
    ]
    sample_indices = sorted(
        {
            str(row["sample_id"] if "sample_id" in row else row["sample_index"])
            for row in selected
        }
    )
    if not sample_indices:
        raise ValueError("bootstrap requires at least one per-video row")

    expected = {(str(row["method"]), int(row["qp"])) for row in selected}
    methods = {method for method, _ in expected}
    if methods != {anchor_method, proposed_method}:
        raise ValueError(
            f"bootstrap requires methods {anchor_method!r} and {proposed_method!r}"
        )
    anchor_qps = {qp for method, qp in expected if method == anchor_method}
    proposed_qps = {qp for method, qp in expected if method == proposed_method}
    if anchor_qps != proposed_qps:
        raise ValueError("bootstrap methods must contain identical QP sets")
    sample_positions = {
        sample_index: position for position, sample_index in enumerate(sample_indices)
    }
    if quality_key == "psnr_db":
        quality_source = "mse" if psnr_aggregation == "psnr_from_mean_video_mse" else "psnr_db"
    elif quality_key == "top1_percent":
        quality_source = "top1"
    else:
        quality_source = quality_key
    values_by_operating_point = {
        key: {
            metric: np.full(len(sample_indices), np.nan, dtype=np.float64)
            for metric in ("bpp", quality_source)
        }
        for key in expected
    }
    for row in selected:
        sample_index = str(row["sample_id"] if "sample_id" in row else row["sample_index"])
        position = sample_positions[sample_index]
        key = (str(row["method"]), int(row["qp"]))
        values = values_by_operating_point[key]
        if math.isfinite(float(values["bpp"][position])):
            raise ValueError(
                "paired bootstrap received duplicate rows for "
                f"sample={sample_index}, method={key[0]}, QP={key[1]}"
            )
        for metric in values:
            values[metric][position] = float(row[metric])
    incomplete = sorted(
        sample_index
        for sample_index, position in sample_positions.items()
        if any(
            not all(math.isfinite(float(values[metric][position])) for metric in values)
            for values in values_by_operating_point.values()
        )
    )
    if incomplete:
        raise ValueError(
            "paired bootstrap requires every video at every method/QP; "
            f"incomplete sample indices: {incomplete[:8]}"
        )

    def aggregate(draw: np.ndarray) -> list[dict[str, float | int | str]]:
        aggregated: list[dict[str, float | int | str]] = []
        for (method, qp), values in sorted(values_by_operating_point.items()):
            quality_value = float(values[quality_source][draw].mean())
            if quality_key == "psnr_db" and quality_source == "mse":
                quality_value = -10.0 * math.log10(max(quality_value, 1e-12))
            elif quality_key == "top1_percent":
                quality_value *= 100.0
            aggregated.append({
                "method": method,
                "qp": qp,
                "bpp": float(values["bpp"][draw].mean()),
                quality_key: quality_value,
            })
        return aggregated

    point_rows = aggregate(np.arange(len(sample_indices)))
    point_estimate = calculate_bd_rate(
        point_rows,
        quality_key,
        anchor_method=anchor_method,
        proposed_method=proposed_method,
        apply_monotone_envelope=apply_monotone_envelope,
    )
    result: dict[str, Any] = {
        "method": "paired_video_bootstrap",
        "anchor_method": anchor_method,
        "proposed_method": proposed_method,
        "quality_metric": quality_key,
        "psnr_aggregation": psnr_aggregation if quality_key == "psnr_db" else None,
        "curve_policy": (
            "monotone_pareto_envelope" if apply_monotone_envelope else "raw"
        ),
        "confidence_level": float(confidence_level),
        "seed": int(seed),
        "samples_requested": int(samples),
        "samples_valid": 0,
        "samples_invalid": int(samples),
        "valid_fraction": 0.0,
        "point_estimate_percent": point_estimate,
        "median_percent": None,
        "lower_percent": None,
        "upper_percent": None,
    }
    if samples == 0:
        return result

    generator = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(samples):
        draw = generator.integers(0, len(sample_indices), size=len(sample_indices))
        estimate = calculate_bd_rate(
            aggregate(draw),
            quality_key,
            anchor_method=anchor_method,
            proposed_method=proposed_method,
            apply_monotone_envelope=apply_monotone_envelope,
        )
        if estimate is not None and math.isfinite(estimate):
            estimates.append(float(estimate))

    result["samples_valid"] = len(estimates)
    result["samples_invalid"] = samples - len(estimates)
    result["valid_fraction"] = len(estimates) / samples if samples else 0.0
    if estimates:
        alpha = (1.0 - confidence_level) / 2.0
        result.update(
            {
                "median_percent": float(np.quantile(estimates, 0.5)),
                "lower_percent": float(np.quantile(estimates, alpha)),
                "upper_percent": float(np.quantile(estimates, 1.0 - alpha)),
            }
        )
    return result
