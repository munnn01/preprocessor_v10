"""Acceptance checks for using a learned rate proxy in preprocessor training."""

from __future__ import annotations

import math
from collections.abc import Sequence


def audit_proxy_metrics(
    metrics: dict[str, float],
    qps: Sequence[int],
    *,
    max_rate_mape_percent: float = 20.0,
    min_pair_direction_accuracy: float = 0.60,
    max_real_delta_percent: float = 0.0,
    min_real_down_fraction: float = 0.55,
) -> dict:
    """Gate a proxy using real-codec rate descent and paired ordering per QP."""

    thresholds = {
        "max_rate_mape_percent": float(max_rate_mape_percent),
        "min_pair_direction_accuracy": float(min_pair_direction_accuracy),
        "max_real_delta_percent": float(max_real_delta_percent),
        "min_real_down_fraction": float(min_real_down_fraction),
    }
    if not qps:
        raise ValueError("audit requires at least one QP")
    if not 0.0 <= thresholds["min_pair_direction_accuracy"] <= 1.0:
        raise ValueError("min_pair_direction_accuracy must be in [0, 1]")
    if not 0.0 <= thresholds["min_real_down_fraction"] <= 1.0:
        raise ValueError("min_real_down_fraction must be in [0, 1]")
    if thresholds["max_rate_mape_percent"] < 0:
        raise ValueError("max_rate_mape_percent must be nonnegative")

    rows = []
    for qp in (int(value) for value in qps):
        names = {
            "rate_mape_percent": f"qp{qp}_rate_mape_percent",
            "pair_direction_accuracy": f"qp{qp}_pair_direction_accuracy",
            "probe_real_delta_percent": f"qp{qp}_probe_real_delta_percent",
            "probe_real_down_fraction": f"qp{qp}_probe_real_down_fraction",
        }
        values = {name: metrics.get(key) for name, key in names.items()}
        finite = all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in values.values()
        )
        reasons = []
        if not finite:
            reasons.append("missing_or_nonfinite_metric")
        else:
            if values["rate_mape_percent"] > thresholds["max_rate_mape_percent"]:
                reasons.append("rate_mape_too_high")
            if values["pair_direction_accuracy"] < thresholds["min_pair_direction_accuracy"]:
                reasons.append("pair_direction_too_low")
            if values["probe_real_delta_percent"] >= thresholds["max_real_delta_percent"]:
                reasons.append("real_bpp_did_not_decrease")
            if values["probe_real_down_fraction"] < thresholds["min_real_down_fraction"]:
                reasons.append("real_down_fraction_too_low")
        rows.append({"qp": qp, **values, "passed": not reasons, "reasons": reasons})

    finite_rows = [row for row in rows if row["reasons"] != ["missing_or_nonfinite_metric"]]
    if len(finite_rows) == len(rows):
        mean_mape = sum(row["rate_mape_percent"] for row in rows) / len(rows)
        mean_pair = sum(row["pair_direction_accuracy"] for row in rows) / len(rows)
        mean_down = sum(row["probe_real_down_fraction"] for row in rows) / len(rows)
        mean_real_delta = sum(row["probe_real_delta_percent"] for row in rows) / len(rows)
        score = (
            mean_mape
            + 25.0 * (1.0 - mean_pair)
            + 25.0 * (1.0 - mean_down)
            + 2.0 * max(mean_real_delta, 0.0)
        )
    else:
        mean_mape = mean_pair = mean_down = mean_real_delta = None
        score = math.inf
    return {
        "feasible": all(row["passed"] for row in rows),
        "score": score,
        "thresholds": thresholds,
        "mean_rate_mape_percent": mean_mape,
        "mean_pair_direction_accuracy": mean_pair,
        "mean_probe_real_delta_percent": mean_real_delta,
        "mean_probe_real_down_fraction": mean_down,
        "per_qp": rows,
    }
