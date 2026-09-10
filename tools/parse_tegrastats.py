"""Summarize Jetson tegrastats module-power samples without guessing missing data."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


POWER_PATTERNS = (
    re.compile(r"\bVDD_IN\s+(\d+(?:\.\d+)?)mW(?:/(\d+(?:\.\d+)?)mW)?"),
    re.compile(r"\bPOM_5V_IN\s+(\d+(?:\.\d+)?)mW(?:/(\d+(?:\.\d+)?)mW)?"),
)


def parse_power_mw(text: str) -> tuple[str | None, list[float]]:
    """Return the first supported rail name and instantaneous milliwatt samples."""

    for pattern in POWER_PATTERNS:
        values = [float(match.group(1)) for match in pattern.finditer(text)]
        if values:
            return pattern.pattern.split("\\b")[1].split("\\s")[0], values
    return None, []


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(round(probability * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[index]


def summarize(path: Path) -> dict:
    rail, values = parse_power_mw(path.read_text(encoding="utf-8", errors="replace"))
    if not values:
        raise ValueError("no VDD_IN or legacy POM_5V_IN milliwatt samples found")
    return {
        "scientific_status": "measured_from_tegrastats_log",
        "source": str(path.resolve()),
        "rail": rail,
        "samples": len(values),
        "power_w": {
            "mean": statistics.fmean(values) / 1000.0,
            "median": statistics.median(values) / 1000.0,
            "p95": percentile(values, 0.95) / 1000.0,
            "minimum": min(values) / 1000.0,
            "maximum": max(values) / 1000.0,
        },
        "note": "Total-module rail samples; subtract an independently measured idle baseline only if the protocol declares that transformation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
