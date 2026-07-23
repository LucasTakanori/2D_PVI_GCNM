#!/usr/bin/env python3
"""Compare subject010 reference, coordinate-3ch, and Newton+coordinate-6ch BP runs."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def _stats(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    subject = "subject010"
    modes = ("waveform", "fiducials")
    bases = {
        "reference_parquet": root / "artifacts/pvi_bp_us120_reference_parquet_v1",
        "coordinate_3ch": root / "artifacts/subject010_coordinate_direct_s1_s2_ds2_bp_v1",
        "newton_coordinate_6ch": root / "artifacts/subject010_newton_coordinate_6ch_bp_v1",
    }
    targets = {
        "reference_parquet": lambda mode: f"reference-parquet-crt-img-to-{mode}",
        "coordinate_3ch": lambda mode: f"gcnm-coordinate-3ch-crt-image-to-{mode}",
        "newton_coordinate_6ch": lambda mode: f"gcnm-newton_coordinate-6ch-crt-image-to-{mode}",
    }
    results = {}
    for representation, base in bases.items():
        results[representation] = {}
        for mode in modes:
            path = base / targets[representation](mode) / "main/statistics" / f"{subject}_statistics.json"
            results[representation][mode] = _stats(path)
    for representation in results:
        for mode in modes:
            stats = results[representation][mode]
            stats["dbp_r"] = math.sqrt(max(0.0, float(stats["dbp_r2"])))
            stats["sbp_r"] = math.sqrt(max(0.0, float(stats["sbp_r2"])))
            stats["bp_accuracy"] = 0.5 * (stats["dbp_r"] + stats["sbp_r"])
    metrics = (
        "bp_accuracy", "dbp_r", "sbp_r", "dbp_r2", "sbp_r2",
        "dbp_cc", "sbp_cc", "amae", "armse", "dbp_mae", "sbp_mae",
    )
    deltas = {
        mode: {
            metric: float(
                results["newton_coordinate_6ch"][mode][metric]
                - results["coordinate_3ch"][mode][metric]
            )
            for metric in metrics
        }
        for mode in modes
    }
    correlation_gate = all(
        deltas[mode][metric] >= 0
        for mode in modes
        for metric in ("bp_accuracy", "dbp_r", "sbp_r", "dbp_r2", "sbp_r2")
    )
    calibration_gate = all(
        deltas[mode][metric] >= 0
        for mode in modes
        for metric in ("dbp_cc", "sbp_cc")
    ) and all(deltas[mode]["amae"] <= 0 for mode in modes)
    if correlation_gate and calibration_gate:
        recommendation = "retain raw Newton+coordinate 6ch for the cohort rollout"
    elif correlation_gate:
        recommendation = (
            "retain the 6ch information, but repair calibration with separate stems or "
            "validation-only affine calibration before cohort rollout"
        )
    else:
        recommendation = (
            "raw fusion loses correlation; pilot separate Newton/coordinate stems before rollout"
        )
    report = {
        "schema": "pvi-subject010-channel-contract-comparison-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject": subject,
        "results": results,
        "deltas_6ch_minus_3ch": deltas,
        "six_channel_passes_correlation_gate": correlation_gate,
        "six_channel_passes_calibration_and_error_gate": calibration_gate,
        "recommendation": recommendation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
