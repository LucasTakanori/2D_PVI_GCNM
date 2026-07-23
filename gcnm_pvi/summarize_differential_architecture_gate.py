#!/usr/bin/env python3
"""Summarize completed differential candidates against one-step PVI Newton."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VARIANTS = ("coordinate_direct", "coordinate_global", "coordinate_residual")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    newton = None
    for variant in VARIANTS:
        path = args.results_root / variant / "evaluation" / "report.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if newton is None:
            newton = report["saved_newton"]
        for stage in report["faithful_gcnm"]:
            metrics = stage["metrics"]
            relative = stage["relative_to_saved_one_step_newton"]
            rows.append(
                {
                    "variant": variant,
                    "stage": stage["stage"],
                    "classification": relative["classification"],
                    "newton_relative_improvements": relative["improvements"],
                    "element_nrmse": metrics["element_nrmse"],
                    "image_correlation": metrics["image_correlation"],
                    "localization_dice": metrics["localization_dice_at_true_volume"],
                    "sign_accuracy": metrics["support_sign_accuracy"],
                    "rms_ratio": metrics["prediction_to_truth_rms"],
                    "frame_dynamics_ratio": metrics["frame_difference_rms_ratio"],
                    "voltage_residual_rms": stage["physics"][
                        "post_gcn_voltage_residual_rms_mean"
                    ],
                }
            )
    if not rows or newton is None:
        raise ValueError(f"no completed architecture reports under {args.results_root}")
    baseline = newton["metrics"]
    lines = [
        "# Differential GCNM architecture gate",
        "",
        "All candidates use the same signed differential beat and are compared with",
        "the production PVI one-step Newton reconstruction from the identical voltage.",
        "",
        "| Model | Stage | NRMSE | Image corr | Dice | Sign acc | RMS ratio | Dynamic ratio | Voltage residual | Newton improvements | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| PVI one-step Newton | 1 | {baseline['element_nrmse']:.3f} | "
            f"{baseline['image_correlation']:.3f} | "
            f"{baseline['localization_dice_at_true_volume']:.3f} | "
            f"{baseline['support_sign_accuracy']:.3f} | "
            f"{baseline['prediction_to_truth_rms']:.3f} | "
            f"{baseline['frame_difference_rms_ratio']:.3f} | "
            f"{newton['nonlinear_voltage_residual_rms_mean']:.3e} | – | baseline |"
        ),
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['stage']} | {row['element_nrmse']:.3f} | "
            f"{row['image_correlation']:.3f} | {row['localization_dice']:.3f} | "
            f"{row['sign_accuracy']:.3f} | {row['rms_ratio']:.3f} | "
            f"{row['frame_dynamics_ratio']:.3f} | {row['voltage_residual_rms']:.3e} | "
            f"{row['newton_relative_improvements']}/7 | {row['classification']} |"
        )
    lines.extend(
        [
            "",
            "A promising label is relative to Newton, not a claim of perfect inverse recovery.",
            "Non-finite, blank/static, or explosive candidates fail the sanity gate.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps({"newton": newton, "candidates": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
