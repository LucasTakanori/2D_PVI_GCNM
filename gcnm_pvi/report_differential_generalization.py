#!/usr/bin/env python3
"""Write a concise clean/augmented differential generalization report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYS = (
    "element_nrmse",
    "image_correlation",
    "localization_dice_at_true_volume",
    "support_sign_accuracy",
    "prediction_to_truth_rms",
    "frame_difference_rms_ratio",
)


def _rows(report: dict, condition: str) -> list[dict]:
    rows = []
    newton = report["saved_newton"]
    rows.append(
        {
            "condition": condition,
            "model": "PVI one-step Newton",
            "stage": 1,
            **{key: newton["metrics"][key] for key in KEYS},
            "voltage_residual": newton["nonlinear_voltage_residual_rms_mean"],
            "improvements": None,
            "classification": "baseline",
        }
    )
    for stage in report["faithful_gcnm"]:
        relative = stage["relative_to_saved_one_step_newton"]
        rows.append(
            {
                "condition": condition,
                "model": "coordinate direct",
                "stage": stage["stage"],
                **{key: stage["metrics"][key] for key in KEYS},
                "voltage_residual": stage["physics"][
                    "post_gcn_voltage_residual_rms_mean"
                ],
                "improvements": relative["improvements"],
                "classification": relative["classification"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--gif-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clean = json.loads(
        (args.result_root / "evaluation_clean/report.json").read_text(encoding="utf-8")
    )
    augmented = json.loads(
        (args.result_root / "evaluation_augmented/report.json").read_text(encoding="utf-8")
    )
    gif = json.loads(args.gif_sidecar.read_text(encoding="utf-8"))
    rows = _rows(clean, "clean") + _rows(augmented, "augmented")
    lines = [
        "# Differential coordinate GCNM small-cohort generalization",
        "",
        "Training uses 20 disjoint synthetic anatomies. Validation and exact-nonlinear",
        "test use 5 unseen anatomies each. Newton is computed from the identical voltage",
        "and is never a label.",
        "",
        "| Condition | Model | Stage | NRMSE | Image corr | Dice | Sign acc | RMS ratio | Dynamic ratio | Voltage residual | Newton improvements |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        improvement = "–" if row["improvements"] is None else f"{row['improvements']}/7"
        lines.append(
            f"| {row['condition']} | {row['model']} | {row['stage']} | "
            f"{row['element_nrmse']:.3f} | {row['image_correlation']:.3f} | "
            f"{row['localization_dice_at_true_volume']:.3f} | "
            f"{row['support_sign_accuracy']:.3f} | "
            f"{row['prediction_to_truth_rms']:.3f} | "
            f"{row['frame_difference_rms_ratio']:.3f} | "
            f"{row['voltage_residual']:.3e} | {improvement} |"
        )
    lines.extend(
        [
            "",
            f"Three-beat GIF: `{gif['gif']}`",
            "",
            "This is the final synthetic gate. Subject006 export and BP training remain",
            "blocked pending visual/metric review.",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps({"rows": rows, "gif": gif}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
