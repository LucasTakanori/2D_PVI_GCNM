#!/usr/bin/env python3
"""Aggregate faithful experiment reports into a compact comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _row(name: str, category: str, stage: int, metrics: dict) -> dict:
    return {
        "experiment": name,
        "category": category,
        "stage": stage,
        "element_correlation": metrics["element_correlation"],
        "image_correlation": metrics["image_correlation"],
        "localization_dice": metrics["localization_dice_at_true_volume"],
        "element_rmse": metrics["element_rmse"],
        "image_rmse": metrics["image_rmse"],
        "background_rms": metrics["background_rms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--evaluation", default="evaluation_nonlinear")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    reports: list[tuple[str, dict]] = []
    for experiment_dir in sorted(args.results_root.iterdir()):
        path = experiment_dir / args.evaluation / "report.json"
        if path.exists():
            reports.append((experiment_dir.name, json.loads(path.read_text())))
    if not reports:
        raise RuntimeError(f"no reports found under {args.results_root}")

    rows: list[dict] = []
    first = reports[0][1]
    if first.get("saved_newton"):
        rows.append(_row("saved_newton", "baseline", 0, first["saved_newton"]))
    for stage in first.get("iterative_lm", []):
        rows.append(
            _row("iterative_lm", "physics_control", stage["stage"], stage["metrics"])
        )
    for name, report in reports:
        for stage in report.get("faithful_gcnm", []):
            rows.append(_row(name, "faithful_gcnm", stage["stage"], stage["metrics"]))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"evaluation": args.evaluation, "rows": rows}, indent=2))
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"evaluation": args.evaluation, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
