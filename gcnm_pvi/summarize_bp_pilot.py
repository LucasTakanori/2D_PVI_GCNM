#!/usr/bin/env python3
"""Summarize the 24 split-matched US120 BP pilot experiments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from gcnm_pvi.bp_comparison import _fiducials, metrics


GCNM_PATTERN = re.compile(
    r"gcnm-(coordinate|global_voltage_slots)-(3ch|6ch)-(crt|crs)-image-to-(waveform|fiducials)$"
)
ORIGINAL_PATTERN = re.compile(
    r"original-pvi-(crt|crs)-image-to-(waveform|fiducials)$"
)


def _read_run(root: Path, target: str, subject: str) -> dict:
    branch = root / target / "main"
    paths = {
        "contract": branch / "run_contracts" / f"{subject}.json",
        "verification": branch / "verification" / f"{subject}.json",
        "results": branch / "results" / f"{subject}_results.csv",
        "statistics": branch / "statistics" / f"{subject}_statistics.json",
        "checkpoint": branch / "checkpoints" / f"{subject}_checkpoints.pth",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete BP run {target}/{subject}: {missing}")
    verification = json.loads(paths["verification"].read_text(encoding="utf-8"))
    if verification.get("status") != "pass":
        raise ValueError(f"inference verification failed for {target}/{subject}")
    derived, target_values = metrics(paths["results"])
    frame = pd.read_csv(paths["results"])
    return {
        "contract": json.loads(paths["contract"].read_text(encoding="utf-8")),
        "verification": verification,
        "native_pvi_ml_statistics": json.loads(
            paths["statistics"].read_text(encoding="utf-8")
        ),
        "derived_fiducial_metrics": derived,
        "test_rows": int(len(frame)),
        "target_fiducials": target_values,
        "results_path": str(paths["results"].resolve()),
        "checkpoint_path": str(paths["checkpoint"].resolve()),
    }


def _aggregate(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "iqr": [
            float(np.percentile(array, 25)),
            float(np.percentile(array, 75)),
        ],
    }


def summarize(root: Path, subjects: tuple[str, ...]) -> dict:
    targets = sorted(path.name for path in root.iterdir() if path.is_dir())
    expected_gcnm = {
        f"gcnm-{family}-3ch-{architecture}-image-to-{output}"
        for family in ("coordinate", "global_voltage_slots")
        for architecture in ("crt", "crs")
        for output in ("waveform", "fiducials")
    }
    expected_original = {
        f"original-pvi-{architecture}-image-to-{output}"
        for architecture in ("crt", "crs")
        for output in ("waveform", "fiducials")
    }
    expected = expected_gcnm | expected_original
    missing_targets = sorted(expected - set(targets))
    if missing_targets:
        raise FileNotFoundError(f"pilot artifact root lacks targets: {missing_targets}")

    runs: dict[str, dict[str, dict]] = {}
    internal: dict[tuple[str, str], dict] = {}
    split_hashes = set()
    for target in sorted(expected):
        runs[target] = {}
        for subject in subjects:
            record = _read_run(root, target, subject)
            split_hashes.add(record["contract"]["split_manifest_sha256"])
            internal[(target, subject)] = record
            runs[target][subject] = {
                key: value
                for key, value in record.items()
                if key != "target_fiducials"
            }
    if len(split_hashes) != 1:
        raise ValueError(f"pilot runs use multiple split hashes: {sorted(split_hashes)}")

    comparisons = {}
    flat_rows = []
    for target in sorted(expected_gcnm):
        match = GCNM_PATTERN.fullmatch(target)
        assert match is not None
        family, channel_mode, architecture, output_mode = match.groups()
        original_target = f"original-pvi-{architecture}-image-to-{output_mode}"
        key = f"{family}-{channel_mode}-{architecture}-{output_mode}"
        subject_reports = {}
        deltas = {"dbp_mae_mmhg": [], "sbp_mae_mmhg": []}
        for subject in subjects:
            gcnm = internal[(target, subject)]
            original = internal[(original_target, subject)]
            if gcnm["test_rows"] != original["test_rows"] or not np.array_equal(
                gcnm["target_fiducials"], original["target_fiducials"]
            ):
                raise ValueError(
                    f"{target}/{subject} and {original_target} do not share exact targets/order"
                )
            paired = {}
            for pressure in ("dbp", "sbp"):
                value = (
                    gcnm["derived_fiducial_metrics"][pressure]["mae_mmhg"]
                    - original["derived_fiducial_metrics"][pressure]["mae_mmhg"]
                )
                paired[f"{pressure}_mae_mmhg"] = value
                deltas[f"{pressure}_mae_mmhg"].append(value)
            subject_reports[subject] = {
                "gcnm": gcnm["derived_fiducial_metrics"],
                "original_pvi": original["derived_fiducial_metrics"],
                "paired_difference_gcnm_minus_original": paired,
                "exact_test_target_order_match": True,
            }
            flat_rows.append(
                {
                    "family": family,
                    "channel_mode": channel_mode,
                    "architecture": architecture,
                    "output_mode": output_mode,
                    "subject": subject,
                    **paired,
                }
            )
        comparisons[key] = {
            "subjects": subject_reports,
            "aggregate_paired_difference": {
                metric: _aggregate(values) for metric, values in deltas.items()
            },
        }
    return {
        "schema": "pvi-gcnm-us120-bp-pilot-summary-v1",
        "status": "pass",
        "artifact_root": str(root.resolve()),
        "mask_key": "mask05",
        "channel_mode": "3ch",
        "subjects": list(subjects),
        "run_count": len(expected) * len(subjects),
        "gcnm_run_count": len(expected_gcnm) * len(subjects),
        "split_matched_newton_run_count": len(expected_original) * len(subjects),
        "split_manifest_sha256": split_hashes.pop(),
        "runs": runs,
        "paired_comparisons": comparisons,
        "flat_paired_rows": flat_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.csv.exists():
        raise FileExistsError("immutable BP pilot summary output already exists")
    report = summarize(args.artifact_root, ("subject006", "subject010"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(report.pop("flat_paired_rows")).to_csv(args.csv, index=False)
    print(
        json.dumps(
            {
                "status": report["status"],
                "runs": report["run_count"],
                "output": str(args.output),
                "csv": str(args.csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
