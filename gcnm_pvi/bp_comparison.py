"""Compute paired per-subject BP metrics for original-PVI and GCNM inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float).reshape(-1), np.asarray(y, float).reshape(-1)
    covariance = np.mean((x - x.mean()) * (y - y.mean()))
    denominator = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    return float(2 * covariance / denominator) if denominator > 0 else float("nan")


def _fiducials(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pred_cols = sorted((c for c in frame if c.startswith("pred_")), key=lambda x: int(x[5:]))
    target_cols = sorted((c for c in frame if c.startswith("target_")), key=lambda x: int(x[7:]))
    prediction = frame[pred_cols].to_numpy(float)
    target = frame[target_cols].to_numpy(float)
    if prediction.shape[1] > 2:
        prediction = np.column_stack((prediction.min(1), prediction.max(1)))
        target = np.column_stack((target.min(1), target.max(1)))
    if prediction.shape[1] != 2:
        raise ValueError("prediction CSV must contain waveform or [DBP,SBP] values")
    return prediction, target


def metrics(path: Path) -> tuple[dict, np.ndarray]:
    prediction, target = _fiducials(pd.read_csv(path))
    error = prediction - target
    result = {}
    for index, name in enumerate(("dbp", "sbp")):
        absolute = np.abs(error[:, index])
        result[name] = {
            "mae_mmhg": float(absolute.mean()),
            "error_sd_mmhg": float(error[:, index].std(ddof=1)),
            "within_5_mmhg": float(np.mean(absolute <= 5)),
            "within_10_mmhg": float(np.mean(absolute <= 10)),
            "within_15_mmhg": float(np.mean(absolute <= 15)),
            "r2": float(r2_score(target[:, index], prediction[:, index])),
            "concordance_correlation": concordance_correlation(
                target[:, index], prediction[:, index]
            ),
        }
    return result, target


def compare(gcnm_root: Path, original_root: Path) -> dict:
    gcnm_files = {path.name.split("_results")[0]: path for path in gcnm_root.rglob("*_results.csv")}
    original_files = {path.name.split("_results")[0]: path for path in original_root.rglob("*_results.csv")}
    subjects = sorted(set(gcnm_files) & set(original_files))
    if not subjects:
        raise FileNotFoundError("no paired subject result CSV files found")
    per_subject = {}
    differences: dict[str, list[float]] = {"dbp_mae_mmhg": [], "sbp_mae_mmhg": []}
    for subject in subjects:
        gcnm, target_gcnm = metrics(gcnm_files[subject])
        original, target_original = metrics(original_files[subject])
        if target_gcnm.shape != target_original.shape or not np.allclose(target_gcnm, target_original):
            raise ValueError(f"{subject} does not share identical source targets/windows")
        delta = {
            "dbp_mae_mmhg": gcnm["dbp"]["mae_mmhg"] - original["dbp"]["mae_mmhg"],
            "sbp_mae_mmhg": gcnm["sbp"]["mae_mmhg"] - original["sbp"]["mae_mmhg"],
        }
        for key, value in delta.items():
            differences[key].append(value)
        per_subject[subject] = {"gcnm": gcnm, "original_pvi": original, "paired_difference": delta}
    aggregate = {}
    for key, values in differences.items():
        array = np.asarray(values)
        aggregate[key] = {
            "median": float(np.median(array)),
            "iqr": [float(np.percentile(array, 25)), float(np.percentile(array, 75))],
        }
    return {"paired_subjects": len(subjects), "per_subject": per_subject, "aggregate": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcnm-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.gcnm_root, args.original_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
