#!/usr/bin/env python3
"""Validate the recovered raw-period order against a processed PVI HDF file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    per_channel = [float(np.corrcoef(pred[ch], ref[ch])[0, 1]) for ch in range(pred.shape[0])]
    error = pred - ref
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_rmse": float(np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(ref**2))),
        "global_correlation": float(np.corrcoef(pred.ravel(), ref.ravel())[0, 1]),
        "mean_channel_correlation": float(np.mean(per_channel)),
        "minimum_channel_correlation": float(np.min(per_channel)),
    }


def validate(candidate_npz: Path, match_npy: Path, h5_path: Path) -> dict:
    candidates = np.load(candidate_npz)
    match = np.asarray(np.load(match_npy), dtype=np.int64)
    trial_index = np.asarray(candidates["trial_index"][match], dtype=np.int16)
    report: dict[str, object] = {
        "candidate_periods": int(len(candidates["duration"])),
        "matched_periods": int(len(match)),
        "skipped_candidate_indices": np.setdiff1d(
            np.arange(len(candidates["duration"])), match
        ).tolist(),
        "matched_periods_per_trial": {
            str(int(trial)): int(np.count_nonzero(trial_index == trial))
            for trial in np.unique(trial_index)
        },
        "method": (
            "Monotonic minimum-cost match on standardized per-period pviLP "
            "resistance/reactance means."
        ),
    }
    with h5py.File(h5_path, "r") as h5:
        for component in ("pviHP", "pviLP"):
            for field in ("resistance", "reactance"):
                key = f"{component}_{field}"
                pred = np.asarray(candidates[key][:, match, :]).reshape(32, -1)
                ref = np.asarray(h5[f"data/{component}/{field}"])
                report[key] = _metrics(pred, ref)
        ref_duration = np.asarray(h5["stats/pviHP/duration"]).ravel()
    duration = np.asarray(candidates["duration"])[match]
    report["duration"] = {
        "correlation": float(np.corrcoef(duration, ref_duration)[0, 1]),
        "candidate_range_s": [float(np.min(duration)), float(np.max(duration))],
        "hdf_range_s": [float(np.min(ref_duration)), float(np.max(ref_duration))],
    }
    report["known_limitation"] = (
        "The archived HDF was produced after manual peak review. Monotonic period "
        "selection recovers LP order almost exactly, but spurious automatic minima "
        "need interval merges to reproduce every HP boundary exactly."
    )
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument(
        "--candidate-npz",
        type=Path,
        default=root / "data" / "subject006_segmented_candidates" / "segmented_periods.npz",
    )
    parser.add_argument(
        "--match-npy",
        type=Path,
        default=root / "data" / "subject006_segmented_candidates" / "hdf_period_match.npy",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=root / "data" / "subject006_segmented_candidates" / "alignment_validation.json",
    )
    args = parser.parse_args()
    report = validate(args.candidate_npz, args.match_npy, args.h5)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
