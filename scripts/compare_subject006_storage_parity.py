#!/usr/bin/env python3
"""Compare deterministic native-HDF5 and reference-Parquet CRT artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader]
    return header, np.asarray(rows, dtype=np.float64)


def _model_maximum_difference(hdf5_path: Path, parquet_path: Path) -> float:
    hdf5 = torch.load(hdf5_path, map_location="cpu", weights_only=False)["model"]
    parquet = torch.load(parquet_path, map_location="cpu", weights_only=False)["model"]
    if hdf5.keys() != parquet.keys():
        raise ValueError("checkpoint model parameter names differ")
    return max(
        float(torch.max(torch.abs(hdf5[key] - parquet[key])).item())
        for key in hdf5
    )


def _compare_seed(artifact_root: Path, seed: int) -> dict:
    subject = "subject006"
    branches = {
        "hdf5": artifact_root / f"seed{seed}/subject006-crt-img-to-waveform/main",
        "parquet": artifact_root / f"seed{seed}/reference-parquet-crt-img-to-waveform/main",
    }
    files = {}
    for storage, branch in branches.items():
        files[storage] = {
            "history": branch / "history" / f"{subject}_history.csv",
            "results": branch / "results" / f"{subject}_results.csv",
            "statistics": branch / "statistics" / f"{subject}_statistics.json",
            "checkpoint": branch / "checkpoints" / f"{subject}_checkpoints.pth",
        }
        missing = [str(path) for path in files[storage].values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing {storage} parity artifacts: {missing}")

    result_header_h5, results_h5 = _csv(files["hdf5"]["results"])
    result_header_pq, results_pq = _csv(files["parquet"]["results"])
    history_header_h5, history_h5 = _csv(files["hdf5"]["history"])
    history_header_pq, history_pq = _csv(files["parquet"]["history"])
    if result_header_h5 != result_header_pq or results_h5.shape != results_pq.shape:
        raise ValueError("HDF5 and Parquet result schemas differ")
    output_size = results_h5.shape[1] // 2
    targets_h5, targets_pq = results_h5[:, output_size:], results_pq[:, output_size:]
    predictions_h5, predictions_pq = results_h5[:, :output_size], results_pq[:, :output_size]
    target_max = float(np.max(np.abs(targets_h5 - targets_pq)))
    prediction_max = float(np.max(np.abs(predictions_h5 - predictions_pq)))
    common_epochs = min(len(history_h5), len(history_pq))
    comparable_history = (
        history_header_h5 == history_header_pq
        and history_h5.shape[1] == history_pq.shape[1]
        and common_epochs > 0
    )
    history_max = (
        float(np.max(np.abs(history_h5[:common_epochs] - history_pq[:common_epochs])))
        if comparable_history
        else None
    )
    statistics = {
        storage: json.loads(paths["statistics"].read_text(encoding="utf-8"))
        for storage, paths in files.items()
    }
    metric_keys = ("amae", "armse", "dbp_mae", "sbp_mae", "dbp_r2", "sbp_r2")
    metric_deltas = {
        key: float(statistics["parquet"][key] - statistics["hdf5"][key])
        for key in metric_keys
    }
    model_max = _model_maximum_difference(
        files["hdf5"]["checkpoint"], files["parquet"]["checkpoint"]
    )
    exact = bool(
        target_max == 0.0
        and prediction_max == 0.0
        and len(history_h5) == len(history_pq)
        and history_max == 0.0
        and model_max == 0.0
    )
    numerical_match = bool(
        target_max == 0.0
        and prediction_max <= 1e-4
        and len(history_h5) == len(history_pq)
        and history_max is not None
        and history_max <= 1e-4
        and max(abs(value) for value in metric_deltas.values()) <= 1e-4
        and model_max <= 1e-6
    )
    report = {
        "schema": "pvi-hdf5-parquet-seeded-parity-seed-v1",
        "subject": subject,
        "seed": seed,
        "storage_parity": exact or numerical_match,
        "classification": "bitwise_exact" if exact else "numerically_equivalent" if numerical_match else "diverged",
        "result_shape": list(results_h5.shape),
        "target_maximum_absolute_difference": target_max,
        "prediction_maximum_absolute_difference": prediction_max,
        "history_rows": {"hdf5": len(history_h5), "parquet": len(history_pq)},
        "common_history_maximum_absolute_difference": history_max,
        "model_parameter_maximum_absolute_difference": model_max,
        "metric_deltas_parquet_minus_hdf5": metric_deltas,
        "checkpoint_sha256": {
            storage: _sha256(paths["checkpoint"]) for storage, paths in files.items()
        },
        "statistics": statistics,
        "artifacts": {
            storage: {name: str(path.resolve()) for name, path in paths.items()}
            for storage, paths in files.items()
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [_compare_seed(args.artifact_root, seed) for seed in args.seeds]
    metric_keys = tuple(reports[0]["metric_deltas_parquet_minus_hdf5"])
    report = {
        "schema": "pvi-hdf5-parquet-seeded-parity-v1",
        "subject": "subject006",
        "seeds": args.seeds,
        "available_determinism": (
            "all RNGs, initialization and DataLoader permutations fixed; "
            "MaxPool3D CUDA backward is warn-only because PyTorch has no deterministic kernel"
        ),
        "all_storage_parity": all(item["storage_parity"] for item in reports),
        "classifications": {
            str(item["seed"]): item["classification"] for item in reports
        },
        "metric_delta_summary_parquet_minus_hdf5": {
            key: {
                "values": [item["metric_deltas_parquet_minus_hdf5"][key] for item in reports],
                "median": float(np.median([item["metric_deltas_parquet_minus_hdf5"][key] for item in reports])),
                "minimum": float(np.min([item["metric_deltas_parquet_minus_hdf5"][key] for item in reports])),
                "maximum": float(np.max([item["metric_deltas_parquet_minus_hdf5"][key] for item in reports])),
            }
            for key in metric_keys
        },
        "per_seed": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
