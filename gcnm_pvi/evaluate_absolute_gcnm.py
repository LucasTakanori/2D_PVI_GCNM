#!/usr/bin/env python3
"""Evaluate one-frame absolute-voltage GCNM anatomy and pulse recovery."""

from __future__ import annotations

import argparse
import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from gcnm_pvi.anatomical_phantoms import element_positions
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.iterative_physics import FixedZeroCurrentLMSolver
from gcnm_pvi.runtime import build_runtime
from gcnm_pvi.train_faithful_gcnm import _make_dataset, _model, _predict


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left).ravel()
    right = np.asarray(right).ravel()
    if np.std(left) < 1e-15 or np.std(right) < 1e-15:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _metrics(prediction: np.ndarray, truth: np.ndarray, resting: np.ndarray) -> dict:
    absolute_error = prediction - truth
    predicted_dynamic = prediction - resting
    true_dynamic = truth - resting
    true_dynamic_rms = max(float(np.sqrt(np.mean(true_dynamic**2))), 1e-12)
    frame_correlations = [
        _correlation(predicted, target)
        for predicted, target in zip(predicted_dynamic, true_dynamic)
    ]
    predicted_temporal = prediction - prediction[:1]
    true_temporal = truth - truth[:1]
    true_temporal_rms = max(float(np.sqrt(np.mean(true_temporal**2))), 1e-12)
    return {
        "absolute_rmse_s_m": float(np.sqrt(np.mean(absolute_error**2))),
        "absolute_correlation": _correlation(prediction, truth),
        "dynamic_rmse_s_m": float(np.sqrt(np.mean((predicted_dynamic - true_dynamic) ** 2))),
        "dynamic_nrmse": float(
            np.sqrt(np.mean((predicted_dynamic - true_dynamic) ** 2))
            / true_dynamic_rms
        ),
        "dynamic_correlation": _correlation(predicted_dynamic, true_dynamic),
        "dynamic_frame_correlation_median": float(np.nanmedian(frame_correlations)),
        "dynamic_rms_ratio": float(
            np.sqrt(np.mean(predicted_dynamic**2)) / true_dynamic_rms
        ),
        "temporal_delta_nrmse": float(
            np.sqrt(np.mean((predicted_temporal - true_temporal) ** 2))
            / true_temporal_rms
        ),
        "temporal_delta_correlation": _correlation(
            predicted_temporal, true_temporal
        ),
        "temporal_delta_rms_ratio": float(
            np.sqrt(np.mean(predicted_temporal**2)) / true_temporal_rms
        ),
        "negative_absolute_elements": int(np.count_nonzero(prediction <= 0)),
    }


def _forward_many(physics, conductivity: np.ndarray, workers: int) -> np.ndarray:
    parts = [
        part
        for part in np.array_split(np.arange(len(conductivity)), max(1, workers))
        if len(part)
    ]
    lanes = [physics, *(copy.deepcopy(physics) for _ in range(len(parts) - 1))]

    def run(lane: int, indices: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.asarray(lanes[lane].solve(conductivity[index]), dtype=np.float64)
                for index in indices
            ]
        )

    with ThreadPoolExecutor(max_workers=len(parts)) as executor:
        results = [
            future.result()
            for future in (
                executor.submit(run, lane, indices)
                for lane, indices in enumerate(parts)
            )
        ]
    return np.concatenate(results, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-conductivity", type=float, default=1e-4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable absolute evaluation exists: {args.output}")

    cfg = GcnmConfig.from_yaml(args.config)
    runtime = build_runtime(cfg, include_forward=False)
    with np.load(args.test) as source:
        required = ("sigma", "sigma_baseline", "sigma_resting", "V")
        missing = [key for key in required if key not in source]
        if missing:
            raise KeyError(f"absolute evaluation pack is missing {missing}")
        truth = np.asarray(source["sigma"], dtype=np.float64)
        baseline = np.asarray(source["sigma_baseline"], dtype=np.float64)
        resting = np.asarray(source["sigma_resting"], dtype=np.float64)
        measured = np.asarray(source["V"], dtype=np.float64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("physics_contract", {}).get("measurement_contract") != "absolute":
        raise ValueError("checkpoint was not trained with the absolute-voltage contract")
    fixed = FixedZeroCurrentLMSolver(
        runtime["physics_inv"],
        baseline[0],
        regularizer=runtime["mappings"].laplace,
        hyper_pvi=cfg.hyper_pvi,
        lambda_lm=cfg.lambda_lm,
        step_size=cfg.lm_step_size,
    )
    direction, diagnostics, _ = fixed.solve_many_absolute(measured)
    positions = element_positions(runtime["mesh_inv"]).astype(np.float32)
    dataset = _make_dataset(
        truth,
        baseline,
        direction,
        positions,
        runtime["edge_index"],
        scale=float(checkpoint["scale"]),
        use_coordinates=bool(checkpoint["use_coordinates"]),
        positive_weight=0.0,
        voltage=measured if checkpoint.get("use_voltage_mlp", False) else None,
        voltage_scale=float(checkpoint.get("voltage_scale", 1.0)),
        reference=resting,
    )
    model = _model(
        checkpoint["output_mode"],
        checkpoint["channels"],
        int(checkpoint["in_channels"]),
        use_voltage_mlp=bool(checkpoint.get("use_voltage_mlp", False)),
        measurements=measured.shape[1],
        voltage_latent=int(checkpoint.get("voltage_latent", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    prediction = _predict(model, dataset, float(checkpoint["scale"]))
    clipped = np.maximum(prediction, args.minimum_conductivity)
    workers = min(int(os.environ.get("GCNM_PHYSICS_WORKERS", "1")), len(clipped))
    predicted_voltage = _forward_many(runtime["physics_inv"], clipped, workers)
    voltage_residual = predicted_voltage - measured
    report = {
        "schema": "pvi-gcnm-absolute-one-frame-evaluation-v1",
        "contract": "V_absolute_filtered(t) -> sigma_absolute(t)",
        "samples": len(truth),
        "metrics": _metrics(prediction, truth, resting),
        "absolute_voltage_residual_rms_v": float(
            np.sqrt(np.mean(voltage_residual**2))
        ),
        "first_lm_residual_rms_v": float(
            np.mean([item.voltage_residual_rms for item in diagnostics])
        ),
        "physics_workers": workers,
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "predictions.npz",
        truth=truth.astype(np.float32),
        resting=resting.astype(np.float32),
        prediction=prediction.astype(np.float32),
        predicted_voltage=predicted_voltage.astype(np.float32),
        measured_voltage=measured.astype(np.float32),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
