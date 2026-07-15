#!/usr/bin/env python3
"""Validate the production inverse directly against aligned PVI HDF images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_differential import reconstruct_newton_series
from gcnm_pvi.runtime import build_runtime


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def validate_hdf_hp(
    config: Path,
    h5_path: Path,
    *,
    start: int = 0,
    frames: int = 500,
) -> dict:
    """Compare images reconstructed from HDF pviHP resistance to HDF pviHP/img.

    The production inverse uses only the real voltage.  Since resistance is
    ``-real(vmeas) / stim_current`` and both the inverse and movmean high-pass
    are linear, the stored high-pass voltage can be recovered exactly as
    ``-stim_current * pviHP/resistance``.  No raw-time cardiac alignment is
    needed for this comparison.
    """
    cfg = GcnmConfig.from_yaml(config)
    runtime = build_runtime(cfg, include_forward=False)
    stop = start + frames
    with h5py.File(h5_path, "r") as f:
        resistance = np.asarray(f["data/pviHP/resistance"][:, start:stop])
        reference = np.asarray(f["data/pviHP/img"][:, :, start:stop])

    voltage = -float(cfg.stim_current) * resistance
    sigma = reconstruct_newton_series(
        runtime["physics_inv"],
        voltage,
        hyper_pvi=cfg.hyper_pvi,
        regularizer=runtime["mappings"].laplace,
        vmeas_ref=np.zeros(voltage.shape[0]),
    )
    prediction = runtime["mappings"].elem_to_image_grid(sigma)

    pred_mask = np.isfinite(prediction[:, :, 0])
    ref_mask = np.isfinite(reference[:, :, 0])
    joint = np.isfinite(prediction) & np.isfinite(reference)
    pred = prediction[joint]
    ref = reference[joint]
    error = pred - ref
    denom = float(np.sqrt(np.mean(ref**2)))

    frame_corr = []
    for frame in range(prediction.shape[2]):
        valid = np.isfinite(prediction[:, :, frame]) & np.isfinite(reference[:, :, frame])
        frame_corr.append(_correlation(prediction[:, :, frame][valid], reference[:, :, frame][valid]))

    intersection = int(np.count_nonzero(pred_mask & ref_mask))
    union = int(np.count_nonzero(pred_mask | ref_mask))
    return {
        "config": str(config.resolve()),
        "h5": str(h5_path.resolve()),
        "component": "pviHP",
        "frame_start": int(start),
        "frames": int(prediction.shape[2]),
        "element_count": int(sigma.shape[0]),
        "image_shape": list(prediction.shape[:2]),
        "finite_pixels": int(np.count_nonzero(pred_mask)),
        "mask_iou": float(intersection / union),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_rmse": float(np.sqrt(np.mean(error**2)) / denom),
        "correlation": _correlation(pred, ref),
        "median_frame_correlation": float(np.nanmedian(frame_corr)),
        "minimum_frame_correlation": float(np.nanmin(frame_corr)),
        "optimal_scale": float(np.dot(pred, ref) / np.dot(pred, pred)),
        "prediction_range": [float(np.min(pred)), float(np.max(pred))],
        "reference_range": [float(np.min(ref)), float(np.max(ref))],
        "interpretation": (
            "This is a frame-matched inverse/mapping check using HDF high-pass "
            "resistance; it is independent of raw-to-cardiac-period alignment."
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "subject006_pvi08_production.yaml",
    )
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=root / "data" / "subject006_hdf_reconstruction_validation.json",
    )
    args = parser.parse_args()
    report = validate_hdf_hp(
        args.config,
        args.h5,
        start=args.start,
        frames=args.frames,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
