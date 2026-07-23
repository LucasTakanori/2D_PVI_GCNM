#!/usr/bin/env python3
"""Compare optimized coordinate reconstruction with the dense reference path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from gcnm_pvi.representations import CoordinateReconstructor, DiffusionSlotReconstructor


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--source-name", default="subject006_baseline")
    parser.add_argument("--device", default=None)
    parser.add_argument("--family", choices=["coordinate", "diffusion"], default="coordinate")
    args = parser.parse_args()
    registry = json.loads(
        (root / "data/registries/main_b045_v1.json").read_text(encoding="utf-8")
    )
    record = next(
        item for item in registry["records"] if item["source_name"] == args.source_name
    )
    with h5py.File(record["source_hdf5"], "r") as handle:
        voltage = 0.01 * np.asarray(
            handle["data/pviHP/resistance"][:, : args.frames], dtype=np.float64
        ).T
    ring = record["ring"]
    name = f"{args.family}_b045_{ring}_seed0"
    model_dir = root / "models/mesh_representations" / args.family / ring
    if args.family == "coordinate":
        reconstructor = CoordinateReconstructor(
            Path(record["config"]), model_dir, name, device=args.device
        )
    else:
        reconstructor = DiffusionSlotReconstructor(
            Path(record["config"]),
            model_dir / f"{name}_localizer.pt",
            model_dir / f"{name}_refiner.pt",
            root / "data/mesh_training/diffusion" / ring / "default_finger_prior.npz",
            device=args.device,
        )
    forward = reconstructor.runtime["physics_inv"]._forward(reconstructor.baseline)
    started = time.perf_counter()
    jacobian_reference = forward.compute_jacobian()
    jacobian_reference_seconds = time.perf_counter() - started
    started = time.perf_counter()
    jacobian_optimized = reconstructor.runtime["physics_inv"].jacobian_from_forward(forward)
    jacobian_optimized_seconds = time.perf_counter() - started
    started = time.perf_counter()
    reference = reconstructor.reconstruct_reference(voltage)
    reference_seconds = time.perf_counter() - started
    started = time.perf_counter()
    optimized = reconstructor.reconstruct(voltage)
    optimized_seconds = time.perf_counter() - started
    report = {
        "frames": args.frames,
        "jacobian_reference_seconds": jacobian_reference_seconds,
        "jacobian_optimized_seconds": jacobian_optimized_seconds,
        "jacobian_speedup": jacobian_reference_seconds / jacobian_optimized_seconds,
        "jacobian_relative_l2": float(
            np.linalg.norm(jacobian_optimized - jacobian_reference)
            / np.linalg.norm(jacobian_reference)
        ),
        "jacobian_maximum_absolute": float(
            np.max(np.abs(jacobian_optimized - jacobian_reference))
        ),
        "reference_seconds": reference_seconds,
        "optimized_seconds": optimized_seconds,
        "speedup": reference_seconds / optimized_seconds,
        "stages": [],
    }
    for index in range(2):
        difference = optimized[index] - reference[index]
        report["stages"].append(
            {
                "stage": index + 1,
                "relative_l2": float(
                    np.linalg.norm(difference)
                    / max(float(np.linalg.norm(reference[index])), 1e-12)
                ),
                "maximum_absolute": float(np.max(np.abs(difference))),
            }
        )
    for key in ("stage_1_forward_voltage_rms", "stage_2_forward_voltage_rms"):
        report[f"{key}_maximum_absolute"] = float(
            np.max(np.abs(np.asarray(optimized[2][key]) - np.asarray(reference[2][key])))
        )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
