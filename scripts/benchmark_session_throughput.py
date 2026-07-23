#!/usr/bin/env python3
"""Benchmark two real PVI session streams concurrently on separate GPUs."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _run(payload: dict) -> dict:
    # Imports occur after process spawn so each worker creates its own CUDA context.
    from gcnm_pvi.representations import CoordinateReconstructor, DiffusionSlotReconstructor

    record = payload["record"]
    family = payload["family"]
    device = f"cuda:{payload['gpu']}"
    ring = record["ring"]
    name = f"{family}_b045_{ring}_seed0"
    model_dir = ROOT / "models" / "mesh_representations" / family / ring
    if family == "coordinate":
        reconstructor = CoordinateReconstructor(
            Path(record["config"]), model_dir, name, device=device
        )
    else:
        reconstructor = DiffusionSlotReconstructor(
            Path(record["config"]),
            model_dir / f"{name}_localizer.pt",
            model_dir / f"{name}_refiner.pt",
            ROOT / "data" / "mesh_training" / "diffusion" / ring / "default_finger_prior.npz",
            device=device,
        )
    with h5py.File(record["source_hdf5"], "r") as handle:
        resistance = np.asarray(
            handle["data/pviHP/resistance"][:, : payload["frames"]],
            dtype=np.float64,
        )
    voltage = 0.01 * resistance.T

    parity_frames = min(8, len(voltage))
    reference = reconstructor.reconstruct_reference(voltage[:parity_frames])
    optimized = reconstructor.reconstruct(voltage[:parity_frames])
    parity = []
    for stage in range(2):
        difference = optimized[stage] - reference[stage]
        parity.append(
            {
                "stage": stage + 1,
                "relative_l2": float(
                    np.linalg.norm(difference)
                    / max(float(np.linalg.norm(reference[stage])), 1e-12)
                ),
                "maximum_absolute": float(np.max(np.abs(difference))),
            }
        )

    started = time.perf_counter()
    for first in range(0, len(voltage), payload["chunk_frames"]):
        reconstructor.reconstruct(voltage[first : first + payload["chunk_frames"]])
    seconds = time.perf_counter() - started
    return {
        "source_name": record["source_name"],
        "gpu": payload["gpu"],
        "frames": len(voltage),
        "seconds": seconds,
        "frames_per_second": len(voltage) / seconds,
        "parity": parity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["coordinate", "diffusion"], required=True)
    parser.add_argument("--frames", type=int, default=8192)
    parser.add_argument("--chunk-frames", type=int, default=256)
    parser.add_argument(
        "--source-names", nargs=2,
        default=["subject006_baseline", "subject010_baseline"],
    )
    args = parser.parse_args()
    registry = json.loads(
        (ROOT / "data" / "registries" / "main_b045_v1.json").read_text()
    )
    by_name = {record["source_name"]: record for record in registry["records"]}
    records = [by_name[name] for name in args.source_names]
    if any(record["ring"] != "US120" for record in records):
        raise ValueError("throughput pilot requires two US120 sessions")
    payloads = [
        {
            "record": record,
            "family": args.family,
            "gpu": gpu,
            "frames": args.frames,
            "chunk_frames": args.chunk_frames,
        }
        for gpu, record in enumerate(records)
    ]
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        sessions = list(executor.map(_run, payloads))
    wall_seconds = time.perf_counter() - started
    total_frames = sum(item["frames"] for item in sessions)
    maximum_parity_l2 = max(
        stage["relative_l2"] for item in sessions for stage in item["parity"]
    )
    report = {
        "family": args.family,
        "gpus": 2,
        "chunk_frames": args.chunk_frames,
        "sessions": sessions,
        "total_frames": total_frames,
        "wall_seconds": wall_seconds,
        "aggregate_frames_per_second": total_frames / wall_seconds,
        "estimated_seconds_per_8000_frames": 8000.0 / (total_frames / wall_seconds),
        "maximum_parity_relative_l2": maximum_parity_l2,
        "acceptance": {
            "parity_relative_l2_at_most": 1e-4,
            "estimated_seconds_per_8000_frames_at_most": 900.0,
            "pass": maximum_parity_l2 <= 1e-4
            and 8000.0 / (total_frames / wall_seconds) <= 900.0,
        },
    }
    print(json.dumps(report, indent=2), flush=True)
    if not report["acceptance"]["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
