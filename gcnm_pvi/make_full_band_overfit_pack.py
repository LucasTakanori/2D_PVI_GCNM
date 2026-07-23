"""Build one immutable absolute-voltage/absolute-conductivity overfit beat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--anatomy-id", type=int, required=True)
    parser.add_argument("--beat-id", type=int)
    parser.add_argument(
        "--voltage-key",
        choices=["V", "V_clean_absolute_filtered"],
        default="V",
        help="use augmented canonical voltage or clean absolute voltage for an ablation",
    )
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"immutable overfit root exists: {args.output_root}")

    full_path = args.dataset_root / "full" / f"{args.split}.npz"
    with np.load(full_path) as full:
        required = (
            "sigma",
            "sigma_absolute",
            "sigma_resting",
            "V",
            "V_absolute_filtered",
            "anatomy_id",
            "beat_id",
            "sample_index",
        )
        missing = [key for key in required if key not in full]
        if missing:
            raise KeyError(f"absolute full-band archive is missing {missing}")
        np.testing.assert_array_equal(full["sigma"], full["sigma_absolute"])
        np.testing.assert_array_equal(full["V"], full["V_absolute_filtered"])
        anatomy = np.asarray(full["anatomy_id"])
        beats = np.asarray(full["beat_id"])
        samples = np.asarray(full["sample_index"])
        available = np.unique(beats[anatomy == args.anatomy_id])
        if not len(available):
            raise ValueError(f"anatomy {args.anatomy_id} is absent from {args.split}")
        beat_id = int(available[0] if args.beat_id is None else args.beat_id)
        indices = np.flatnonzero((anatomy == args.anatomy_id) & (beats == beat_id))
        indices = indices[np.argsort(samples[indices])]
        if len(indices) != 50 or not np.array_equal(samples[indices], np.arange(50)):
            raise ValueError("selected overfit beat must contain samples 0..49 exactly once")
        sigma = np.asarray(full["sigma"][indices], dtype=np.float32)
        voltage = np.asarray(full[args.voltage_key][indices], dtype=np.float32)
        resting = np.asarray(full["sigma_resting"][indices], dtype=np.float32)
        baseline = np.full_like(sigma, 0.7, dtype=np.float32)
        payload = {
            "sigma": sigma,
            "sigma_baseline": baseline,
            "V": voltage,
            "sigma_resting": resting,
            "anatomy_id": anatomy[indices].astype(np.int32),
            "beat_id": beats[indices].astype(np.int32),
            "sample_index": samples[indices].astype(np.int16),
        }

    args.output_root.mkdir(parents=True)
    for name in ("train", "validation"):
        np.savez_compressed(args.output_root / f"{name}.npz", **payload)
    report = {
        "schema": "pvi-gcnm-absolute-full-band-overfit-v2",
        "source_full": str(full_path.resolve()),
        "source_split": args.split,
        "anatomy_id": args.anatomy_id,
        "beat_id": beat_id,
        "samples": 50,
        "contract": "one frame: V_absolute_filtered -> sigma_absolute",
        "voltage_key": args.voltage_key,
        "validation_is_training_copy": True,
        "purpose": "architecture learnability/overfit gate only",
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
