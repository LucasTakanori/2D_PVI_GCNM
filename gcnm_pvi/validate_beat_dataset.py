"""Validate one immutable 1,000-whole-beat, 50-sample GCNM training pack."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


EXPECTED = {
    "train": {"beats": 800, "samples": 40_000, "mode": "linearized"},
    "validation": {"beats": 100, "samples": 5_000, "mode": "linearized"},
    "test": {"beats": 100, "samples": 5_000, "mode": "nonlinear"},
}


def validate(root: Path) -> dict:
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata["beat_resolution"]["generated_samples_per_beat"] != 50:
        raise ValueError("dataset was not generated at 50 samples per beat")
    if metadata["beat_resolution"]["sample_stride"] != 1:
        raise ValueError("dataset does not retain every generated beat sample")
    reports = {}
    for split, expected in EXPECTED.items():
        path = root / f"{split}.npz"
        with np.load(path, mmap_mode="r") as arrays:
            required = {
                "sigma", "sigma_baseline", "V", "V_template", "beat_id",
                "sample_index", "phase_index",
            }
            missing = sorted(required - set(arrays.files))
            if missing:
                raise KeyError(f"{path} is missing {missing}")
            count = len(arrays["sigma"])
            if count != expected["samples"]:
                raise ValueError(f"{split} has {count} samples, expected {expected['samples']}")
            if arrays["V"].shape != (count, 32):
                raise ValueError(f"{split} voltage shape is {arrays['V'].shape}")
            if arrays["V_template"].shape != (count, 32):
                raise ValueError(f"{split} template shape is {arrays['V_template'].shape}")
            beat_ids = np.asarray(arrays["beat_id"])
            sample_index = np.asarray(arrays["sample_index"])
            if not np.array_equal(sample_index, np.asarray(arrays["phase_index"])):
                raise ValueError(f"{split} sample_index and phase_index differ")
            per_beat = Counter(int(value) for value in beat_ids)
            if len(per_beat) != expected["beats"] or set(per_beat.values()) != {50}:
                raise ValueError(f"{split} is not {expected['beats']} complete 50-sample beats")
            for beat_id in sorted(per_beat):
                selected = sample_index[beat_ids == beat_id]
                if not np.array_equal(selected, np.arange(50)):
                    raise ValueError(f"{split} beat {beat_id} is not ordered 0..49")
            if not all(np.all(np.isfinite(arrays[key])) for key in ("sigma", "V", "V_template")):
                raise ValueError(f"{split} contains non-finite training values")

        anatomy_path = root / f"{split}_anatomy.json"
        anatomy = json.loads(anatomy_path.read_text(encoding="utf-8"))
        if len(anatomy) != expected["samples"]:
            raise ValueError(f"{split} anatomy count differs from array count")
        modes = {record["simulation_mode"] for record in anatomy}
        if modes != {expected["mode"]}:
            raise ValueError(f"{split} simulation modes are {sorted(modes)}")
        reports[split] = {
            "beats": expected["beats"],
            "samples_per_beat": 50,
            "samples": expected["samples"],
            "simulation_mode": expected["mode"],
        }
    return {
        "schema": "gcnm-beats1000x50-validation-v1",
        "root": str(root.resolve()),
        "valid": True,
        "splits": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.root)
    output = args.root / "validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
