#!/usr/bin/env python3
"""Validate a canonical full-band differential GCNM cohort before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gcnm_pvi.mesh_registry import sha256_file
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.runtime import build_runtime


SPLITS = ("train", "validation", "test")
REQUIRED = (
    "sigma",
    "sigma_baseline",
    "V",
    "V_augmented",
    "V_template",
    "anatomy_id",
    "beat_id",
    "sample_index",
)


def validate(root: Path, expected: dict[str, int]) -> dict:
    root = Path(root)
    if (root / "_INCOMPLETE").exists():
        raise RuntimeError(f"differential cohort is incomplete: {root}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract") != "one frame: V_delta_reference -> sigma_delta_reference":
        raise ValueError("differential cohort has the wrong training contract")
    config_path = Path(manifest["config"])
    cfg = GcnmConfig.from_yaml(config_path)
    runtime = build_runtime(cfg, include_forward=False)
    expected_elements = int(runtime["mappings"].num_elements)
    reports = {}
    split_ids = {}
    for split in SPLITS:
        path = root / f"{split}.npz"
        with np.load(path) as source:
            missing = [key for key in REQUIRED if key not in source]
            if missing:
                raise KeyError(f"{path} lacks {missing}")
            if split == "test" and "newton" not in source:
                raise KeyError("test archive lacks PVI one-step Newton control")
            count = len(source["sigma"])
            for key in REQUIRED:
                if len(source[key]) != count:
                    raise ValueError(f"{path} {key} length mismatch")
            if source["sigma"].shape != (count, expected_elements):
                raise ValueError(
                    f"{path} has conductivity shape {source['sigma'].shape}; "
                    f"ring config requires {(count, expected_elements)}"
                )
            if source["V"].shape != (count, 32):
                raise ValueError(f"{path} has unexpected measurement shape {source['V'].shape}")
            if source["sigma_baseline"].shape != source["sigma"].shape:
                raise ValueError(f"{path} baseline/target shapes differ")
            if split == "test" and source["newton"].shape != source["sigma"].shape:
                raise ValueError(f"{path} Newton/target shapes differ")
            if not np.all(np.isfinite(source["sigma"])) or not np.all(np.isfinite(source["V"])):
                raise ValueError(f"{path} contains non-finite truth or clean voltage")
            if not np.allclose(source["sigma_baseline"], 0.7):
                raise ValueError(f"{path} inverse baseline is not 0.7 S/m")
            anatomy = np.asarray(source["anatomy_id"])
            beat = np.asarray(source["beat_id"])
            sample = np.asarray(source["sample_index"])
            split_ids[split] = set(int(value) for value in np.unique(anatomy))
            for anatomy_id in split_ids[split]:
                selected = np.flatnonzero(anatomy == anatomy_id)
                if len(selected) != 250 or len(np.unique(beat[selected])) != 5:
                    raise ValueError(f"{path} anatomy {anatomy_id} is not five beats")
                if not np.allclose(source["sigma"][selected[0]], 0.0, atol=2e-7):
                    raise ValueError(f"{path} anatomy {anatomy_id} truth reference is not zero")
                if not np.allclose(source["V"][selected[0]], 0.0, atol=2e-9):
                    raise ValueError(f"{path} anatomy {anatomy_id} voltage reference is not zero")
                for beat_id in np.unique(beat[selected]):
                    frames = np.flatnonzero(beat == beat_id)
                    frames = frames[np.argsort(sample[frames])]
                    if len(frames) != 50 or not np.array_equal(sample[frames], np.arange(50)):
                        raise ValueError(f"{path} beat {beat_id} is incomplete or unordered")
            observed = len(split_ids[split])
            if observed != expected[split]:
                raise ValueError(f"{split} has {observed} anatomies, expected {expected[split]}")
            reports[split] = {
                "frames": count,
                "anatomies": observed,
                "beats": int(len(np.unique(beat))),
                "truth_rms_s_m": float(np.sqrt(np.mean(source["sigma"] ** 2))),
                "clean_voltage_rms_v": float(np.sqrt(np.mean(source["V"] ** 2))),
                "sha256": sha256_file(path),
            }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                raise ValueError(f"anatomies leak between {left}/{right}: {overlap}")
    with np.load(root / "test.npz") as clean, np.load(root / "test_augmented.npz") as augmented:
        for key in ("sigma", "anatomy_id", "beat_id", "sample_index"):
            if not np.array_equal(clean[key], augmented[key]):
                raise ValueError(f"augmented test changes {key}")
        if np.array_equal(clean["V"], augmented["V"]):
            raise ValueError("augmented robustness voltage equals clean voltage")
    return {
        "schema": "pvi-gcnm-differential-cohort-validation-v1",
        "status": "pass",
        "root": str(root.resolve()),
        "source_schema": manifest.get("source_schema"),
        "contract": manifest["contract"],
        "ring": config_path.stem,
        "num_elements": expected_elements,
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-anatomies", type=int, default=160)
    parser.add_argument("--validation-anatomies", type=int, default=20)
    parser.add_argument("--test-anatomies", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(
        args.root,
        {
            "train": args.train_anatomies,
            "validation": args.validation_anatomies,
            "test": args.test_anatomies,
        },
    )
    output = args.output or args.root / "validation.json"
    if output.exists():
        raise FileExistsError(f"immutable validation output exists: {output}")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
