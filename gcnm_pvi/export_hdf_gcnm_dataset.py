#!/usr/bin/env python3
"""Build leakage-safe subject006 GCNM pseudo-label packs from aligned HDF data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_differential import reconstruct_newton_series
from gcnm_pvi.runtime import build_runtime


SPLIT_TRIALS = {
    "train": np.arange(1, 8),
    "validation": np.arange(8, 10),
    "test": np.arange(10, 12),
}


def _load_period_trials(candidate_npz: Path, match_npy: Path, num_periods: int) -> np.ndarray:
    candidates = np.load(candidate_npz)
    match = np.asarray(np.load(match_npy), dtype=np.int64)
    if match.shape != (num_periods,):
        raise ValueError(f"period match has shape {match.shape}; expected {(num_periods,)}")
    return np.asarray(candidates["trial_index"][match], dtype=np.int16)


def export_dataset(
    config: Path,
    h5_path: Path,
    candidate_npz: Path,
    match_npy: Path,
    out_dir: Path,
    *,
    phase_stride: int = 5,
) -> dict:
    cfg = GcnmConfig.from_yaml(config)
    runtime = build_runtime(cfg, include_forward=False)

    with h5py.File(h5_path, "r") as f:
        num_periods = int(f["metadata/num_periods"][()].item())
        period_length = int(f["metadata/period_length"][()].item())
        clean_ranges = np.asarray(f["masks/mask01"], dtype=np.int64)
        if not np.all(clean_ranges[:, 0] == clean_ranges[:, 1]):
            raise ValueError("mask01 is not a list of individual clean periods")
        clean_periods = np.unique(clean_ranges[:, 0])
        resistance = f["data/pviHP/resistance"]
        phases = np.arange(0, period_length, phase_stride, dtype=np.int16)
        flat_indices = np.concatenate(
            [(period - 1) * period_length + phases for period in clean_periods]
        )
        voltage = -float(cfg.stim_current) * np.asarray(resistance[:, flat_indices])

    period_trials = _load_period_trials(candidate_npz, match_npy, num_periods)
    sample_periods = np.repeat(clean_periods, len(phases)).astype(np.int16)
    sample_phases = np.tile(phases, len(clean_periods)).astype(np.int16)
    sample_trials = period_trials[sample_periods - 1]

    # These are deterministic production pseudo-labels, not independent
    # conductivity ground truth.  They make the data path testable while the
    # synthetic/anatomical ground-truth strategy is developed.
    sigma = reconstruct_newton_series(
        runtime["physics_inv"],
        voltage,
        hyper_pvi=cfg.hyper_pvi,
        regularizer=runtime["mappings"].laplace,
        vmeas_ref=np.zeros(voltage.shape[0]),
    ).T
    voltage = voltage.T

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split, trials in SPLIT_TRIALS.items():
        select = np.isin(sample_trials, trials)
        counts[split] = int(np.count_nonzero(select))
        split_path = out_dir / f"{split}.npz"
        if not split_path.exists():
            np.savez_compressed(
                split_path,
                sigma=sigma[select].astype(np.float32),
                V=voltage[select].astype(np.float32),
                period_id=sample_periods[select],
                phase=sample_phases[select],
                trial_index=sample_trials[select],
            )

    train_validation = np.isin(sample_trials, np.concatenate([SPLIT_TRIALS["train"], SPLIT_TRIALS["validation"]]))
    tv_order = np.concatenate(
        [
            np.flatnonzero(np.isin(sample_trials, SPLIT_TRIALS["train"])),
            np.flatnonzero(np.isin(sample_trials, SPLIT_TRIALS["validation"])),
        ]
    )
    assert int(np.count_nonzero(train_validation)) == len(tv_order)
    train_validation_path = out_dir / "train_validation.npz"
    if not train_validation_path.exists():
        np.savez_compressed(
            train_validation_path,
            sigma=sigma[tv_order].astype(np.float32),
            V=voltage[tv_order].astype(np.float32),
            period_id=sample_periods[tv_order],
            phase=sample_phases[tv_order],
            trial_index=sample_trials[tv_order],
            validation_start=np.asarray([counts["train"]], dtype=np.int64),
        )

    metadata = {
        "source_h5": str(h5_path.resolve()),
        "config": str(config.resolve()),
        "label_type": "MATLAB-production one-step Newton pseudo-label",
        "ground_truth_warning": (
            "sigma repeats the deterministic PVI inverse and is suitable for "
            "pipeline validation/distillation only, not superiority claims."
        ),
        "clean_mask": "mask01",
        "clean_periods": int(len(clean_periods)),
        "period_length": period_length,
        "phase_stride": phase_stride,
        "phases": phases.tolist(),
        "samples": counts,
        "split_trials": {key: value.tolist() for key, value in SPLIT_TRIALS.items()},
        "leakage_control": (
            "Splits are disjoint acquisition trials; all phases from a cardiac "
            "period remain in the same split."
        ),
        "period_alignment": {
            "candidate_npz": str(candidate_npz.resolve()),
            "match_npy": str(match_npy.resolve()),
            "candidate_periods": 1310,
            "matched_hdf_periods": num_periods,
        },
        "element_count": int(sigma.shape[1]),
        "voltage_channels": int(voltage.shape[1]),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs" / "subject006_pvi08_production.yaml")
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
    parser.add_argument("--out-dir", type=Path, default=root / "data" / "subject006_gcnm_hdf")
    parser.add_argument("--phase-stride", type=int, default=5)
    args = parser.parse_args()
    metadata = export_dataset(
        args.config,
        args.h5,
        args.candidate_npz,
        args.match_npy,
        args.out_dir,
        phase_stride=args.phase_stride,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
