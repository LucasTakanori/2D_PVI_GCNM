#!/usr/bin/env python3
"""Build an immutable one-beat differential GCNM architecture-gate pack.

The v2 synthetic archive deliberately retains both absolute and referenced
quantities.  This adapter selects one complete 50-sample beat and makes the
training contract explicit without modifying the source archive::

    V_delta_reference -> sigma_delta_reference

The production PVI one-step Newton result is recomputed from the *same* voltage
array selected for the GCNM.  It is a baseline/control, never a training label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_differential import production_reconstruction_matrix
from gcnm_pvi.runtime import build_runtime


FRAMES_PER_BEAT = 50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_complete_beat(
    anatomy_ids: np.ndarray,
    beat_ids: np.ndarray,
    sample_indices: np.ndarray,
    *,
    anatomy_id: int | None,
    beat_id: int | None,
) -> tuple[np.ndarray, int, int]:
    anatomies = np.unique(anatomy_ids)
    if anatomy_id is not None:
        anatomies = np.asarray([anatomy_id], dtype=anatomy_ids.dtype)
    for anatomy in anatomies:
        available = np.unique(beat_ids[anatomy_ids == anatomy])
        if beat_id is not None:
            available = np.asarray([beat_id], dtype=beat_ids.dtype)
        for beat in available:
            selected = np.flatnonzero(
                (anatomy_ids == anatomy) & (beat_ids == beat)
            )
            selected = selected[np.argsort(sample_indices[selected])]
            if len(selected) == FRAMES_PER_BEAT and np.array_equal(
                sample_indices[selected], np.arange(FRAMES_PER_BEAT)
            ):
                return selected, int(anatomy), int(beat)
    raise ValueError("no selected anatomy/beat contains samples 0..49 exactly once")


def build_pack(
    *,
    config: Path,
    source_path: Path,
    output_root: Path,
    voltage_key: str,
    anatomy_id: int | None = None,
    beat_id: int | None = None,
) -> dict:
    if output_root.exists():
        raise FileExistsError(f"immutable differential pilot root exists: {output_root}")
    with np.load(source_path) as source:
        required = (
            "sigma_delta_reference",
            voltage_key,
            "anatomy_id",
            "beat_id",
            "sample_index",
        )
        missing = [key for key in required if key not in source]
        if missing:
            raise KeyError(f"{source_path} lacks differential pilot fields {missing}")
        selected, chosen_anatomy, chosen_beat = _select_complete_beat(
            np.asarray(source["anatomy_id"]),
            np.asarray(source["beat_id"]),
            np.asarray(source["sample_index"]),
            anatomy_id=anatomy_id,
            beat_id=beat_id,
        )
        sigma = np.asarray(
            source["sigma_delta_reference"][selected], dtype=np.float32
        )
        voltage = np.asarray(source[voltage_key][selected], dtype=np.float32)
        sample_index = np.asarray(source["sample_index"][selected], dtype=np.int16)

    # Use the exact production PVI inverse settings from the ring config.  This
    # is intentionally independent of the learned nonlinear GCNM stages.
    cfg = GcnmConfig.from_yaml(config)
    runtime = build_runtime(cfg, include_forward=False)
    inverse = production_reconstruction_matrix(
        runtime["physics_inv"],
        hyper_pvi=cfg.hyper_pvi,
        regularizer=runtime["mappings"].laplace,
        sigma_init=0.7,
    )
    # Synthetic forward voltages already use the physical FEM sign. Archived
    # real PVI voltage uses the opposite saved/display convention before the
    # MATLAB ``-D1`` operator. Convert the synthetic physical vector first so
    # the Newton control has the same physical conductivity sign as the truth.
    newton = (inverse @ (-voltage.astype(np.float64)).T).T.astype(np.float32)
    baseline = np.full_like(sigma, 0.7, dtype=np.float32)
    payload = {
        "sigma": sigma,
        "sigma_baseline": baseline,
        "V": voltage,
        "newton": newton,
        "anatomy_id": np.full(FRAMES_PER_BEAT, chosen_anatomy, dtype=np.int32),
        "beat_id": np.full(FRAMES_PER_BEAT, chosen_beat, dtype=np.int32),
        "sample_index": sample_index,
    }
    if not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(voltage)):
        raise ValueError("selected differential truth/voltage contains non-finite values")
    if not np.array_equal(sigma[0], np.zeros_like(sigma[0])):
        raise ValueError("differential conductivity reference frame is not exactly zero")
    if not np.array_equal(voltage[0], np.zeros_like(voltage[0])):
        raise ValueError("differential voltage reference frame is not exactly zero")

    output_root.mkdir(parents=True, exist_ok=False)
    for split in ("train", "validation", "test"):
        np.savez_compressed(output_root / f"{split}.npz", **payload)
    truth_rms = float(np.sqrt(np.mean(sigma * sigma)))
    newton_rms = float(np.sqrt(np.mean(newton * newton)))
    manifest = {
        "schema": "pvi-gcnm-differential-one-beat-pilot-v1",
        "source": str(source_path.resolve()),
        "source_sha256": _sha256(source_path),
        "config": str(config.resolve()),
        "config_sha256": _sha256(config),
        "anatomy_id": chosen_anatomy,
        "beat_id": chosen_beat,
        "samples_per_beat": FRAMES_PER_BEAT,
        "contract": "V_delta_reference -> sigma_delta_reference",
        "voltage_key": voltage_key,
        "voltage_units": "V",
        "conductivity_units": "S/m",
        "signed_target": True,
        "noise_policy": (
            "clean referenced voltage architecture gate"
            if voltage_key == "V_clean_delta_reference"
            else "augmented referenced voltage robustness gate"
        ),
        "newton_control": (
            "PVI production one-step matrix at homogeneous 0.7 S/m applied "
            "to the identical selected voltage after the documented synthetic "
            "physical-to-PVI-saved sign conversion"
        ),
        "synthetic_physical_to_pvi_saved_voltage_sign": -1.0,
        "newton_is_training_label": False,
        "train_validation_test_are_identical": True,
        "purpose": "architecture learnability and Newton-relative comparison only",
        "rms": {
            "voltage_v": float(np.sqrt(np.mean(voltage * voltage))),
            "truth_s_m": truth_rms,
            "newton_s_m": newton_rms,
            "newton_to_truth": newton_rms / max(truth_rms, 1e-12),
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=root / "configs/rings_b045/US120.yaml"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--voltage-key",
        choices=["V_clean_delta_reference", "V_delta_reference"],
        default="V_clean_delta_reference",
    )
    parser.add_argument("--anatomy-id", type=int)
    parser.add_argument("--beat-id", type=int)
    args = parser.parse_args()
    report = build_pack(
        config=args.config,
        source_path=args.dataset_root / "full" / f"{args.source_split}.npz",
        output_root=args.output_root,
        voltage_key=args.voltage_key,
        anatomy_id=args.anatomy_id,
        beat_id=args.beat_id,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
