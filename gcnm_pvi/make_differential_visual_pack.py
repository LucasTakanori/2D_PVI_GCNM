#!/usr/bin/env python3
"""Create a canonical multi-beat differential NPZ for GCNM visual checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_differential import production_reconstruction_matrix
from gcnm_pvi.runtime import build_runtime


def _beat_templates(voltage: np.ndarray, beat: np.ndarray) -> np.ndarray:
    output = np.empty_like(voltage)
    for beat_id in np.unique(beat):
        selected = np.flatnonzero(beat == beat_id)
        values = voltage[selected]
        centered = values - np.mean(values, axis=0, keepdims=True)
        if np.any(np.abs(centered) > 1e-15):
            u, singular, vh = np.linalg.svd(centered, full_matrices=False)
            spatial = vh[0]
            temporal = singular[0] * u[:, 0]
            template = temporal[int(np.argmax(np.abs(temporal)))] * spatial
        else:
            template = np.zeros(voltage.shape[1], dtype=voltage.dtype)
        output[selected] = template
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=root / "configs/rings_b045/US120.yaml"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--voltage-key",
        choices=["V_clean_delta_reference", "V_delta_reference"],
        default="V_clean_delta_reference",
    )
    parser.add_argument("--anatomy-id", type=int)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"immutable visual pack exists for {args.output}")

    with np.load(args.source) as source:
        anatomy_ids = np.asarray(source["anatomy_id"])
        chosen = int(np.unique(anatomy_ids)[0] if args.anatomy_id is None else args.anatomy_id)
        selected = np.flatnonzero(anatomy_ids == chosen)
        order = np.lexsort(
            (np.asarray(source["sample_index"])[selected], np.asarray(source["beat_id"])[selected])
        )
        selected = selected[order]
        sigma = np.asarray(source["sigma_delta_reference"][selected], dtype=np.float32)
        voltage = np.asarray(source[args.voltage_key][selected], dtype=np.float32)
        beat = np.asarray(source["beat_id"][selected], dtype=np.int32)
        sample = np.asarray(source["sample_index"][selected], dtype=np.int16)
    available = np.unique(beat)
    if len(available) < 3:
        raise ValueError("visual pack requires at least three beats from one anatomy")
    for value in available:
        if not np.array_equal(sample[beat == value], np.arange(50)):
            raise ValueError(f"beat {int(value)} does not contain ordered samples 0..49")

    cfg = GcnmConfig.from_yaml(args.config)
    runtime = build_runtime(cfg, include_forward=False)
    inverse = production_reconstruction_matrix(
        runtime["physics_inv"],
        hyper_pvi=cfg.hyper_pvi,
        regularizer=runtime["mappings"].laplace,
        sigma_init=0.7,
    )
    newton = (inverse @ (-voltage.astype(np.float64)).T).T.astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sigma=sigma,
        sigma_baseline=np.full_like(sigma, 0.7),
        V=voltage,
        V_template=_beat_templates(voltage, beat).astype(np.float32),
        newton=newton,
        anatomy_id=np.full(len(sigma), chosen, dtype=np.int32),
        beat_id=beat,
        sample_index=sample,
    )
    report = {
        "schema": "pvi-gcnm-differential-visual-pack-v1",
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "contract": "V_delta_reference -> sigma_delta_reference",
        "voltage_key": args.voltage_key,
        "anatomy_id": chosen,
        "beats": available.tolist(),
        "frames": int(len(sigma)),
        "newton_control": (
            "production one-step PVI on identical voltage after synthetic "
            "physical-to-PVI-saved sign conversion"
        ),
        "synthetic_physical_to_pvi_saved_voltage_sign": -1.0,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
