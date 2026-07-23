#!/usr/bin/env python3
"""Create a compact canonical differential view of a full-band v2 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_differential import production_reconstruction_matrix
from gcnm_pvi.make_differential_visual_pack import _beat_templates
from gcnm_pvi.runtime import build_runtime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_split(
    dataset_root: Path,
    split: str,
    *,
    source_schema: str,
    voltage_key: str,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load full differential fields from either retained source schema."""

    if source_schema == "pvi-gcnm-continuous-full-hp-lp-beats-v2":
        source_path = dataset_root / "full" / f"{split}.npz"
        with np.load(source_path) as source:
            output = {
                "sigma": np.asarray(source["sigma_delta_reference"], dtype=np.float32),
                "voltage": np.asarray(source[voltage_key], dtype=np.float32),
                "augmented": np.asarray(source["V_delta_reference"], dtype=np.float32),
                "anatomy": np.asarray(source["anatomy_id"], dtype=np.int32),
                "beat": np.asarray(source["beat_id"], dtype=np.int32),
                "sample": np.asarray(source["sample_index"], dtype=np.int16),
            }
        return output, {"full": _sha256(source_path)}
    if source_schema != "pvi-gcnm-continuous-hp-lp-beats-v1":
        raise ValueError(f"unsupported source schema {source_schema!r}")

    hp_path = dataset_root / "hp" / f"{split}.npz"
    lp_path = dataset_root / "lp" / f"{split}.npz"
    clean_key = "V_clean" if voltage_key == "V_clean_delta_reference" else "V"
    with np.load(hp_path) as hp, np.load(lp_path) as lp:
        for key in ("anatomy_id", "beat_id", "sample_index"):
            if not np.array_equal(hp[key], lp[key]):
                raise ValueError(f"HP/LP {key} identities differ in {split}")
        output = {
            "sigma": np.asarray(hp["sigma"] + lp["sigma"], dtype=np.float32),
            "voltage": np.asarray(hp[clean_key] + lp[clean_key], dtype=np.float32),
            "augmented": np.asarray(hp["V"] + lp["V"], dtype=np.float32),
            "anatomy": np.asarray(hp["anatomy_id"], dtype=np.int32),
            "beat": np.asarray(hp["beat_id"], dtype=np.int32),
            "sample": np.asarray(hp["sample_index"], dtype=np.int16),
        }
    return output, {"hp": _sha256(hp_path), "lp": _sha256(lp_path)}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=root / "configs/rings_b045/US120.yaml"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--voltage-key",
        choices=["V_clean_delta_reference", "V_delta_reference"],
        default="V_clean_delta_reference",
    )
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"immutable differential cohort exists: {args.output_root}")

    source_metadata = json.loads(
        (args.dataset_root / "metadata.json").read_text(encoding="utf-8")
    )
    source_schema = source_metadata.get("schema")

    cfg = GcnmConfig.from_yaml(args.config)
    runtime = build_runtime(cfg, include_forward=False)
    inverse = production_reconstruction_matrix(
        runtime["physics_inv"],
        hyper_pvi=cfg.hyper_pvi,
        regularizer=runtime["mappings"].laplace,
        sigma_init=0.7,
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    incomplete = args.output_root / "_INCOMPLETE"
    incomplete.write_text("differential cohort view in progress\n", encoding="utf-8")
    splits = {}
    source_hashes = {}
    for split in ("train", "validation", "test"):
        values, source_hashes[split] = _load_split(
            args.dataset_root,
            split,
            source_schema=source_schema,
            voltage_key=args.voltage_key,
        )
        sigma = values["sigma"]
        voltage = values["voltage"]
        anatomy = values["anatomy"]
        beat = values["beat"]
        sample = values["sample"]
        augmented = values["augmented"]
        payload = {
            "sigma": sigma,
            "sigma_baseline": np.full_like(sigma, 0.7),
            "V": voltage,
            "V_template": _beat_templates(voltage, beat).astype(np.float32),
            "V_augmented": augmented,
            "anatomy_id": anatomy,
            "beat_id": beat,
            "sample_index": sample,
        }
        if split == "test":
            payload["newton"] = (
                inverse @ (-voltage.astype(np.float64)).T
            ).T.astype(np.float32)
        for anatomy_id in np.unique(anatomy):
            first = np.flatnonzero(anatomy == anatomy_id)[0]
            if not np.allclose(sigma[first], 0.0, atol=2e-7):
                raise ValueError(f"{split} anatomy {anatomy_id} truth reference is not zero")
            if not np.allclose(voltage[first], 0.0, atol=2e-9):
                raise ValueError(f"{split} anatomy {anatomy_id} voltage reference is not zero")
        np.savez_compressed(args.output_root / f"{split}.npz", **payload)
        if split == "test":
            augmented_payload = dict(payload)
            augmented_payload["V"] = augmented
            augmented_payload["V_template"] = _beat_templates(
                augmented, beat
            ).astype(np.float32)
            augmented_payload["newton"] = (
                inverse @ (-augmented.astype(np.float64)).T
            ).T.astype(np.float32)
            np.savez_compressed(
                args.output_root / "test_augmented.npz", **augmented_payload
            )
        splits[split] = {
            "frames": int(len(sigma)),
            "anatomies": int(len(np.unique(anatomy))),
            "beats": int(len(np.unique(beat))),
            "truth_rms_s_m": float(np.sqrt(np.mean(sigma * sigma))),
            "clean_voltage_rms_v": float(np.sqrt(np.mean(voltage * voltage))),
            "augmented_voltage_rms_v": float(np.sqrt(np.mean(augmented * augmented))),
        }
    manifest = {
        "schema": "pvi-gcnm-differential-cohort-v1",
        "source_root": str(args.dataset_root.resolve()),
        "source_schema": source_schema,
        "source_metadata_sha256": _sha256(args.dataset_root / "metadata.json"),
        "source_archive_sha256": source_hashes,
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "contract": "one frame: V_delta_reference -> sigma_delta_reference",
        "legacy_component_identity": (
            "sigma_full=sigma_hp+sigma_lp; V_full=V_hp+V_lp"
            if source_schema == "pvi-gcnm-continuous-hp-lp-beats-v1"
            else None
        ),
        "training_voltage_key": args.voltage_key,
        "noise_policy": "clean training plus retained augmented voltage robustness view",
        "augmented_robustness_archive": "test_augmented.npz",
        "synthetic_physical_to_pvi_saved_voltage_sign": -1.0,
        "newton_is_training_label": False,
        "splits": splits,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    incomplete.unlink()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
