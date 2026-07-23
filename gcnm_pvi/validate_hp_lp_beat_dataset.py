#!/usr/bin/env python3
"""Fail-fast validation for continuous full-band and HP/LP synthetic packs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gcnm_pvi.mesh_registry import sha256_file


LEGACY_COMPONENTS = ("hp", "lp")
FULL_COMPONENTS = ("full", "hp", "lp")
SPLITS = ("train", "validation", "test")
REQUIRED = (
    "sigma",
    "sigma_baseline",
    "V",
    "V_clean",
    "V_template",
    "tissue_labels",
    "anatomy_id",
    "beat_id",
    "sample_index",
)
FULL_REQUIRED = (
    "sigma_absolute",
    "sigma_delta_reference",
    "sigma_reference",
    "sigma_resting",
    "V_absolute_raw",
    "V_clean_absolute_raw",
    "V_absolute_filtered",
    "V_clean_absolute_filtered",
    "V_reference_filtered",
    "V_delta_reference",
    "V_clean_delta_reference",
)


def _validate_archive(
    path: Path, *, component: str, require_newton: bool
) -> dict:
    with np.load(path) as source:
        required = REQUIRED + (FULL_REQUIRED if component == "full" else ())
        missing = [key for key in required if key not in source]
        if missing:
            raise KeyError(f"{path} is missing {missing}")
        count = len(source["sigma"])
        if count == 0 or count % 250:
            raise ValueError(f"{path} has {count} frames, not anatomy-aligned groups of 250")
        for key in required:
            if len(source[key]) != count:
                raise ValueError(f"{path} {key} count differs from sigma")
        if source["sigma"].ndim != 2 or source["V"].ndim != 2:
            raise ValueError(f"{path} has invalid conductivity/voltage dimensions")
        if source["V"].shape[1] != 32:
            raise ValueError(f"{path} must contain 32 PVI measurements")
        if source["sigma_baseline"].shape != source["sigma"].shape:
            raise ValueError(f"{path} baseline/target shapes differ")
        if not np.allclose(source["sigma_baseline"], 0.7):
            raise ValueError(f"{path} training baseline is not homogeneous 0.7 S/m")
        for key in ("sigma", "V", "V_clean", "V_template"):
            if not np.all(np.isfinite(source[key])):
                raise ValueError(f"{path} {key} contains non-finite values")
        target = np.asarray(source["sigma"])
        if component == "full":
            if np.any(target <= 0):
                raise ValueError(f"{path} absolute conductivity is not positive")
        elif not np.any(target < 0) or not np.any(target > 0):
            raise ValueError(f"{path} does not preserve both target signs")
        anatomy_ids = np.asarray(source["anatomy_id"])
        beat_ids = np.asarray(source["beat_id"])
        sample_indices = np.asarray(source["sample_index"])
        for anatomy in np.unique(anatomy_ids):
            selected = np.flatnonzero(anatomy_ids == anatomy)
            if len(selected) != 250:
                raise ValueError(f"{path} anatomy {anatomy} has {len(selected)} frames")
            beats = np.unique(beat_ids[selected])
            if len(beats) != 5:
                raise ValueError(f"{path} anatomy {anatomy} does not have five beats")
            if component == "full":
                if not np.allclose(
                    source["sigma_delta_reference"][selected[0]], 0.0, atol=2e-7
                ):
                    raise ValueError(
                        f"{path} anatomy {anatomy} first delta frame is not zero"
                    )
                if not np.allclose(
                    target[selected[0]], source["sigma_reference"][selected[0]]
                ):
                    raise ValueError(
                        f"{path} anatomy {anatomy} first absolute frame differs from reference"
                    )
            elif not np.allclose(target[selected[0]], 0.0, atol=2e-7):
                raise ValueError(f"{path} anatomy {anatomy} first frame is not the reference")
            for beat in beats:
                frames = np.flatnonzero(beat_ids == beat)
                frames = frames[np.argsort(sample_indices[frames])]
                if len(frames) != 50 or not np.array_equal(
                    sample_indices[frames], np.arange(50)
                ):
                    raise ValueError(f"{path} beat {beat} is not a complete 50-sample beat")
                template = source["V_template"][frames]
                if not np.allclose(template, template[:1]):
                    raise ValueError(f"{path} beat {beat} template changes within the beat")
        if require_newton and "newton" not in source:
            raise KeyError(f"{path} test archive lacks production Newton controls")
        absolute_identity_error = None
        voltage_identity_error = None
        if component == "full":
            for key in FULL_REQUIRED:
                if not np.all(np.isfinite(source[key])):
                    raise ValueError(f"{path} {key} contains non-finite values")
            absolute_identity_error = float(
                np.max(
                    np.abs(
                        source["sigma_reference"]
                        + source["sigma_delta_reference"]
                        - source["sigma"]
                    )
                )
            )
            voltage_identity_error = float(
                np.max(
                    np.abs(
                        source["V_reference_filtered"]
                        + source["V_delta_reference"]
                        - source["V_absolute_filtered"]
                    )
                )
            )
            if absolute_identity_error > 2e-6:
                raise ValueError(
                    f"{path} absolute/reference identity error is {absolute_identity_error}"
                )
            # The absolute voltage scale changes with ring geometry. These
            # three arrays are serialized independently as float32, so the
            # correct identity tolerance is a small number of float32 ULPs at
            # the observed magnitude rather than one US120-specific constant.
            voltage_scale = float(
                max(np.max(np.abs(source["V_absolute_filtered"])), 1.0)
            )
            voltage_identity_tolerance = float(
                max(2e-8, 4.0 * np.finfo(np.float32).eps * voltage_scale)
            )
            if voltage_identity_error > voltage_identity_tolerance:
                raise ValueError(
                    f"{path} absolute/reference voltage error is "
                    f"{voltage_identity_error} (tolerance={voltage_identity_tolerance})"
                )
        return {
            "frames": count,
            "anatomies": int(len(np.unique(anatomy_ids))),
            "beats": int(len(np.unique(beat_ids))),
            "target_rms": float(np.sqrt(np.mean(target * target))),
            "negative_target_fraction": float(np.mean(target < 0)),
            "voltage_rms": float(np.sqrt(np.mean(source["V"] ** 2))),
            "absolute_identity_max_error": absolute_identity_error,
            "voltage_identity_max_error": voltage_identity_error,
            "voltage_identity_tolerance": (
                voltage_identity_tolerance if component == "full" else None
            ),
            "sha256": sha256_file(path),
        }


def validate(root: Path, *, require_exact: bool = True) -> dict:
    root = Path(root)
    if (root / "_INCOMPLETE").exists():
        raise RuntimeError(f"synthetic dataset is incomplete: {root}")
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    schema = metadata.get("schema")
    if schema == "pvi-gcnm-continuous-hp-lp-beats-v1":
        components = LEGACY_COMPONENTS
    elif schema == "pvi-gcnm-continuous-full-hp-lp-beats-v2":
        components = FULL_COMPONENTS
    else:
        raise ValueError(f"unsupported synthetic metadata schema: {schema!r}")
    reports = {}
    split_anatomies: dict[str, set[int]] = {}
    for split in SPLITS:
        reports[split] = {}
        component_ids = None
        for component in components:
            archive = root / component / f"{split}.npz"
            anatomy_json = root / component / (
                f"{split}_anatomies.json"
                if component == "full"
                else f"{split}_anatomy.json"
            )
            report = _validate_archive(
                archive, component=component, require_newton=split == "test"
            )
            records = json.loads(anatomy_json.read_text(encoding="utf-8"))
            expected_records = (
                report["anatomies"] if component == "full" else report["frames"]
            )
            if len(records) != expected_records:
                raise ValueError(f"{anatomy_json} record count differs from archive")
            if component == "full":
                for record in records:
                    for key in (
                        "model",
                        "beat_waveforms",
                        "beat_durations_s",
                        "acquisition_effects",
                    ):
                        if key not in record:
                            raise KeyError(f"{anatomy_json} record is missing {key}")
            with np.load(archive) as source:
                ids = np.asarray(source["anatomy_id"])
                identity = np.column_stack(
                    (ids, source["beat_id"], source["sample_index"])
                )
            if component_ids is None:
                component_ids = identity
                split_anatomies[split] = set(int(value) for value in np.unique(ids))
            elif not np.array_equal(component_ids, identity):
                raise ValueError(f"full/HP/LP sample identities differ in {split}")
            reports[split][component] = {
                **report,
                "anatomy_sha256": sha256_file(anatomy_json),
            }
        if components == FULL_COMPONENTS:
            with (
                np.load(root / "full" / f"{split}.npz") as full,
                np.load(root / "hp" / f"{split}.npz") as hp,
                np.load(root / "lp" / f"{split}.npz") as lp,
            ):
                if not np.allclose(
                    full["sigma_delta_reference"],
                    hp["sigma"] + lp["sigma"],
                    atol=2e-6,
                ):
                    raise ValueError(
                        f"referenced full conductivity does not equal HP+LP in {split}"
                    )
                if not np.allclose(
                    full["V_delta_reference"], hp["V"] + lp["V"], atol=2e-9
                ):
                    raise ValueError(
                        f"referenced full voltage does not equal HP+LP in {split}"
                    )
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = split_anatomies[left] & split_anatomies[right]
            if overlap:
                raise ValueError(f"virtual anatomies leak between {left}/{right}: {overlap}")
    if require_exact:
        gate = metadata.get("linearized_acceleration_gate", {})
        if not gate.get("verified"):
            raise ValueError("linearized acceleration lacks exact-nonlinear verification")
        if metadata.get("split_modes", {}).get("validation") != "nonlinear":
            raise ValueError("validation split is not exact nonlinear")
        if metadata.get("split_modes", {}).get("test") != "nonlinear":
            raise ValueError("test split is not exact nonlinear")
    expected = {"train": 160, "validation": 20, "test": 20}
    observed = {split: len(split_anatomies[split]) for split in SPLITS}
    if require_exact and observed != expected:
        raise ValueError(f"production anatomy counts are {observed}, expected {expected}")
    return {
        "schema": "pvi-gcnm-full-hp-lp-dataset-validation-v2",
        "source_schema": schema,
        "status": "pass",
        "root": str(root.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "exact_nonlinear_gate_required": require_exact,
        "anatomy_counts": observed,
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-unverified-exact", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.root, require_exact=not args.allow_unverified_exact)
    output = args.output or args.root / "validation.json"
    if output.exists():
        raise FileExistsError(f"immutable validation output already exists: {output}")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
