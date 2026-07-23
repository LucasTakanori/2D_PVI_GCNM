"""Fail-fast validation for the mask05 single-subject BP experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as pads

from gcnm_pvi.pvi_splits import SPLIT_LABELS, validate_split


EXPECTED_SHAPES = {
    "hp_s1": [1, 40, 40, 250],
    "hp_s2": [1, 40, 40, 250],
    "lp_s1": [1, 40, 40, 250],
    "lp_s2": [1, 40, 40, 250],
    "bp_waveform": [50],
    "stats": [2, 5],
}
METADATA_COLUMNS = [
    "sample_id",
    "subject",
    "session",
    "source_name",
    "source_order",
    "mask_start",
    "mask_stop",
    "num_periods",
]


def _included_subjects(registry: dict) -> set[str]:
    return {
        str(record["subject"])
        for record in registry["records"]
        if not record.get("exclusion_reason")
    }


def _check_fixed_float_list(schema: pa.Schema, name: str, width: int) -> None:
    field_type = schema.field(name).type
    if not pa.types.is_fixed_size_list(field_type):
        raise ValueError(f"{name} must be a fixed-size list, got {field_type}")
    if field_type.list_size != width or not pa.types.is_float32(field_type.value_type):
        raise ValueError(
            f"{name} must be fixed_size_list<float32>[{width}], got {field_type}"
        )


def _read_root(root: Path, expected_family: str) -> tuple[dict, list[dict]]:
    root = Path(root)
    if (root / "_INCOMPLETE").exists():
        raise ValueError(f"representation root is incomplete: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"representation manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("representation_family") != expected_family:
        raise ValueError(
            f"{root} family is {manifest.get('representation_family')!r}; "
            f"expected {expected_family!r}"
        )
    if manifest.get("mask_key") != "mask05":
        raise ValueError(f"{root} was not exported with mask05")
    if manifest.get("mesh_variant") != "b045":
        raise ValueError(f"{root} mesh variant is not b045")
    if manifest.get("stage_columns") != {
        "hp_s1": "HP stage 1",
        "hp_s2": "HP stage 2",
        "lp_s1": "LP stage 1",
        "lp_s2": "LP stage 2",
    }:
        raise ValueError(f"{root} has invalid literal stage columns")
    source_signal = manifest.get("source_signal", {})
    if source_signal.get("hp_resistance") != "data/pviHP/resistance":
        raise ValueError(f"{root} has an invalid HP source")
    if source_signal.get("lp_resistance") != "data/pviLP/resistance":
        raise ValueError(f"{root} has an invalid LP source")
    if source_signal.get("reference_rule") != (
        "subtract first frame independently within each mask05 window"
    ):
        raise ValueError(f"{root} has an invalid component reference rule")
    if manifest.get("tensor_shapes") != EXPECTED_SHAPES:
        raise ValueError(f"{root} has unexpected tensor shapes")

    shards = root / "shards"
    if not shards.is_dir() or not any(shards.glob("*.parquet")):
        raise FileNotFoundError(f"no Parquet shards found under {shards}")
    dataset = pads.dataset(str(shards), format="parquet")
    stage_fields = ["hp_s1", "hp_s2", "lp_s1", "lp_s2"]
    for column in METADATA_COLUMNS + stage_fields + ["bp_waveform", "stats"]:
        if column not in dataset.schema.names:
            raise ValueError(f"{root} is missing Parquet column {column}")
    for field in stage_fields:
        _check_fixed_float_list(dataset.schema, field, 40 * 40 * 250)
    _check_fixed_float_list(dataset.schema, "bp_waveform", 50)
    _check_fixed_float_list(dataset.schema, "stats", 10)
    rows = dataset.to_table(columns=METADATA_COLUMNS).to_pylist()
    if len(rows) != int(manifest.get("row_count", -1)):
        raise ValueError(
            f"{root} has {len(rows)} Parquet rows but manifest records "
            f"{manifest.get('row_count')}"
        )
    return manifest, rows


def _metadata_by_id(rows: list[dict], root: Path) -> dict[str, tuple]:
    output = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in output:
            raise ValueError(f"duplicate sample_id in {root}: {sample_id}")
        output[sample_id] = tuple(row[column] for column in METADATA_COLUMNS[1:])
    return output


def preflight(
    coordinate_root: Path,
    global_voltage_slots_root: Path,
    split_manifest: Path,
    registry_path: Path,
    expected_subjects: int = 91,
    subjects: set[str] | None = None,
) -> dict:
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    expected = _included_subjects(registry)
    if subjects is not None:
        requested = {str(subject).lower() for subject in subjects}
        missing = sorted(requested - expected)
        if missing:
            raise ValueError(f"requested subjects absent from registry: {missing}")
        expected = requested
    if len(expected) != expected_subjects:
        raise ValueError(
            f"registry contains {len(expected)} included subjects; expected {expected_subjects}"
        )

    coordinate_manifest, coordinate_rows = _read_root(coordinate_root, "coordinate")
    vessel_manifest, vessel_rows = _read_root(
        global_voltage_slots_root, "global_voltage_slots"
    )
    coordinate_metadata = _metadata_by_id(coordinate_rows, coordinate_root)
    vessel_metadata = _metadata_by_id(vessel_rows, global_voltage_slots_root)
    if coordinate_metadata != vessel_metadata:
        coordinate_ids, vessel_ids = set(coordinate_metadata), set(vessel_metadata)
        raise ValueError(
            "coordinate/global-voltage-slot sample metadata differ "
            f"(coordinate_only={len(coordinate_ids - vessel_ids)}, "
            f"global_voltage_slots_only={len(vessel_ids - coordinate_ids)})"
        )

    observed = {str(row["subject"]) for row in coordinate_rows}
    if observed != expected:
        raise ValueError(
            "Parquet/registry subject sets differ "
            f"(registry_only={sorted(expected - observed)}, "
            f"parquet_only={sorted(observed - expected)})"
        )

    split_path = Path(split_manifest)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("mask_key") != "mask05":
        raise ValueError("split manifest is not mask05")
    assignments = split.get("assignments", {})
    sample_ids = set(coordinate_metadata)
    if set(assignments) != sample_ids:
        raise ValueError(
            "split/Parquet sample IDs differ "
            f"(split_only={len(set(assignments) - sample_ids)}, "
            f"parquet_only={len(sample_ids - set(assignments))})"
        )
    invalid = set(assignments.values()) - SPLIT_LABELS
    if invalid:
        raise ValueError(f"split manifest has invalid labels: {sorted(invalid)}")
    validate_split(coordinate_rows, assignments)

    counts_by_subject = {}
    for subject in sorted(expected):
        subject_ids = {
            str(row["sample_id"])
            for row in coordinate_rows
            if str(row["subject"]) == subject
        }
        counts = {
            label: sum(assignments[sample_id] == label for sample_id in subject_ids)
            for label in sorted(SPLIT_LABELS)
        }
        if not counts["train"] or not counts["test"]:
            raise ValueError(f"{subject} does not have non-empty train and test partitions")
        counts_by_subject[subject] = counts

    return {
        "status": "pass",
        "mask_key": "mask05",
        "subject_count": len(observed),
        "sample_count": len(sample_ids),
        "coordinate_rows": len(coordinate_rows),
        "global_voltage_slots_rows": len(vessel_rows),
        "coordinate_manifest": str((Path(coordinate_root) / "manifest.json").resolve()),
        "global_voltage_slots_manifest": str(
            (Path(global_voltage_slots_root) / "manifest.json").resolve()
        ),
        "split_manifest": str(split_path.resolve()),
        "split_counts": {
            label: sum(value == label for value in assignments.values())
            for label in sorted(SPLIT_LABELS)
        },
        "subjects": counts_by_subject,
        "representation_rows_recorded": {
            "coordinate": int(coordinate_manifest["row_count"]),
            "global_voltage_slots": int(vessel_manifest["row_count"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinate-root", type=Path, required=True)
    parser.add_argument("--global-voltage-slots-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--expected-subjects", type=int, default=91)
    parser.add_argument("--subject", action="append", default=[])
    args = parser.parse_args()
    report = preflight(
        args.coordinate_root,
        args.global_voltage_slots_root,
        args.split_manifest,
        args.registry,
        args.expected_subjects,
        set(args.subject) if args.subject else None,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
