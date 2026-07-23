#!/usr/bin/env python3
"""Manifests and finalization checks for the 91-subject coordinate rollout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.dataset as pads

from gcnm_pvi.newton_coordinate_hdf5_dataset import (
    validate_coordinate_hdf5_contract,
)
from gcnm_pvi.pvi_splits import validate_split


RINGS = (
    "US060", "US065", "US070", "US075", "US080", "US085", "US090",
    "US095", "US100", "US105", "US110", "US115", "US120", "US125",
    "US130",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _registry(path: Path) -> tuple[dict, list[dict], list[str]]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    records = [row for row in registry["records"] if not row.get("exclusion_reason")]
    records.sort(key=lambda row: int(row["source_order"]))
    subjects = sorted({str(row["subject"]) for row in records})
    rings = sorted({str(row["ring"]) for row in records})
    if len(records) != 216 or len(subjects) != 91 or tuple(rings) != RINGS:
        raise ValueError(
            f"registry contract differs: sessions={len(records)}, "
            f"subjects={len(subjects)}, rings={rings}"
        )
    for row in records:
        for key in ("source_hdf5", "mesh_forward", "mesh_inverse", "mapping_40", "config"):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(f"{row['source_name']} lacks {key}: {row[key]}")
    return registry, records, subjects


def build_manifests(registry_path: Path, output_root: Path) -> dict:
    registry, records, subjects = _registry(registry_path)
    output_root.mkdir(parents=True, exist_ok=True)
    rings = []
    for ring in RINGS:
        ring_records = [row for row in records if row["ring"] == ring]
        rings.append(
            {
                "ring": ring,
                "subjects": sorted({row["subject"] for row in ring_records}),
                "sessions": len(ring_records),
                "config": ring_records[0]["config"],
                "synthetic_source": (
                    "data/hp_lp_beats_US120_v1"
                    if ring == "US120"
                    else f"data/hp_lp_beats_main_b045_v1/{ring}"
                ),
                "differential_pack": (
                    "data/differential_US120_1000beats_clean_v1"
                    if ring == "US120"
                    else f"data/differential_main_b045_1000beats_clean_v1/{ring}"
                ),
                "checkpoint_dir": (
                    "models/differential_US120_1000beats_v1/coordinate_direct"
                    if ring == "US120"
                    else f"models/differential_main_b045_1000beats_v1/{ring}/coordinate_direct"
                ),
                "model_name": "coordinate_direct",
            }
        )
    ring_manifest = {
        "schema": "pvi-gcnm-coordinate-main-b045-rings-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "registry": str(registry_path.resolve()),
        "registry_sha256": _sha256(registry_path),
        "subject_count": len(subjects),
        "session_count": len(records),
        "ring_count": len(rings),
        "synthetic_contract": {
            "virtual_anatomies": 200,
            "retained_beats": 1000,
            "samples_per_beat": 50,
            "frames": 50000,
            "train_validation_test_anatomies": [160, 20, 20],
            "train_mode": "linearized",
            "validation_test_mode": "nonlinear",
            "training_voltage": "clean differential voltage from first retained frame",
        },
        "rings": rings,
    }
    ring_path = output_root / "coordinate_main_b045_rings_v1.json"
    ring_path.write_text(json.dumps(ring_manifest, indent=2) + "\n", encoding="utf-8")
    subject_path = output_root / "coordinate_main_b045_subjects_v1.txt"
    subject_path.write_text("\n".join(subjects) + "\n", encoding="utf-8")

    rows = []
    contracts = (
        ("coordinate", "3ch", "waveform"),
        ("coordinate", "3ch", "fiducials"),
        ("newton_coordinate", "6ch", "waveform"),
        ("newton_coordinate", "6ch", "fiducials"),
    )
    coordinate_root = "gcnm_parquet/coordinate_direct_main_b045_v1"
    reference_registry = "data/registries/main_b045_v1.json"
    for family, channels, output_mode in contracts:
        for subject in subjects:
            target = f"gcnm-{family}-{channels}-crt-image-to-{output_mode}"
            rows.append(
                {
                    "task_id": len(rows),
                    "subject": subject,
                    "family": family,
                    "architecture": "crt",
                    "output_mode": output_mode,
                    "channel_mode": channels,
                    "coordinate_root": coordinate_root,
                    "reference_registry": reference_registry,
                    "target": target,
                    "mask_key": "mask05",
                    "seed": 0,
                }
            )
    if len(rows) != 364:
        raise RuntimeError(f"expected 364 CRT runs, built {len(rows)}")
    tsv_path = output_root / "coordinate_main_b045_bp_364_v1.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    matrix = {
        "schema": "pvi-gcnm-coordinate-main-b045-bp-matrix-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject_count": 91,
        "run_count": 364,
        "architecture": "crt",
        "mask_key": "mask05",
        "seed": 0,
        "contracts": [
            {"family": family, "channel_mode": channels, "output_mode": output_mode, "runs": 91}
            for family, channels, output_mode in contracts
        ],
        "runs": rows,
    }
    json_path = output_root / "coordinate_main_b045_bp_364_v1.json"
    json_path.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    return {
        "rings": str(ring_path),
        "subjects": str(subject_path),
        "bp_tsv": str(tsv_path),
        "bp_json": str(json_path),
    }


def merge_coordinate_parts(root: Path, registry_path: Path) -> dict:
    registry, records, subjects = _registry(registry_path)
    incomplete = root / "_INCOMPLETE"
    if not incomplete.is_file():
        raise FileNotFoundError(f"rollout marker is missing: {incomplete}")
    if (root / "manifest.json").exists():
        raise FileExistsError(f"immutable final manifest already exists: {root / 'manifest.json'}")
    shard_root = root / "shards"
    shard_root.mkdir(parents=True, exist_ok=False)
    manifests = []
    for ring in RINGS:
        part = root / "ring_parts" / ring
        manifest_path = part / "manifest.json"
        if (part / "_INCOMPLETE").exists() or not manifest_path.is_file():
            raise RuntimeError(f"ring export is incomplete: {part}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests.append((ring, part, manifest))
        for index, source in enumerate(sorted((part / "shards").glob("*.parquet"))):
            destination = shard_root / f"{ring}-part-{index:05d}.parquet"
            os.link(source, destination)
    sessions = sorted(
        [row for _ring, _part, manifest in manifests for row in manifest["source_sessions"]],
        key=lambda row: int(next(item["source_order"] for item in records if item["source_name"] == row["source_name"])),
    )
    if len(sessions) != 216 or {row["subject"] for row in sessions} != set(subjects):
        raise RuntimeError("merged coordinate sessions do not match the 91-subject registry")
    source_names = [row["source_name"] for row in sessions]
    if len(source_names) != len(set(source_names)):
        raise RuntimeError("coordinate ring parts contain duplicate source sessions")
    checkpoint_hashes = {}
    for _ring, _part, manifest in manifests:
        checkpoint_hashes.update(manifest.get("checkpoint_hashes", {}))
    template = manifests[0][2]
    final = {
        **{key: template[key] for key in (
            "schema", "version", "representation_family", "mesh_variant", "mask_key",
            "stored_fields", "bp_channel_contract", "derivative", "source_signal", "tensor_shapes",
        )},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": sum(int(manifest["row_count"]) for _ring, _part, manifest in manifests),
        "session_count": len(sessions),
        "subject_count": len(subjects),
        "source_sessions": sessions,
        "parquet_shards": [str(path.relative_to(root)) for path in sorted(shard_root.glob("*.parquet"))],
        "checkpoint_hashes": checkpoint_hashes,
        "registry": str(registry_path.resolve()),
        "registry_sha256": _sha256(registry_path),
        "workbook_sha256": registry.get("workbook_sha256"),
        "ring_parts": {ring: str(part.resolve()) for ring, part, _manifest in manifests},
        "merge": "hard links; tensors are stored once and ring manifests remain auditable",
        "excluded_sessions": [],
    }
    (root / "manifest.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    incomplete.unlink()
    return {"rows": final["row_count"], "sessions": len(sessions), "subjects": len(subjects)}


def merge_reference_parts(root: Path, registry_path: Path) -> dict:
    _registry_payload, records, subjects = _registry(registry_path)
    incomplete = root / "_INCOMPLETE"
    if not incomplete.is_file():
        raise FileNotFoundError(f"rollout marker is missing: {incomplete}")
    if (root / "manifest.json").exists():
        raise FileExistsError(f"immutable final manifest already exists: {root / 'manifest.json'}")
    shard_root = root / "shards"
    shard_root.mkdir(parents=True, exist_ok=False)
    manifests = []
    for subject in subjects:
        part = root / "subject_parts" / subject
        manifest_path = part / "manifest.json"
        if (part / "_INCOMPLETE").exists() or not manifest_path.is_file():
            raise RuntimeError(f"reference subject export is incomplete: {part}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("subjects") != [subject]:
            raise RuntimeError(f"reference part has wrong subject identity: {part}")
        manifests.append((subject, part, manifest))
        for index, source in enumerate(sorted((part / "shards").glob("*.parquet"))):
            os.link(source, shard_root / f"{subject}-part-{index:05d}.parquet")
    sessions = sorted(
        [row for _subject, _part, manifest in manifests for row in manifest["source_sessions"]],
        key=lambda row: int(row["source_order"]),
    )
    if len(sessions) != 216 or {row["subject"] for row in sessions} != set(subjects):
        raise RuntimeError("merged reference sessions do not match the registry")
    if len({row["source_name"] for row in sessions}) != len(sessions):
        raise RuntimeError("reference ring parts contain duplicate sessions")
    template = manifests[0][2]
    final = {
        **{key: template[key] for key in (
            "schema", "version", "representation", "input_mode", "bioz_channel_order",
            "mask_key", "period_length", "tensor_shapes", "shard_rows",
        )},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": sum(int(manifest["row_count"]) for _subject, _part, manifest in manifests),
        "session_count": len(sessions),
        "subjects": subjects,
        "parquet_shards": [str(path.relative_to(root)) for path in sorted(shard_root.glob("*.parquet"))],
        "registry": str(registry_path.resolve()),
        "registry_sha256": _sha256(registry_path),
        "source_sessions": sessions,
        "subject_parts": {
            subject: str(part.resolve()) for subject, part, _manifest in manifests
        },
        "merge": "hard links; tensors are stored once and subject manifests remain auditable",
    }
    (root / "manifest.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    incomplete.unlink()
    return {"rows": final["row_count"], "sessions": len(sessions), "subjects": len(subjects)}


def preflight(coordinate_root: Path, reference_root: Path, split_path: Path) -> dict:
    identity_columns = [
        "sample_id", "subject", "session", "source_name", "source_order",
        "mask_start", "mask_stop", "num_periods",
    ]
    tables = []
    for root in (coordinate_root, reference_root):
        if (root / "_INCOMPLETE").exists():
            raise RuntimeError(f"incomplete Parquet root: {root}")
        tables.append(pads.dataset(str(root / "shards"), format="parquet").to_table(
            columns=[*identity_columns, "bp_waveform", "stats"]
        ).sort_by([
            ("source_order", "ascending"), ("mask_start", "ascending"), ("mask_stop", "ascending")
        ]))
    if not tables[0]["bp_waveform"].equals(tables[1]["bp_waveform"]):
        raise ValueError("coordinate and reference BP waveforms differ")
    if not tables[0]["stats"].equals(tables[1]["stats"]):
        raise ValueError("coordinate and reference stats differ")
    left, right = (table.select(identity_columns).to_pylist() for table in tables)
    # The first seven fields form the stable sample identity. Reference v1's
    # legacy num_periods field stores window length; it is not model-facing.
    identity = lambda row: tuple(row[key] for key in identity_columns[:-1])
    if [identity(row) for row in left] != [identity(row) for row in right]:
        raise ValueError("coordinate and reference Parquet sample identities/order differ")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    sample_ids = {row["sample_id"] for row in left}
    if set(split["assignments"]) != sample_ids:
        raise ValueError("frozen split and exported Parquet sample IDs differ")
    validate_split(left, split["assignments"])
    subjects = sorted({row["subject"] for row in left})
    if len(subjects) != 91:
        raise ValueError(f"expected 91 subjects, found {len(subjects)}")
    return {"status": "pass", "rows": len(left), "subjects": len(subjects)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-manifests")
    build.add_argument("--registry", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    merge = sub.add_parser("merge-coordinate")
    merge.add_argument("--root", type=Path, required=True)
    merge.add_argument("--registry", type=Path, required=True)
    merge_reference = sub.add_parser("merge-reference")
    merge_reference.add_argument("--root", type=Path, required=True)
    merge_reference.add_argument("--registry", type=Path, required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--coordinate-root", type=Path, required=True)
    check.add_argument("--reference-root", type=Path, required=True)
    check.add_argument("--split-manifest", type=Path, required=True)
    check_hdf5 = sub.add_parser("preflight-hdf5")
    check_hdf5.add_argument("--coordinate-root", type=Path, required=True)
    check_hdf5.add_argument("--registry", type=Path, required=True)
    check_hdf5.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build-manifests":
        report = build_manifests(args.registry, args.output_root)
    elif args.command == "merge-coordinate":
        report = merge_coordinate_parts(args.root, args.registry)
    elif args.command == "merge-reference":
        report = merge_reference_parts(args.root, args.registry)
    elif args.command == "preflight":
        report = preflight(args.coordinate_root, args.reference_root, args.split_manifest)
    else:
        report = validate_coordinate_hdf5_contract(
            args.coordinate_root, args.registry, args.split_manifest
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
