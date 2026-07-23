"""Validate coordinate-direct S1/S2 Parquet and its frozen subject split."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from gcnm_pvi.export_coordinate_direct_parquet import SCHEMA
from gcnm_pvi.pvi_splits import validate_split


def validate(root: Path, split_manifest: Path | None = None) -> dict:
    root = Path(root)
    if (root / "_INCOMPLETE").exists():
        raise RuntimeError(f"incomplete export: {root}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unexpected representation schema {manifest.get('schema')}")
    if manifest.get("bp_channel_contract") != ["s1", "s2", "d_s2_dt"]:
        raise ValueError("representation does not declare [S1,S2,dS2/dt]")
    dataset = pads.dataset(str(root / "shards"), format="parquet")
    required = {
        "s1", "s2", "bp_waveform", "stats", "subject", "session",
        "source_name", "sample_id", "source_order", "mask_start", "mask_stop",
        "num_periods",
    }
    missing = sorted(required - set(dataset.schema.names))
    if missing:
        raise ValueError(f"missing Parquet columns: {missing}")
    metadata_table = dataset.to_table(
        columns=[
            "sample_id", "subject", "session", "source_name", "mask_start",
            "mask_stop",
        ]
    )
    if len(metadata_table) != int(manifest["row_count"]):
        raise ValueError("Parquet/manifest row counts differ")
    metadata = metadata_table.to_pylist()
    sample_ids = [row["sample_id"] for row in metadata]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs are not unique")
    if any(int(row["mask_stop"]) - int(row["mask_start"]) != 5 for row in metadata):
        raise ValueError("a row is not a mask05 five-period window")
    if split_manifest is not None:
        split = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
        assignments = split["assignments"]
        selected_assignments = {key: assignments[key] for key in sample_ids}
        validate_split(metadata, selected_assignments)
        counts = {
            name: sum(value == name for value in selected_assignments.values())
            for name in ("train", "test")
        }
    else:
        counts = None

    shapes = manifest["tensor_shapes"]
    expected_width = int(np.prod(shapes["s1"]))
    if dataset.schema.field("s1").type.list_size != expected_width:
        raise ValueError("S1 fixed-list width differs from manifest")
    if dataset.schema.field("s2").type.list_size != expected_width:
        raise ValueError("S2 fixed-list width differs from manifest")
    # Stream all fixed-size tensors by row group. The old implementation
    # materialized roughly 500 GB at once and left most CPUs idle. Independent
    # Zstd row groups are safe to validate concurrently and yield identical
    # finite-mask/RMS checks.
    row_groups = []
    for path in sorted((root / "shards").glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        row_groups.extend((path, index) for index in range(parquet.num_row_groups))

    def metrics(task: tuple[Path, int]) -> tuple[int, float, int, float]:
        path, row_group = task
        table = pq.ParquetFile(path).read_row_group(
            row_group, columns=["s1", "s2"], use_threads=False
        )
        s1 = table["s1"].combine_chunks().values.to_numpy(zero_copy_only=False)
        s2 = table["s2"].combine_chunks().values.to_numpy(zero_copy_only=False)
        finite_1 = np.isfinite(s1)
        finite_2 = np.isfinite(s2)
        if not np.array_equal(finite_1, finite_2):
            raise ValueError(f"S1/S2 finite masks differ in {path} row group {row_group}")
        values_1 = s1[finite_1].astype(np.float64)
        values_2 = s2[finite_2].astype(np.float64)
        return (
            len(values_1), float(np.dot(values_1, values_1)),
            len(values_2), float(np.dot(values_2, values_2)),
        )

    with ThreadPoolExecutor(max_workers=min(64, max(1, len(row_groups)))) as executor:
        parts = list(executor.map(metrics, row_groups))
    count_1 = sum(item[0] for item in parts)
    count_2 = sum(item[2] for item in parts)
    if not count_1 or not count_2:
        raise ValueError("representations contain no finite mesh pixels")
    rms = {
        "s1": float(np.sqrt(sum(item[1] for item in parts) / count_1)),
        "s2": float(np.sqrt(sum(item[3] for item in parts) / count_2)),
    }
    if min(rms.values()) <= 0 or not np.isfinite(list(rms.values())).all():
        raise ValueError("representation RMS is zero or non-finite")
    report = {
        "status": "pass",
        "schema": SCHEMA,
        "rows": len(metadata_table),
        "subjects": sorted({row["subject"] for row in metadata}),
        "sessions": sorted({row["session"] for row in metadata}),
        "split_counts": counts,
        "activation_rms": rms,
        "channel_contract": ["s1", "s2", "d_s2_dt"],
    }
    (root / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.root, args.split_manifest), indent=2))


if __name__ == "__main__":
    main()
