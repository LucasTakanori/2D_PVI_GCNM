"""Validate Parquet metadata/targets against immutable source HDF5 sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

from gcnm_pvi.mesh_registry import sha256_file
from gcnm_pvi.pvi_splits import stable_sample_id


VALIDATION_COLUMNS = (
    "hp_s1",
    "hp_s2",
    "lp_s1",
    "lp_s2",
    "bp_waveform",
    "stats",
    "sample_id",
    "source_name",
    "mask_start",
    "mask_stop",
)


def _parquet_inventory(shards: Path) -> tuple[list[tuple[Path, int]], int]:
    """Return deterministic shard row counts without reading column payloads."""
    paths = sorted(shards.glob("*.parquet"))
    if not paths:
        raise ValueError(f"no Parquet shards found under {shards}")
    inventory = []
    total = 0
    for path in paths:
        rows = int(pq.ParquetFile(path).metadata.num_rows)
        inventory.append((path, rows))
        total += rows
    return inventory, total


def _sample_rows(
    inventory: list[tuple[Path, int]], total_rows: int, max_rows: int
):
    """Yield evenly spaced rows, reading only their containing row groups."""
    if max_rows < 0:
        raise ValueError("max_rows must be non-negative")
    checked = min(total_rows, max_rows)
    if checked == 0:
        return
    targets = np.linspace(0, total_rows - 1, checked, dtype=np.int64)
    target_index = 0
    global_offset = 0
    for path, shard_rows in inventory:
        shard_stop = global_offset + shard_rows
        local_targets = []
        while target_index < len(targets) and int(targets[target_index]) < shard_stop:
            local_targets.append(int(targets[target_index]) - global_offset)
            target_index += 1
        global_offset = shard_stop
        if not local_targets:
            continue

        parquet = pq.ParquetFile(path)
        row_group_start = 0
        local_index = 0
        for row_group_index in range(parquet.metadata.num_row_groups):
            row_group_rows = int(parquet.metadata.row_group(row_group_index).num_rows)
            row_group_stop = row_group_start + row_group_rows
            selected = []
            while (
                local_index < len(local_targets)
                and local_targets[local_index] < row_group_stop
            ):
                selected.append(local_targets[local_index] - row_group_start)
                local_index += 1
            if selected:
                table = parquet.read_row_group(
                    row_group_index, columns=list(VALIDATION_COLUMNS)
                )
                for offset in selected:
                    yield table.slice(offset, 1)
            row_group_start = row_group_stop
            if local_index == len(local_targets):
                break


def validate(root: Path, max_rows: int = 32) -> dict:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    inventory, total_rows = _parquet_inventory(root / "shards")
    if total_rows != int(manifest["row_count"]):
        raise ValueError("Parquet row count differs from manifest")
    failures = []
    hashes_verified = 0
    for mapping_name in ("checkpoint_hashes", "config_hashes", "mesh_hashes"):
        for raw_path, expected_hash in manifest.get(mapping_name, {}).items():
            path = Path(raw_path)
            if not path.is_file() or sha256_file(path) != expected_hash:
                failures.append(f"{mapping_name} mismatch: {path}")
            else:
                hashes_verified += 1
    registry_path = Path(manifest.get("registry", ""))
    if manifest.get("registry_sha256"):
        if not registry_path.is_file() or sha256_file(registry_path) != manifest["registry_sha256"]:
            failures.append(f"registry hash mismatch: {registry_path}")
        else:
            hashes_verified += 1

    source_cache = {}
    expected_order = []
    for session_record in manifest["source_sessions"]:
        source_path = Path(session_record["source_hdf5"])
        source_name = source_path.stem.removesuffix("_masked")
        expected_hash = session_record.get("source_hdf5_sha256")
        if expected_hash:
            if not source_path.is_file() or sha256_file(source_path) != expected_hash:
                failures.append(f"source HDF5 hash mismatch: {source_path}")
            else:
                hashes_verified += 1
        with h5py.File(source_path, "r") as handle:
            periods = int(np.asarray(handle["metadata/num_periods"]).item())
            bp = np.asarray(handle["data/bp/signal"], dtype=np.float32).reshape(periods, 50)
            stats = np.vstack(
                (
                    np.asarray(handle["stats/pviHP/duration"]).reshape(-1),
                    np.asarray(handle["stats/pviHP/tMax"]).reshape(-1),
                )
            ).astype(np.float32)
            masks = np.asarray(handle["masks/mask05"], dtype=np.int64)
            masks[:, 0] -= 1
        source_cache[source_name] = {"bp": bp, "stats": stats}
        expected_order.extend(
            (source_name, int(start), int(stop)) for start, stop in masks
        )
    if len(expected_order) != total_rows:
        failures.append(
            f"source mask total {len(expected_order)} differs from Parquet rows {total_rows}"
        )
    checked = min(total_rows, max_rows)
    for row_index, row in enumerate(_sample_rows(inventory, total_rows, max_rows)):
        source_name = row["source_name"][0].as_py()
        start = int(row["mask_start"][0].as_py())
        stop = int(row["mask_stop"][0].as_py())
        source = source_cache.get(source_name)
        if source is None:
            failures.append(f"unknown source_name {source_name}")
            continue
        bp = source["bp"][stop - 1]
        stats = source["stats"][:, start:stop].reshape(-1)
        stored_bp = np.asarray(
            row["bp_waveform"][0].values.to_numpy(zero_copy_only=False),
            dtype=np.float32,
        )
        stored_stats = np.asarray(
            row["stats"][0].values.to_numpy(zero_copy_only=False),
            dtype=np.float32,
        )
        sample_id = str(row["sample_id"][0].as_py())
        stages = {
            name: np.asarray(
                row[name][0].values.to_numpy(zero_copy_only=False),
                dtype=np.float32,
            ).reshape(1, 40, 40, 250)
            for name in ("hp_s1", "hp_s2", "lp_s1", "lp_s2")
        }
        if not np.array_equal(bp, stored_bp):
            failures.append(f"{source_name}:{start}-{stop} BP mismatch")
        if not np.array_equal(stats, stored_stats):
            failures.append(f"{source_name}:{start}-{stop} stats mismatch")
        if sample_id != stable_sample_id(source_name, "mask05", start, stop):
            failures.append(f"{source_name}:{start}-{stop} sample_id mismatch")
        if checked == total_rows and row_index < len(expected_order):
            if (source_name, start, stop) != expected_order[row_index]:
                failures.append(
                    f"row {row_index} ordering mismatch: "
                    f"{(source_name, start, stop)} != {expected_order[row_index]}"
                )
        reference_mask = np.isfinite(stages["hp_s1"])
        for name, values in stages.items():
            if not np.array_equal(reference_mask, np.isfinite(values)):
                failures.append(
                    f"{source_name}:{start}-{stop} {name} finite-mask mismatch"
                )
            finite = values[np.isfinite(values)]
            if not len(finite) or float(np.sqrt(np.mean(finite * finite))) <= 0:
                failures.append(f"{source_name}:{start}-{stop} {name} has zero RMS")
            finite_by_pixel = np.isfinite(values[0])
            if not np.array_equal(
                np.any(finite_by_pixel, axis=2), np.all(finite_by_pixel, axis=2)
            ):
                failures.append(
                    f"{source_name}:{start}-{stop} {name} NaNs vary over time"
                )
        for component in ("hp", "lp"):
            first = stages[f"{component}_s1"]
            second = stages[f"{component}_s2"]
            finite = np.isfinite(first) & np.isfinite(second)
            if not np.any(first[finite] != second[finite]):
                failures.append(
                    f"{source_name}:{start}-{stop} {component} s1 and s2 are identical"
                )
    if failures:
        raise ValueError("; ".join(failures))
    return {
        "root": str(root.resolve()),
        "rows": total_rows,
        "rows_checked_against_hdf5": checked,
        "hashes_verified": hashes_verified,
        "full_order_checked": checked == total_rows,
        "stage_distinctness_checked": True,
        "tensor_shapes": manifest["tensor_shapes"],
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(validate(args.root, args.max_rows), indent=2))


if __name__ == "__main__":
    main()
