#!/usr/bin/env python3
"""Build the scratch-resident model-ready six-channel PW Parquet cache.

Each row stores four base image streams, ``[Newton HP, GCNM S1]`` and
``[Newton LP, GCNM S2]``.  The training preprocessor derives the final six
channels as ``[Newton HP, d(Newton LP), d2(Newton LP), S1, S2, d(S2)]``.
Only rows selected by the frozen population-within split are materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PERIOD_LENGTH = 50
WINDOW_PERIODS = 5
IMAGE_SIDE = 40
BASE_CHANNELS = 2
IMAGE_WIDTH = BASE_CHANNELS * IMAGE_SIDE * IMAGE_SIDE * PERIOD_LENGTH * WINDOW_PERIODS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_ids_from_cache(root: Path) -> set[str]:
    """Project stable IDs from an existing Parquet payload without image I/O."""

    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if not (root / "_SUCCESS").is_file() or not manifest_path.is_file():
        raise RuntimeError(f"skip-source cache is incomplete: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_ids: set[str] = set()
    for relative_paths in manifest.get("files", {}).values():
        if not isinstance(relative_paths, list):
            continue
        for relative in relative_paths:
            path = root / relative
            for sample_id in pq.read_table(
                path,
                columns=["sample_id"],
                use_threads=False,
            )["sample_id"].to_pylist():
                if sample_id in sample_ids:
                    raise ValueError(f"duplicate sample in skip-source cache: {sample_id}")
                sample_ids.add(str(sample_id))
    return sample_ids


def schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("pviHP", pa.list_(pa.float32(), IMAGE_WIDTH)),
            pa.field("pviLP", pa.list_(pa.float32(), IMAGE_WIDTH)),
            pa.field("bp", pa.list_(pa.float32(), PERIOD_LENGTH)),
            pa.field("stats", pa.list_(pa.float32(), 2 * WINDOW_PERIODS)),
            pa.field("sample_id", pa.string()),
            pa.field("subject", pa.string()),
            pa.field("session", pa.string()),
            pa.field("source_name", pa.string()),
            pa.field("mask_start", pa.int32()),
            pa.field("mask_stop", pa.int32()),
        ]
    )


def fixed_list(values: np.ndarray, width: int) -> pa.FixedSizeListArray:
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), width)


def column_array(table: pa.Table, name: str, shape: tuple[int, ...]) -> np.ndarray:
    values = table[name].combine_chunks().values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape((table.num_rows, *shape))


def row_table(rows: list[dict]) -> pa.Table:
    hp = np.stack([row["pviHP"] for row in rows])
    lp = np.stack([row["pviLP"] for row in rows])
    bp = np.stack([row["bp"] for row in rows])
    stats = np.stack([row["stats"] for row in rows])
    return pa.Table.from_arrays(
        [
            fixed_list(hp, IMAGE_WIDTH),
            fixed_list(lp, IMAGE_WIDTH),
            fixed_list(bp, PERIOD_LENGTH),
            fixed_list(stats, 2 * WINDOW_PERIODS),
            pa.array([row["sample_id"] for row in rows]),
            pa.array([row["subject"] for row in rows]),
            pa.array([row["session"] for row in rows]),
            pa.array([row["source_name"] for row in rows]),
            pa.array([row["mask_start"] for row in rows], type=pa.int32()),
            pa.array([row["mask_stop"] for row in rows], type=pa.int32()),
        ],
        schema=schema(),
    )


def process_shard(payload: dict) -> dict:
    coordinate_path = Path(payload["coordinate_path"])
    output_root = Path(payload["output_root"])
    assignments = payload["assignments"]
    sources = payload["sources"]
    row_group_size = int(payload["row_group_size"])
    relative = coordinate_path.name.replace(".parquet", "")
    final_paths = {
        split: output_root / split / f"{relative}-{split}.parquet"
        for split in ("train", "test")
    }
    report_path = output_root / "reports" / f"{relative}.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if all(
            report["counts"][split] == 0 or final_paths[split].is_file()
            for split in ("train", "test")
        ):
            return report

    output_root.joinpath("reports").mkdir(parents=True, exist_ok=True)
    for path in final_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    partial_paths = {split: path.with_suffix(".partial.parquet") for split, path in final_paths.items()}
    for path in partial_paths.values():
        if path.exists():
            path.unlink()

    writers: dict[str, pq.ParquetWriter | None] = {"train": None, "test": None}
    buffers: dict[str, list[dict]] = {"train": [], "test": []}
    counts = Counter()
    current_source = None
    current_hdf = None

    def flush(split: str, *, force: bool = False) -> None:
        while len(buffers[split]) >= row_group_size or (force and buffers[split]):
            count = row_group_size if len(buffers[split]) >= row_group_size else len(buffers[split])
            selected = buffers[split][:count]
            del buffers[split][:count]
            table = row_table(selected)
            if writers[split] is None:
                writers[split] = pq.ParquetWriter(
                    partial_paths[split], schema(), compression="zstd", compression_level=1
                )
            writers[split].write_table(table, row_group_size=count)

    parquet = pq.ParquetFile(coordinate_path)
    try:
        for group_index in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(
                group_index,
                columns=[
                    "s1", "s2", "bp_waveform", "stats", "sample_id", "subject",
                    "session", "source_name", "mask_start", "mask_stop",
                ],
            )
            s1 = column_array(table, "s1", (1, IMAGE_SIDE, IMAGE_SIDE, 250))
            s2 = column_array(table, "s2", (1, IMAGE_SIDE, IMAGE_SIDE, 250))
            bp = column_array(table, "bp_waveform", (PERIOD_LENGTH,))
            stats = column_array(table, "stats", (2, WINDOW_PERIODS))
            metadata = table.select(
                ["sample_id", "subject", "session", "source_name", "mask_start", "mask_stop"]
            ).to_pylist()
            for index, meta in enumerate(metadata):
                sample_id = str(meta["sample_id"])
                split = assignments.get(sample_id)
                if split == "excluded":
                    counts["excluded"] += 1
                    continue
                if split not in {"train", "test"}:
                    raise RuntimeError(f"missing/invalid split for {sample_id}")
                source_name = str(meta["source_name"])
                if source_name != current_source:
                    if current_hdf is not None:
                        current_hdf.close()
                    current_hdf = h5py.File(sources[source_name], "r")
                    current_source = source_name
                start = int(meta["mask_start"])
                stop = int(meta["mask_stop"])
                if stop - start != WINDOW_PERIODS:
                    raise ValueError(f"non-mask05 coordinate row: {sample_id}")
                frame_start, frame_stop = start * PERIOD_LENGTH, stop * PERIOD_LENGTH
                newton_hp = np.asarray(
                    current_hdf["data/pviHP/img"][:, :, frame_start:frame_stop],
                    dtype=np.float32,
                )[None]
                newton_lp = np.asarray(
                    current_hdf["data/pviLP/img"][:, :, frame_start:frame_stop],
                    dtype=np.float32,
                )[None]
                if newton_hp.shape != (1, IMAGE_SIDE, IMAGE_SIDE, 250):
                    raise ValueError(f"invalid Newton window for {sample_id}: {newton_hp.shape}")
                buffers[split].append(
                    {
                        "pviHP": np.concatenate((newton_hp, s1[index]), axis=0),
                        "pviLP": np.concatenate((newton_lp, s2[index]), axis=0),
                        "bp": bp[index],
                        "stats": stats[index],
                        **meta,
                    }
                )
                counts[split] += 1
                flush(split)
    finally:
        if current_hdf is not None:
            current_hdf.close()
        for split in ("train", "test"):
            flush(split, force=True)
            if writers[split] is not None:
                writers[split].close()

    for split in ("train", "test"):
        if counts[split]:
            partial_paths[split].replace(final_paths[split])
        elif partial_paths[split].exists():
            partial_paths[split].unlink()
    report = {
        "coordinate_shard": str(coordinate_path.resolve()),
        "counts": {key: int(counts[key]) for key in ("train", "test", "excluded")},
        "outputs": {
            split: str(final_paths[split].resolve()) if counts[split] else None
            for split in ("train", "test")
        },
    }
    temporary_report = report_path.with_suffix(".partial.json")
    temporary_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_report.replace(report_path)
    return report


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinate-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--row-group-size", type=int, default=64)
    parser.add_argument(
        "--skip-sample-ids-from-cache-root",
        type=Path,
        help="materialize only split rows absent from this validated Parquet cache",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    marker = args.output_root / "_INCOMPLETE"
    if args.resume:
        if not marker.is_file() or (args.output_root / "manifest.json").exists():
            raise RuntimeError("resume requires an incomplete cache without a final manifest")
    else:
        if args.output_root.exists():
            raise FileExistsError(f"immutable cache root exists: {args.output_root}")
        args.output_root.mkdir(parents=True)
        marker.write_text("population six-channel cache build in progress\n", encoding="utf-8")

    coordinate_manifest = json.loads(
        (args.coordinate_root / "manifest.json").read_text(encoding="utf-8")
    )
    if coordinate_manifest.get("schema") != "pvi-gcnm-coordinate-direct-parquet-v1":
        raise ValueError("coordinate root has the wrong schema")
    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if split_manifest.get("schema") not in {
        "pvi-gcnm-population-within-split-v1",
        "pvi-gcnm-population-disjoint-split-v1",
    }:
        raise ValueError("split manifest has the wrong schema")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    records = [row for row in registry["records"] if not row.get("exclusion_reason")]
    sources = {str(row["source_name"]): str(Path(row["source_hdf5"]).resolve()) for row in records}
    if len(sources) != len(records):
        raise ValueError("registry source names are not unique")
    for path in sources.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    coordinate_shards = sorted((args.coordinate_root / "shards").glob("*.parquet"))
    if not coordinate_shards:
        raise FileNotFoundError("coordinate root contains no Parquet shards")
    payloads = []
    skip_ids = (
        sample_ids_from_cache(args.skip_sample_ids_from_cache_root)
        if args.skip_sample_ids_from_cache_root is not None
        else set()
    )
    source_assignments = split_manifest["assignments"]
    unknown_skip_ids = skip_ids - set(source_assignments)
    if unknown_skip_ids:
        raise ValueError(
            f"skip-source cache contains {len(unknown_skip_ids)} identities absent from split"
        )
    all_assignments = {
        sample_id: "excluded" if sample_id in skip_ids else assignment
        for sample_id, assignment in source_assignments.items()
    }
    for path in coordinate_shards:
        metadata = pq.read_table(path, columns=["sample_id", "source_name"])
        sample_ids = metadata["sample_id"].to_pylist()
        source_names = set(metadata["source_name"].to_pylist())
        payloads.append(
            {
                "coordinate_path": str(path),
                "output_root": str(args.output_root),
                # Send only this shard's identities to its worker instead of
                # pickling the complete 162k-row manifest for every task.
                "assignments": {sample_id: all_assignments[sample_id] for sample_id in sample_ids},
                "sources": {name: sources[name] for name in source_names},
                "row_group_size": args.row_group_size,
            }
        )
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(payloads)), mp_context=context
    ) as executor:
        reports = list(executor.map(process_shard, payloads))

    totals = Counter()
    for report in reports:
        totals.update(report["counts"])
    expected_counter = Counter(all_assignments.values())
    expected = {
        key: int(expected_counter[key]) for key in ("train", "test", "excluded")
    }
    actual = {key: int(totals[key]) for key in ("train", "test", "excluded")}
    if actual != expected:
        raise RuntimeError(f"cache split counts differ: actual={actual}, expected={expected}")

    files = {
        split: sorted(str(path.relative_to(args.output_root)) for path in (args.output_root / split).glob("*.parquet"))
        for split in ("train", "test")
    }
    manifest = {
        "schema": "pvi-gcnm-population-6ch-parquet-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "representation": "Newton plus coordinate-direct GCNM",
        "channel_contract": [
            "newton_hp", "d_newton_lp_dt", "d2_newton_lp_dt2", "s1", "s2", "d_s2_dt"
        ],
        "stored_base_fields": {"pviHP": ["newton_hp", "s1"], "pviLP": ["newton_lp", "s2"]},
        "tensor_shapes": {
            "pviHP": [2, 40, 40, 250],
            "pviLP": [2, 40, 40, 250],
            "bp": [50],
            "stats": [2, 5],
        },
        "mask_key": "mask05",
        "split_mode": (
            "payload-completion"
            if args.skip_sample_ids_from_cache_root is not None
            else str(split_manifest.get("split_mode", "within"))
        ),
        "seed": int(split_manifest["seed"]),
        "counts": actual,
        "row_group_size": args.row_group_size,
        "files": files,
        "coordinate_root": str(args.coordinate_root.resolve()),
        "coordinate_manifest_sha256": sha256_file(args.coordinate_root / "manifest.json"),
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry),
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "payload_completion": (
            {
                "skip_source_root": str(args.skip_sample_ids_from_cache_root.resolve()),
                "skip_source_manifest_sha256": sha256_file(
                    args.skip_sample_ids_from_cache_root / "manifest.json"
                ),
                "skipped_existing_rows": len(skip_ids),
                "materialized_rows": actual["train"] + actual["test"],
            }
            if args.skip_sample_ids_from_cache_root is not None
            else None
        ),
        "build": {"workers": min(args.workers, len(payloads)), "source_shards": len(payloads)},
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    marker.unlink()
    (args.output_root / "_SUCCESS").write_text("validated\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "counts": actual, "files": {k: len(v) for k, v in files.items()}}, indent=2))


if __name__ == "__main__":
    main()
