#!/usr/bin/env python3
"""Benchmark exact random-access versus fixed-batch Parquet on real rows."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import DataLoader

from gcnm_pvi.population_6ch_parquet import (
    PopulationSixChannelSplit,
    PrecomputedPviBatchSampler,
    RowGroupBatchSampler,
)
from scripts.build_population_6ch_exact_pvi_cache import (
    pvi_stratified_order,
    require_scratch_path,
)


BYTES_PER_SAMPLE = (2 * 2 * 40 * 40 * 250 + 50 + 10) * 4


def build_benchmark_files(source_path: Path, root: Path, rows: int) -> dict:
    paths = {
        "row_group_1": root / "row_group_1" / "benchmark.parquet",
        "row_group_32": root / "row_group_32" / "benchmark.parquet",
    }
    schedule_path = root / "schedule.npy"
    marker = root / "_SUCCESS"
    if marker.is_file() and schedule_path.is_file() and all(path.is_file() for path in paths.values()):
        schedule = np.load(schedule_path, mmap_mode="r")
        if schedule.shape == (1, rows):
            return {"paths": paths, "schedule": schedule_path}

    root.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
    if schedule_path.exists():
        schedule_path.unlink()

    source = pq.ParquetFile(source_path)
    writers = {
        name: pq.ParquetWriter(path, source.schema_arrow, compression="zstd", compression_level=1)
        for name, path in paths.items()
    }
    groups: OrderedDict[str, list[int]] = OrderedDict()
    written = 0
    try:
        for row_group in range(source.metadata.num_row_groups):
            table = source.read_row_group(row_group, use_threads=False)
            remaining = rows - written
            if remaining <= 0:
                break
            if table.num_rows > remaining:
                table = table.slice(0, remaining)
            writers["row_group_1"].write_table(table, row_group_size=1)
            writers["row_group_32"].write_table(table, row_group_size=32)
            source_names = table["source_name"].to_pylist()
            for local, source_name in enumerate(source_names):
                groups.setdefault(str(source_name), []).append(written + local)
            written += table.num_rows
    finally:
        for writer in writers.values():
            writer.close()
    if written != rows:
        raise RuntimeError(f"benchmark source has only {written} rows, requested {rows}")

    order = pvi_stratified_order(
        range(rows),
        list(groups.values()),
        cluster_size=48,
        rng=random.Random(42),
    )
    np.save(schedule_path, np.asarray(order, dtype=np.int32)[None, :])
    marker.write_text("validated\n", encoding="utf-8")
    return {"paths": paths, "schedule": schedule_path}


def benchmark_loader(name: str, loader: DataLoader, rows: int) -> dict:
    started = time.perf_counter()
    observed = 0
    for batch in loader:
        observed += int(batch["bp"].shape[0])
    elapsed = time.perf_counter() - started
    if observed != rows:
        raise RuntimeError(f"{name} loader returned {observed} rows, expected {rows}")
    gib = rows * BYTES_PER_SAMPLE / (1024**3)
    return {
        "seconds": elapsed,
        "samples_per_second": rows / elapsed,
        "uncompressed_gib_per_second": gib / elapsed,
        "projected_full_train_minutes": (103_738 / (rows / elapsed)) / 60,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    source_path = require_scratch_path(args.source_file)
    root = require_scratch_path(args.scratch_root)
    built = build_benchmark_files(source_path, root, args.rows)

    common = {
        "batch_size": 32,
        "num_workers": args.workers,
        "persistent_workers": args.workers > 0,
        "prefetch_factor": 2 if args.workers > 0 else None,
    }
    loaders = {}
    fixed = PopulationSixChannelSplit(
        root,
        "row_group_32",
        relative_files=[str(built["paths"]["row_group_32"].relative_to(root))],
    )
    loaders["fixed_row_group_32"] = DataLoader(
        fixed,
        batch_sampler=RowGroupBatchSampler(
            fixed.bounds, 32, shuffle=True, seed=42
        ),
        num_workers=common["num_workers"],
        persistent_workers=common["persistent_workers"],
        **({"prefetch_factor": common["prefetch_factor"]} if args.workers > 0 else {}),
    )
    exact = PopulationSixChannelSplit(
        root,
        "row_group_1",
        relative_files=[str(built["paths"]["row_group_1"].relative_to(root))],
    )
    loaders["exact_random_row_group_1"] = DataLoader(
        exact,
        batch_sampler=PrecomputedPviBatchSampler(
            built["schedule"], row_count=args.rows, batch_size=32
        ),
        num_workers=common["num_workers"],
        persistent_workers=common["persistent_workers"],
        **({"prefetch_factor": common["prefetch_factor"]} if args.workers > 0 else {}),
    )

    results = {
        name: benchmark_loader(name, loader, args.rows)
        for name, loader in loaders.items()
    }
    print(
        json.dumps(
            {
                "rows": args.rows,
                "workers": args.workers,
                "scratch_root": str(root),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
