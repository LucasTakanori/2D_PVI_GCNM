#!/usr/bin/env python3
"""Build a scratch-resident, exact-PVI-scheduled six-channel Parquet cache.

Every processed sample is stored exactly once in a one-row Parquet row group.
All train and test permutations are generated before training with the same
two-level source-cluster shuffle used by PVI-ML's ``PviBatchSampler``.  The
schedule arrays are memory-mapped during training, so no HDF5 processing or
runtime sampling work remains in the model loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import random
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import reduce
from operator import concat
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


SOURCE_SCHEMA = "pvi-gcnm-population-6ch-parquet-v1"
OUTPUT_SCHEMA = "pvi-gcnm-population-6ch-parquet-v3"
SCHEDULE_SCHEMA = "pvi-gcnm-precomputed-pvi-batch-schedules-v1"
METADATA_COLUMNS = ["sample_id", "subject", "session", "source_name"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_scratch_path(path: Path, *, scratch_root: Path = Path("/mmfs1/scratch")) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(scratch_root.resolve())
    except ValueError as exc:
        raise ValueError(f"large cache I/O must remain under {scratch_root}: {resolved}") from exc
    return resolved


def pvi_stratified_order(
    global_indices: list[int] | range,
    grouping: list[list[int]],
    *,
    cluster_size: int,
    rng: random.Random,
) -> list[int]:
    """Literal schedule equivalent of ``PviBatchSampler._stratified_random``."""

    selected = set(global_indices)
    file_subgroups = [list(selected & set(group)) for group in grouping]
    rng.shuffle(file_subgroups)
    if not file_subgroups:
        return []

    flattened: list[int] = []
    for first in range(0, len(file_subgroups), cluster_size):
        cluster = file_subgroups[first : first + cluster_size]
        if not cluster:
            raise RuntimeError("empty PVI source cluster")
        combined = reduce(concat, cluster)
        rng.shuffle(combined)
        flattened.extend(combined)

    if len(flattened) != len(selected) or set(flattened) != selected:
        raise RuntimeError("PVI schedule is not a permutation of the requested indices")
    return flattened


def _valid_random_access_file(path: Path, expected_rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        metadata = pq.ParquetFile(path).metadata
    except Exception:
        return False
    return (
        int(metadata.num_rows) == int(expected_rows)
        and int(metadata.num_row_groups) == int(expected_rows)
    )


def repack_random_access_file(payload: dict) -> dict:
    source_path = Path(payload["source_path"])
    output_path = Path(payload["output_path"])
    expected_rows = int(payload["expected_rows"])
    report = {
        "split": payload["split"],
        "part": int(payload["part"]),
        "path": str(output_path),
        "rows": expected_rows,
    }
    if _valid_random_access_file(output_path, expected_rows):
        report["status"] = "reused"
        return report

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(".partial.parquet")
    if partial.exists():
        partial.unlink()

    source = pq.ParquetFile(source_path)
    writer = pq.ParquetWriter(
        partial,
        source.schema_arrow,
        compression="zstd",
        compression_level=1,
    )
    rows_written = 0
    try:
        for row_group in range(source.metadata.num_row_groups):
            table = source.read_row_group(row_group, use_threads=False)
            writer.write_table(table, row_group_size=1)
            rows_written += table.num_rows
    finally:
        writer.close()
    if rows_written != expected_rows:
        raise RuntimeError(
            f"random-access repack row mismatch for {source_path}: "
            f"{rows_written} != {expected_rows}"
        )
    partial.replace(output_path)
    report["status"] = "written"
    return report


def _scan_groupings(
    output_root: Path,
    files: dict[str, list[str]],
) -> tuple[
    dict[str, list[list[int]]],
    list[str],
    dict[str, int],
    dict[str, str],
    dict[str, dict[str, int]],
]:
    by_split: dict[str, OrderedDict[str, list[int]]] = {}
    source_order: list[str] = []
    known_sources: set[str] = set()
    counts: dict[str, int] = {}
    identity_digests: dict[str, str] = {}
    local_index_by_id: dict[str, dict[str, int]] = {}

    for split in ("train", "test"):
        groups: OrderedDict[str, list[int]] = OrderedDict()
        seen_ids: set[str] = set()
        split_local_index_by_id: dict[str, int] = {}
        digest = hashlib.sha256()
        offset = 0
        for relative_path in files[split]:
            table = pq.read_table(
                output_root / relative_path,
                columns=METADATA_COLUMNS,
                use_threads=False,
            )
            for row in table.to_pylist():
                sample_id = str(row["sample_id"])
                if sample_id in seen_ids:
                    raise ValueError(f"duplicate {split} sample_id: {sample_id}")
                seen_ids.add(sample_id)
                split_local_index_by_id[sample_id] = offset
                digest.update(sample_id.encode("utf-8"))
                digest.update(b"\n")
                source_name = str(row["source_name"])
                groups.setdefault(source_name, []).append(offset)
                if source_name not in known_sources:
                    known_sources.add(source_name)
                    source_order.append(source_name)
                offset += 1
        by_split[split] = groups
        counts[split] = offset
        identity_digests[split] = digest.hexdigest()
        local_index_by_id[split] = split_local_index_by_id

    groupings = {
        split: [by_split[split].get(source_name, []) for source_name in source_order]
        for split in ("train", "test")
    }
    return groupings, source_order, counts, identity_digests, local_index_by_id


def _reference_pvi_schedule_inputs(
    source_manifest: dict,
    *,
    local_groupings: dict[str, list[list[int]]],
    local_source_order: list[str],
    local_index_by_id: dict[str, dict[str, int]],
) -> tuple[dict[str, dict], list[str], dict]:
    """Recover the exact global index space used by the frozen PVI split.

    PVI's sampler intersects each source grouping with the sorted global
    ``Subset.indices`` and only then remaps the shuffled result to local subset
    indices.  The frozen split manifest records those samples in the original
    PVI dataset order, so the same global-index operation can be performed once
    ahead of training and translated to the row positions in this cache.
    """

    split_manifest_value = source_manifest.get("split_manifest")
    split_manifest_digest = source_manifest.get("split_manifest_sha256")
    if not split_manifest_value or not split_manifest_digest:
        return (
            {
                split: {
                    "global_indices": list(range(len(local_index_by_id[split]))),
                    "grouping": local_groupings[split],
                    "global_to_local": None,
                }
                for split in ("train", "test")
            },
            local_source_order,
            {
                "index_space": "cache-local",
                "reason": "source manifest did not reference a frozen PVI split manifest",
            },
        )

    split_manifest_path = Path(split_manifest_value).resolve()
    if not split_manifest_path.is_file():
        raise FileNotFoundError(split_manifest_path)
    observed_digest = sha256_file(split_manifest_path)
    if observed_digest != split_manifest_digest:
        raise ValueError("frozen PVI split manifest checksum differs from source cache")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split_manifest.get("schema") != "pvi-gcnm-population-within-split-v1":
        raise ValueError("frozen split manifest has the wrong schema")

    source_groups: OrderedDict[str, list[int]] = OrderedDict()
    selected_global: dict[str, list[int]] = {"train": [], "test": []}
    global_to_local: dict[str, dict[int, int]] = {"train": {}, "test": {}}
    seen_selected_ids: dict[str, set[str]] = {"train": set(), "test": set()}
    identities = split_manifest.get("identities", [])
    if len(identities) != int(split_manifest.get("row_count", -1)):
        raise ValueError("frozen split identity count differs from its row count")

    for global_index, identity in enumerate(identities):
        source_name = str(identity["source_name"])
        source_groups.setdefault(source_name, []).append(global_index)
        split = str(identity["assignment"])
        if split not in {"train", "test"}:
            continue
        sample_id = str(identity["sample_id"])
        try:
            local_index = local_index_by_id[split][sample_id]
        except KeyError as exc:
            raise ValueError(
                f"frozen {split} sample is absent from random-access cache: {sample_id}"
            ) from exc
        if sample_id in seen_selected_ids[split]:
            raise ValueError(f"duplicate frozen {split} sample identity: {sample_id}")
        seen_selected_ids[split].add(sample_id)
        selected_global[split].append(global_index)
        global_to_local[split][global_index] = local_index

    for split in ("train", "test"):
        cache_ids = set(local_index_by_id[split])
        if seen_selected_ids[split] != cache_ids:
            raise ValueError(
                f"frozen PVI and cache {split} identities differ: "
                f"pvi_only={len(seen_selected_ids[split] - cache_ids)}, "
                f"cache_only={len(cache_ids - seen_selected_ids[split])}"
            )

    grouping = list(source_groups.values())
    schedule_inputs = {
        split: {
            "global_indices": selected_global[split],
            "grouping": grouping,
            "global_to_local": global_to_local[split],
        }
        for split in ("train", "test")
    }
    return (
        schedule_inputs,
        list(source_groups),
        {
            "index_space": "original-pvi-global-indices",
            "split_manifest": str(split_manifest_path),
            "split_manifest_sha256": observed_digest,
            "source_count": len(source_groups),
            "global_sample_count": len(identities),
        },
    )


def _valid_schedule(path: Path, *, epochs: int, rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        array = np.load(path, mmap_mode="r")
    except Exception:
        return False
    return array.dtype == np.int32 and array.shape == (epochs, rows)


def build_schedule_bank(
    *,
    output_root: Path,
    schedule_inputs: dict[str, dict],
    counts: dict[str, int],
    epochs: int,
    batch_size: int,
    cluster_size: int,
    seed: int,
) -> dict:
    schedule_root = output_root / "schedules"
    schedule_root.mkdir(parents=True, exist_ok=True)
    paths = {
        split: schedule_root / f"{split}_order.npy" for split in ("train", "test")
    }
    if all(
        _valid_schedule(paths[split], epochs=epochs, rows=counts[split])
        for split in ("train", "test")
    ):
        return {
            "files": {
                split: str(paths[split].relative_to(output_root))
                for split in ("train", "test")
            },
            "sha256": {split: sha256_file(paths[split]) for split in ("train", "test")},
            "status": "reused",
        }

    partials = {
        split: schedule_root / f"{split}_order.partial.npy"
        for split in ("train", "test")
    }
    for partial in partials.values():
        if partial.exists():
            partial.unlink()
    arrays = {
        split: np.lib.format.open_memmap(
            partials[split],
            mode="w+",
            dtype=np.int32,
            shape=(epochs, counts[split]),
        )
        for split in ("train", "test")
    }

    # PVI constructs the train loader and then the test loader once per epoch.
    # A single RNG preserves that call order and all random-state consumption.
    rng = random.Random(int(seed))
    for epoch in range(epochs):
        for split in ("train", "test"):
            spec = schedule_inputs[split]
            global_order = pvi_stratified_order(
                spec["global_indices"],
                spec["grouping"],
                cluster_size=cluster_size,
                rng=rng,
            )
            remap = spec["global_to_local"]
            order = (
                global_order
                if remap is None
                else [remap[global_index] for global_index in global_order]
            )
            arrays[split][epoch, :] = np.asarray(order, dtype=np.int32)
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == epochs:
            print(f"precomputed PVI schedules: {epoch + 1}/{epochs} epochs", flush=True)
    for split in ("train", "test"):
        arrays[split].flush()
        del arrays[split]
        os.replace(partials[split], paths[split])

    return {
        "files": {
            split: str(paths[split].relative_to(output_root))
            for split in ("train", "test")
        },
        "sha256": {split: sha256_file(paths[split]) for split in ("train", "test")},
        "status": "written",
    }


def build_exact_cache(
    *,
    source_root: Path,
    output_root: Path,
    workers: int,
    epochs: int,
    batch_size: int,
    cluster_size: int,
    seed: int,
    resume: bool,
    enforce_scratch: bool = True,
) -> dict:
    if enforce_scratch:
        source_root = require_scratch_path(source_root)
        output_root = require_scratch_path(output_root)
    else:
        source_root = Path(source_root).resolve()
        output_root = Path(output_root).resolve()

    source_manifest_path = source_root / "manifest.json"
    if not (source_root / "_SUCCESS").is_file() or not source_manifest_path.is_file():
        raise RuntimeError(f"source cache is incomplete: {source_root}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"expected source schema {SOURCE_SCHEMA}")
    if int(source_manifest.get("seed", -1)) != int(seed):
        raise ValueError("source cache seed differs from schedule seed")
    if epochs < 1 or batch_size < 1 or cluster_size < 1:
        raise ValueError("epochs, batch size, and cluster size must be positive")

    marker = output_root / "_INCOMPLETE"
    if resume:
        if not marker.is_file() or (output_root / "manifest.json").exists():
            raise RuntimeError("resume requires an incomplete cache without a manifest")
    else:
        if output_root.exists():
            raise FileExistsError(f"immutable exact cache exists: {output_root}")
        output_root.mkdir(parents=True)
        marker.write_text("exact PVI Parquet cache build in progress\n", encoding="utf-8")

    payloads = []
    output_files: dict[str, list[str]] = {"train": [], "test": []}
    for split in ("train", "test"):
        for part, relative_source in enumerate(source_manifest["files"][split]):
            source_path = source_root / relative_source
            expected_rows = int(pq.ParquetFile(source_path).metadata.num_rows)
            relative_output = f"{split}/random-access-part-{part:05d}.parquet"
            output_files[split].append(relative_output)
            payloads.append(
                {
                    "split": split,
                    "part": part,
                    "source_path": str(source_path),
                    "output_path": str(output_root / relative_output),
                    "expected_rows": expected_rows,
                }
            )

    worker_count = min(max(1, int(workers)), len(payloads))
    reports = []
    if worker_count == 1:
        for payload in payloads:
            report = repack_random_access_file(payload)
            reports.append(report)
            print(
                f"[{report['split']}] part {report['part']:05d}: "
                f"{report['status']} {report['rows']:,} one-row groups",
                flush=True,
            )
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            futures = {
                executor.submit(repack_random_access_file, payload): payload
                for payload in payloads
            }
            for future in as_completed(futures):
                report = future.result()
                reports.append(report)
                print(
                    f"[{report['split']}] part {report['part']:05d}: "
                    f"{report['status']} {report['rows']:,} one-row groups",
                    flush=True,
                )
    reports.sort(key=lambda report: (report["split"], report["part"]))

    (
        local_groupings,
        local_source_order,
        counts,
        identity_digests,
        local_index_by_id,
    ) = _scan_groupings(output_root, output_files)
    for split in ("train", "test"):
        expected = int(source_manifest["counts"][split])
        if counts[split] != expected:
            raise RuntimeError(f"{split} count differs: {counts[split]} != {expected}")

    schedule_inputs, source_order, reference_layout = _reference_pvi_schedule_inputs(
        source_manifest,
        local_groupings=local_groupings,
        local_source_order=local_source_order,
        local_index_by_id=local_index_by_id,
    )
    schedules = build_schedule_bank(
        output_root=output_root,
        schedule_inputs=schedule_inputs,
        counts=counts,
        epochs=epochs,
        batch_size=batch_size,
        cluster_size=cluster_size,
        seed=seed,
    )
    output_counts = {
        "train": counts["train"],
        "test": counts["test"],
        "excluded": int(source_manifest["counts"].get("excluded", 0)),
    }
    manifest = {
        key: value
        for key, value in source_manifest.items()
        if key not in {"schema", "created_utc", "files", "row_group_size", "build"}
    }
    manifest.update(
        {
            "schema": OUTPUT_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "counts": output_counts,
            "files": output_files,
            "row_group_size": 1,
            "random_access": {
                "one_sample_per_row_group": True,
                "identity_sha256": identity_digests,
            },
            "batch_layout": {
                "strategy": "precomputed-exact-pvi-batch-schedules",
                "reference": "PviBatchSampler._stratified_random",
                "grouping_key": "source_name",
                "source_order": source_order,
                "cluster_size": cluster_size,
                "batch_size": batch_size,
                "epochs": epochs,
                "seed": seed,
                "rng": "python random.Random",
                "generation_call_order": "train then test for every epoch",
                "reference_index_layout": reference_layout,
                "schedule_schema": SCHEDULE_SCHEMA,
                "schedule_files": schedules["files"],
                "schedule_sha256": schedules["sha256"],
            },
            "source_cache": str(source_root),
            "source_cache_schema": SOURCE_SCHEMA,
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "build": {
                "workers": worker_count,
                "parts": len(reports),
                "parts_written": sum(report["status"] == "written" for report in reports),
                "parts_reused": sum(report["status"] == "reused" for report in reports),
                "schedule_status": schedules["status"],
                "large_io_root": str(output_root),
            },
        }
    )
    temporary_manifest = output_root / "manifest.partial.json"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(output_root / "manifest.json")
    marker.unlink()
    (output_root / "_SUCCESS").write_text("validated\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--epochs",
        type=int,
        default=501,
        help="schedule rows; 500 training epochs require one extra terminal-test row",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--cluster-size",
        type=int,
        default=48,
        help="PVI main uses max_cache=50, hence max(5, max_cache-2)=48",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = build_exact_cache(
        source_root=args.source_root,
        output_root=args.output_root,
        workers=args.workers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        cluster_size=args.cluster_size,
        seed=args.seed,
        resume=args.resume,
        enforce_scratch=True,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "schema": manifest["schema"],
                "counts": manifest["counts"],
                "files": {key: len(value) for key, value in manifest["files"].items()},
                "schedule_files": manifest["batch_layout"]["schedule_files"],
                "large_io_root": manifest["build"]["large_io_root"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
