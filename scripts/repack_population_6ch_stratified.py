#!/usr/bin/env python3
"""Materialize PVI-style stratified batches as immutable Parquet row groups.

The source six-channel cache is organized by reconstruction shard and therefore
produces batches dominated by one recording.  PVI-ML's ``PviBatchSampler``
instead shuffles recording groups in clusters and mixes their samples before
forming batches.  This script performs that expensive mixing once, ahead of
training.  Every output Parquet row group is one ready-to-stream training
batch; training only needs to shuffle row-group order between epochs.

The source cache is never modified.  Builds are deterministic and resumable at
the output-file level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import random
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import pyarrow as pa
import pyarrow.parquet as pq


SOURCE_SCHEMA = "pvi-gcnm-population-6ch-parquet-v1"
OUTPUT_SCHEMA = "pvi-gcnm-population-6ch-parquet-v2"
METADATA_COLUMNS = ["sample_id", "subject", "session", "source_name"]


@dataclass(frozen=True)
class RowReference:
    path: str
    row_group: int
    local_index: int
    sample_id: str
    subject: str
    session: str
    source_name: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_digest(references: list[RowReference]) -> str:
    digest = hashlib.sha256()
    for reference in references:
        digest.update(reference.sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def scan_split(
    source_root: Path,
    source_manifest: dict,
    split: str,
) -> dict[str, list[RowReference]]:
    """Index source rows by the same file/recording key used by PVI-ML."""

    grouped: dict[str, list[RowReference]] = defaultdict(list)
    seen: set[str] = set()
    expected_schema: pa.Schema | None = None
    relative_paths = source_manifest.get("files", {}).get(split, [])
    if not relative_paths:
        raise FileNotFoundError(f"source manifest contains no {split} files")

    for relative_path in relative_paths:
        path = (source_root / relative_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        parquet = pq.ParquetFile(path)
        if expected_schema is None:
            expected_schema = parquet.schema_arrow
        elif parquet.schema_arrow != expected_schema:
            raise ValueError(f"Parquet schema mismatch: {path}")
        for row_group in range(parquet.metadata.num_row_groups):
            metadata = parquet.read_row_group(
                row_group, columns=METADATA_COLUMNS, use_threads=False
            ).to_pylist()
            for local_index, row in enumerate(metadata):
                sample_id = str(row["sample_id"])
                if sample_id in seen:
                    raise ValueError(f"duplicate {split} sample_id: {sample_id}")
                seen.add(sample_id)
                source_name = str(row["source_name"])
                grouped[source_name].append(
                    RowReference(
                        path=str(path),
                        row_group=row_group,
                        local_index=local_index,
                        sample_id=sample_id,
                        subject=str(row["subject"]),
                        session=str(row["session"]),
                        source_name=source_name,
                    )
                )
    return dict(grouped)


def build_stratified_plan(
    grouped: dict[str, list[RowReference]],
    *,
    cluster_size: int,
    seed: int,
) -> tuple[list[RowReference], list[list[str]]]:
    """Reproduce PVI source clustering while keeping source reads monotonic.

    PVI-ML shuffles source groups, combines at most ``cluster_size`` groups,
    and then shuffles every sample in that cluster.  Here we shuffle the source
    membership labels identically but retain the original order *within* each
    source.  Batch source composition therefore has the same distribution as
    PVI's sampler while the one-time repack can stream each source efficiently.
    """

    if cluster_size < 1:
        raise ValueError("cluster_size must be positive")
    if not grouped:
        return [], []

    rng = random.Random(int(seed))
    source_names = sorted(grouped)
    rng.shuffle(source_names)
    plan: list[RowReference] = []
    clusters: list[list[str]] = []

    for first in range(0, len(source_names), cluster_size):
        cluster = source_names[first : first + cluster_size]
        clusters.append(cluster)
        labels = [
            source_name
            for source_name in cluster
            for _ in range(len(grouped[source_name]))
        ]
        rng.shuffle(labels)
        offsets = Counter()
        for source_name in labels:
            offset = offsets[source_name]
            plan.append(grouped[source_name][offset])
            offsets[source_name] += 1

    expected = sum(len(rows) for rows in grouped.values())
    if len(plan) != expected:
        raise RuntimeError(f"stratified plan lost rows: {len(plan)} != {expected}")
    if len({row.sample_id for row in plan}) != expected:
        raise RuntimeError("stratified plan duplicated sample identities")
    return plan, clusters


class _RowGroupCache:
    def __init__(self, max_groups: int) -> None:
        self.max_groups = max(1, int(max_groups))
        self.parquet_files: dict[str, pq.ParquetFile] = {}
        self.tables: OrderedDict[tuple[str, int], pa.Table] = OrderedDict()

    def get_row(self, reference: RowReference) -> pa.Table:
        key = (reference.path, reference.row_group)
        table = self.tables.get(key)
        if table is None:
            parquet = self.parquet_files.get(reference.path)
            if parquet is None:
                parquet = pq.ParquetFile(reference.path)
                self.parquet_files[reference.path] = parquet
            table = parquet.read_row_group(reference.row_group, use_threads=False)
            self.tables[key] = table
            while len(self.tables) > self.max_groups:
                self.tables.popitem(last=False)
        else:
            self.tables.move_to_end(key)
        return table.slice(reference.local_index, 1)


def _batch_audit(references: list[RowReference]) -> dict:
    return {
        "size": len(references),
        "unique_sources": len({row.source_name for row in references}),
        "unique_subjects": len({row.subject for row in references}),
        "unique_sessions": len({(row.subject, row.session) for row in references}),
    }


def _completed_part_is_valid(path: Path, *, rows: int, row_groups: int) -> bool:
    if not path.is_file():
        return False
    try:
        metadata = pq.ParquetFile(path).metadata
    except Exception:
        return False
    return int(metadata.num_rows) == rows and int(metadata.num_row_groups) == row_groups


def write_output_part(payload: dict) -> dict:
    references: list[RowReference] = payload["references"]
    output_path = Path(payload["output_path"])
    batch_size = int(payload["batch_size"])
    cache_groups = int(payload["cache_groups"])
    expected_groups = (len(references) + batch_size - 1) // batch_size
    report = {
        "split": payload["split"],
        "part": int(payload["part"]),
        "path": str(output_path),
        "rows": len(references),
        "row_groups": expected_groups,
        "ordered_sample_sha256": _ordered_digest(references),
        "batches": [
            _batch_audit(references[first : first + batch_size])
            for first in range(0, len(references), batch_size)
        ],
    }
    if _completed_part_is_valid(
        output_path, rows=len(references), row_groups=expected_groups
    ):
        report["status"] = "reused"
        return report

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(".partial.parquet")
    if partial.exists():
        partial.unlink()

    cache = _RowGroupCache(cache_groups)
    writer: pq.ParquetWriter | None = None
    try:
        for first in range(0, len(references), batch_size):
            batch = references[first : first + batch_size]
            rows = [cache.get_row(reference) for reference in batch]
            table = pa.concat_tables(rows).combine_chunks()
            if writer is None:
                writer = pq.ParquetWriter(
                    partial,
                    table.schema,
                    compression="zstd",
                    compression_level=1,
                )
            writer.write_table(table, row_group_size=len(batch))
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError(f"empty output payload for {output_path}")
    partial.replace(output_path)
    report["status"] = "written"
    return report


def _distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

    return {
        "min": int(ordered[0]),
        "p10": percentile(0.10),
        "median": float(median(ordered)),
        "mean": float(mean(ordered)),
        "p90": percentile(0.90),
        "max": int(ordered[-1]),
    }


def _layout_audit(reports: list[dict], split: str) -> dict:
    batches = [
        batch
        for report in reports
        if report["split"] == split
        for batch in report["batches"]
    ]
    full = [batch for batch in batches if batch["size"] == max(x["size"] for x in batches)]
    return {
        "batch_count": len(batches),
        "full_batch_count": len(full),
        "batch_size": _distribution([batch["size"] for batch in batches]),
        "unique_sources_per_full_batch": _distribution(
            [batch["unique_sources"] for batch in full]
        ),
        "unique_subjects_per_full_batch": _distribution(
            [batch["unique_subjects"] for batch in full]
        ),
        "unique_sessions_per_full_batch": _distribution(
            [batch["unique_sessions"] for batch in full]
        ),
    }


def repack_cache(
    *,
    source_root: Path,
    output_root: Path,
    batch_size: int,
    cluster_size: int,
    row_groups_per_file: int,
    workers: int,
    seed: int,
    resume: bool,
) -> dict:
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    source_manifest_path = source_root / "manifest.json"
    if not (source_root / "_SUCCESS").is_file() or not source_manifest_path.is_file():
        raise RuntimeError(f"source cache is incomplete: {source_root}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"expected source schema {SOURCE_SCHEMA}")
    if int(source_manifest.get("seed", -1)) != int(seed):
        raise ValueError("source cache seed differs from requested layout seed")
    if batch_size < 1 or cluster_size < 1 or row_groups_per_file < 1:
        raise ValueError("batch and layout sizes must be positive")

    marker = output_root / "_INCOMPLETE"
    if resume:
        if not marker.is_file() or (output_root / "manifest.json").exists():
            raise RuntimeError("resume requires an incomplete cache without a manifest")
    else:
        if output_root.exists():
            raise FileExistsError(f"immutable output cache exists: {output_root}")
        output_root.mkdir(parents=True)
        marker.write_text("pre-stratified Parquet repack in progress\n", encoding="utf-8")

    plans: dict[str, list[RowReference]] = {}
    cluster_orders: dict[str, list[list[str]]] = {}
    for split, split_seed in (("train", seed), ("test", seed + 1_000_003)):
        grouped = scan_split(source_root, source_manifest, split)
        plan, clusters = build_stratified_plan(
            grouped, cluster_size=cluster_size, seed=split_seed
        )
        expected = int(source_manifest["counts"][split])
        if len(plan) != expected:
            raise RuntimeError(f"{split} plan count differs: {len(plan)} != {expected}")
        plans[split] = plan
        cluster_orders[split] = clusters

    payloads = []
    rows_per_file = batch_size * row_groups_per_file
    for split in ("train", "test"):
        plan = plans[split]
        for part, first in enumerate(range(0, len(plan), rows_per_file)):
            references = plan[first : first + rows_per_file]
            payloads.append(
                {
                    "split": split,
                    "part": part,
                    "references": references,
                    "output_path": str(
                        output_root / split / f"stratified-part-{part:05d}.parquet"
                    ),
                    "batch_size": batch_size,
                    "cache_groups": cluster_size + 8,
                }
            )

    worker_count = min(max(1, int(workers)), len(payloads))
    if worker_count == 1:
        reports = []
        for payload in payloads:
            report = write_output_part(payload)
            reports.append(report)
            print(
                f"[{report['split']}] part {report['part']:05d}: "
                f"{report['status']} {report['rows']:,} rows",
                flush=True,
            )
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            futures = {executor.submit(write_output_part, payload): payload for payload in payloads}
            reports = []
            for future in as_completed(futures):
                report = future.result()
                reports.append(report)
                print(
                    f"[{report['split']}] part {report['part']:05d}: "
                    f"{report['status']} {report['rows']:,} rows",
                    flush=True,
                )
    reports.sort(key=lambda row: (row["split"], row["part"]))

    files = {
        split: [
            str(Path(report["path"]).relative_to(output_root))
            for report in reports
            if report["split"] == split
        ]
        for split in ("train", "test")
    }
    output_counts = {
        split: sum(report["rows"] for report in reports if report["split"] == split)
        for split in ("train", "test")
    }
    output_counts["excluded"] = int(source_manifest["counts"].get("excluded", 0))
    expected_counts = {
        key: int(source_manifest["counts"][key])
        for key in ("train", "test", "excluded")
    }
    if output_counts != expected_counts:
        raise RuntimeError(
            f"repacked counts differ: actual={output_counts}, expected={expected_counts}"
        )

    manifest = {
        key: value
        for key, value in source_manifest.items()
        if key
        not in {
            "schema",
            "created_utc",
            "files",
            "row_group_size",
            "build",
        }
    }
    manifest.update(
        {
            "schema": OUTPUT_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "counts": output_counts,
            "row_group_size": batch_size,
            "files": files,
            "batch_layout": {
                "strategy": "pvi-source-clustered-prestratified",
                "grouping_key": "source_name",
                "cluster_size": cluster_size,
                "batch_size": batch_size,
                "seed": seed,
                "within_source_order": "source-cache-order",
                "epoch_behavior": "fixed batch membership; deterministic row-group-order shuffle",
                "cluster_source_order": cluster_orders,
                "ordered_sample_sha256": {
                    split: _ordered_digest(plans[split]) for split in ("train", "test")
                },
            },
            "layout_audit": {
                split: _layout_audit(reports, split) for split in ("train", "test")
            },
            "source_cache": str(source_root),
            "source_cache_schema": SOURCE_SCHEMA,
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "build": {
                "workers": worker_count,
                "row_groups_per_file": row_groups_per_file,
                "parts": len(reports),
                "parts_written": sum(report["status"] == "written" for report in reports),
                "parts_reused": sum(report["status"] == "reused" for report in reports),
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cluster-size", type=int, default=30)
    parser.add_argument("--row-groups-per-file", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = repack_cache(
        source_root=args.source_root,
        output_root=args.output_root,
        batch_size=args.batch_size,
        cluster_size=args.cluster_size,
        row_groups_per_file=args.row_groups_per_file,
        workers=args.workers,
        seed=args.seed,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "counts": manifest["counts"],
                "files": {key: len(value) for key, value in manifest["files"].items()},
                "layout_audit": manifest["layout_audit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
