#!/usr/bin/env python3
"""Build an exact-PVI disjoint index view over the six-channel Parquet cache.

The immutable v3 cache already stores every overlap-filtered sample in one-row
Parquet groups.  This builder writes only disjoint train/test row-index arrays
and ahead-of-time PVI sampler schedules.  No image tensor is copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scripts.build_population_6ch_exact_pvi_cache import (
    SCHEDULE_SCHEMA,
    build_schedule_bank,
    require_scratch_path,
    sha256_file,
)


SOURCE_SCHEMAS = {
    "pvi-gcnm-population-6ch-parquet-v1",
    "pvi-gcnm-population-6ch-parquet-v3",
}
SPLIT_SCHEMA = "pvi-gcnm-population-disjoint-split-v1"
OUTPUT_SCHEMA = "pvi-gcnm-population-6ch-parquet-view-v4"
METADATA_COLUMNS = [
    "sample_id",
    "subject",
    "session",
    "source_name",
    "mask_start",
    "mask_stop",
]


def _write_npy_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def _scan_physical_rows(
    source_files: list[Path],
) -> tuple[dict[str, tuple[int, dict]], str]:
    by_id: dict[str, tuple[int, dict]] = {}
    digest = hashlib.sha256()
    physical_index = 0
    for path in source_files:
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != parquet.metadata.num_row_groups:
            raise ValueError(f"source file is not one-row random access: {path}")
        rows = pq.read_table(
            path,
            columns=METADATA_COLUMNS,
            use_threads=False,
        ).to_pylist()
        for row in rows:
            sample_id = str(row["sample_id"])
            if sample_id in by_id:
                raise ValueError(f"duplicate source-cache sample_id: {sample_id}")
            by_id[sample_id] = (physical_index, row)
            digest.update(sample_id.encode("utf-8"))
            digest.update(b"\n")
            physical_index += 1
    return by_id, digest.hexdigest()


def build_disjoint_view(
    *,
    source_root: Path,
    additional_source_roots: list[Path] | None,
    split_manifest_path: Path,
    output_root: Path,
    epochs: int,
    batch_size: int,
    cluster_size: int,
    seed: int,
    resume: bool,
    enforce_scratch: bool = True,
) -> dict:
    if enforce_scratch:
        source_root = require_scratch_path(source_root)
        additional_source_roots = [
            require_scratch_path(path) for path in (additional_source_roots or [])
        ]
        output_root = require_scratch_path(output_root)
        split_manifest_path = require_scratch_path(split_manifest_path)
    else:
        source_root = Path(source_root).resolve()
        additional_source_roots = [
            Path(path).resolve() for path in (additional_source_roots or [])
        ]
        output_root = Path(output_root).resolve()
        split_manifest_path = Path(split_manifest_path).resolve()

    source_roots = [source_root, *(additional_source_roots or [])]
    source_caches = []
    for cache_root in source_roots:
        source_manifest_path = cache_root / "manifest.json"
        if not (cache_root / "_SUCCESS").is_file() or not source_manifest_path.is_file():
            raise RuntimeError(f"source cache is incomplete: {cache_root}")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest.get("schema") not in SOURCE_SCHEMAS:
            raise ValueError(f"unsupported source cache schema at {cache_root}")
        if int(source_manifest.get("seed", -1)) != int(seed):
            raise ValueError("source cache and requested seed differ")
        relative_files = [
            *source_manifest.get("files", {}).get("train", []),
            *source_manifest.get("files", {}).get("test", []),
        ]
        source_caches.append(
            {
                "root": cache_root,
                "manifest_path": source_manifest_path,
                "manifest": source_manifest,
                "relative_files": relative_files,
                "physical_files": [cache_root / relative for relative in relative_files],
            }
        )
    source = source_caches[0]["manifest"]
    split = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split.get("schema") != SPLIT_SCHEMA:
        raise ValueError(f"expected split schema {SPLIT_SCHEMA}")
    if int(split.get("seed", -1)) != int(seed):
        raise ValueError("split manifest and requested seed differ")
    if epochs < 1 or batch_size < 1 or cluster_size < 1:
        raise ValueError("epochs, batch size, and cluster size must be positive")

    marker = output_root / "_INCOMPLETE"
    if resume:
        if not marker.is_file() or (output_root / "manifest.json").exists():
            raise RuntimeError("resume requires an incomplete view without a manifest")
    else:
        if output_root.exists():
            raise FileExistsError(f"immutable disjoint view exists: {output_root}")
        output_root.mkdir(parents=True)
        marker.write_text("disjoint six-channel index view build in progress\n", encoding="utf-8")

    physical_files = [
        path for cache in source_caches for path in cache["physical_files"]
    ]
    physical_by_id, physical_identity_digest = _scan_physical_rows(physical_files)
    expected_physical = sum(
        int(cache["manifest"]["counts"]["train"])
        + int(cache["manifest"]["counts"]["test"])
        for cache in source_caches
    )
    if len(physical_by_id) != expected_physical:
        raise RuntimeError(
            f"source physical count differs: {len(physical_by_id)} != {expected_physical}"
        )

    identities = split.get("identities", [])
    if len(identities) != int(split.get("row_count", -1)):
        raise ValueError("disjoint split identity count differs from row_count")
    source_groups: OrderedDict[str, list[int]] = OrderedDict()
    selected_global: dict[str, list[int]] = {"train": [], "test": []}
    global_to_local: dict[str, dict[int, int]] = {"train": {}, "test": {}}
    physical_selection: dict[str, list[int]] = {"train": [], "test": []}
    seen_active: set[str] = set()
    for global_index, identity in enumerate(identities):
        source_name = str(identity["source_name"])
        source_groups.setdefault(source_name, []).append(global_index)
        assignment = str(identity["assignment"])
        if assignment == "excluded":
            continue
        if assignment not in {"train", "test"}:
            raise ValueError(f"invalid disjoint assignment: {assignment}")
        sample_id = str(identity["sample_id"])
        try:
            physical_index, physical = physical_by_id[sample_id]
        except KeyError as exc:
            raise ValueError(f"active disjoint sample absent from source cache: {sample_id}") from exc
        if sample_id in seen_active:
            raise ValueError(f"duplicate active disjoint sample: {sample_id}")
        seen_active.add(sample_id)
        for field in ("subject", "source_name", "mask_start", "mask_stop"):
            if str(identity[field]) != str(physical[field]):
                raise ValueError(
                    f"disjoint/source identity mismatch for {sample_id}: {field}"
                )
        local_index = len(physical_selection[assignment])
        physical_selection[assignment].append(physical_index)
        selected_global[assignment].append(global_index)
        global_to_local[assignment][global_index] = local_index

    if seen_active != set(physical_by_id):
        raise ValueError(
            "disjoint active set differs from exact cache: "
            f"split_only={len(seen_active - set(physical_by_id))}, "
            f"cache_only={len(set(physical_by_id) - seen_active)}"
        )
    train_physical = set(physical_selection["train"])
    test_physical = set(physical_selection["test"])
    if train_physical & test_physical:
        raise RuntimeError("disjoint physical selections overlap")
    if train_physical | test_physical != set(range(expected_physical)):
        raise RuntimeError("disjoint physical selections do not cover the source cache")

    index_root = output_root / "indices"
    index_root.mkdir(parents=True, exist_ok=True)
    selection_paths = {
        split_name: index_root / f"{split_name}_physical_rows.npy"
        for split_name in ("train", "test")
    }
    for split_name, path in selection_paths.items():
        values = np.asarray(physical_selection[split_name], dtype=np.int32)
        if not path.is_file():
            _write_npy_atomic(path, values)
        observed = np.load(path, mmap_mode="r")
        if observed.dtype != np.int32 or not np.array_equal(observed, values):
            raise RuntimeError(f"stored {split_name} physical selection differs")

    counts = {
        "train": len(physical_selection["train"]),
        "test": len(physical_selection["test"]),
        "excluded": int(split["counts"]["excluded"]),
    }
    for split_name in ("train", "test"):
        if counts[split_name] != int(split["counts"][split_name]):
            raise RuntimeError(f"{split_name} count differs from disjoint manifest")
    schedule_inputs = {
        split_name: {
            "global_indices": selected_global[split_name],
            "grouping": list(source_groups.values()),
            "global_to_local": global_to_local[split_name],
        }
        for split_name in ("train", "test")
    }
    schedules = build_schedule_bank(
        output_root=output_root,
        schedule_inputs=schedule_inputs,
        counts=counts,
        epochs=epochs,
        batch_size=batch_size,
        cluster_size=cluster_size,
        seed=seed,
    )

    manifest = {
        "schema": OUTPUT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "representation": source.get("representation"),
        "channel_contract": source.get("channel_contract"),
        "stored_base_fields": source.get("stored_base_fields"),
        "tensor_shapes": source.get("tensor_shapes"),
        "mask_key": source.get("mask_key", "mask05"),
        "split_mode": "disjoint",
        "seed": int(seed),
        "counts": counts,
        "row_group_size": 1,
        "test_subjects": split["test_subjects"],
        "train_subjects": split["train_subjects"],
        "split_manifest": str(split_manifest_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "data_view": {
            "strategy": "immutable-physical-row-index",
            "source_caches": [
                {
                    "root": str(cache["root"]),
                    "schema": cache["manifest"]["schema"],
                    "manifest_sha256": sha256_file(cache["manifest_path"]),
                    "files": [str(path) for path in cache["physical_files"]],
                }
                for cache in source_caches
            ],
            "source_files": [str(path) for path in physical_files],
            "source_physical_rows": expected_physical,
            "source_identity_sha256": physical_identity_digest,
            "selection_files": {
                split_name: str(path.relative_to(output_root))
                for split_name, path in selection_paths.items()
            },
            "selection_sha256": {
                split_name: sha256_file(path)
                for split_name, path in selection_paths.items()
            },
            "payload_copied": False,
        },
        "batch_layout": {
            "strategy": "precomputed-exact-pvi-batch-schedules",
            "reference": "PviBatchSampler._stratified_random",
            "grouping_key": "source_name",
            "source_order": list(source_groups),
            "cluster_size": int(cluster_size),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "seed": int(seed),
            "rng": "python random.Random",
            "generation_call_order": "train then test for every epoch",
            "reference_index_layout": {
                "index_space": "original-pvi-global-indices",
                "split_manifest": str(split_manifest_path),
                "split_manifest_sha256": sha256_file(split_manifest_path),
                "source_count": len(source_groups),
                "global_sample_count": len(identities),
            },
            "schedule_schema": SCHEDULE_SCHEMA,
            "schedule_files": schedules["files"],
            "schedule_sha256": schedules["sha256"],
        },
        "build": {
            "schedule_status": schedules["status"],
            "large_io_root": str(output_root),
            "payload_copied": False,
        },
    }
    temporary_manifest = output_root / "manifest.partial.json"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(output_root / "manifest.json")
    marker.unlink()
    (output_root / "_SUCCESS").write_text("validated\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--additional-source-root",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=501)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cluster-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = build_disjoint_view(
        source_root=args.source_root,
        additional_source_roots=args.additional_source_root,
        split_manifest_path=args.split_manifest,
        output_root=args.output_root,
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
                "test_subjects": manifest["test_subjects"],
                "payload_copied": manifest["data_view"]["payload_copied"],
                "schedule_files": manifest["batch_layout"]["schedule_files"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
