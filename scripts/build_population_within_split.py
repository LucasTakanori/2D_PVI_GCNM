#!/usr/bin/env python3
"""Freeze the exact pvi_ml population-within mask05 partition by sample ID.

The split itself is produced by pvi_ml's ``PviLazyDataset`` and
``GraphBipartitePartitioner``.  This script only translates its global masks
back to stable ``(source, start, stop)`` identities so downstream Parquet
caches never have to recompute the partition.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def stable_sample_id(source_name: str, mask_key: str, start: int, stop: int) -> str:
    value = f"{source_name}|{mask_key}|{int(start)}|{int(stop)}".encode()
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinate_ids(root: Path) -> set[str]:
    import pyarrow.dataset as pads

    table = pads.dataset(str(root / "shards"), format="parquet").to_table(
        columns=["sample_id"]
    )
    return set(table["sample_id"].to_pylist())


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pvi-ml-root",
        type=Path,
        default=repo.parent / "fundational_pvi",
    )
    parser.add_argument(
        "--coordinate-root",
        type=Path,
        default=repo / "gcnm_parquet/coordinate_direct_main_b045_v1",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=repo / "data/registries/main_b045_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "data/manifests/pw_population_within_mask05_seed42_v1.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"immutable split manifest exists: {args.output}")
    for path in (args.coordinate_root / "manifest.json", args.registry):
        if not path.is_file():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(args.pvi_ml_root.resolve()))
    from src.pipeline.data_discovery import PviDatasetInventory
    from src.pipeline.data_preparation_lazy import PviLazyDataset
    from src.utils.primitives import InputMode, OutputMode, SequenceMask

    # pvi_ml is intentionally verbose while surveying and removing overlaps.
    # Preserve a concise audit log without emitting hundreds of thousands of
    # progress updates into the scheduler log.
    transcript = io.StringIO()
    with contextlib.redirect_stdout(transcript), contextlib.redirect_stderr(transcript):
        inventory = PviDatasetInventory(branch="main")
        dataset = PviLazyDataset(
            ds_files=inventory,
            input_mode=InputMode("img"),
            output_mode=OutputMode("waveform"),
            mask_key=SequenceMask("mask05"),
            max_cache=1,
            persistent_handle=False,
        ).build()
        dataset.set_partition(
            test_size=0.1,
            shuffle=True,
            split_mode="within",
            random_state=args.seed,
        )
        dataset.get_partition()

    train = set(dataset.train_mask)
    test = set(dataset.test_mask)
    if train & test:
        raise RuntimeError("pvi_ml returned overlapping train and test masks")

    assignments: dict[str, str] = {}
    identities: list[dict] = []
    subject_counts: dict[str, Counter] = {}
    for source, local, global_mask in zip(
        dataset.mappings.files,
        dataset.mappings.masks_local,
        dataset.mappings.masks_global,
        strict=True,
    ):
        start, stop = (int(local[0]), int(local[1]))
        sample_id = stable_sample_id(source.name, "mask05", start, stop)
        label = "train" if global_mask in train else "test" if global_mask in test else "excluded"
        if sample_id in assignments:
            raise RuntimeError(f"duplicate stable identity: {source.name} {local}")
        assignments[sample_id] = label
        subject = str(source.subject)
        subject_counts.setdefault(subject, Counter())[label] += 1
        identities.append(
            {
                "sample_id": sample_id,
                "subject": subject,
                "source_name": str(source.name),
                "mask_start": start,
                "mask_stop": stop,
                "assignment": label,
            }
        )

    parquet_ids = coordinate_ids(args.coordinate_root)
    assignment_ids = set(assignments)
    if parquet_ids != assignment_ids:
        raise RuntimeError(
            "pvi_ml inventory and coordinate Parquet identities differ: "
            f"pvi_only={len(assignment_ids - parquet_ids)}, "
            f"parquet_only={len(parquet_ids - assignment_ids)}"
        )

    counts = Counter(assignments.values())
    payload = {
        "schema": "pvi-gcnm-population-within-split-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "pvi_ml PviLazyDataset split_mode=within with GraphBipartitePartitioner",
        "branch": "main",
        "mask_key": "mask05",
        "test_size": 0.1,
        "seed": args.seed,
        "subject_count": len(subject_counts),
        "source_count": len(dataset.raws),
        "row_count": len(assignments),
        "counts": {key: int(counts[key]) for key in ("train", "test", "excluded")},
        "subjects": {
            subject: {key: int(values[key]) for key in ("train", "test", "excluded")}
            for subject, values in sorted(subject_counts.items())
        },
        "coordinate_root": str(args.coordinate_root.resolve()),
        "coordinate_manifest_sha256": sha256_file(args.coordinate_root / "manifest.json"),
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry),
        "pvi_ml_root": str(args.pvi_ml_root.resolve()),
        "assignments": assignments,
        "identities": identities,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()
