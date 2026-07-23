"""Stable sample identifiers and leakage-safe subject split manifests."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


SPLIT_VERSION = 1
SESSION_ORDER = {"baseline": 0, "valsalva": 1, "pressor": 2}
SPLIT_LABELS = {"train", "test", "excluded"}


def stable_sample_id(source_name: str, mask_key: str, start: int, stop: int) -> str:
    payload = f"{source_name}|{mask_key}|{int(start)}|{int(stop)}".encode()
    return hashlib.sha256(payload).hexdigest()


def _compute_intervals_overlap(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    """Mirror pvi_ml.graph_partitioner.compute_intervals_overlap exactly."""
    left_starts = np.array([item[0] for item in left])[:, np.newaxis]
    left_ends = np.array([item[-1] for item in left])[:, np.newaxis]
    right_starts = np.array([item[0] for item in right])
    right_ends = np.array([item[-1] for item in right])
    overlap = (left_starts < right_ends) & (right_starts < left_ends)
    if overlap.sum() > 0:
        rows = overlap.sum(axis=1).argsort()[::-1]
        columns = overlap.sum(axis=0).argsort()[::-1]
        overlap = overlap[np.ix_(rows, columns)]
        left = [left[index] for index in rows]
        right = [right[index] for index in columns]
    return overlap, left, right


def _pvi_ml_graph_split(
    intervals: list[tuple[int, int]], test_size: float, seed: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Reproduce the current pvi_ml GraphBipartitePartitioner split.

    This deliberately retains its seed-42 sklearn initialization, conflict
    sorting, ratio-preserving alternating removals, and skip size of one.
    Windows removed to eliminate train/test overlap are handled as exclusions
    by :func:`deterministic_subject_split`.
    """
    from sklearn.model_selection import train_test_split

    if len(intervals) < 20:
        train, test = intervals, intervals.copy()
    elif test_size == 0.0:
        train, test = intervals, []
    elif test_size == 1.0:
        train, test = [], intervals
    else:
        train, test = train_test_split(
            intervals, test_size=test_size, random_state=seed, shuffle=True
        )
        train, test = sorted(train), sorted(test)
    if not train or not test:
        return train, test

    overlap, train, test = _compute_intervals_overlap(train, test)
    original_ratio = len(train) / len(test)
    keep_train = set(range(len(train)))
    keep_test = set(range(len(test)))
    selected_train, selected_test = train.copy(), test.copy()
    for _ in itertools.count():
        if len(selected_train) <= 1 or len(selected_test) <= 1 or overlap.sum() == 0:
            break
        current_ratio = len(selected_train) / len(selected_test)
        if current_ratio >= original_ratio:
            remove = overlap.sum(axis=1).argsort()[-1:]
            overlap[remove, :] = False
            keep_train -= set(remove)
        else:
            remove = overlap.sum(axis=0).argsort()[-1:]
            overlap[:, remove] = False
            keep_test -= set(remove)
        selected_train = [train[index] for index in keep_train]
        selected_test = [test[index] for index in keep_test]

    remaining, _, _ = _compute_intervals_overlap(selected_train, selected_test)
    if remaining.sum() > 0:
        raise RuntimeError(
            f"pvi_ml-compatible graph split retained {int(remaining.sum())} overlaps"
        )
    return sorted(selected_train), sorted(selected_test)


def _global_subject_intervals(
    rows: list[dict],
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], dict]]:
    """Apply pvi_ml's composite-session offsets to one subject's local masks."""
    by_session: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_session[(str(row["session"]), str(row["source_name"]))].append(row)
    ordered_sessions = sorted(
        by_session, key=lambda key: (SESSION_ORDER.get(key[0], 99), key[1])
    )
    intervals: list[tuple[int, int]] = []
    lookup: dict[tuple[int, int], dict] = {}
    offset = 0
    for key in ordered_sessions:
        session_rows = sorted(
            by_session[key], key=lambda row: (int(row["mask_start"]), int(row["mask_stop"]))
        )
        period_counts = {
            int(row["num_periods"]) for row in session_rows if "num_periods" in row
        }
        if len(period_counts) > 1:
            raise ValueError(f"inconsistent num_periods for {key[1]}")
        inferred_periods = max(int(row["mask_stop"]) for row in session_rows)
        num_periods = period_counts.pop() if period_counts else inferred_periods
        if num_periods < inferred_periods:
            raise ValueError(f"mask exceeds num_periods for {key[1]}")
        for row in session_rows:
            interval = (
                int(row["mask_start"]) + offset,
                int(row["mask_stop"]) + offset,
            )
            if interval in lookup:
                raise ValueError(f"duplicate global mask {interval} for {key[1]}")
            intervals.append(interval)
            lookup[interval] = row
        offset += num_periods
    return intervals, lookup


def validate_split(rows: Iterable[dict], assignments: dict[str, str]) -> None:
    rows = list(rows)
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("sample_id values are not unique")
    missing = sorted(set(ids) - set(assignments))
    if missing:
        raise ValueError(f"split manifest is missing {len(missing)} sample IDs")
    invalid = {value for value in assignments.values() if value not in SPLIT_LABELS}
    if invalid:
        raise ValueError(f"invalid split labels: {sorted(invalid)}")
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_name"])].append(row)
    for source, source_rows in by_source.items():
        train = sorted(
            (int(row["mask_start"]), int(row["mask_stop"]))
            for row in source_rows
            if assignments[str(row["sample_id"])] == "train"
        )
        test = sorted(
            (int(row["mask_start"]), int(row["mask_stop"]))
            for row in source_rows
            if assignments[str(row["sample_id"])] == "test"
        )
        left = right = 0
        while left < len(train) and right < len(test):
            a, b = train[left], test[right]
            if a[0] < b[1] and b[0] < a[1]:
                raise ValueError(f"overlapping windows cross partitions in {source}")
            if a[1] <= b[0]:
                left += 1
            else:
                right += 1


def deterministic_subject_split(
    rows: Iterable[dict], test_size: float = 0.1, seed: int = 42
) -> dict:
    rows = list(rows)
    subjects = sorted({str(row["subject"]) for row in rows})
    assignments: dict[str, str] = {}
    for subject in subjects:
        subject_rows = [row for row in rows if str(row["subject"]) == subject]
        intervals, lookup = _global_subject_intervals(subject_rows)
        train, test = _pvi_ml_graph_split(intervals, float(test_size), int(seed))
        train_counts, test_counts = Counter(train), Counter(test)
        for interval, row in lookup.items():
            if train_counts[interval] and test_counts[interval]:
                raise RuntimeError(f"mask {interval} remained in both train and test")
            label = (
                "train" if train_counts[interval]
                else "test" if test_counts[interval]
                else "excluded"
            )
            assignments[str(row["sample_id"])] = label
    validate_split(rows, assignments)
    return {
        "schema": "pvi-gcnm-split-manifest-v1",
        "version": SPLIT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "deterministic-pvi-ml-graph-bipartite",
        "seed": int(seed),
        "test_size": float(test_size),
        "assignments": assignments,
        "counts": {
            label: sum(value == label for value in assignments.values())
            for label in sorted(SPLIT_LABELS)
        },
    }


def recover_subject_split(rows: Iterable[dict], checkpoint: Path) -> dict:
    """Translate a pvi_ml dataset checkpoint's global masks to sample IDs.

    Rows must include ``num_periods``. Session ordering follows pvi_ml's
    ``SessionName`` enumeration, which is part of the historical split contract.
    """
    import torch

    rows = list(rows)
    subjects = {str(row["subject"]) for row in rows}
    if len(subjects) != 1:
        raise ValueError("checkpoint recovery accepts exactly one subject")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = saved.get("dataset", saved)
    active_masks = {tuple(int(x) for x in mask) for mask in state.get("active_mask", [])}
    train_masks = {tuple(int(x) for x in mask) for mask in state.get("train_mask", [])}
    test_masks = {tuple(int(x) for x in mask) for mask in state.get("test_mask", [])}
    if not active_masks or not train_masks or not test_masks:
        raise ValueError(f"{checkpoint} does not contain active/train/test masks")
    if train_masks & test_masks:
        raise ValueError(f"{checkpoint} has masks in both train and test")
    if not (train_masks | test_masks) <= active_masks:
        raise ValueError(f"{checkpoint} train/test masks are not subsets of active_mask")

    assignments: dict[str, str] = {}
    translated_active: set[tuple[int, int]] = set()
    offset = 0
    by_session: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_session[(str(row["session"]), str(row["source_name"]))].append(row)
    ordered = sorted(by_session, key=lambda key: (SESSION_ORDER.get(key[0], 99), key[1]))
    for key in ordered:
        session_rows = by_session[key]
        period_counts = {int(row["num_periods"]) for row in session_rows}
        if len(period_counts) != 1:
            raise ValueError(f"inconsistent num_periods for {key[1]}")
        for row in session_rows:
            global_mask = (int(row["mask_start"]) + offset, int(row["mask_stop"]) + offset)
            translated_active.add(global_mask)
            if global_mask in train_masks:
                assignments[str(row["sample_id"])] = "train"
            elif global_mask in test_masks:
                assignments[str(row["sample_id"])] = "test"
            else:
                assignments[str(row["sample_id"])] = "excluded"
        offset += period_counts.pop()
    missing_from_source = active_masks - translated_active
    extra_in_source = translated_active - active_masks
    if missing_from_source or extra_in_source:
        raise ValueError(
            "checkpoint/source active masks differ "
            f"(checkpoint_only={len(missing_from_source)}, source_only={len(extra_in_source)})"
        )
    validate_split(rows, assignments)
    return {
        "schema": "pvi-gcnm-split-manifest-v1",
        "version": SPLIT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "recovered-pvi-ml-checkpoint",
        "source_checkpoint": str(Path(checkpoint).resolve()),
        "source_checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "assignments": assignments,
        "counts": {
            "train": sum(value == "train" for value in assignments.values()),
            "test": sum(value == "test" for value in assignments.values()),
            "excluded": sum(value == "excluded" for value in assignments.values()),
        },
    }


def rows_from_parquet(root: Path) -> list[dict]:
    import pyarrow.dataset as pads

    dataset = pads.dataset(str(Path(root) / "shards"), format="parquet")
    columns = [
        "sample_id", "subject", "session", "source_name", "mask_start", "mask_stop",
        "num_periods",
    ]
    return dataset.to_table(columns=columns).to_pylist()


def rows_from_registry(registry_path: Path) -> list[dict]:
    """Read mask05 metadata directly from immutable source HDF5 sessions."""
    import h5py

    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    rows: list[dict] = []
    records = sorted(registry["records"], key=lambda item: int(item["source_order"]))
    for record in records:
        if record.get("exclusion_reason"):
            continue
        source_path = Path(record["source_hdf5"])
        with h5py.File(source_path, "r") as handle:
            num_periods = int(np.asarray(handle["metadata/num_periods"]).item())
            masks = np.asarray(handle["masks/mask05"], dtype=np.int64)
        if masks.ndim != 2 or masks.shape[1] != 2:
            raise ValueError(f"invalid mask05 shape in {source_path}: {masks.shape}")
        masks[:, 0] -= 1  # pvi_ml converts the MATLAB-inclusive start to Python.
        if np.any(masks[:, 1] - masks[:, 0] != 5):
            raise ValueError(f"non-five-period mask05 window in {source_path}")
        for start, stop in masks:
            rows.append(
                {
                    "sample_id": stable_sample_id(
                        str(record["source_name"]), "mask05", int(start), int(stop)
                    ),
                    "subject": str(record["subject"]),
                    "session": str(record["session"]),
                    "source_name": str(record["source_name"]),
                    "mask_start": int(start),
                    "mask_stop": int(stop),
                    "num_periods": num_periods,
                }
            )
    return rows


def _checkpoint_subject(path: Path) -> str | None:
    match = re.match(r"(subject\d{3})_checkpoints(?:_.*)?\.pth$", path.name)
    return None if match is None else match.group(1)


def checkpoint_candidates(roots: Iterable[Path]) -> dict[str, list[Path]]:
    """Index historical per-subject checkpoints in caller-specified priority order."""
    output: dict[str, list[Path]] = defaultdict(list)
    suffix_priority = {"_checkpoints.pth": 0, "_checkpoints_best.pth": 1}
    for root_index, root in enumerate(roots):
        root = Path(root)
        if not root.exists():
            continue
        paths = root.rglob("subject*_checkpoints*.pth") if root.is_dir() else [root]
        ranked = []
        for path in paths:
            subject = _checkpoint_subject(path)
            if subject is None:
                continue
            rank = next(
                (value for suffix, value in suffix_priority.items() if path.name.endswith(suffix)),
                2,
            )
            ranked.append((root_index, rank, str(path), subject, path))
        for _, _, _, subject, path in sorted(ranked):
            output[subject].append(path)
    return dict(output)


def build_frozen_split_manifest(
    rows: Iterable[dict],
    checkpoint_roots: Iterable[Path] = (),
    test_size: float = 0.1,
    seed: int = 42,
) -> dict:
    """Recover compatible historical splits and deterministically fill the rest."""
    rows = list(rows)
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject"])].append(row)
    candidates = checkpoint_candidates(checkpoint_roots)
    assignments: dict[str, str] = {}
    provenance: dict[str, dict] = {}
    for subject in sorted(by_subject):
        subject_rows = by_subject[subject]
        failures = []
        recovered = None
        for checkpoint in candidates.get(subject, []):
            try:
                recovered = recover_subject_split(subject_rows, checkpoint)
                break
            except (KeyError, RuntimeError, TypeError, ValueError) as error:
                failures.append({"checkpoint": str(checkpoint.resolve()), "reason": str(error)})
        if recovered is not None:
            assignments.update(recovered["assignments"])
            provenance[subject] = {
                "method": recovered["method"],
                "source_checkpoint": recovered["source_checkpoint"],
                "source_checkpoint_sha256": recovered["source_checkpoint_sha256"],
                "counts": recovered["counts"],
                "rejected_candidates": failures,
            }
        else:
            fallback = deterministic_subject_split(subject_rows, test_size=test_size, seed=seed)
            assignments.update(fallback["assignments"])
            provenance[subject] = {
                "method": fallback["method"],
                "seed": int(seed),
                "test_size": float(test_size),
                "counts": fallback["counts"],
                "rejected_candidates": failures,
            }
    validate_split(rows, assignments)
    return {
        "schema": "pvi-gcnm-split-manifest-v1",
        "version": SPLIT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "historical-when-compatible-otherwise-deterministic",
        "mask_key": "mask05",
        "seed": int(seed),
        "test_size": float(test_size),
        "subject_count": len(by_subject),
        "assignments": assignments,
        "subjects": provenance,
        "counts": {
            label: sum(value == label for value in assignments.values())
            for label in sorted(SPLIT_LABELS)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parquet-root", type=Path)
    source.add_argument("--registry", type=Path)
    parser.add_argument(
        "--checkpoint-root", type=Path, action="append", default=[],
        help="checkpoint file/directory; repeat in preferred search order",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--subject",
        action="append",
        default=[],
        help="limit the frozen manifest to this subject; repeat as needed",
    )
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = (
        rows_from_parquet(args.parquet_root)
        if args.parquet_root is not None
        else rows_from_registry(args.registry)
    )
    if args.subject:
        requested = {str(subject).lower() for subject in args.subject}
        available = {str(row["subject"]).lower() for row in rows}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"requested subjects are absent from the source: {missing}")
        rows = [row for row in rows if str(row["subject"]).lower() in requested]
    if not rows:
        raise ValueError("split source selected zero mask05 rows")
    manifest = build_frozen_split_manifest(
        rows, args.checkpoint_root, args.test_size, args.seed
    )
    if args.registry is not None:
        manifest["source"] = {
            "kind": "mesh-registry-source-hdf5",
            "path": str(args.registry.resolve()),
            "sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest(),
            "row_count": len(rows),
            "subjects": sorted({str(row["subject"]) for row in rows}),
        }
    else:
        parquet_manifest = args.parquet_root / "manifest.json"
        manifest["source"] = {
            "kind": "gcnm-parquet",
            "path": str(args.parquet_root.resolve()),
            "manifest_sha256": (
                hashlib.sha256(parquet_manifest.read_bytes()).hexdigest()
                if parquet_manifest.is_file()
                else None
            ),
            "row_count": len(rows),
            "subjects": sorted({str(row["subject"]) for row in rows}),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
