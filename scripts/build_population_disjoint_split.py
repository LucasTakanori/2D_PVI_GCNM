#!/usr/bin/env python3
"""Freeze or recover a subject-disjoint split over PVI mask05 identities.

The overlap-filtered active samples and their original global order come from
the already validated population-within split manifest. Subject selection can
either use a literal seeded implementation of PVI-ML's disjoint loop or the
held-out subjects stored in an archived PVI checkpoint. The latter can restore
the full legacy identity cohort, including rows excluded by a newer split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path


SOURCE_SCHEMA = "pvi-gcnm-population-within-split-v1"
OUTPUT_SCHEMA = "pvi-gcnm-population-disjoint-split-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_test_subjects(
    identities: list[dict],
    *,
    test_size: float,
    seed: int,
    requested_subjects: list[str] | None = None,
    all_identities_active: bool = False,
) -> tuple[list[str], OrderedDict[str, int]]:
    """Apply PVI's subject-order shuffle and cumulative 1.2x ratio guard."""

    active_counts: OrderedDict[str, int] = OrderedDict()
    for identity in identities:
        if not all_identities_active and identity["assignment"] == "excluded":
            continue
        subject = str(identity["subject"])
        active_counts.setdefault(subject, 0)
        active_counts[subject] += 1
    if len(active_counts) < 5:
        raise ValueError("PVI disjoint splitting requires at least five subjects")

    if requested_subjects:
        requested = list(dict.fromkeys(map(str, requested_subjects)))
        unknown = set(requested) - set(active_counts)
        if unknown:
            raise ValueError(f"unknown requested test subjects: {sorted(unknown)}")
        if not requested or len(requested) == len(active_counts):
            raise ValueError("requested test subjects must leave non-empty train and test sets")
        return requested, active_counts

    shuffled = list(active_counts)
    random.Random(int(seed)).shuffle(shuffled)
    selected: list[str] = []
    num_test = 0
    total = sum(active_counts.values())
    for subject in shuffled:
        candidate_count = active_counts[subject]
        projected_ratio = (num_test + candidate_count) / total
        if num_test > 0 and projected_ratio > 1.2 * float(test_size):
            break
        selected.append(subject)
        num_test += candidate_count
    if not selected or len(selected) == len(active_counts):
        raise RuntimeError("PVI disjoint selection produced an empty partition")
    return selected, active_counts


def build_disjoint_manifest(
    *,
    source_manifest_path: Path,
    output_path: Path,
    seed: int,
    test_size: float = 0.1,
    requested_subjects: list[str] | None = None,
    all_identities_active: bool = False,
    reference: dict | None = None,
) -> dict:
    source_manifest_path = Path(source_manifest_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"immutable disjoint split manifest exists: {output_path}")
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"expected source split schema {SOURCE_SCHEMA}")
    identities = source.get("identities", [])
    if len(identities) != int(source.get("row_count", -1)):
        raise ValueError("source split identity count differs from row_count")

    test_subjects, active_counts = select_test_subjects(
        identities,
        test_size=test_size,
        seed=seed,
        requested_subjects=requested_subjects,
        all_identities_active=all_identities_active,
    )
    test_set = set(test_subjects)
    assignments: dict[str, str] = {}
    output_identities: list[dict] = []
    subject_counts: dict[str, Counter] = {}
    counts = Counter()
    for identity in identities:
        subject = str(identity["subject"])
        if not all_identities_active and identity["assignment"] == "excluded":
            assignment = "excluded"
        else:
            assignment = "test" if subject in test_set else "train"
        sample_id = str(identity["sample_id"])
        if sample_id in assignments:
            raise ValueError(f"duplicate stable sample identity: {sample_id}")
        assignments[sample_id] = assignment
        output_identities.append({**identity, "assignment": assignment})
        subject_counts.setdefault(subject, Counter())[assignment] += 1
        counts[assignment] += 1

    for subject, values in subject_counts.items():
        if values["train"] and values["test"]:
            raise RuntimeError(f"subject leakage across disjoint split: {subject}")
    train_subjects = [subject for subject in active_counts if subject not in test_set]
    active_total = counts["train"] + counts["test"]
    payload = {
        "schema": OUTPUT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": (
            "full frozen identity cohort plus archived PVI checkpoint holdout"
            if all_identities_active and requested_subjects
            else "frozen PVI active mask plus literal PviLazyDataset disjoint selection"
        ),
        "branch": source.get("branch", "main"),
        "mask_key": source.get("mask_key", "mask05"),
        "split_mode": "disjoint",
        "test_size": float(test_size),
        "seed": int(seed),
        "selection": (
            "explicit-held-out-subjects"
            if requested_subjects
            else "seeded-literal-pvi-disjoint"
        ),
        "active_policy": (
            "all-frozen-identities"
            if all_identities_active
            else "source-manifest-active-only"
        ),
        "subject_count": len(active_counts),
        "source_count": int(source.get("source_count", 0)),
        "row_count": len(output_identities),
        "counts": {
            key: int(counts[key]) for key in ("train", "test", "excluded")
        },
        "active_test_ratio": float(counts["test"] / active_total),
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "subjects": {
            subject: {
                key: int(values[key]) for key in ("train", "test", "excluded")
            }
            for subject, values in sorted(subject_counts.items())
        },
        "source_split_manifest": str(source_manifest_path),
        "source_split_manifest_sha256": sha256_file(source_manifest_path),
        "coordinate_root": source.get("coordinate_root"),
        "coordinate_manifest_sha256": source.get("coordinate_manifest_sha256"),
        "registry": source.get("registry"),
        "registry_sha256": source.get("registry_sha256"),
        "pvi_ml_root": source.get("pvi_ml_root"),
        "reference": reference,
        "assignments": assignments,
        "identities": output_identities,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return payload


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=repo / "data/manifests/pw_population_within_mask05_seed42_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument(
        "--test-subject",
        action="append",
        dest="test_subjects",
        help="explicit held-out subject; repeat to reproduce an archived PD split",
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        help="legacy PVI checkpoint whose dataset.test_subgroups define the holdout",
    )
    parser.add_argument(
        "--all-identities-active",
        action="store_true",
        help="reproduce legacy PD runs that used every frozen mask05 identity",
    )
    args = parser.parse_args()
    reference = None
    requested_subjects = args.test_subjects
    if args.reference_checkpoint is not None:
        if requested_subjects:
            parser.error("use either --reference-checkpoint or --test-subject, not both")
        import torch

        checkpoint_path = args.reference_checkpoint.resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        dataset_state = checkpoint.get("dataset", {})
        requested_subjects = list(dataset_state.get("test_subgroups", []))
        if not requested_subjects:
            raise ValueError(
                f"reference checkpoint does not store test_subgroups: {checkpoint_path}"
            )
        reference = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "train_rows": len(dataset_state.get("train_mask", [])),
            "test_rows": len(dataset_state.get("test_mask", [])),
            "active_rows": len(dataset_state.get("active_mask", [])),
            "test_subgroups": requested_subjects,
        }
    manifest = build_disjoint_manifest(
        source_manifest_path=args.source_manifest,
        output_path=args.output,
        seed=args.seed,
        test_size=args.test_size,
        requested_subjects=requested_subjects,
        all_identities_active=args.all_identities_active,
        reference=reference,
    )
    if reference is not None:
        for split_name in ("train", "test"):
            expected = int(reference[f"{split_name}_rows"])
            observed = int(manifest["counts"][split_name])
            if observed != expected:
                raise RuntimeError(
                    f"legacy {split_name} count mismatch: {observed} != {expected}"
                )
        if sum(manifest["counts"].values()) != int(reference["active_rows"]):
            raise RuntimeError("legacy active-row count differs from generated manifest")
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(args.output.resolve()),
                "counts": manifest["counts"],
                "test_subjects": manifest["test_subjects"],
                "active_test_ratio": manifest["active_test_ratio"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
