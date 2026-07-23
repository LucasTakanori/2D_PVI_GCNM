#!/usr/bin/env python3
"""Deterministically merge independently generated full/HP/LP anatomy shards."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


COMPONENTS = ("full", "hp", "lp")
SPLITS = ("train", "validation", "test")


def _record_path(root: Path, component: str, split: str) -> Path:
    suffix = "anatomies" if component == "full" else "anatomy"
    return root / component / f"{split}_{suffix}.json"


def _compatible(reference: dict, candidate: dict, path: Path) -> None:
    for key in (
        "schema",
        "ring",
        "components",
        "stored_conductivity_contract",
        "stored_voltage_contract",
        "temporal_processing",
        "augmentation",
        "noise",
        "linearization_reference",
        "finger_config_sha256",
        "config_sha256",
        "mesh_inverse_sha256",
        "mesh_forward_sha256",
        "mapping_40_sha256",
        "split_modes",
    ):
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"shard metadata mismatch for {key}: {path}")


def merge(shards: list[Path], output_root: Path) -> dict:
    if output_root.exists():
        raise FileExistsError(f"immutable merged root exists: {output_root}")
    if len(shards) < 2:
        raise ValueError("at least two independent shards are required")
    metadata = []
    for root in shards:
        if (root / "_INCOMPLETE").exists():
            raise RuntimeError(f"shard is incomplete: {root}")
        metadata.append(json.loads((root / "metadata.json").read_text(encoding="utf-8")))
    for root, candidate in zip(shards[1:], metadata[1:]):
        _compatible(metadata[0], candidate, root)

    # Map (shard, split, local anatomy) to one globally unique contiguous ID.
    id_maps: dict[tuple[int, str], dict[int, int]] = {}
    next_id = 0
    for split in SPLITS:
        for shard_index, root in enumerate(shards):
            with np.load(root / "full" / f"{split}.npz") as source:
                local_ids = sorted(int(value) for value in np.unique(source["anatomy_id"]))
            id_maps[(shard_index, split)] = {
                old: next_id + offset for offset, old in enumerate(local_ids)
            }
            next_id += len(local_ids)

    output_root.mkdir(parents=True, exist_ok=False)
    incomplete = output_root / "_INCOMPLETE"
    incomplete.write_text("deterministic shard merge in progress\n", encoding="utf-8")
    split_ids: dict[str, list[int]] = {}
    counts: dict[str, dict] = {}
    for component in COMPONENTS:
        component_root = output_root / component
        component_root.mkdir()
        for split in SPLITS:
            pieces: dict[str, list[np.ndarray]] = {}
            records = []
            for shard_index, root in enumerate(shards):
                mapping = id_maps[(shard_index, split)]
                with np.load(root / component / f"{split}.npz") as source:
                    for key in source.files:
                        values = np.asarray(source[key]).copy()
                        if key == "anatomy_id":
                            values = np.asarray([mapping[int(value)] for value in values], dtype=np.int32)
                        elif key == "beat_id":
                            # Every anatomy contains exactly five retained beats.
                            local_anatomy = np.asarray(source["anatomy_id"])
                            local_beat = np.asarray(source["beat_id"])
                            values = np.asarray(
                                [mapping[int(a)] * 5 + int(b) % 5 for a, b in zip(local_anatomy, local_beat)],
                                dtype=np.int32,
                            )
                        pieces.setdefault(key, []).append(values)
                shard_records = json.loads(
                    _record_path(root, component, split).read_text(encoding="utf-8")
                )
                for record in shard_records:
                    record = copy.deepcopy(record)
                    old_anatomy = int(record["anatomy_id"])
                    record["anatomy_id"] = mapping[old_anatomy]
                    if "beat_id" in record:
                        record["beat_id"] = mapping[old_anatomy] * 5 + int(record["beat_id"]) % 5
                    records.append(record)
            merged = {key: np.concatenate(values, axis=0) for key, values in pieces.items()}
            order = np.lexsort(
                (merged["sample_index"], merged["beat_id"], merged["anatomy_id"])
            )
            merged = {key: values[order] for key, values in merged.items()}
            np.savez_compressed(component_root / f"{split}.npz", **merged)
            if component == "full":
                records.sort(key=lambda item: int(item["anatomy_id"]))
            else:
                records.sort(
                    key=lambda item: (
                        int(item["anatomy_id"]),
                        int(item["beat_id"]),
                        int(item["sample_index"]),
                    )
                )
            _record_path(output_root, component, split).write_text(
                json.dumps(records), encoding="utf-8"
            )
            unique_ids = sorted(int(value) for value in np.unique(merged["anatomy_id"]))
            if component == "full":
                split_ids[split] = unique_ids
            elif unique_ids != split_ids[split]:
                raise RuntimeError(f"component anatomy identities differ in {split}")
            counts.setdefault(split, {})[component] = {
                "samples": int(len(merged["sigma"])),
                "retained_beats": int(len(np.unique(merged["beat_id"]))),
                "anatomies": len(unique_ids),
                "negative_target_fraction": float(np.mean(merged["sigma"] < 0)),
                "voltage_rms": float(np.sqrt(np.mean(merged["V"] ** 2))),
            }

    merged_metadata = copy.deepcopy(metadata[0])
    merged_metadata["created_utc"] = datetime.now(timezone.utc).isoformat()
    merged_metadata["seed"] = None
    merged_metadata["shard_merge"] = {
        "method": "independent process shards, deterministic identity remap and sort",
        "source_roots": [str(path.resolve()) for path in shards],
        "source_seeds": [item.get("seed") for item in metadata],
        "shards": len(shards),
    }
    merged_metadata["split_anatomy_ids"] = split_ids
    merged_metadata["counts"] = {
        "virtual_anatomies": sum(len(value) for value in split_ids.values()),
        "retained_beats": sum(len(value) for value in split_ids.values()) * 5,
        "retained_samples": sum(len(value) for value in split_ids.values()) * 250,
        "splits": counts,
    }
    comparisons = []
    verified = True
    for shard_index, item in enumerate(metadata):
        gate = item.get("linearized_acceleration_gate", {})
        verified = verified and bool(gate.get("verified", False))
        for comparison in gate.get("comparisons", []):
            comparisons.append({"shard": shard_index, **comparison})
    merged_metadata["linearized_acceleration_gate"] = {
        "verified": verified,
        "comparisons": comparisons,
        "scope": "every independently generated shard",
    }
    (output_root / "metadata.json").write_text(
        json.dumps(merged_metadata, indent=2) + "\n", encoding="utf-8"
    )
    incomplete.unlink()
    return {
        "schema": "pvi-gcnm-sharded-dataset-merge-v1",
        "output_root": str(output_root.resolve()),
        "shards": len(shards),
        "anatomy_counts": {key: len(value) for key, value in split_ids.items()},
        "frames": {key: len(value) * 250 for key, value in split_ids.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.shard, args.output_root), indent=2))


if __name__ == "__main__":
    main()
