#!/usr/bin/env python3
"""Export archived Newton images or BioZ windows to immutable Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gcnm_pvi.pvi_splits import stable_sample_id


SCHEMA = "pvi-reference-parquet-v1"
PERIOD_LENGTH = 50
MASK_KEY = "mask05"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed(values: list[np.ndarray], size: int) -> pa.FixedSizeListArray:
    matrix = np.stack(values).astype(np.float32, copy=False).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(matrix, type=pa.float32()), size)


def _write_shard(rows: list[dict], path: Path, tensor_shapes: dict[str, list[int]]) -> None:
    if not rows:
        return
    hp_size = int(np.prod(tensor_shapes["pviHP"]))
    lp_size = int(np.prod(tensor_shapes["pviLP"]))
    bp_size = int(np.prod(tensor_shapes["bp_waveform"]))
    stats_size = int(np.prod(tensor_shapes["stats"]))
    table = pa.table(
        {
            "pviHP": _fixed([row["pviHP"] for row in rows], hp_size),
            "pviLP": _fixed([row["pviLP"] for row in rows], lp_size),
            "bp_waveform": _fixed([row["bp_waveform"] for row in rows], bp_size),
            "stats": _fixed([row["stats"] for row in rows], stats_size),
            "subject": pa.array([row["subject"] for row in rows], pa.string()),
            "session": pa.array([row["session"] for row in rows], pa.string()),
            "source_name": pa.array([row["source_name"] for row in rows], pa.string()),
            "sample_id": pa.array([row["sample_id"] for row in rows], pa.string()),
            "source_order": pa.array([row["source_order"] for row in rows], pa.int32()),
            "mask_start": pa.array([row["mask_start"] for row in rows], pa.int32()),
            "mask_stop": pa.array([row["mask_stop"] for row in rows], pa.int32()),
            "num_periods": pa.array([row["num_periods"] for row in rows], pa.int16()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", use_dictionary=False)


def _source_hashes(manifest_path: Path | None) -> dict[str, str]:
    if manifest_path is None:
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        str(Path(row["source_hdf5"]).resolve()): row["source_hdf5_sha256"]
        for row in manifest.get("source_sessions", [])
    }


def _tensor_shapes(input_mode: str) -> dict[str, list[int]]:
    input_shape = [1, 40, 40, 250] if input_mode == "img" else [64, 250]
    return {
        "pviHP": input_shape,
        "pviLP": input_shape,
        "bp_waveform": [50],
        "stats": [2, 5],
    }


def _window(handle: h5py.File, start: int, stop: int, input_mode: str) -> dict[str, np.ndarray]:
    frame_start, frame_stop = start * PERIOD_LENGTH, stop * PERIOD_LENGTH
    if stop - start != 5:
        raise ValueError(f"mask05 window must span five periods, found [{start}, {stop})")

    def component(name: str) -> np.ndarray:
        group = handle[f"data/{name}"]
        if input_mode == "img":
            # pvi_ml adds a leading channel dimension to archived Newton images.
            return np.asarray(group["img"][:, :, frame_start:frame_stop], dtype=np.float32)[None]
        # Preserve pvi_ml's literal impedance contract: reactance rows followed by
        # resistance rows (see utils.h5io.format_tensors_pvi).
        reactance = np.asarray(group["reactance"][:, frame_start:frame_stop], dtype=np.float32)
        resistance = np.asarray(group["resistance"][:, frame_start:frame_stop], dtype=np.float32)
        return np.concatenate((reactance, resistance), axis=0)

    bp_start = (stop - 1) * PERIOD_LENGTH
    bp = np.asarray(handle["data/bp/signal"][0, bp_start:frame_stop], dtype=np.float32)
    stats = np.stack(
        [
            np.asarray(handle["stats/pviHP/duration"][0, start:stop], dtype=np.float32),
            np.asarray(handle["stats/pviHP/tMax"][0, start:stop], dtype=np.float32),
        ]
    )
    return {"pviHP": component("pviHP"), "pviLP": component("pviLP"), "bp_waveform": bp, "stats": stats}


def export_reference_parquet(
    *,
    registry_path: Path,
    output_root: Path,
    subjects: list[str],
    input_mode: str,
    shard_rows: int,
    source_hash_manifest: Path | None,
    ring: str | None = None,
) -> dict:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"immutable output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    incomplete = output_root / "_INCOMPLETE"
    incomplete.write_text("reference export in progress\n", encoding="utf-8")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    wanted = {subject.lower() for subject in subjects}
    records = [
        row for row in registry["records"]
        if row["subject"].lower() in wanted and not row.get("exclusion_reason")
        and (ring is None or row["ring"] == ring)
    ]
    records.sort(key=lambda row: int(row["source_order"]))
    found = {row["subject"].lower() for row in records}
    if found != wanted:
        raise ValueError(f"registry subjects mismatch: requested={sorted(wanted)}, found={sorted(found)}")

    tensor_shapes = _tensor_shapes(input_mode)
    known_hashes = _source_hashes(source_hash_manifest)
    rows: list[dict] = []
    sample_ids: set[str] = set()
    source_reports: list[dict] = []
    shard_paths: list[str] = []
    shard_id = 0
    total_rows = 0

    for record in records:
        source = Path(record["source_hdf5"]).resolve()
        source_key = str(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        with h5py.File(source, "r") as handle:
            masks = np.asarray(handle[f"masks/{MASK_KEY}"], dtype=np.int64)
            masks[:, 0] -= 1
            source_rows = 0
            for start_raw, stop_raw in masks:
                start, stop = int(start_raw), int(stop_raw)
                sample_id = stable_sample_id(record["source_name"], MASK_KEY, start, stop)
                if sample_id in sample_ids:
                    raise ValueError(f"duplicate sample_id {sample_id}")
                sample_ids.add(sample_id)
                tensors = _window(handle, start, stop, input_mode)
                rows.append(
                    {
                        **tensors,
                        "subject": record["subject"],
                        "session": record["session"],
                        "source_name": record["source_name"],
                        "sample_id": sample_id,
                        "source_order": int(record["source_order"]),
                        "mask_start": start,
                        "mask_stop": stop,
                        "num_periods": stop - start,
                    }
                )
                source_rows += 1
                total_rows += 1
                if len(rows) >= shard_rows:
                    shard = output_root / "shards" / f"part-{shard_id:05d}.parquet"
                    _write_shard(rows, shard, tensor_shapes)
                    shard_paths.append(str(shard.relative_to(output_root)))
                    rows.clear()
                    shard_id += 1
        source_reports.append(
            {
                "subject": record["subject"],
                "session": record["session"],
                "source_name": record["source_name"],
                "source_order": int(record["source_order"]),
                "source_hdf5": source_key,
                "source_hdf5_sha256": known_hashes.get(source_key) or _sha256(source),
                "rows": source_rows,
            }
        )

    if rows:
        shard = output_root / "shards" / f"part-{shard_id:05d}.parquet"
        _write_shard(rows, shard, tensor_shapes)
        shard_paths.append(str(shard.relative_to(output_root)))

    manifest = {
        "schema": SCHEMA,
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "representation": "archived-newton-pvi" if input_mode == "img" else "archived-bioz-impedance",
        "input_mode": input_mode,
        "bioz_channel_order": "reactance_then_resistance" if input_mode == "bioz" else None,
        "mask_key": MASK_KEY,
        "period_length": PERIOD_LENGTH,
        "tensor_shapes": tensor_shapes,
        "row_count": total_rows,
        "session_count": len(records),
        "subjects": sorted(found),
        "shard_rows": shard_rows,
        "parquet_shards": shard_paths,
        "registry": str(registry_path.resolve()),
        "registry_sha256": _sha256(registry_path),
        "source_sessions": source_reports,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    incomplete.unlink()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument(
        "--all-subjects",
        action="store_true",
        help="export every included subject in the authoritative registry",
    )
    parser.add_argument(
        "--ring",
        choices=[
            "US060", "US065", "US070", "US075", "US080", "US085", "US090",
            "US095", "US100", "US105", "US110", "US115", "US120", "US125",
            "US130",
        ],
        help="limit --all-subjects to subjects registered to one ring",
    )
    parser.add_argument("--input-mode", choices=["img", "bioz"], required=True)
    parser.add_argument("--shard-rows", type=int)
    parser.add_argument("--source-hash-manifest", type=Path)
    args = parser.parse_args()
    if args.ring and not args.all_subjects:
        parser.error("--ring requires --all-subjects")
    if args.all_subjects and args.subject:
        parser.error("use either --all-subjects or repeated --subject, not both")
    if not args.all_subjects and not args.subject:
        parser.error("at least one --subject or --all-subjects is required")
    subjects = args.subject
    if args.all_subjects:
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
        subjects = sorted(
            {
                str(record["subject"])
                for record in registry["records"]
                if not record.get("exclusion_reason")
                and (args.ring is None or record["ring"] == args.ring)
            }
        )
    shard_rows = args.shard_rows or (32 if args.input_mode == "img" else 2048)
    report = export_reference_parquet(
        registry_path=args.registry,
        output_root=args.output_root,
        subjects=subjects,
        input_mode=args.input_mode,
        shard_rows=shard_rows,
        source_hash_manifest=args.source_hash_manifest,
        ring=args.ring,
    )
    print(json.dumps({"root": str(args.output_root.resolve()), "rows": report["row_count"], "shards": len(report["parquet_shards"])}, indent=2))


if __name__ == "__main__":
    main()
