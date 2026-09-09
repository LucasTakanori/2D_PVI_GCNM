#!/usr/bin/env python3
"""Export ten beats of the exact six-channel CRT image input for every subject.

The export is intentionally split into prepare, export-subject, and finalize
steps so subjects can be rendered in parallel as a Slurm array.  PNG files are
lossless visualizations of the native 40 x 40 reconstruction grids.  A
float32 NumPy tensor is also written for every beat so the original numerical
values remain available without color quantization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, PngImagePlugin


PERIOD_LENGTH = 50
WINDOW_PERIODS = 5
WINDOWS_PER_SUBJECT = 2
BEATS_PER_SUBJECT = WINDOW_PERIODS * WINDOWS_PER_SUBJECT
NATIVE_SHAPE = (40, 40)
CHANNEL_NAMES = (
    "newton_hp",
    "d_newton_lp_dt",
    "d2_newton_lp_dt2",
    "gcnm_s1",
    "gcnm_s2",
    "d_gcnm_s2_dt",
)
PNG_FILENAMES = tuple(
    f"{index:02d}_{name}.png" for index, name in enumerate(CHANNEL_NAMES, 1)
)
METADATA_COLUMNS = (
    "subject",
    "session",
    "source_name",
    "sample_id",
    "source_order",
    "mask_start",
    "mask_stop",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _subject_number(subject: str) -> int:
    return int(subject.removeprefix("subject"))


def _registry_records(registry_path: Path) -> tuple[dict, list[dict]]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    records = [
        row for row in registry["records"] if not row.get("exclusion_reason")
    ]
    records.sort(
        key=lambda row: (_subject_number(str(row["subject"])), int(row["source_order"]))
    )
    return registry, records


def _scan_parquet_metadata(shards_root: Path) -> dict[str, list[dict]]:
    """Scan only small metadata columns and retain exact row-group locations."""

    by_subject: dict[str, list[dict]] = {}
    shard_paths = sorted(shards_root.glob("*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"no Parquet shards found under {shards_root}")
    for shard_path in shard_paths:
        parquet = pq.ParquetFile(shard_path)
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(
                row_group, columns=list(METADATA_COLUMNS), use_threads=False
            )
            for row_index, row in enumerate(table.to_pylist()):
                row["shard"] = shard_path.name
                row["row_group"] = row_group
                row["row_index"] = row_index
                by_subject.setdefault(str(row["subject"]).lower(), []).append(row)
    return by_subject


def _select_windows(rows: list[dict]) -> list[dict]:
    """Choose the first two deterministic non-overlapping five-beat windows."""

    rows = sorted(
        rows,
        key=lambda row: (
            int(row["source_order"]),
            int(row["mask_start"]),
            str(row["sample_id"]),
        ),
    )
    selected: list[dict] = []
    for row in rows:
        interval = (int(row["mask_start"]), int(row["mask_stop"]))
        if interval[1] - interval[0] != WINDOW_PERIODS:
            continue
        overlaps = any(
            row["source_name"] == previous["source_name"]
            and interval[0] < int(previous["mask_stop"])
            and int(previous["mask_start"]) < interval[1]
            for previous in selected
        )
        if not overlaps:
            selected.append(row)
        if len(selected) == WINDOWS_PER_SUBJECT:
            return selected
    raise ValueError("subject does not have two non-overlapping five-beat windows")


def prepare(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    coordinate_root = args.coordinate_root.resolve()
    registry_path = args.registry.resolve()
    manifest_path = coordinate_root / "manifest.json"
    shards_root = coordinate_root / "shards"
    coordinate_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry, records = _registry_records(registry_path)
    ring_by_subject: dict[str, str] = {}
    source_by_key: dict[tuple[str, str], dict] = {}
    for record in records:
        subject = str(record["subject"]).lower()
        ring = str(record["ring"])
        if subject in ring_by_subject and ring_by_subject[subject] != ring:
            raise ValueError(f"{subject} is assigned to multiple rings")
        ring_by_subject[subject] = ring
        source_by_key[(subject, str(record["source_name"]))] = record

    parquet_rows = _scan_parquet_metadata(shards_root)
    subjects = sorted(ring_by_subject, key=_subject_number)
    if len(subjects) != 91:
        raise ValueError(f"expected 91 subjects, found {len(subjects)}")
    selections: list[dict] = []
    for subject in subjects:
        if subject not in parquet_rows:
            raise ValueError(f"coordinate Parquet contains no rows for {subject}")
        windows = _select_windows(parquet_rows[subject])
        serialized_windows: list[dict] = []
        for window in windows:
            key = (subject, str(window["source_name"]))
            if key not in source_by_key:
                raise ValueError(f"registry source missing for {key}")
            record = source_by_key[key]
            serialized_windows.append(
                {
                    **window,
                    "source_hdf5": str(Path(record["source_hdf5"]).resolve()),
                }
            )
        selections.append(
            {
                "subject": subject,
                "mesh": ring_by_subject[subject],
                "windows": serialized_windows,
            }
        )

    output_root.mkdir(parents=True)
    (output_root / "_INCOMPLETE").write_text(_utc_now() + "\n", encoding="utf-8")
    selection_manifest = {
        "schema": "pvi-six-channel-ten-beat-selection-v1",
        "created_utc": _utc_now(),
        "coordinate_root": str(coordinate_root),
        "coordinate_manifest": str(manifest_path.resolve()),
        "coordinate_manifest_sha256": _sha256(manifest_path),
        "registry": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "registry_schema": registry.get("schema"),
        "gcnm_hidden_channels": 64,
        "subjects": selections,
        "subject_count": len(selections),
        "beats_per_subject": BEATS_PER_SUBJECT,
        "samples_per_beat": PERIOD_LENGTH,
        "channel_count": len(CHANNEL_NAMES),
        "channel_order": list(CHANNEL_NAMES),
        "png_filenames": list(PNG_FILENAMES),
        "png_encoding": {
            "format": "lossless RGB PNG",
            "shape": list(NATIVE_SHAPE),
            "resampling": "none",
            "color_map": "signed blue-white-red",
            "scale": "symmetric per subject and channel over all selected frames",
        },
        "raw_array": {
            "filename": "channels_float32.npy",
            "dtype": "float32",
            "shape": [PERIOD_LENGTH, len(CHANNEL_NAMES), *NATIVE_SHAPE],
            "axis_order": ["sample", "channel", "row", "column"],
        },
        "expected_png_count": len(selections)
        * BEATS_PER_SUBJECT
        * PERIOD_LENGTH
        * len(CHANNEL_NAMES),
    }
    _write_json(output_root / "selection_manifest.json", selection_manifest)
    readme = f"""# Six-channel PVI reconstruction image export

This export contains the exact six image channels supplied to the fused CRT
training pipeline for all {len(selections)} patients.  It uses the completed
64-hidden-channel ring-specific GCNM rollout.

The directory hierarchy is `mesh_<ring>/<patient>/beat_###/sample_###/`.
Every sample directory contains these six separate lossless PNG images:

1. Newton high-pass reconstruction
2. First temporal derivative of the Newton low-pass reconstruction
3. Second temporal derivative of the Newton low-pass reconstruction
4. GCNM stage-1 reconstruction
5. GCNM stage-2 reconstruction
6. First temporal derivative of the GCNM stage-2 reconstruction

Each patient has 10 non-overlapping beats and every beat has 50 samples.  PNGs
retain the native 40 x 40 reconstruction grid without interpolation or lossy
compression.  Their signed blue-white-red scale is fixed for a given
patient/channel across all 500 frames, so time points are directly comparable.
The scale is recorded in each patient manifest and embedded in each PNG.
Pixels outside the finite-element reconstruction domain are NaN in the source
arrays and are displayed with the neutral center color in the PNG files.

For exact scientific reuse, every beat also contains `channels_float32.npy`
with shape `(50, 6, 40, 40)`.  These arrays contain the original float32 values
before visualization scaling or color quantization.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    print(f"prepared {output_root}")
    print(f"subjects={len(selections)} expected_pngs={selection_manifest['expected_png_count']}")


def _central_difference(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    result[..., 1:-1] = (values[..., 2:] - values[..., :-2]) / np.float32(2.0)
    return result


def _read_coordinate_window(
    coordinate_root: Path, window: dict
) -> tuple[np.ndarray, np.ndarray]:
    shard_path = coordinate_root / "shards" / str(window["shard"])
    table = pq.ParquetFile(shard_path).read_row_group(
        int(window["row_group"]), columns=["sample_id", "s1", "s2"]
    )
    row_index = int(window["row_index"])
    sample_id = table["sample_id"][row_index].as_py()
    if sample_id != window["sample_id"]:
        raise ValueError(
            f"Parquet row identity changed: {sample_id} != {window['sample_id']}"
        )
    arrays: list[np.ndarray] = []
    for name in ("s1", "s2"):
        values = np.asarray(table[name][row_index].values, dtype=np.float32)
        arrays.append(values.reshape(1, 40, 40, 250)[0])
    return arrays[0], arrays[1]


def _read_newton_window(window: dict) -> tuple[np.ndarray, np.ndarray]:
    start = int(window["mask_start"]) * PERIOD_LENGTH
    stop = int(window["mask_stop"]) * PERIOD_LENGTH
    with h5py.File(window["source_hdf5"], "r") as handle:
        hp = np.asarray(handle["data/pviHP/img"][:, :, start:stop], dtype=np.float32)
        lp = np.asarray(handle["data/pviLP/img"][:, :, start:stop], dtype=np.float32)
    expected = (*NATIVE_SHAPE, WINDOW_PERIODS * PERIOD_LENGTH)
    if hp.shape != expected or lp.shape != expected:
        raise ValueError(f"invalid Newton window shapes: hp={hp.shape}, lp={lp.shape}")
    return hp, lp


def _build_channels(
    coordinate_root: Path, windows: list[dict]
) -> tuple[np.ndarray, list[dict]]:
    channel_windows: list[np.ndarray] = []
    beat_sources: list[dict] = []
    for window_number, window in enumerate(windows, 1):
        s1, s2 = _read_coordinate_window(coordinate_root, window)
        newton_hp, newton_lp = _read_newton_window(window)
        channels = np.stack(
            (
                newton_hp,
                _central_difference(newton_lp),
                _central_difference(_central_difference(newton_lp)),
                s1,
                s2,
                _central_difference(s2),
            ),
            axis=0,
        )
        if channels.shape != (len(CHANNEL_NAMES), 40, 40, 250):
            raise ValueError(f"unexpected channel shape {channels.shape}")
        channel_windows.append(channels)
        for local_beat in range(WINDOW_PERIODS):
            beat_sources.append(
                {
                    "window_number": window_number,
                    "sample_id": window["sample_id"],
                    "source_name": window["source_name"],
                    "session": window["session"],
                    "source_period_index_zero_based": int(window["mask_start"])
                    + local_beat,
                }
            )
    return np.concatenate(channel_windows, axis=-1), beat_sources


def _color_lut() -> np.ndarray:
    anchor_x = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    anchor_rgb = np.asarray(
        [
            [5, 48, 97],
            [69, 117, 180],
            [247, 247, 247],
            [214, 96, 77],
            [103, 0, 31],
        ],
        dtype=np.float64,
    )
    x = np.linspace(0.0, 1.0, 256)
    return np.stack(
        [np.interp(x, anchor_x, anchor_rgb[:, channel]) for channel in range(3)],
        axis=-1,
    ).round().astype(np.uint8)


COLOR_LUT = _color_lut()


def _write_png(
    path: Path,
    values: np.ndarray,
    scale: float,
    metadata_values: dict[str, str],
) -> None:
    finite = np.isfinite(values)
    normalized = np.zeros_like(values, dtype=np.float32)
    normalized[finite] = np.clip(
        values[finite] / np.float32(scale), -1.0, 1.0
    )
    indices = np.rint((normalized + 1.0) * 127.5).astype(np.uint8)
    image = Image.fromarray(COLOR_LUT[indices], mode="RGB")
    metadata = PngImagePlugin.PngInfo()
    for key, value in metadata_values.items():
        metadata.add_text(key, value)
    image.save(path, format="PNG", compress_level=9, pnginfo=metadata)


def _validate_subject(subject_root: Path) -> dict:
    png_paths = list(subject_root.glob("beat_*/sample_*/*.png"))
    array_paths = sorted(subject_root.glob("beat_*/channels_float32.npy"))
    if len(png_paths) != BEATS_PER_SUBJECT * PERIOD_LENGTH * len(CHANNEL_NAMES):
        raise ValueError(f"{subject_root}: invalid PNG count {len(png_paths)}")
    if len(array_paths) != BEATS_PER_SUBJECT:
        raise ValueError(f"{subject_root}: invalid NumPy array count {len(array_paths)}")
    for path in array_paths:
        array = np.load(path, mmap_mode="r")
        expected = (PERIOD_LENGTH, len(CHANNEL_NAMES), *NATIVE_SHAPE)
        if array.shape != expected or array.dtype != np.float32:
            raise ValueError(f"{path}: invalid array {array.shape} {array.dtype}")
    return {
        "png_count": len(png_paths),
        "npy_count": len(array_paths),
        "sample_directory_count": BEATS_PER_SUBJECT * PERIOD_LENGTH,
    }


def export_subject(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    selection = json.loads(
        (output_root / "selection_manifest.json").read_text(encoding="utf-8")
    )
    if args.subject is not None:
        matches = [
            row
            for row in selection["subjects"]
            if row["subject"].lower() == args.subject.lower()
        ]
        if len(matches) != 1:
            raise ValueError(f"unknown or ambiguous subject: {args.subject}")
        selected_subject = matches[0]
    else:
        selected_subject = selection["subjects"][args.subject_index]

    subject = selected_subject["subject"]
    mesh = selected_subject["mesh"]
    mesh_root = output_root / f"mesh_{mesh}"
    final_root = mesh_root / subject
    if final_root.exists():
        validation = _validate_subject(final_root)
        print(f"{subject} already complete: {validation}")
        return
    mesh_root.mkdir(parents=True, exist_ok=True)
    staging_root = mesh_root / f".{subject}.tmp-{os.getpid()}"
    if staging_root.exists():
        raise FileExistsError(staging_root)
    staging_root.mkdir()

    try:
        channels, beat_sources = _build_channels(
            Path(selection["coordinate_root"]), selected_subject["windows"]
        )
        # channels: (channel, row, column, 500 samples)
        finite_absolute = np.where(
            np.isfinite(channels), np.abs(channels), np.nan
        )
        scales = np.nanmax(finite_absolute, axis=(1, 2, 3)).astype(np.float64)
        scales[~np.isfinite(scales) | (scales <= 0)] = 1.0
        subject_manifest = {
            "schema": "pvi-six-channel-patient-images-v1",
            "created_utc": _utc_now(),
            "subject": subject,
            "mesh": mesh,
            "gcnm_hidden_channels": selection["gcnm_hidden_channels"],
            "beats": BEATS_PER_SUBJECT,
            "samples_per_beat": PERIOD_LENGTH,
            "channel_order": list(CHANNEL_NAMES),
            "png_filenames": list(PNG_FILENAMES),
            "symmetric_color_scales": {
                name: float(scale) for name, scale in zip(CHANNEL_NAMES, scales)
            },
            "source_windows": selected_subject["windows"],
            "beat_sources": beat_sources,
        }
        _write_json(staging_root / "subject_manifest.json", subject_manifest)

        for beat_index in range(BEATS_PER_SUBJECT):
            beat_root = staging_root / f"beat_{beat_index + 1:03d}"
            beat_root.mkdir()
            start = beat_index * PERIOD_LENGTH
            stop = start + PERIOD_LENGTH
            # Store sample-major exact raw values: (50, 6, 40, 40).
            beat_array = np.ascontiguousarray(
                np.moveaxis(channels[..., start:stop], -1, 0), dtype=np.float32
            )
            array_path = beat_root / "channels_float32.npy"
            np.save(array_path, beat_array, allow_pickle=False)
            beat_manifest = {
                **beat_sources[beat_index],
                "beat_number": beat_index + 1,
                "array_file": array_path.name,
                "array_sha256": _sha256(array_path),
                "array_shape": list(beat_array.shape),
                "array_dtype": str(beat_array.dtype),
            }
            _write_json(beat_root / "beat_manifest.json", beat_manifest)

            tasks: list[tuple[Path, np.ndarray, float, dict[str, str]]] = []
            for sample_index in range(PERIOD_LENGTH):
                sample_root = beat_root / f"sample_{sample_index + 1:03d}"
                sample_root.mkdir()
                global_sample = start + sample_index
                for channel_index, (channel_name, filename) in enumerate(
                    zip(CHANNEL_NAMES, PNG_FILENAMES)
                ):
                    metadata_values = {
                        "subject": subject,
                        "mesh": mesh,
                        "beat": str(beat_index + 1),
                        "sample": str(sample_index + 1),
                        "channel_index": str(channel_index + 1),
                        "channel_name": channel_name,
                        "symmetric_scale": f"{scales[channel_index]:.17g}",
                        "native_shape": "40x40",
                        "encoding": "lossless RGB PNG",
                    }
                    tasks.append(
                        (
                            sample_root / filename,
                            channels[channel_index, :, :, global_sample],
                            float(scales[channel_index]),
                            metadata_values,
                        )
                    )
            with ThreadPoolExecutor(max_workers=args.png_workers) as pool:
                list(pool.map(lambda task: _write_png(*task), tasks))

        validation = _validate_subject(staging_root)
        subject_manifest["validation"] = validation
        subject_manifest["completed_utc"] = _utc_now()
        _write_json(staging_root / "subject_manifest.json", subject_manifest)
        staging_root.rename(final_root)
        print(f"completed {mesh}/{subject}: {validation}")
    except BaseException:
        # Keep failed staging data for diagnosis and never replace a valid output.
        print(f"ERROR: incomplete staging directory retained at {staging_root}")
        raise


def finalize(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    selection_path = output_root / "selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    totals = {"png_count": 0, "npy_count": 0, "sample_directory_count": 0}
    completed: list[dict] = []
    for row in selection["subjects"]:
        subject_root = output_root / f"mesh_{row['mesh']}" / row["subject"]
        if not subject_root.is_dir():
            raise FileNotFoundError(subject_root)
        validation = _validate_subject(subject_root)
        for key in totals:
            totals[key] += validation[key]
        completed.append(
            {"mesh": row["mesh"], "subject": row["subject"], **validation}
        )
    if totals["png_count"] != selection["expected_png_count"]:
        raise ValueError(
            f"root PNG count {totals['png_count']} != {selection['expected_png_count']}"
        )
    validation = {
        "schema": "pvi-six-channel-ten-beat-validation-v1",
        "status": "pass",
        "completed_utc": _utc_now(),
        "subject_count": len(completed),
        "mesh_count": len({row["mesh"] for row in completed}),
        "beats_per_subject": BEATS_PER_SUBJECT,
        "samples_per_beat": PERIOD_LENGTH,
        "channel_count": len(CHANNEL_NAMES),
        **totals,
        "subjects": completed,
    }
    _write_json(output_root / "validation.json", validation)
    incomplete = output_root / "_INCOMPLETE"
    if incomplete.exists():
        incomplete.unlink()
    (output_root / "_SUCCESS").write_text(_utc_now() + "\n", encoding="utf-8")
    print(json.dumps({key: validation[key] for key in (
        "status", "subject_count", "mesh_count", "png_count", "npy_count"
    )}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--coordinate-root", type=Path, required=True)
    prepare_parser.add_argument("--registry", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.set_defaults(function=prepare)

    subject_parser = subparsers.add_parser("export-subject")
    subject_parser.add_argument("--output-root", type=Path, required=True)
    group = subject_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject-index", type=int)
    group.add_argument("--subject")
    subject_parser.add_argument(
        "--png-workers", type=int, default=4, help="parallel lossless PNG encoders"
    )
    subject_parser.set_defaults(function=export_subject)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output-root", type=Path, required=True)
    finalize_parser.set_defaults(function=finalize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
