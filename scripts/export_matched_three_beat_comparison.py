#!/usr/bin/env python3
"""Repackage a three-beat Figure 26 export for direct 10-beat comparison.

The source Figure 26 tensors already contain the same first three experimental
beats as the original ten-beat audit.  This exporter keeps those tensors and
recreates the original audit's visualization contract:

* native 40 x 40 lossless PNGs with no resampling;
* the same signed blue-white-red lookup table;
* the same neutral color outside the reconstruction domain;
* the original per-subject, per-channel symmetric scales; and
* the original beat/sample/channel directory and filename layout.

The three Newton-derived control channels must agree exactly with the original
audit before a subject is published.  The three GCNM-derived channels retain
the source Figure 26 values but use the corresponding original audit scales,
so equal numerical values map to equal RGB pixels across exports.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, PngImagePlugin


BEATS_PER_SUBJECT = 3
SAMPLES_PER_BEAT = 50
NATIVE_SHAPE = (40, 40)
CONTROL_CHANNEL_COUNT = 3
OUTPUT_CHANNEL_NAMES = (
    "newton_hp",
    "d_newton_lp_dt",
    "d2_newton_lp_dt2",
    "gcnm_s1",
    "gcnm_s2",
    "d_gcnm_s2_dt",
)
PNG_FILENAMES = tuple(
    f"{index:02d}_{name}.png" for index, name in enumerate(OUTPUT_CHANNEL_NAMES, 1)
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
    """Use the exact PNG transformation from the original ten-beat export."""

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


def _link_or_copy(source: Path, destination: Path) -> str:
    """Reuse an identical control PNG, falling back to a byte copy if needed."""

    try:
        os.link(source, destination)
        return "hardlink"
    except OSError as error:
        if error.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES}:
            raise
        shutil.copy2(source, destination)
        return "copy"


def _source_subject_manifest(source_root: Path, subject: str) -> tuple[Path, dict]:
    path = source_root / subject / "subject_manifest.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _reference_subject_manifest(
    reference_root: Path, ring: str, subject: str
) -> tuple[Path, dict]:
    path = reference_root / f"mesh_{ring}" / subject / "subject_manifest.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _validate_selection_pair(source_row: dict, reference_row: dict) -> list[dict]:
    """Return the exact reference beat records after proving beat identity."""

    reference_beats = reference_row["beat_sources"][:BEATS_PER_SUBJECT]
    source_periods = [int(value) for value in source_row["source_period_indices_zero_based"]]
    if len(source_periods) != BEATS_PER_SUBJECT:
        raise ValueError(
            f"{source_row['subject']} has {len(source_periods)} source periods"
        )
    actual_periods = [
        int(beat["source_period_index_zero_based"]) for beat in reference_beats
    ]
    if actual_periods != source_periods:
        raise ValueError(
            f"{source_row['subject']} period mismatch: "
            f"source={source_periods}, reference={actual_periods}"
        )
    for beat in reference_beats:
        for key in ("source_name", "session"):
            if str(beat[key]) != str(source_row[key]):
                raise ValueError(
                    f"{source_row['subject']} {key} mismatch: "
                    f"source={source_row[key]!r}, reference={beat[key]!r}"
                )
    return reference_beats


def prepare(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve()
    reference_root = args.reference_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    for root in (source_root, reference_root):
        if not (root / "_SUCCESS").is_file():
            raise FileNotFoundError(f"validated export marker is missing: {root}/_SUCCESS")

    source_selection_path = source_root / "selection_manifest.json"
    reference_selection_path = reference_root / "selection_manifest.json"
    source_selection = json.loads(source_selection_path.read_text(encoding="utf-8"))
    reference_selection = json.loads(
        reference_selection_path.read_text(encoding="utf-8")
    )
    if int(source_selection["beats_per_subject"]) != BEATS_PER_SUBJECT:
        raise ValueError("source export is not a three-beat export")
    if int(reference_selection["beats_per_subject"]) != 10:
        raise ValueError("reference export is not the ten-beat audit")
    if list(reference_selection["channel_order"]) != list(OUTPUT_CHANNEL_NAMES):
        raise ValueError("reference channel order does not match the expected contract")

    reference_rows = {
        str(row["subject"]).lower(): row for row in reference_selection["subjects"]
    }
    subjects: list[dict] = []
    for source_row in source_selection["subjects"]:
        subject = str(source_row["subject"]).lower()
        ring = str(source_row["ring"])
        if subject not in reference_rows:
            raise ValueError(f"{subject} is absent from the reference selection")
        reference_selection_row = reference_rows[subject]
        if str(reference_selection_row["mesh"]) != ring:
            raise ValueError(f"{subject} ring mismatch")

        source_manifest_path, source_manifest = _source_subject_manifest(
            source_root, subject
        )
        reference_manifest_path, reference_manifest = _reference_subject_manifest(
            reference_root, ring, subject
        )
        reference_beats = _validate_selection_pair(source_row, reference_manifest)
        source_channel_order = list(source_manifest["channel_order"])
        if len(source_channel_order) != len(OUTPUT_CHANNEL_NAMES):
            raise ValueError(f"{subject} source does not contain six channels")
        if source_channel_order[:CONTROL_CHANNEL_COUNT] != list(
            OUTPUT_CHANNEL_NAMES[:CONTROL_CHANNEL_COUNT]
        ):
            raise ValueError(f"{subject} control channel order changed")

        source_array_path = source_root / subject / str(source_manifest["array_file"])
        if not source_array_path.is_file():
            raise FileNotFoundError(source_array_path)
        subjects.append(
            {
                "subject": subject,
                "ring": ring,
                "session": source_row["session"],
                "source_name": source_row["source_name"],
                "source_period_indices_zero_based": [
                    int(value)
                    for value in source_row["source_period_indices_zero_based"]
                ],
                "source_subject_root": str((source_root / subject).resolve()),
                "source_subject_manifest": str(source_manifest_path.resolve()),
                "source_subject_manifest_sha256": _sha256(source_manifest_path),
                "source_array": str(source_array_path.resolve()),
                "source_array_sha256": source_manifest["array_sha256"],
                "source_channel_order": source_channel_order,
                "reference_subject_root": str(reference_manifest_path.parent.resolve()),
                "reference_subject_manifest": str(reference_manifest_path.resolve()),
                "reference_subject_manifest_sha256": _sha256(reference_manifest_path),
                "reference_beat_sources": reference_beats,
            }
        )
    subjects.sort(key=lambda row: _subject_number(row["subject"]))
    if len(subjects) != 91:
        raise ValueError(f"expected 91 subjects, found {len(subjects)}")

    output_root.mkdir(parents=True)
    (output_root / "_INCOMPLETE").write_text(_utc_now() + "\n", encoding="utf-8")
    manifest = {
        "schema": "pvi-gcnm-three-beat-ten-beat-matched-comparison-v1",
        "created_utc": _utc_now(),
        "source_export": str(source_root),
        "source_schema": source_selection["schema"],
        "source_selection_manifest": str(source_selection_path.resolve()),
        "source_selection_manifest_sha256": _sha256(source_selection_path),
        "reference_export": str(reference_root),
        "reference_schema": reference_selection["schema"],
        "reference_selection_manifest": str(reference_selection_path.resolve()),
        "reference_selection_manifest_sha256": _sha256(reference_selection_path),
        "subject_count": len(subjects),
        "beats_per_subject": BEATS_PER_SUBJECT,
        "samples_per_beat": SAMPLES_PER_BEAT,
        "channel_count": len(OUTPUT_CHANNEL_NAMES),
        "channel_order": list(OUTPUT_CHANNEL_NAMES),
        "png_filenames": list(PNG_FILENAMES),
        "beat_contract": (
            "beat_001 through beat_003 are the exact first three beats of the "
            "corresponding subject in the ten-beat reference export"
        ),
        "channel_contract": {
            "control_channels": list(OUTPUT_CHANNEL_NAMES[:CONTROL_CHANNEL_COUNT]),
            "method_channels": list(OUTPUT_CHANNEL_NAMES[CONTROL_CHANNEL_COUNT:]),
            "source_channel_order": list(source_selection["channel_order"]),
            "mapping": {
                source: output
                for source, output in zip(
                    source_selection["channel_order"], OUTPUT_CHANNEL_NAMES
                )
            },
        },
        "png_encoding": {
            "format": "lossless RGB PNG",
            "shape": list(NATIVE_SHAPE),
            "resampling": "none",
            "color_map": "signed blue-white-red",
            "nan_color": "neutral center color [247, 247, 247]",
            "scale": (
                "exact per-subject/per-channel symmetric scale copied from the "
                "ten-beat reference export"
            ),
        },
        "raw_array": {
            "filename": "channels_float32.npy",
            "dtype": "float32",
            "shape": [SAMPLES_PER_BEAT, len(OUTPUT_CHANNEL_NAMES), *NATIVE_SHAPE],
            "axis_order": ["sample", "channel", "row", "column"],
        },
        "subjects": subjects,
        "expected_png_count": len(subjects)
        * BEATS_PER_SUBJECT
        * SAMPLES_PER_BEAT
        * len(OUTPUT_CHANNEL_NAMES),
    }
    _write_json(output_root / "selection_manifest.json", manifest)
    (output_root / "README.md").write_text(
        "# Three-beat export matched to the original ten-beat audit\n\n"
        "This package is a comparison-safe rendering of the source Figure 26 "
        "three-beat export. For every subject, `beat_001` through `beat_003` are "
        "the exact same source/session/period beats as the first three beats in "
        "the original ten-beat audit.\n\n"
        "The six channel roles, native 40 x 40 PNG encoding, signed color lookup "
        "table, outside-domain color, directory layout, and per-subject/channel "
        "symmetric scales are copied from the original export. Therefore the same "
        "numeric value in a given subject and channel maps to the same RGB value "
        "in both packages. Comparisons are valid within the same subject/channel; "
        "the original contract still uses different scales between subjects and "
        "between channel roles.\n\n"
        "Channels 1-3 are common Newton-derived controls and are required to match "
        "the original float tensors exactly. Channels 4-6 contain the source "
        "Figure 26 GCNM method outputs. Exact float32 data are stored per beat in "
        "`channels_float32.npy` with shape `(50, 6, 40, 40)`.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "prepared": str(output_root),
                "subjects": len(subjects),
                "expected_png_count": manifest["expected_png_count"],
            },
            indent=2,
        )
    )


def _select_subject(manifest: dict, args: argparse.Namespace) -> dict:
    if args.subject is not None:
        matches = [
            row
            for row in manifest["subjects"]
            if row["subject"] == args.subject.lower()
        ]
        if len(matches) != 1:
            raise ValueError(f"unknown subject: {args.subject}")
        return matches[0]
    return manifest["subjects"][args.subject_index]


def _validate_subject(subject_root: Path) -> dict:
    png_paths = list(subject_root.glob("beat_*/sample_*/*.png"))
    array_paths = sorted(subject_root.glob("beat_*/channels_float32.npy"))
    expected_pngs = (
        BEATS_PER_SUBJECT * SAMPLES_PER_BEAT * len(OUTPUT_CHANNEL_NAMES)
    )
    if len(png_paths) != expected_pngs:
        raise ValueError(f"{subject_root}: invalid PNG count {len(png_paths)}")
    if len(array_paths) != BEATS_PER_SUBJECT:
        raise ValueError(f"{subject_root}: invalid array count {len(array_paths)}")
    for path in array_paths:
        array = np.load(path, mmap_mode="r")
        expected_shape = (SAMPLES_PER_BEAT, len(OUTPUT_CHANNEL_NAMES), *NATIVE_SHAPE)
        if array.shape != expected_shape or array.dtype != np.float32:
            raise ValueError(f"{path}: invalid array {array.shape} {array.dtype}")
    with Image.open(png_paths[0]) as image:
        if image.size != NATIVE_SHAPE or image.mode != "RGB":
            raise ValueError(
                f"{png_paths[0]}: invalid PNG configuration {image.size} {image.mode}"
            )
    return {
        "png_count": len(png_paths),
        "npy_count": len(array_paths),
        "sample_directory_count": BEATS_PER_SUBJECT * SAMPLES_PER_BEAT,
    }


def export_subject(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    manifest = json.loads(
        (output_root / "selection_manifest.json").read_text(encoding="utf-8")
    )
    row = _select_subject(manifest, args)
    subject = row["subject"]
    ring = row["ring"]
    mesh_root = output_root / f"mesh_{ring}"
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

    source_manifest_path = Path(row["source_subject_manifest"])
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    reference_manifest_path = Path(row["reference_subject_manifest"])
    reference_manifest = json.loads(
        reference_manifest_path.read_text(encoding="utf-8")
    )
    source_channels = np.load(Path(row["source_array"]), mmap_mode="r")
    expected_source_shape = (
        BEATS_PER_SUBJECT,
        SAMPLES_PER_BEAT,
        len(OUTPUT_CHANNEL_NAMES),
        *NATIVE_SHAPE,
    )
    if source_channels.shape != expected_source_shape or source_channels.dtype != np.float32:
        raise ValueError(
            f"{row['source_array']}: invalid source tensor "
            f"{source_channels.shape} {source_channels.dtype}"
        )
    scales = np.asarray(
        [
            float(reference_manifest["symmetric_color_scales"][name])
            for name in OUTPUT_CHANNEL_NAMES
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(scales) & (scales > 0)):
        raise ValueError(f"{subject}: invalid reference scales {scales}")

    link_modes: set[str] = set()
    try:
        beat_manifests: list[dict] = []
        for beat_index in range(BEATS_PER_SUBJECT):
            beat_number = beat_index + 1
            reference_beat_root = (
                Path(row["reference_subject_root"]) / f"beat_{beat_number:03d}"
            )
            reference_array = np.load(
                reference_beat_root / "channels_float32.npy", mmap_mode="r"
            )
            source_beat = np.asarray(source_channels[beat_index])
            if not np.array_equal(
                source_beat[:, :CONTROL_CHANNEL_COUNT],
                reference_array[:, :CONTROL_CHANNEL_COUNT],
                equal_nan=True,
            ):
                raise ValueError(
                    f"{subject} beat {beat_number}: Newton control channels differ "
                    "from the ten-beat reference"
                )

            beat_root = staging_root / f"beat_{beat_number:03d}"
            beat_root.mkdir()
            beat_array = np.ascontiguousarray(source_beat, dtype=np.float32)
            array_path = beat_root / "channels_float32.npy"
            np.save(array_path, beat_array, allow_pickle=False)
            reference_beat_manifest = json.loads(
                (reference_beat_root / "beat_manifest.json").read_text(encoding="utf-8")
            )
            beat_manifest = {
                key: reference_beat_manifest[key]
                for key in (
                    "window_number",
                    "sample_id",
                    "source_name",
                    "session",
                    "source_period_index_zero_based",
                    "beat_number",
                )
            }
            beat_manifest.update(
                {
                    "array_file": array_path.name,
                    "array_sha256": _sha256(array_path),
                    "array_shape": list(beat_array.shape),
                    "array_dtype": str(beat_array.dtype),
                }
            )
            _write_json(beat_root / "beat_manifest.json", beat_manifest)
            beat_manifests.append(beat_manifest)

            render_tasks: list[
                tuple[Path, np.ndarray, float, dict[str, str]]
            ] = []
            for sample_index in range(SAMPLES_PER_BEAT):
                sample_number = sample_index + 1
                sample_root = beat_root / f"sample_{sample_number:03d}"
                sample_root.mkdir()
                reference_sample_root = (
                    reference_beat_root / f"sample_{sample_number:03d}"
                )
                for channel_index, (channel_name, filename) in enumerate(
                    zip(OUTPUT_CHANNEL_NAMES, PNG_FILENAMES)
                ):
                    destination = sample_root / filename
                    if channel_index < CONTROL_CHANNEL_COUNT:
                        link_modes.add(
                            _link_or_copy(reference_sample_root / filename, destination)
                        )
                        continue
                    render_tasks.append(
                        (
                            destination,
                            source_beat[sample_index, channel_index],
                            float(scales[channel_index]),
                            {
                                "subject": subject,
                                "mesh": ring,
                                "beat": str(beat_number),
                                "sample": str(sample_number),
                                "channel_index": str(channel_index + 1),
                                "channel_name": channel_name,
                                "symmetric_scale": f"{scales[channel_index]:.17g}",
                                "native_shape": "40x40",
                                "encoding": "lossless RGB PNG",
                            },
                        )
                    )
            with ThreadPoolExecutor(max_workers=args.png_workers) as pool:
                list(pool.map(lambda task: _write_png(*task), render_tasks))

        subject_manifest = {
            "schema": "pvi-gcnm-three-beat-ten-beat-matched-subject-v1",
            "created_utc": _utc_now(),
            "subject": subject,
            "mesh": ring,
            "beats": BEATS_PER_SUBJECT,
            "samples_per_beat": SAMPLES_PER_BEAT,
            "channel_order": list(OUTPUT_CHANNEL_NAMES),
            "source_channel_order": row["source_channel_order"],
            "png_filenames": list(PNG_FILENAMES),
            "symmetric_color_scales": {
                name: float(scale) for name, scale in zip(OUTPUT_CHANNEL_NAMES, scales)
            },
            "scale_reference_subject_manifest": str(reference_manifest_path),
            "scale_reference_subject_manifest_sha256": row[
                "reference_subject_manifest_sha256"
            ],
            "source_subject_manifest": str(source_manifest_path),
            "source_subject_manifest_sha256": row["source_subject_manifest_sha256"],
            "source_array": row["source_array"],
            "source_array_sha256": row["source_array_sha256"],
            "beat_sources": row["reference_beat_sources"],
            "beat_manifests": beat_manifests,
            "control_png_materialization": sorted(link_modes),
            "comparison_contract": {
                "beat_identity": "exact source/session/period match",
                "control_float_identity": "exact including NaN masks",
                "control_png_identity": "byte-identical reuse from reference export",
                "method_channel_scaling": (
                    "original ten-beat per-subject/per-channel symmetric scales"
                ),
                "png_shape": list(NATIVE_SHAPE),
                "resampling": "none",
            },
        }
        validation = _validate_subject(staging_root)
        subject_manifest["validation"] = validation
        subject_manifest["completed_utc"] = _utc_now()
        _write_json(staging_root / "subject_manifest.json", subject_manifest)
        staging_root.rename(final_root)
        print(f"completed {ring}/{subject}: {validation}")
    except BaseException:
        print(f"ERROR: incomplete staging directory retained at {staging_root}")
        raise


def finalize(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    manifest = json.loads(
        (output_root / "selection_manifest.json").read_text(encoding="utf-8")
    )
    totals = {"png_count": 0, "npy_count": 0, "sample_directory_count": 0}
    completed: list[dict] = []
    for row in manifest["subjects"]:
        subject_root = output_root / f"mesh_{row['ring']}" / row["subject"]
        if not subject_root.is_dir():
            raise FileNotFoundError(subject_root)
        validation = _validate_subject(subject_root)
        for key in totals:
            totals[key] += validation[key]
        completed.append(
            {"mesh": row["ring"], "subject": row["subject"], **validation}
        )
    if totals["png_count"] != manifest["expected_png_count"]:
        raise ValueError(
            f"PNG count {totals['png_count']} != {manifest['expected_png_count']}"
        )
    validation = {
        "schema": "pvi-gcnm-three-beat-ten-beat-matched-validation-v1",
        "status": "pass",
        "completed_utc": _utc_now(),
        "subject_count": len(completed),
        "mesh_count": len({row["mesh"] for row in completed}),
        "beats_per_subject": BEATS_PER_SUBJECT,
        "samples_per_beat": SAMPLES_PER_BEAT,
        "channel_count": len(OUTPUT_CHANNEL_NAMES),
        **totals,
        "subjects": completed,
    }
    _write_json(output_root / "validation.json", validation)
    incomplete = output_root / "_INCOMPLETE"
    if incomplete.exists():
        incomplete.unlink()
    (output_root / "_SUCCESS").write_text(_utc_now() + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: validation[key]
                for key in (
                    "status",
                    "subject_count",
                    "mesh_count",
                    "png_count",
                    "npy_count",
                )
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-root", type=Path, required=True)
    prepare_parser.add_argument("--reference-root", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.set_defaults(function=prepare)

    subject_parser = subparsers.add_parser("export-subject")
    subject_parser.add_argument("--output-root", type=Path, required=True)
    group = subject_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject-index", type=int)
    group.add_argument("--subject")
    subject_parser.add_argument("--png-workers", type=int, default=8)
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
