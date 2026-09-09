#!/usr/bin/env python3
"""Finalize the one-beat-per-ring GCNM training export."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from export_us120_training_truth_gcnm_beats import comparison_image, save_colorbar


DEFAULT_ROOT = Path("figures/gcnm_training_one_beat_per_mesh")
RINGS = [f"US{size:03d}" for size in range(60, 131, 5)]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--reference-root",
        type=Path,
        help=(
            "Prior export whose selected examples, rendering contract, and ground "
            "truth must match exactly."
        ),
    )
    args = parser.parse_args()
    root = args.root
    reference_root = args.reference_root.resolve() if args.reference_root else None
    reference_manifest = None
    reference_examples: dict[str, dict[str, object]] = {}
    if reference_root is not None:
        reference_manifest = json.loads(
            (reference_root / "manifest.json").read_text(encoding="utf-8")
        )
        reference_examples = {
            str(row["ring"]): row for row in reference_manifest["selected_examples"]
        }
    anatomy_gallery = root / "anatomies"
    waveform_gallery = root / "waveforms"
    anatomy_gallery.mkdir(exist_ok=True)
    waveform_gallery.mkdir(exist_ok=True)

    combined_index: list[dict[str, object]] = []
    combined_waveforms: list[dict[str, object]] = []
    ring_manifests: list[dict[str, object]] = []
    shared_limit: float | None = None
    physics_mesh_mode: str | None = None
    model_name: str | None = None

    for ring in RINGS:
        ring_root = root / ring
        manifest = json.loads((ring_root / "manifest.json").read_text(encoding="utf-8"))
        ring_physics_mode = str(manifest.get("physics_mesh_mode", "coarse"))
        ring_model_name = str(manifest.get("model_name", "coordinate_direct"))
        if physics_mesh_mode is None:
            physics_mesh_mode = ring_physics_mode
            model_name = ring_model_name
        elif ring_physics_mode != physics_mesh_mode or ring_model_name != model_name:
            raise ValueError(
                f"{ring}: inconsistent model/physics contract "
                f"{ring_model_name}/{ring_physics_mode}"
            )
        index_rows = read_rows(ring_root / "index.csv")
        if len(index_rows) != 1:
            raise ValueError(f"{ring}: expected one exported beat, found {len(index_rows)}")

        row = index_rows[0]
        beat_root = ring_root / row["folder"]
        anatomy_id = int(row["anatomy_id"])
        beat_id = int(row["beat_id"])
        limit = float(manifest["shared_color_limit_s_per_m"])
        if shared_limit is None:
            shared_limit = limit
        elif limit != shared_limit:
            raise ValueError(f"{ring}: inconsistent shared color limit {limit}")

        truth_paths = sorted((beat_root / "ground_truth_delta_sigma").glob("sample_*.png"))
        gcnm_paths = sorted((beat_root / "gcnm_stage2_delta_sigma").glob("sample_*.png"))
        if len(truth_paths) != 50 or len(gcnm_paths) != 50:
            raise ValueError(
                f"{ring}: expected 50 truth and 50 GCNM frames, "
                f"found {len(truth_paths)} and {len(gcnm_paths)}"
            )

        if reference_root is not None:
            reference_row = reference_examples.get(ring)
            if reference_row is None:
                raise ValueError(f"{ring}: missing from reference export")
            reference_identity = (
                int(reference_row["anatomy_id"]),
                int(reference_row["beat_id"]),
            )
            if (anatomy_id, beat_id) != reference_identity:
                raise ValueError(
                    f"{ring}: selected beat changed from {reference_identity} to "
                    f"{(anatomy_id, beat_id)}"
                )
            reference_ring_root = reference_root / ring
            reference_ring_manifest = json.loads(
                (reference_ring_root / "manifest.json").read_text(encoding="utf-8")
            )
            for key in (
                "dataset_sha256",
                "image_size_pixels",
                "shared_color_limit_s_per_m",
            ):
                if manifest[key] != reference_ring_manifest[key]:
                    raise ValueError(f"{ring}: comparison contract changed for {key}")
            reference_beat_root = reference_ring_root / str(reference_row.get(
                "folder", reference_ring_manifest["examples"][0]["folder"]
            ))
            reference_truth_paths = sorted(
                (reference_beat_root / "ground_truth_delta_sigma").glob("sample_*.png")
            )
            if len(reference_truth_paths) != len(truth_paths):
                raise ValueError(f"{ring}: reference ground-truth frame count changed")
            for new_truth, old_truth in zip(
                truth_paths, reference_truth_paths, strict=True
            ):
                if new_truth.read_bytes() != old_truth.read_bytes():
                    raise ValueError(
                        f"{ring}: rendered ground truth differs at {new_truth.name}"
                    )
            with np.load(beat_root / "element_values.npz") as new_values, np.load(
                reference_beat_root / "element_values.npz"
            ) as reference_values:
                np.testing.assert_array_equal(
                    new_values["ground_truth_delta_sigma"],
                    reference_values["ground_truth_delta_sigma"],
                    err_msg=f"{ring}: numerical ground truth changed",
                )

        comparison_root = beat_root / "side_by_side"
        comparison_root.mkdir(exist_ok=True)
        for sample, (truth_path, gcnm_path) in enumerate(
            zip(truth_paths, gcnm_paths, strict=True), start=1
        ):
            with Image.open(truth_path) as truth_source, Image.open(gcnm_path) as gcnm_source:
                comparison = comparison_image(
                    truth_source.convert("RGB"),
                    gcnm_source.convert("RGB"),
                    ring=ring,
                    anatomy_id=anatomy_id,
                    beat_id=beat_id,
                    sample_one_based=sample,
                )
                comparison.save(
                    comparison_root / f"sample_{sample:02d}.png",
                    optimize=True,
                )

        save_colorbar(ring_root / "shared_colorbar.png", limit)
        shutil.copy2(beat_root / "anatomy.png", anatomy_gallery / f"{ring}_anatomy.png")
        shutil.copy2(beat_root / "waveform.png", waveform_gallery / f"{ring}_waveform.png")

        combined_index.append(
            {
                **row,
                "folder": f"{ring}/{row['folder']}",
            }
        )
        for waveform_row in read_rows(ring_root / "all_waveforms.csv"):
            combined_waveforms.append({"ring": ring, **waveform_row})
        ring_manifests.append(manifest)

    if shared_limit is None:
        raise RuntimeError("No ring exports were found")

    write_rows(
        root / "index.csv",
        combined_index,
        [
            "order",
            "ring",
            "split",
            "anatomy_id",
            "beat_id",
            "frames",
            "gcnm_stage2_nrmse",
            "gcnm_stage2_correlation",
            "folder",
        ],
    )
    write_rows(
        root / "all_waveforms.csv",
        combined_waveforms,
        [
            "ring",
            "anatomy_id",
            "beat_id",
            "sample_zero_based",
            "sample_one_based",
            "ground_truth_rms_delta_sigma_s_per_m",
            "gcnm_stage2_rms_delta_sigma_s_per_m",
        ],
    )
    save_colorbar(root / "shared_colorbar.png", shared_limit)

    root_manifest = {
        "schema": "gcnm-training-truth-stage2-one-beat-per-ring-v1",
        "rings": RINGS,
        "ring_count": len(RINGS),
        "selection": (
            "one training anatomy with two visible rasterized artery lumens and one "
            "complete 50-sample beat per ring"
        ),
        "selected_examples": [
            {
                "ring": row["ring"],
                "anatomy_id": int(row["anatomy_id"]),
                "beat_id": int(row["beat_id"]),
            }
            for row in combined_index
        ],
        "frames_per_ring": 50,
        "ground_truth_frames_per_ring": 50,
        "gcnm_stage2_frames_per_ring": 50,
        "comparison_frames_per_ring": 50,
        "shared_color_limit_s_per_m": shared_limit,
        "model_name": model_name,
        "physics_mesh_mode": physics_mesh_mode,
        "comparison_contract": (
            "same synthetic anatomy IDs, beat IDs, samples, image size, and color "
            "scale as the prior two-visible-arteries export"
        ),
        "reference_export": str(reference_root) if reference_root else None,
        "ground_truth_identity_verified": reference_root is not None,
        "waveform_definition": "RMS differential conductivity across inverse-mesh elements",
        "ring_manifests": ring_manifests,
    }
    (root / "manifest.json").write_text(
        json.dumps(root_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (root / "README.md").write_text(
        "# Synthetic truth vs retrained projected-fine GCNM\n\n"
        "This export repeats the exact synthetic anatomy and beat selections from "
        "the prior two-visible-arteries comparison. Each ring contains 50 matched "
        "ground-truth frames, literal stage-2 GCNM frames, side-by-side panels, "
        "element arrays, and waveform summaries.\n\n"
        f"Model: `{model_name}`\n\n"
        f"Inference/training physics: `{physics_mesh_mode}`\n\n"
        "The finalization audit verified that every numerical ground-truth array "
        "and rendered ground-truth PNG is identical to the reference export.\n",
        encoding="utf-8",
    )
    (root / "_SUCCESS").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
