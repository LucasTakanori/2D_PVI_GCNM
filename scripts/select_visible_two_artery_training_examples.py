#!/usr/bin/env python3
"""Select one training anatomy per ring with two clearly rasterized artery lumens."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from scipy import ndimage

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_mesh_maps import MeshMappings


RINGS = [f"US{size:03d}" for size in range(60, 131, 5)]
OUTPUT = Path("figures/gcnm_training_two_visible_arteries_selection.csv")
PREVIEW = Path("figures/gcnm_training_two_visible_arteries_selection_preview.png")
TISSUE_COLORS = [
    "#EEF1F4", "#D9A066", "#E8CE8A", "#C0556B", "#C3CBD4",
    "#3E8E8A", "#B7263F", "#8C7B6B", "#E08A9B",
]


def paths(ring: str) -> tuple[Path, Path, Path, Path]:
    if ring == "US120":
        delta = Path("data/differential_US120_1000beats_clean_v1/train.npz")
        hp = Path("data/hp_lp_beats_US120_v1/hp/train.npz")
        records = Path("data/hp_lp_beats_US120_v1/hp/train_anatomy.json")
    else:
        delta = Path(f"data/differential_main_b045_1000beats_clean_v1/{ring}/train.npz")
        hp = Path(f"data/hp_lp_beats_main_b045_v1/{ring}/hp/train.npz")
        records = Path(
            f"data/hp_lp_beats_main_b045_v1/{ring}/full/train_anatomies.json"
        )
    return delta, hp, records, Path(f"configs/rings_b045/{ring}.yaml")


def score_ring(ring: str) -> list[dict[str, object]]:
    delta_path, hp_path, records_path, config_path = paths(ring)
    mappings = MeshMappings(GcnmConfig.from_yaml(config_path).mappings_h5)
    records = {
        int(record["anatomy_id"]): record
        for record in json.loads(records_path.read_text(encoding="utf-8"))
    }
    rows: list[dict[str, object]] = []
    structure = np.ones((3, 3), dtype=np.uint8)

    with np.load(delta_path) as delta, np.load(hp_path) as hp:
        anatomy_ids = np.asarray(delta["anatomy_id"])
        beat_ids = np.asarray(delta["beat_id"])
        sample_indices = np.asarray(delta["sample_index"])
        sigma_all = np.asarray(delta["sigma"], dtype=np.float32)
        tissue_all = np.asarray(hp["tissue_labels"], dtype=np.uint8)
        np.testing.assert_array_equal(hp["anatomy_id"], anatomy_ids)

        for anatomy_id in np.unique(anatomy_ids):
            anatomy_id = int(anatomy_id)
            anatomy_indices = np.flatnonzero(anatomy_ids == anatomy_id)
            beat_id = int(np.min(beat_ids[anatomy_indices]))
            indices = np.flatnonzero(
                (anatomy_ids == anatomy_id) & (beat_ids == beat_id)
            )
            indices = indices[np.argsort(sample_indices[indices])]
            if len(indices) != 50:
                continue

            tissue = mappings.categorical_to_image_grid(
                tissue_all[indices[0]], num_classes=len(TISSUE_COLORS)
            )
            lumen_components, count = ndimage.label(tissue == 6, structure=structure)
            components = []
            for component in range(1, count + 1):
                mask = lumen_components == component
                size = int(np.count_nonzero(mask))
                if size:
                    center = np.asarray(ndimage.center_of_mass(mask), dtype=np.float64)
                    components.append((size, component, mask, center))
            components.sort(reverse=True, key=lambda item: item[0])
            if len(components) < 2:
                continue

            sigma = np.asarray(sigma_all[indices], dtype=np.float64)
            peak = int(np.argmax(np.sqrt(np.mean(sigma * sigma, axis=1))))
            peak_grid = mappings.elem_to_image_grid(sigma[peak])
            first, second = components[:2]
            peaks = [float(np.max(np.abs(peak_grid[item[2]]))) for item in (first, second)]
            distance = float(np.linalg.norm(first[3] - second[3]))
            record = records[anatomy_id]
            if "model" in record:
                arteries = record["model"]["arteries"]
                lumen_areas = [
                    float(
                        np.pi
                        * artery["radius_x_mm"]
                        * artery["radius_y_mm"]
                        * artery["lumen_fraction"] ** 2
                    )
                    for artery in arteries
                ]
            else:
                lumen_areas = [
                    float(np.pi * vessel["axis_a"] * vessel["axis_b"])
                    for vessel in record["vessels"]
                ]
            second_pixels = int(second[0])
            min_peak = float(min(peaks))
            min_area = float(min(lumen_areas))
            if min_peak < 0.01:
                continue
            score = (
                100000.0 * second_pixels * min_peak
                + 100.0 * min_area
                + 10.0 * distance
            )
            rows.append(
                {
                    "ring": ring,
                    "anatomy_id": anatomy_id,
                    "beat_id": beat_id,
                    "peak_sample_one_based": peak + 1,
                    "artery_1_pixels": int(first[0]),
                    "artery_2_pixels": second_pixels,
                    "artery_centroid_distance_pixels": distance,
                    "minimum_lumen_area_mm2": min_area,
                    "minimum_peak_delta_sigma_s_per_m": min_peak,
                    "visibility_score": score,
                }
            )
    return sorted(rows, key=lambda row: float(row["visibility_score"]), reverse=True)


def main() -> None:
    selected: list[dict[str, object]] = []
    for ring in RINGS:
        ranked = score_ring(ring)
        if not ranked:
            raise RuntimeError(f"{ring}: no training anatomy had two rasterized artery lumens")
        winner = ranked[0]
        selected.append(winner)
        alternatives = ", ".join(
            f"{row['anatomy_id']} ({row['artery_1_pixels']}+{row['artery_2_pixels']} px)"
            for row in ranked[:5]
        )
        print(f"{ring}: {alternatives}", flush=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    render_preview(selected)
    print(f"wrote {OUTPUT}")


def render_preview(selected: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(5, 6, figsize=(18, 15), dpi=220)
    tissue_cmap = ListedColormap(TISSUE_COLORS)
    tissue_norm = BoundaryNorm(np.arange(-0.5, 9.5), tissue_cmap.N)
    for pair, row in enumerate(selected):
        ring = str(row["ring"])
        anatomy_id = int(row["anatomy_id"])
        beat_id = int(row["beat_id"])
        delta_path, hp_path, _records_path, config_path = paths(ring)
        mappings = MeshMappings(GcnmConfig.from_yaml(config_path).mappings_h5)
        with np.load(delta_path) as delta, np.load(hp_path) as hp:
            indices = np.flatnonzero(
                (delta["anatomy_id"] == anatomy_id) & (delta["beat_id"] == beat_id)
            )
            indices = indices[np.argsort(delta["sample_index"][indices])]
            tissue = mappings.categorical_to_image_grid(
                hp["tissue_labels"][indices[0]], num_classes=len(TISSUE_COLORS)
            )
            peak = int(row["peak_sample_one_based"]) - 1
            truth = mappings.elem_to_image_grid(delta["sigma"][indices[peak]])
        row_index, column_pair = divmod(pair, 3)
        anatomy_axis = axes[row_index, 2 * column_pair]
        truth_axis = axes[row_index, 2 * column_pair + 1]
        anatomy_axis.imshow(tissue, cmap=tissue_cmap, norm=tissue_norm, origin="lower")
        truth_axis.imshow(
            truth, cmap="RdBu_r", vmin=-0.03, vmax=0.03, origin="lower"
        )
        anatomy_axis.set_title(f"{ring} anatomy {anatomy_id}")
        truth_axis.set_title(f"GT peak sample {peak + 1}")
        anatomy_axis.axis("off")
        truth_axis.axis("off")
    figure.suptitle("Selected training anatomies with two visible artery lumens", fontsize=20)
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig(PREVIEW, facecolor="white", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
