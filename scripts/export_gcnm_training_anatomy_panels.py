#!/usr/bin/env python3
"""Export publication panels for distinct GCNM training anatomies and beats."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_mesh_maps import MeshMappings


FRAMES_PER_BEAT = 50
TISSUE_COLORS = [
    "#EEF1F4",
    "#D9A066",
    "#E8CE8A",
    "#C0556B",
    "#C3CBD4",
    "#3E8E8A",
    "#B7263F",
    "#8C7B6B",
    "#E08A9B",
]
TISSUE_NAMES = {
    1: "Skin",
    2: "Fat",
    3: "Muscle",
    4: "Cortical bone",
    5: "Ligament / tendon",
    6: "Artery lumen",
    7: "Bone marrow",
    8: "Artery wall",
}
INK = "#15243A"
MUTED = "#5B6B7F"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/hp_lp_beats_US120_v1"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rings_b045/US120.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("figures/gcnm_training_anatomy_examples_US120"),
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    figure.savefig(
        path,
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)


def clean_image_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def style_waveform_axis(axis: plt.Axes, y_max: float) -> None:
    axis.set_xlim(1, FRAMES_PER_BEAT)
    axis.set_ylim(0, y_max)
    axis.set_xlabel("Sample in beat")
    axis.set_ylabel(r"RMS $\Delta\sigma$ (S/m)")
    axis.grid(color="#CBD3DD", linewidth=0.6, alpha=0.75)
    axis.tick_params(colors=MUTED, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#CBD3DD")


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and not args.overwrite:
        raise FileExistsError(f"output folder already exists: {args.output_root}")
    if args.count < 1:
        raise ValueError("count must be positive")

    cfg = GcnmConfig.from_yaml(args.config)
    mappings = MeshMappings(cfg.mappings_h5)
    rng = np.random.default_rng(args.seed)

    hp_path = args.dataset_root / "hp" / "train.npz"
    lp_path = args.dataset_root / "lp" / "train.npz"
    examples: list[dict[str, object]] = []

    with np.load(hp_path) as hp, np.load(lp_path) as lp:
        anatomy_ids = np.asarray(hp["anatomy_id"])
        beat_ids = np.asarray(hp["beat_id"])
        sample_indices = np.asarray(hp["sample_index"])
        np.testing.assert_array_equal(anatomy_ids, lp["anatomy_id"])
        np.testing.assert_array_equal(beat_ids, lp["beat_id"])
        np.testing.assert_array_equal(sample_indices, lp["sample_index"])

        unique_anatomies = np.unique(anatomy_ids)
        if args.count > len(unique_anatomies):
            raise ValueError(
                f"requested {args.count} anatomies but only {len(unique_anatomies)} are available"
            )
        selected_anatomies = sorted(
            int(value) for value in rng.choice(unique_anatomies, args.count, replace=False)
        )

        for order, anatomy_id in enumerate(selected_anatomies, start=1):
            available_beats = np.unique(beat_ids[anatomy_ids == anatomy_id])
            beat_id = int(rng.choice(available_beats))
            indices = np.flatnonzero(
                (anatomy_ids == anatomy_id) & (beat_ids == beat_id)
            )
            indices = indices[np.argsort(sample_indices[indices])]
            if len(indices) != FRAMES_PER_BEAT or not np.array_equal(
                sample_indices[indices], np.arange(FRAMES_PER_BEAT)
            ):
                raise ValueError(f"incomplete anatomy {anatomy_id}, beat {beat_id}")

            hp_sigma = np.asarray(hp["sigma"][indices], dtype=np.float64)
            lp_sigma = np.asarray(lp["sigma"][indices], dtype=np.float64)
            full_sigma = hp_sigma + lp_sigma
            waveform = np.sqrt(np.mean(full_sigma * full_sigma, axis=1))
            peak = int(np.argmax(waveform))
            tissue = mappings.categorical_to_image_grid(
                hp["tissue_labels"][indices[0]], num_classes=len(TISSUE_COLORS)
            ).astype(np.float64)
            peak_grid = mappings.elem_to_image_grid(full_sigma[peak])
            tissue[~np.isfinite(tissue)] = np.nan
            peak_grid[~np.isfinite(peak_grid)] = np.nan
            examples.append(
                {
                    "order": order,
                    "anatomy_id": anatomy_id,
                    "beat_id": beat_id,
                    "peak_sample_zero_based": peak,
                    "peak_sample_one_based": peak + 1,
                    "waveform": waveform,
                    "tissue": tissue,
                    "peak_grid": peak_grid,
                }
            )

    peak_values = np.concatenate(
        [
            np.abs(example["peak_grid"][np.isfinite(example["peak_grid"])])
            for example in examples
        ]
    )
    conductivity_limit = max(float(np.quantile(peak_values, 0.995)), 1e-8)
    waveform_limit = 1.08 * max(float(np.max(example["waveform"])) for example in examples)

    args.output_root.mkdir(parents=True, exist_ok=args.overwrite)
    cmap_tissue = ListedColormap(TISSUE_COLORS)
    norm_tissue = BoundaryNorm(np.arange(-0.5, 9.5, 1), cmap_tissue.N)
    tissue_handles = [
        Patch(facecolor=TISSUE_COLORS[index], edgecolor="none", label=name)
        for index, name in TISSUE_NAMES.items()
    ]
    samples = np.arange(1, FRAMES_PER_BEAT + 1)
    index_rows = []

    for example in examples:
        order = int(example["order"])
        anatomy_id = int(example["anatomy_id"])
        beat_id = int(example["beat_id"])
        peak_sample = int(example["peak_sample_one_based"])
        tissue = np.asarray(example["tissue"])
        peak_grid = np.asarray(example["peak_grid"])
        waveform = np.asarray(example["waveform"])
        stem = f"{order:02d}_A{anatomy_id:03d}_B{beat_id:04d}"

        anatomy_name = f"{stem}_anatomy.png"
        figure, axis = plt.subplots(figsize=(3.5, 3.5))
        axis.imshow(
            tissue,
            cmap=cmap_tissue,
            norm=norm_tissue,
            origin="upper",
            interpolation="nearest",
        )
        axis.set_title(f"Anatomy {anatomy_id}", color=INK, fontweight="semibold")
        clean_image_axis(axis)
        save_figure(figure, args.output_root / anatomy_name, args.dpi)

        peak_name = f"{stem}_peak_ground_truth.png"
        figure, axis = plt.subplots(figsize=(4.2, 3.5))
        shown = axis.imshow(
            peak_grid,
            cmap="RdBu_r",
            vmin=-conductivity_limit,
            vmax=conductivity_limit,
            origin="upper",
            interpolation="nearest",
        )
        axis.set_title(
            f"Peak ground truth · sample {peak_sample}/50",
            color=INK,
            fontweight="semibold",
        )
        clean_image_axis(axis)
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.03, label=r"$\Delta\sigma$ (S/m)")
        save_figure(figure, args.output_root / peak_name, args.dpi)

        waveform_name = f"{stem}_waveform.png"
        figure, axis = plt.subplots(figsize=(4.4, 3.5))
        axis.plot(samples, waveform, color="#355F8D", linewidth=2.2)
        axis.axvline(peak_sample, color="#B7263F", linestyle="--", linewidth=1.2)
        axis.scatter(
            [peak_sample],
            [waveform[peak_sample - 1]],
            s=28,
            color="#B7263F",
            zorder=3,
        )
        axis.set_title(
            f"Beat {beat_id} · anatomy {anatomy_id}",
            color=INK,
            fontweight="semibold",
        )
        style_waveform_axis(axis, waveform_limit)
        save_figure(figure, args.output_root / waveform_name, args.dpi)

        panel_name = f"{stem}_panel.png"
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(12.0, 3.45),
            gridspec_kw={"width_ratios": [1.0, 1.15, 1.65]},
        )
        axes[0].imshow(
            tissue,
            cmap=cmap_tissue,
            norm=norm_tissue,
            origin="upper",
            interpolation="nearest",
        )
        axes[0].set_title("Anatomy", color=INK, fontweight="semibold")
        clean_image_axis(axes[0])
        shown = axes[1].imshow(
            peak_grid,
            cmap="RdBu_r",
            vmin=-conductivity_limit,
            vmax=conductivity_limit,
            origin="upper",
            interpolation="nearest",
        )
        axes[1].set_title(
            f"Peak ground truth\nsample {peak_sample}/50",
            color=INK,
            fontweight="semibold",
        )
        clean_image_axis(axes[1])
        figure.colorbar(
            shown,
            ax=axes[1],
            fraction=0.046,
            pad=0.03,
            label=r"$\Delta\sigma$ (S/m)",
        )
        axes[2].plot(samples, waveform, color="#355F8D", linewidth=2.2)
        axes[2].axvline(peak_sample, color="#B7263F", linestyle="--", linewidth=1.2)
        axes[2].scatter(
            [peak_sample],
            [waveform[peak_sample - 1]],
            s=28,
            color="#B7263F",
            zorder=3,
        )
        axes[2].set_title("50-sample beat", color=INK, fontweight="semibold")
        style_waveform_axis(axes[2], waveform_limit)
        figure.suptitle(
            f"US120 training example {order:02d} · anatomy {anatomy_id} · beat {beat_id}",
            x=0.02,
            y=1.01,
            ha="left",
            color=INK,
            fontsize=13,
            fontweight="semibold",
        )
        figure.legend(
            handles=tissue_handles,
            loc="lower center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.35, -0.06),
            fontsize=7,
        )
        figure.subplots_adjust(left=0.02, right=0.985, bottom=0.21, top=0.82, wspace=0.48)
        save_figure(figure, args.output_root / panel_name, args.dpi)

        index_rows.append(
            {
                "order": order,
                "ring": "US120",
                "split": "train",
                "anatomy_id": anatomy_id,
                "beat_id": beat_id,
                "peak_sample_zero_based": peak_sample - 1,
                "peak_sample_one_based": peak_sample,
                "peak_rms_delta_sigma_s_per_m": float(waveform[peak_sample - 1]),
                "shared_map_limit_s_per_m": conductivity_limit,
                "shared_waveform_max_s_per_m": waveform_limit,
                "anatomy_image": anatomy_name,
                "peak_ground_truth_image": peak_name,
                "waveform_image": waveform_name,
                "combined_panel": panel_name,
            }
        )

    with (args.output_root / "index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    with (args.output_root / "waveforms.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["sample"] + [
            f"A{int(example['anatomy_id']):03d}_B{int(example['beat_id']):04d}"
            for example in examples
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index in range(FRAMES_PER_BEAT):
            row = {"sample": sample_index + 1}
            for example, key in zip(examples, fieldnames[1:]):
                row[key] = float(example["waveform"][sample_index])
            writer.writerow(row)

    manifest = {
        "schema": "gcnm-training-anatomy-panels-v1",
        "ring": "US120",
        "split": "train",
        "dataset_root": str(args.dataset_root.resolve()),
        "config": str(args.config.resolve()),
        "selection_seed": args.seed,
        "count": args.count,
        "samples_per_beat": FRAMES_PER_BEAT,
        "shared_map_limit_s_per_m": conductivity_limit,
        "shared_waveform_max_s_per_m": waveform_limit,
        "dpi": args.dpi,
        "examples": index_rows,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_root), "examples": len(examples)}))


if __name__ == "__main__":
    main()
