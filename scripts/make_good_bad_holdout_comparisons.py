#!/usr/bin/env python3
"""Create IEEE-style good/bad holdout comparisons for old and new GCNMs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gcnm_pvi.gcnm_mesh_maps import MeshMappings


PANEL_NAMES = (
    r"Ground truth $\Delta\sigma^*$",
    "PVI-style Newton",
    "Old GCNM stage 1",
    "Old GCNM stage 2",
    "New GCNM stage 1",
    "New GCNM stage 2",
)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[finite], b[finite])[0, 1])


def _images(values: np.ndarray, mappings: MeshMappings) -> np.ndarray:
    return np.stack([mappings.elem_to_image_grid(sample) for sample in values])


def _latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-predictions",
        type=Path,
        default=root
        / "data/faithful_results/voltage_vessel_slots_hom_seed0/evaluation_nonlinear/predictions.npz",
    )
    parser.add_argument(
        "--old-predictions",
        type=Path,
        default=root
        / "data/faithful_results/faithful_background025_hom_seed0/evaluation_nonlinear_separated/predictions.npz",
    )
    parser.add_argument(
        "--mappings",
        type=Path,
        default=root
        / "data/ring_meshes/subject006_US120/ring_US120_mappings_40.h5",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "reports/voltage_vessel_good_bad_comparisons",
    )
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--per-group", type=int, default=5)
    args = parser.parse_args()

    with np.load(args.new_predictions) as source:
        truth = np.asarray(source["truth"], dtype=np.float64)
        newton = np.asarray(source["old_newton"], dtype=np.float64)
        new_stages = np.asarray(source["gcnm_stages"], dtype=np.float64)
    with np.load(args.old_predictions) as source:
        old_truth = np.asarray(source["truth"], dtype=np.float64)
        old_newton = np.asarray(source["old_newton"], dtype=np.float64)
        old_stages = np.asarray(source["gcnm_stages"], dtype=np.float64)
    if truth.shape != old_truth.shape or not np.allclose(truth, old_truth, atol=1e-7):
        raise ValueError("old and new predictions do not use identical holdout truth")
    if not np.allclose(newton, old_newton, atol=1e-7):
        raise ValueError("old and new evaluations do not use identical PVI/Newton inputs")
    if new_stages.shape[0] < 2 or old_stages.shape[0] < 2:
        raise ValueError("both prediction packs must contain two GCNM stages")

    mappings = MeshMappings(args.mappings, img_size=40)
    truth_images = _images(truth, mappings)
    panel_arrays = (
        truth_images,
        _images(newton, mappings),
        _images(old_stages[0], mappings),
        _images(old_stages[1], mappings),
        _images(new_stages[0], mappings),
        _images(new_stages[1], mappings),
    )
    correlations = np.stack(
        [
            np.asarray(
                [_corr(image, target) for image, target in zip(values, truth_images)]
            )
            for values in panel_arrays[1:]
        ],
        axis=1,
    )
    score = correlations[:, -1]
    ordered = np.argsort(score)
    split = len(ordered) // 2
    bad_pool, good_pool = ordered[:split], ordered[split:]
    rng = np.random.default_rng(args.seed)
    chosen = {
        "good": np.sort(rng.choice(good_pool, size=args.per_group, replace=False)),
        "bad": np.sort(rng.choice(bad_pool, size=args.per_group, replace=False)),
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7,
            "axes.titlesize": 6.5,
            "figure.titlesize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    tex_lines = [
        "% Auto-generated IEEE-style comparison figures.",
        "% Requires \\usepackage{graphicx}",
    ]
    for group in ("good", "bad"):
        group_indices = sorted(chosen[group], key=lambda index: score[index], reverse=True)
        for rank, sample_index in enumerate(group_indices, start=1):
            images = [values[sample_index] for values in panel_arrays]
            finite_values = np.concatenate(
                [np.abs(image[np.isfinite(image)]).ravel() for image in images]
            )
            limit = max(
                float(np.percentile(finite_values, 99.5)),
                float(np.nanmax(np.abs(images[0]))),
                1e-6,
            )
            figure, axes = plt.subplots(1, 6, figsize=(7.16, 1.72), dpi=300)
            shown = None
            for panel_index, (axis, image, title) in enumerate(
                zip(axes, images, PANEL_NAMES)
            ):
                shown = axis.imshow(
                    image,
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                    origin="lower",
                    interpolation="nearest",
                )
                if panel_index == 0:
                    axis.set_title(title)
                else:
                    axis.set_title(
                        f"{title}\n$r={correlations[sample_index, panel_index - 1]:.3f}$"
                    )
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
            figure.suptitle(
                f"{group.capitalize()} case {rank}: exact-nonlinear holdout sample {sample_index}",
                y=0.98,
            )
            colorbar = figure.colorbar(
                shown,
                ax=axes,
                orientation="horizontal",
                fraction=0.08,
                pad=0.08,
                aspect=45,
            )
            colorbar.set_label(r"Differential conductivity $\Delta\sigma$ (S/m)")
            colorbar.ax.tick_params(labelsize=6, length=2)
            figure.subplots_adjust(left=0.01, right=0.995, top=0.78, bottom=0.27, wspace=0.04)
            stem = f"{group}_{rank:02d}_sample_{sample_index:02d}"
            pdf_path = args.out_dir / f"{stem}.pdf"
            png_path = args.out_dir / f"{stem}.png"
            figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
            figure.savefig(png_path, bbox_inches="tight", pad_inches=0.015, dpi=300)
            plt.close(figure)

            record = {
                "group": group,
                "rank_within_selected_group": rank,
                "sample_index": int(sample_index),
                "selection_seed": args.seed,
                "selection_rule": f"random without replacement from {group} correlation half",
                "pvi_newton_correlation": float(correlations[sample_index, 0]),
                "old_stage_1_correlation": float(correlations[sample_index, 1]),
                "old_stage_2_correlation": float(correlations[sample_index, 2]),
                "new_stage_1_correlation": float(correlations[sample_index, 3]),
                "new_stage_2_correlation": float(correlations[sample_index, 4]),
                "pdf": str(pdf_path.resolve()),
                "png": str(png_path.resolve()),
            }
            records.append(record)
            caption = (
                f"{group.capitalize()} exact-nonlinear holdout case (sample "
                f"{sample_index}), randomly selected from the {group} half according "
                f"to new stage-2 image correlation. All panels show differential "
                f"conductivity on a shared symmetric scale."
            )
            tex_lines.extend(
                [
                    r"\begin{figure*}[t]",
                    r"  \centering",
                    rf"  \includegraphics[width=\textwidth]{{{_latex_escape(pdf_path.name)}}}",
                    rf"  \caption{{{caption}}}",
                    rf"  \label{{fig:{group}-{rank:02d}}}",
                    r"\end{figure*}",
                    "",
                ]
            )

    fieldnames = list(records[0])
    with (args.out_dir / "selection_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    (args.out_dir / "selection_manifest.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    (args.out_dir / "comparison_figures.tex").write_text(
        "\n".join(tex_lines), encoding="utf-8"
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
