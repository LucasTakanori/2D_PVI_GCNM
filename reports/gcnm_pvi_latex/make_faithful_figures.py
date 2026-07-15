#!/usr/bin/env python3
"""Generate report figures from the executed faithful PVI-GCNM ablations."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcnm_pvi.gcnm_mesh_maps import MeshMappings


ABLATION_CANDIDATES = [
    ("faithful_direct_hom_seed0", "Direct", "architecture"),
    ("faithful_residual_hom_seed0", "Residual", "architecture"),
    ("faithful_coords_hom_seed0", "+ coordinates", "architecture"),
    ("faithful_weight1_hom_seed0", r"$\alpha=1$", "weight"),
    ("faithful_weight2_hom_seed0", r"$\alpha=2$", "weight"),
    ("faithful_weight4_hom_seed0", r"$\alpha=4$", "weight"),
    ("faithful_weight8_hom_seed0", r"$\alpha=8$", "weight"),
    ("faithful_background025_hom_seed0", r"BG $0.25$", "background"),
    ("faithful_background05_hom_seed0", r"BG $0.5$", "background"),
    ("faithful_background1_hom_seed0", r"BG $1.0$", "background"),
]


def _selected_experiment() -> str:
    experiment = os.environ.get("FINAL_FAITHFUL_EXPERIMENT")
    if not experiment:
        raise RuntimeError(
            "set FINAL_FAITHFUL_EXPERIMENT to the selected completed faithful "
            "experiment before generating report figures"
        )
    known = {name for name, _label, _group in ABLATION_CANDIDATES}
    if experiment not in known:
        raise ValueError(
            f"FINAL_FAITHFUL_EXPERIMENT={experiment!r} is not in the controlled "
            f"ablation roster: {sorted(known)}"
        )
    return experiment


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "savefig.bbox": "tight",
        }
    )


def _report(experiment: str, evaluation: str = "evaluation_nonlinear") -> dict:
    path = ROOT / "data/faithful_results" / experiment / evaluation / "report.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _final_metrics(experiment: str, evaluation: str = "evaluation_nonlinear") -> dict:
    return _report(experiment, evaluation)["faithful_gcnm"][-1]["metrics"]


def core_comparison() -> None:
    old = json.loads(
        (
            ROOT
            / "data/subject006_anatomical_results/evaluation_nonlinear/report.json"
        ).read_text()
    )
    direct = _report("faithful_direct_hom_seed0")
    residual = _report("faithful_residual_hom_seed0")
    labels = [
        "Saved\nNewton",
        "Iterative\nLM 2",
        "Fixed GCN\n2",
        "Direct GCN\n2",
        "Residual GCN\n2",
    ]
    metrics = [
        direct["saved_newton"],
        direct["iterative_lm"][-1]["metrics"],
        {
            "image_correlation": old["image"]["gcnm_correlation"],
            "image_rmse": old["image"]["gcnm_rmse"],
            "localization_dice_at_true_volume": old["gcnm"][
                "localization_dice_at_true_volume"
            ],
            "background_rms": old["gcnm"]["background_rms"],
        },
        direct["faithful_gcnm"][-1]["metrics"],
        residual["faithful_gcnm"][-1]["metrics"],
    ]
    specifications = [
        ("image_correlation", "Image correlation", True),
        ("localization_dice_at_true_volume", "Localization Dice", True),
        ("image_rmse", "Image RMSE", False),
        ("background_rms", "Background RMS", False),
    ]
    colors = ["#A0A0A0", "#72B7B2", "#E45756", "#4C78A8", "#F58518"]
    figure, axes = plt.subplots(2, 2, figsize=(8.8, 4.2))
    for axis, (key, title, higher) in zip(axes.ravel(), specifications):
        values = [entry[key] for entry in metrics]
        bars = axis.bar(np.arange(len(labels)), values, color=colors)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=8, ha="center")
        axis.tick_params(axis="x", labelsize=6.5, pad=2)
        axis.set_title(title + (" (higher is better)" if higher else " (lower is better)"))
        axis.set_ylim(top=max(values) * 1.18)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}" if value < 0.01 else f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
            )
    figure.suptitle("Exact nonlinear holdout: old method, physics control, and faithful core")
    figure.subplots_adjust(hspace=0.58, wspace=0.3, top=0.88, bottom=0.16)
    figure.savefig(OUT / "faithful_core_comparison.pdf")
    plt.close(figure)


def ablation_comparison(selected_experiment: str) -> None:
    available = []
    missing = []
    for experiment, label, group in ABLATION_CANDIDATES:
        path = ROOT / "data/faithful_results" / experiment / "evaluation_nonlinear/report.json"
        if path.exists():
            available.append((experiment, label, group, _final_metrics(experiment)))
        else:
            missing.append(experiment)
    if missing:
        raise RuntimeError(
            "faithful ablation reports are incomplete; missing exact nonlinear "
            f"reports for: {', '.join(missing)}"
        )
    labels = [
        entry[1] + (" *" if entry[0] == selected_experiment else "")
        for entry in available
    ]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 5.6))
    specifications = [
        ("image_correlation", "Image correlation", True),
        ("localization_dice_at_true_volume", "Localization Dice", True),
        ("image_rmse", "Image RMSE", False),
        ("background_rms", "Background RMS", False),
    ]
    group_colors = {
        "architecture": "#4C78A8",
        "weight": "#F58518",
        "background": "#54A24B",
    }
    for axis, (key, title, higher) in zip(axes.ravel(), specifications):
        values = [entry[3][key] for entry in available]
        colors = [group_colors[entry[2]] for entry in available]
        bars = axis.bar(x, values, color=colors)
        for bar, entry in zip(bars, available):
            if entry[0] == selected_experiment:
                bar.set_edgecolor("black")
                bar.set_linewidth(2.0)
                bar.set_hatch("//")
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_title(title + (" (higher is better)" if higher else " (lower is better)"))
        axis.set_ylim(top=max(values) * 1.16)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}" if value >= 0.01 else f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=5.8,
                rotation=90,
            )
    figure.suptitle(
        "Deployable homogeneous-baseline ablation sequence "
        "(* and hatching mark the selected experiment)"
    )
    figure.legend(
        handles=[
            Patch(facecolor=group_colors["architecture"], label="architecture/features"),
            Patch(facecolor=group_colors["weight"], label="vessel weighting"),
            Patch(facecolor=group_colors["background"], label="background penalty"),
            Patch(facecolor="white", edgecolor="black", hatch="//", label="selected"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=False,
    )
    figure.subplots_adjust(hspace=0.62, wspace=0.28, top=0.82)
    figure.savefig(OUT / "faithful_ablation_comparison.pdf")
    plt.close(figure)


def baseline_comparison() -> None:
    oracle = _final_metrics("faithful_direct_seed0")
    homogeneous = _final_metrics("faithful_direct_hom_seed0")
    keys = ["image_correlation", "localization_dice_at_true_volume", "image_rmse", "background_rms"]
    titles = ["Image correlation", "Localization Dice", "Image RMSE", "Background RMS"]
    figure, axes = plt.subplots(1, 4, figsize=(8.7, 2.5))
    for axis, key, title in zip(axes, keys, titles):
        values = [oracle[key], homogeneous[key]]
        axis.bar(["Oracle\nanatomy", "Homogeneous\ndeployable"], values, color=["#B279A2", "#4C78A8"])
        axis.set_title(title)
        axis.set_ylim(top=max(values) * 1.18)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.4f}" if value < 0.01 else f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    figure.suptitle("Baseline-knowledge ablation on the exact nonlinear holdout", y=1.04)
    figure.savefig(OUT / "faithful_baseline_comparison.pdf")
    plt.close(figure)


def output_gallery(final_experiment: str) -> None:
    mapping = MeshMappings(
        ROOT / "data/ring_meshes/subject006_US120/ring_US120_mappings_40.h5", 40
    )
    faithful = np.load(
        ROOT
        / "data/faithful_results"
        / final_experiment
        / "evaluation_nonlinear/predictions.npz"
    )
    old = np.load(
        ROOT
        / "data/subject006_anatomical_results/evaluation_nonlinear/predictions.npz"
    )
    values = [
        faithful["truth"],
        faithful["old_newton"],
        faithful["lm_stages"][-1],
        old["stages"][-1],
        faithful["gcnm_stages"][0],
        faithful["gcnm_stages"][-1],
    ]
    titles = ["Clean truth", "Saved Newton", "Iterative LM 2", "Old fixed GCN 2", "Faithful GCN 1", "Faithful GCN 2"]
    images = [
        np.stack([mapping.elem_to_image_grid(sample) for sample in array])
        for array in values
    ]
    correlations = np.asarray(
        [
            np.corrcoef(pred[np.isfinite(target)], target[np.isfinite(target)])[0, 1]
            for pred, target in zip(images[-1], images[0])
        ]
    )
    order = np.argsort(correlations)
    selected = [int(order[-1]), int(order[len(order) // 2]), int(order[0])]
    row_names = ["Best", "Median", "Worst"]
    figure, axes = plt.subplots(3, len(values), figsize=(10.0, 5.3))
    for row, (sample, row_name) in enumerate(zip(selected, row_names)):
        row_images = [stack[sample] for stack in images]
        finite = np.concatenate([image[np.isfinite(image)] for image in row_images])
        limit = max(float(np.quantile(np.abs(finite), 0.995)), 1e-8)
        for column, image in enumerate(row_images):
            axes[row, column].imshow(image, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower")
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(titles[column])
        axes[row, 0].text(
            -0.22,
            0.5,
            f"{row_name}\nsample {sample}\nr={correlations[sample]:.3f}",
            rotation=90,
            ha="center",
            va="center",
            transform=axes[row, 0].transAxes,
        )
    figure.suptitle(f"Ground truth and reconstruction evolution: {final_experiment}")
    figure.savefig(OUT / "faithful_output_gallery.pdf")
    plt.close(figure)


def real_pvi_gallery(final_experiment: str) -> None:
    """Compare the selected model with the shared homogeneous-baseline LM control."""

    mapping = MeshMappings(
        ROOT / "data/ring_meshes/subject006_US120/ring_US120_mappings_40.h5", 40
    )
    selected_path = (
        ROOT
        / "data/faithful_results"
        / final_experiment
        / "evaluation_real_pvi/predictions.npz"
    )
    control_path = (
        ROOT
        / "data/faithful_results/faithful_direct_hom_seed0"
        / "evaluation_real_pvi/predictions.npz"
    )
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    if not control_path.exists():
        raise FileNotFoundError(control_path)

    selected = np.load(selected_path)
    control = np.load(control_path)
    reference = np.asarray(selected["truth"])
    if reference.shape != control["truth"].shape or not np.allclose(
        reference, control["truth"], equal_nan=True
    ):
        raise RuntimeError(
            "selected real-PVI evaluation and homogeneous direct LM control do not "
            "use the identical pseudo-reference pack"
        )
    lm_stages = np.asarray(control["lm_stages"])
    gcnm_stages = np.asarray(selected["gcnm_stages"])
    if len(lm_stages) == 0:
        raise RuntimeError("shared homogeneous direct real evaluation has no LM control")
    if len(gcnm_stages) == 0:
        raise RuntimeError("selected real-PVI evaluation has no faithful GCNM stages")

    values = [reference]
    titles = ["PVI Newton\npseudo-reference"]
    values.extend(lm_stages)
    titles.extend([f"Iterative LM {index}" for index in range(1, len(lm_stages) + 1)])
    values.extend(gcnm_stages)
    titles.extend([f"Faithful GCNM {index}" for index in range(1, len(gcnm_stages) + 1)])
    values.append(gcnm_stages[-1] - reference)
    titles.append("Final GCNM minus\npseudo-reference")

    images = [
        np.stack([mapping.elem_to_image_grid(sample) for sample in array])
        for array in values
    ]
    sample_count = len(reference)
    if sample_count >= 10:
        selected_samples = [0, 3, 6, 9]
    else:
        selected_samples = np.linspace(
            0, sample_count - 1, min(4, sample_count), dtype=int
        ).tolist()
    figure, axes = plt.subplots(
        len(selected_samples), len(values), figsize=(10.2, 1.65 * len(selected_samples))
    )
    axes = np.atleast_2d(axes)
    for row, sample in enumerate(selected_samples):
        main_values = np.concatenate(
            [
                stack[sample][np.isfinite(stack[sample])]
                for stack in images[:-1]
            ]
        )
        limit = max(float(np.quantile(np.abs(main_values), 0.995)), 1e-8)
        error_values = images[-1][sample]
        error_limit = max(
            float(np.quantile(np.abs(error_values[np.isfinite(error_values)]), 0.995)),
            1e-8,
        )
        for column, stack in enumerate(images):
            this_limit = error_limit if column == len(images) - 1 else limit
            axes[row, column].imshow(
                stack[sample],
                cmap="RdBu_r",
                vmin=-this_limit,
                vmax=this_limit,
                origin="lower",
            )
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(titles[column])
        axes[row, 0].text(
            -0.22,
            0.5,
            f"held-out\nsample {sample}",
            rotation=90,
            ha="center",
            va="center",
            transform=axes[row, 0].transAxes,
        )
    figure.suptitle(
        "Real subject-6 PVI comparison: the reference is a Newton reconstruction, "
        "not anatomical truth"
    )
    figure.savefig(OUT / "faithful_real_pvi_gallery.pdf")
    plt.close(figure)


def main() -> None:
    _style()
    final_experiment = _selected_experiment()
    core_comparison()
    baseline_comparison()
    ablation_comparison(final_experiment)
    output_gallery(final_experiment)
    real_pvi_gallery(final_experiment)
    print(f"Wrote faithful report figures to {OUT}")


if __name__ == "__main__":
    main()
