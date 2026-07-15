#!/usr/bin/env python3
"""Generate publication figures for the GCNM--PVI technical report."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcnm_pvi.gcnm_mesh_maps import MeshMappings


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def measurement_comparison() -> None:
    labels = ["Injun: 32-electrode\nadjacent protocol", "PVI: 8-electrode\nring protocol"]
    values = [32 * 29, 32]
    colors = ["#4C78A8", "#F58518"]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_yscale("log")
    ax.set_ylabel("Voltage measurements per frame (log scale)")
    ax.set_title("Boundary-information difference")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value}", ha="center")
    ax.text(0.5, 0.88, "29$\\times$ fewer measurements", transform=ax.transAxes, ha="center", weight="bold")
    fig.savefig(OUT / "measurement_comparison.pdf")
    plt.close(fig)


def training_curves() -> None:
    report = json.load(
        open(
            ROOT
            / "data/subject006_anatomical_results/gcnm_subject006_anatomical_training_report.json"
        )
    )
    histories = report["history"]
    fig, axes = plt.subplots(1, len(histories), figsize=(7.2, 2.9), sharey=True)
    if len(histories) == 1:
        axes = [axes]
    for stage, (axis, history) in enumerate(zip(axes, histories), start=1):
        values = np.asarray(history, dtype=float)
        axis.plot(values[:, 0], label="training", color="#4C78A8")
        axis.plot(values[:, 1], label="validation", color="#F58518")
        best = int(np.argmin(values[:, 1]))
        axis.scatter([best], [values[best, 1]], color="#E45756", zorder=3)
        axis.set_title(f"GCNM stage {stage}")
        axis.set_xlabel("Epoch")
        axis.set_yscale("log")
        axis.text(best, values[best, 1] * 1.08, f"best={best}", ha="center", fontsize=7)
    axes[0].set_ylabel("Weighted objective")
    axes[-1].legend(frameon=False)
    fig.suptitle("Greedy stage-wise clean-truth training", y=1.02)
    fig.savefig(OUT / "training_curves.pdf")
    plt.close(fig)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def _dice(prediction: np.ndarray, truth: np.ndarray) -> float:
    scores = []
    for pred, target in zip(prediction, truth):
        support = np.flatnonzero(np.abs(target) > 1e-8)
        chosen = np.argpartition(np.abs(pred), -len(support))[-len(support) :]
        scores.append(len(np.intersect1d(support, chosen)) / len(support))
    return float(np.mean(scores))


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    support = np.abs(truth) > 1e-8
    return {
        "correlation": _corr(prediction, truth),
        "dice": _dice(prediction, truth),
        "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "background": float(np.sqrt(np.mean(prediction[~support] ** 2))),
    }


def _mapping() -> MeshMappings:
    return MeshMappings(
        ROOT / "data/ring_meshes/subject006_US120/ring_US120_mappings_40.h5",
        img_size=40,
    )


def _image_stack(mapping: MeshMappings, values: np.ndarray) -> np.ndarray:
    return np.stack([mapping.elem_to_image_grid(row) for row in values])


def _finite_corr(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    return _corr(left[valid], right[valid])


def experiment_evolution() -> None:
    """Compare the old smoke result and earlier fixed-feature model on one phantom."""
    mapping = _mapping()
    old = np.load(ROOT / "data/subject006_anatomical_results/evaluation/predictions.npz")
    fixed_feature = np.load(
        ROOT / "data/subject006_anatomical_results/evaluation_nonlinear/predictions.npz"
    )
    if not np.allclose(old["truth"][0], fixed_feature["truth"][0]):
        raise RuntimeError(
            "smoke and earlier fixed-feature sample 0 no longer share the same truth"
        )

    values = np.stack(
        [
            fixed_feature["truth"][0],
            fixed_feature["newton"][0],
            old["gcnm"][0],
            fixed_feature["stages"][0, 0],
            fixed_feature["stages"][1, 0],
        ]
    )
    images = _image_stack(mapping, values)
    titles = [
        "Clean ground truth",
        "Physical Newton",
        "Previous smoke\nGCNM (one stage)",
        "Earlier fixed-feature\nGCNM stage 1",
        "Earlier fixed-feature\nGCNM stage 2",
    ]
    finite_values = np.concatenate([image[np.isfinite(image)] for image in images])
    limit = float(np.quantile(np.abs(finite_values), 0.995))

    fig, axes = plt.subplots(2, 5, figsize=(9.0, 3.9))
    for column, (image_value, title) in enumerate(zip(images, titles)):
        shown = axes[0, column].imshow(
            image_value, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower"
        )
        axes[0, column].set_title(title)
        axes[0, column].axis("off")
        error = image_value - images[0]
        axes[1, column].imshow(
            error, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower"
        )
        axes[1, column].axis("off")
    axes[0, 0].text(
        -0.15, 0.5, "Output", rotation=90, va="center", ha="center",
        transform=axes[0, 0].transAxes, weight="bold"
    )
    axes[1, 0].text(
        -0.15, 0.5, "Error from truth", rotation=90, va="center", ha="center",
        transform=axes[1, 0].transAxes, weight="bold"
    )
    colorbar = fig.colorbar(shown, ax=axes, fraction=0.018, pad=0.015)
    colorbar.set_label("Conductivity change")
    fig.suptitle(
        "Earlier fixed-feature experiment evolution on one exact nonlinear phantom",
        y=0.99,
    )
    fig.savefig(OUT / "experiment_evolution.pdf")
    plt.close(fig)


def clean_truth_gallery() -> None:
    """Show objective cases from the earlier fixed-feature nonlinear experiment."""
    mapping = _mapping()
    data = np.load(
        ROOT / "data/subject006_anatomical_results/evaluation_nonlinear/predictions.npz"
    )
    truth_images = _image_stack(mapping, data["truth"])
    newton_images = _image_stack(mapping, data["newton"])
    stage1_images = _image_stack(mapping, data["stages"][0])
    stage2_images = _image_stack(mapping, data["stages"][1])
    correlations = np.asarray(
        [_finite_corr(pred, truth) for pred, truth in zip(stage2_images, truth_images)]
    )
    order = np.argsort(correlations)
    selected = [int(order[-1]), int(order[len(order) // 2]), int(order[0])]
    labels = ["Best", "Median", "Worst"]
    stacks = [truth_images, newton_images, stage1_images, stage2_images]
    titles = [
        "Clean ground truth",
        "Newton",
        "Fixed-feature GCNM 1",
        "Fixed-feature GCNM 2",
    ]

    fig, axes = plt.subplots(3, 4, figsize=(7.4, 5.6))
    for row, (sample, label) in enumerate(zip(selected, labels)):
        row_values = np.stack([stack[sample] for stack in stacks])
        finite_values = np.concatenate([value[np.isfinite(value)] for value in row_values])
        limit = float(np.quantile(np.abs(finite_values), 0.995))
        for column, value in enumerate(row_values):
            axes[row, column].imshow(
                value, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower"
            )
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(titles[column])
        axes[row, 0].text(
            -0.22,
            0.5,
            f"{label}: sample {sample}\n$r={correlations[sample]:.3f}$\nscale $\\pm${limit:.3f}",
            rotation=90,
            va="center",
            ha="center",
            transform=axes[row, 0].transAxes,
        )
    fig.suptitle(
        "Earlier fixed-feature exact-nonlinear holdout: objective case selection",
        y=0.995,
    )
    fig.savefig(OUT / "clean_truth_gallery.pdf")
    plt.close(fig)


def pvi_output_comparison() -> None:
    """Compare archived PVI, production pseudo-labels, and the earlier distilled GCN."""
    mapping = _mapping()
    output = np.load(
        ROOT / "data/subject006_smoke/gcnm_subject006_US120_smoke_training_output.npz"
    )
    train = np.load(ROOT / "data/subject006_gcnm_hdf/smoke/train.npz")
    validation = np.load(ROOT / "data/subject006_gcnm_hdf/smoke/validation.npz")
    periods = np.concatenate([train["period_id"], validation["period_id"]])
    phases = np.concatenate([train["phase"], validation["phase"]])
    pseudo = _image_stack(mapping, output["TRUTHS"])
    prediction = _image_stack(mapping, output["PREDICTIONS"][1])
    h5_path_text = os.environ.get("PVI_SUBJECT006_H5")
    if not h5_path_text:
        raise RuntimeError(
            "set PVI_SUBJECT006_H5 to the private subject006_baseline_masked.h5 "
            "file when regenerating the archived-PVI comparison"
        )
    h5_path = Path(h5_path_text)
    flat_indices = (periods.astype(int) - 1) * 50 + phases.astype(int)
    with h5py.File(h5_path, "r") as handle:
        archived = np.moveaxis(
            np.asarray(handle["data/pviHP/img"][:, :, flat_indices]), -1, 0
        )

    selected = [0, 2, 4, 6]
    main_values = np.concatenate(
        [
            archived[selected][np.isfinite(archived[selected])],
            pseudo[selected][np.isfinite(pseudo[selected])],
            prediction[selected][np.isfinite(prediction[selected])],
        ]
    )
    limit = float(np.quantile(np.abs(main_values), 0.995))
    errors = prediction[selected] - archived[selected]
    error_limit = float(np.quantile(np.abs(errors[np.isfinite(errors)]), 0.995))

    fig, axes = plt.subplots(4, 4, figsize=(7.8, 7.4))
    titles = [
        "Archived PVI image",
        "Production Newton\npseudo-label",
        "Previous one-epoch\nGCN output",
        "GCN minus PVI",
    ]
    for row, sample in enumerate(selected):
        row_images = [archived[sample], pseudo[sample], prediction[sample], errors[row]]
        for column, value in enumerate(row_images):
            this_limit = error_limit if column == 3 else limit
            axes[row, column].imshow(
                value, cmap="RdBu_r", vmin=-this_limit, vmax=this_limit, origin="lower"
            )
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(titles[column])
        axes[row, 0].text(
            -0.18,
            0.5,
            f"Period 1\nphase {int(phases[sample])}",
            rotation=90,
            va="center",
            ha="center",
            transform=axes[row, 0].transAxes,
        )
    fig.suptitle(
        "Earlier PVI-distillation outputs across one cardiac period\n"
        "(PVI and Newton are reconstructions, not anatomical ground truth)",
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        f"shared output scale: $\\pm${limit:.4f}; difference scale: $\\pm${error_limit:.4f}",
        ha="center",
        fontsize=8,
    )
    fig.savefig(OUT / "pvi_output_comparison.pdf")
    plt.close(fig)

    metrics = {
        "samples": int(len(archived)),
        "archived_pvi_vs_production_pseudo_image_correlation": _finite_corr(
            archived, pseudo
        ),
        "archived_pvi_vs_previous_gcn_image_correlation": _finite_corr(
            archived, prediction
        ),
        "production_pseudo_vs_previous_gcn_image_correlation": _finite_corr(
            pseudo, prediction
        ),
        "scope": (
            "Ten train/validation smoke examples; descriptive only, not independent "
            "test performance or anatomical validation."
        ),
    }
    with open(OUT / "qualitative_comparison_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def stage_results() -> None:
    names = ["Linearized holdout", "Exact nonlinear holdout"]
    folders = ["evaluation_linearized", "evaluation_nonlinear"]
    methods = ["Newton", "GCNM stage 1", "GCNM stage 2"]
    colors = ["#A0A0A0", "#4C78A8", "#E45756"]
    all_metrics: list[list[dict[str, float]]] = []
    for folder in folders:
        data = np.load(ROOT / "data/subject006_anatomical_results" / folder / "predictions.npz")
        predictions = [data["newton"], data["stages"][0], data["stages"][1]]
        all_metrics.append([_metrics(pred, data["truth"]) for pred in predictions])

    fig, axes = plt.subplots(2, 2, figsize=(8.1, 5.5))
    specifications = [
        ("correlation", "Conductivity correlation", True),
        ("dice", "Localization Dice", True),
        ("rmse", "Element RMSE", False),
        ("background", "Background RMS", False),
    ]
    x = np.arange(len(names))
    width = 0.24
    for axis, (key, title, higher) in zip(axes.ravel(), specifications):
        for method_index, (method, color) in enumerate(zip(methods, colors)):
            values = [all_metrics[dataset][method_index][key] for dataset in range(2)]
            bars = axis.bar(x + (method_index - 1) * width, values, width, label=method, color=color)
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )
        axis.set_xticks(x, names)
        axis.set_ylim(top=axis.get_ylim()[1] * 1.16)
        axis.set_title(
            title + (" (higher is better)" if higher else " (lower is better)"),
            pad=12,
        )
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(1.1, 1.34))
    fig.subplots_adjust(hspace=0.45, wspace=0.28, top=0.84)
    fig.savefig(OUT / "stage_results.pdf")
    plt.close(fig)


def algorithm_flow() -> None:
    """Compare the executable flow of the three repositories in one figure."""

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 7.4))
    columns = [
        (
            "1. Injun Lee: original 2D GCNM",
            "Solves absolute synthetic EIT",
            [
                "Build 32-electrode PyEIT\nforward + inverse FEM meshes",
                "Generate absolute $\\sigma^*$ phantoms;\nsimulate noisy boundary $V$",
                "Inverse elements $\\rightarrow$ graph $G$;\ninitialize $\\sigma_0=1$",
                "Stage $k$: recompute\n$F(\\sigma_k)$, $J_k$, and residual $r_k$",
                "Solve LM direction\n$(J_k^T J_k+\\lambda I)\\,\\delta\\sigma_k=-J_k^T r_k$",
                "Features $[\\sigma_k,\\delta\\sigma_k]$\n$\\rightarrow$ stage-specific $\\mathrm{GCN}_k$",
                "Train direct $\\sigma_{k+1}$ against\nclean $\\sigma^*$ using MSE",
                "Feed prediction forward;\nrepeat 10 stages, then test",
            ],
            "#4C78A8",
            "GCNM_training.py  |  GCNM_testing.py",
        ),
        (
            "2. Original PVI repository",
            "Solves production differential PVI",
            [
                "8-electrode ring acquisition:\nScioSpec .eit + NOVA BP/ECG",
                "Import + timestamp checks; remap\nchannels $\\rightarrow$ 32 complex $\\Delta V$",
                "Remove transient; 5 Hz filter;\nsynchronize and crop BP/ECG",
                "Peak/IBI alignment +\nmanual trial approval",
                "Subject ring mesh + one-step\nregularized Newton $\\Delta\\sigma$",
                "m2i PVI image + SVD ROI;\nsplit 100-frame LP / HP",
                "Heartbeat segmentation;\nresample every cycle to 50 phases",
                "Physiological/artifact masks;\nclean HDF5 tensors + metadata",
            ],
            "#72B7B2",
            "scionova_01--04  |  pvi_inv_make / pvi_inv_solve",
        ),
        (
            "3. Our faithful 2D PVI-GCNM",
            "Learns clean differential conductivity",
            [
                "Load US120 meshes, PVI FEM, $R$,\n40$\\times$40 map, and graph $G$",
                "Training: randomized anatomy on\nfine/inverse meshes $\\rightarrow$ clean $\\Delta\\sigma^*$",
                "Training: simulate fine-mesh $\\Delta V$ + noise\nInference: ingest real 32-channel $\\Delta V$",
                "Deployable $\\sigma_b=0.7$ S/m;\ninitialize $\\Delta\\sigma_0=0$",
                "Stage $k$: recompute differential\n$F$, residual $r_k$, and Jacobian $J_k$",
                "Solve regularized LM $p_k$; features\n$[\\Delta\\sigma_k,p_k,x,y,\\rho]$",
                "Train/load $\\mathrm{GCN}_k\\rightarrow\\Delta\\sigma_{k+1}$;\nclean labels; repeat 2 stages",
                "Save intermediate conductivity + voltage\nresidual; compare Newton / LM / GCNM",
            ],
            "#F58518",
            "generate_anatomical_dataset.py  |  train/evaluate_faithful_gcnm.py",
        ),
    ]
    for axis, (title, objective, boxes, color, source) in zip(axes, columns):
        axis.set_xlim(0, 1)
        axis.set_ylim(-0.45, len(boxes) + 0.95)
        axis.axis("off")
        axis.set_title(title, weight="bold", fontsize=10.5, pad=25, color=color)
        axis.text(
            0.5,
            len(boxes) + 0.55,
            objective,
            ha="center",
            va="center",
            fontsize=8.2,
            weight="bold",
            color="0.25",
        )
        for index, label in enumerate(boxes):
            y = len(boxes) - index - 0.15
            axis.text(
                0.5,
                y,
                f"{index + 1}.  {label}",
                ha="center",
                va="center",
                fontsize=7.8,
                linespacing=1.18,
                bbox={
                    "boxstyle": "round,pad=0.34",
                    "facecolor": color,
                    "alpha": 0.14,
                    "edgecolor": color,
                    "linewidth": 1.15,
                },
            )
            if index < len(boxes) - 1:
                axis.annotate(
                    "",
                    xy=(0.5, y - 0.76),
                    xytext=(0.5, y - 0.38),
                    arrowprops={"arrowstyle": "-|>", "color": "0.35", "lw": 0.9},
                )
        axis.text(
            0.5,
            -0.22,
            source,
            ha="center",
            va="center",
            fontsize=6.4,
            color="0.35",
            style="italic",
        )

    # Real PVI supplies measured voltage to the learned inverse.  Its Newton image
    # is retained only as a pseudo-reference and never used as a clean target.
    axes[2].annotate(
        "",
        xy=(0.015, 0.665),
        xytext=(-0.17, 0.665),
        xycoords="axes fraction",
        textcoords="axes fraction",
        annotation_clip=False,
        arrowprops={"arrowstyle": "-|>", "color": "#2A7F62", "lw": 1.7},
    )
    axes[2].text(
        -0.075,
        0.695,
        "real 32-channel $\\Delta V$\n(PVI image = pseudo-reference only)",
        transform=axes[2].transAxes,
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#2A7F62",
        weight="bold",
        clip_on=False,
    )

    fig.suptitle(
        "Three-repository algorithm flow: original learned solver, production PVI, and faithful adaptation",
        fontsize=13,
        weight="bold",
        y=0.97,
    )
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.055, top=0.84, wspace=0.17)
    fig.savefig(OUT / "algorithm_flow.pdf")
    fig.savefig(OUT / "algorithm_flow.png", dpi=240)
    plt.close(fig)


def main() -> None:
    _style()
    measurement_comparison()
    training_curves()
    stage_results()
    algorithm_flow()
    experiment_evolution()
    clean_truth_gallery()
    pvi_output_comparison()
    print(f"Wrote report figures to {OUT}")


if __name__ == "__main__":
    main()
