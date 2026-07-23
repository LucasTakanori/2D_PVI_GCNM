#!/usr/bin/env python3
"""Make fair old-dataset comparisons for the core-guided two-stage model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcnm_pvi.gcnm_mesh_maps import MeshMappings


MAPPING = ROOT / "data/ring_meshes/subject006_US120/ring_US120_mappings_40.h5"
PRIMITIVE = "finger_default_primitive_coords_a1_b025_hom_seed0"
CORE_GUIDED = "finger_default_core_guided_hom_seed0"


def _correlation(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred = prediction - prediction.mean(axis=1, keepdims=True)
    target = truth - truth.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1)
    return np.divide(
        np.sum(pred * target, axis=1),
        denominator,
        out=np.zeros(len(pred), dtype=float),
        where=denominator > 1e-12,
    )


def _image(mapping: MeshMappings, values: np.ndarray) -> np.ndarray:
    return mapping.elem_to_image_grid(values)


def _limit(images: list[np.ndarray]) -> float:
    values = np.concatenate([np.abs(image[np.isfinite(image)]) for image in images])
    return max(float(np.percentile(values, 99.5)), 1e-8)


def synthetic_figure(mapping: MeshMappings, primitive, core, out: Path) -> None:
    truth = np.asarray(core["truth"])
    new_stage_2 = np.asarray(core["gcnm_stages"])[1]
    correlations = _correlation(new_stage_2, truth)
    order = np.argsort(correlations)
    selected = [int(order[-1]), int(order[len(order) // 2]), int(order[0])]
    row_names = ["Best", "Median", "Worst"]
    titles = [
        "Clean truth",
        "PVI-style Newton",
        "Primitive coordinate stage 2",
        "Core-guided stage 1",
        "Core-guided stage 2",
    ]
    figure, axes = plt.subplots(3, 5, figsize=(14.2, 8.3), dpi=150)
    for row, (label, index) in enumerate(zip(row_names, selected)):
        images = [
            _image(mapping, truth[index]),
            _image(mapping, np.asarray(core["old_newton"])[index]),
            _image(mapping, np.asarray(primitive["gcnm_stages"])[1, index]),
            _image(mapping, np.asarray(core["gcnm_stages"])[0, index]),
            _image(mapping, new_stage_2[index]),
        ]
        # Newton is intentionally excluded from the display-limit estimate so
        # its broad inverse artifacts cannot wash out the learned maps.
        limit = _limit([images[0], *images[2:]])
        shown = None
        for column, (axis, image, title) in enumerate(zip(axes[row], images, titles)):
            shown = axis.imshow(
                image, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="upper"
            )
            if row == 0:
                axis.set_title(title, fontsize=9)
            axis.axis("off")
            if column == 0:
                axis.text(
                    -0.08,
                    0.5,
                    f"{label}\nsample {index}\nr={correlations[index]:.3f}",
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9,
                )
        colorbar = figure.colorbar(
            shown, ax=list(axes[row]), orientation="horizontal", fraction=0.035, pad=0.02
        )
        colorbar.set_label(f"Delta conductivity (S/m), shared row scale +/-{limit:.4f}")
    figure.suptitle(
        "Same default-finger exact-nonlinear holdout: primitive vs core-guided",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.005,
        "Best/median/worst are selected by core-guided stage-2 image correlation.",
        ha="center",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.10, right=0.99, top=0.91, bottom=0.05, hspace=0.38)
    figure.savefig(out, facecolor="white")
    plt.close(figure)


def real_figure(mapping: MeshMappings, primitive, core, meta, out: Path) -> None:
    periods = [1060, 1061, 1062]
    period_id = np.asarray(meta["period_id"])
    stage_2 = np.asarray(core["gcnm_stages"])[1]
    selected = []
    for period in periods:
        candidates = np.flatnonzero(period_id == period)
        rms = np.sqrt(np.mean(stage_2[candidates] ** 2, axis=1))
        selected.append(int(candidates[int(np.argmax(rms))]))
    titles = [
        "PVI Newton pseudo-reference",
        "Primitive coordinate stage 2",
        "Core-guided stage 1",
        "Core-guided stage 2",
    ]
    figure, axes = plt.subplots(3, 4, figsize=(11.6, 8.2), dpi=150)
    for row, (period, index) in enumerate(zip(periods, selected)):
        images = [
            _image(mapping, np.asarray(core["truth"])[index]),
            _image(mapping, np.asarray(primitive["gcnm_stages"])[1, index]),
            _image(mapping, np.asarray(core["gcnm_stages"])[0, index]),
            _image(mapping, stage_2[index]),
        ]
        reference_limit = _limit([images[0]])
        model_limit = _limit(images[1:])
        for column, (axis, image, title) in enumerate(zip(axes[row], images, titles)):
            limit = reference_limit if column == 0 else model_limit
            axis.imshow(image, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="upper")
            if row == 0:
                axis.set_title(title, fontsize=9)
            axis.axis("off")
            if column == 0:
                axis.text(
                    -0.08,
                    0.5,
                    f"Period {period}\nsample {index}",
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9,
                )
        axes[row, 0].text(
            0.5,
            -0.06,
            f"pseudo-reference scale +/-{reference_limit:.4f} S/m",
            transform=axes[row, 0].transAxes,
            ha="center",
            fontsize=7,
        )
        axes[row, 2].text(
            0.5,
            -0.06,
            f"model shared scale +/-{model_limit:.4f} S/m",
            transform=axes[row, 2].transAxes,
            ha="center",
            fontsize=7,
        )
    figure.suptitle(
        "Subject 006 morphology at model scale (PVI is display-only, not truth)",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        "Each row uses the phase with the largest core-guided stage-2 output in that beat. "
        "PVI and model panels use explicitly separate scales.",
        ha="center",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.10, right=0.99, top=0.91, bottom=0.06, hspace=0.25)
    figure.savefig(out, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "data/faithful_results/finger_default_core_guided_hom_seed0/comparisons",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mapping = MeshMappings(MAPPING, img_size=40)
    base = ROOT / "data/faithful_results"
    with (
        np.load(base / PRIMITIVE / "evaluation_nonlinear/predictions.npz") as primitive_s,
        np.load(base / CORE_GUIDED / "evaluation_nonlinear/predictions.npz") as core_s,
    ):
        synthetic_figure(
            mapping,
            primitive_s,
            core_s,
            args.out_dir / "synthetic_primitive_vs_core_guided.png",
        )
    with (
        np.load(base / PRIMITIVE / "evaluation_real_pvi/predictions.npz") as primitive_r,
        np.load(base / CORE_GUIDED / "evaluation_real_pvi/predictions.npz") as core_r,
        np.load(ROOT / "data/subject006_gcnm_hdf/test.npz") as meta,
    ):
        real_figure(
            mapping,
            primitive_r,
            core_r,
            meta,
            args.out_dir / "subject006_primitive_vs_core_guided_model_scale.png",
        )
    print(args.out_dir)


if __name__ == "__main__":
    main()
