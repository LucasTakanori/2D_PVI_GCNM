#!/usr/bin/env python3
"""Regenerate Gallery A with the exact four columns in the asset specification."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcnm_pvi.gcnm_mesh_maps import MeshMappings


def main() -> None:
    mapping = MeshMappings(
        ROOT / "data/ring_meshes/subject006_US120/ring_US120_mappings_40.h5",
        img_size=40,
    )
    saved = np.load(
        ROOT / "data/subject006_anatomical_results/evaluation_nonlinear/predictions.npz"
    )
    values = np.stack(
        [
            saved["truth"][0],
            saved["newton"][0],
            saved["stages"][0, 0],
            saved["stages"][1, 0],
        ]
    )
    images = np.stack([mapping.elem_to_image_grid(row) for row in values])
    titles = [
        "Clean ground truth",
        "Physical Newton",
        "Fixed-feature GCNM\nstage 1",
        "Fixed-feature GCNM\nstage 2",
    ]
    finite = np.concatenate([image[np.isfinite(image)] for image in images])
    limit = float(np.quantile(np.abs(finite), 0.995))

    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )
    figure, axes = plt.subplots(2, 4, figsize=(8.3, 4.0))
    shown = None
    for column, (image, title) in enumerate(zip(images, titles)):
        shown = axes[0, column].imshow(
            image,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            origin="lower",
        )
        axes[0, column].set_title(title)
        axes[0, column].axis("off")
        axes[1, column].imshow(
            image - images[0],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            origin="lower",
        )
        axes[1, column].axis("off")

    axes[0, 0].text(
        -0.16,
        0.5,
        "Output",
        rotation=90,
        va="center",
        ha="center",
        transform=axes[0, 0].transAxes,
        weight="bold",
    )
    axes[1, 0].text(
        -0.16,
        0.5,
        "Prediction minus truth",
        rotation=90,
        va="center",
        ha="center",
        transform=axes[1, 0].transAxes,
        weight="bold",
    )
    assert shown is not None
    colorbar = figure.colorbar(shown, ax=axes, fraction=0.022, pad=0.018)
    colorbar.set_label(r"Conductivity change $\Delta\sigma$ (S/m)")
    figure.suptitle(
        "Earlier fixed-feature evolution on one shared exact-nonlinear phantom",
        y=0.995,
    )
    output = ASSET_ROOT / "source_images/gallery_a_method_evolution.pdf"
    figure.savefig(output)
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
