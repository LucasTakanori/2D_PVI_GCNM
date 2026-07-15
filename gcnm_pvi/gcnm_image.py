"""Project element conductivity to pixel images and plot comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gcnm_pvi.gcnm_mesh_maps import MeshMappings


def elem_to_image(sigma_elem: np.ndarray, mappings: MeshMappings) -> np.ndarray:
    return mappings.elem_to_image_grid(sigma_elem)


def image_mse(pred_elem: np.ndarray, true_elem: np.ndarray, mappings: MeshMappings) -> float:
    p = mappings.elem_to_image_grid(pred_elem)
    t = mappings.elem_to_image_grid(true_elem)
    return float(np.mean((p - t) ** 2))


def save_image_triplet(
    out_path: Path | str,
    sigma_true: np.ndarray,
    sigma_pred: np.ndarray,
    mappings: MeshMappings,
    title: str = "",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gt = mappings.elem_to_image_grid(sigma_true)
    pr = mappings.elem_to_image_grid(sigma_pred)
    vmin = min(gt.min(), pr.min())
    vmax = max(gt.max(), pr.max())
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))
    im0 = axs[0].imshow(gt, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axs[0].set_title("Ground truth")
    axs[1].imshow(pr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axs[1].set_title("Prediction")
    diff = pr - gt
    lim = np.max(np.abs(diff)) or 1e-6
    axs[2].imshow(diff, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim)
    axs[2].set_title("Difference")
    for ax in axs:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.colorbar(im0, ax=axs, fraction=0.02, pad=0.04)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
