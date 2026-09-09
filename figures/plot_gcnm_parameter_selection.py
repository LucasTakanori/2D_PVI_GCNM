#!/usr/bin/env python3
"""Replot the GCNM parameter-selection data from the exported CSV file."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "gcnm_parameter_selection_data.csv"
OUTPUT = HERE / "gcnm_parameter_selection_replotted.pdf"

with DATA.open(newline="") as stream:
    rows = list(csv.DictReader(stream))

labels = [row["label"] + (" *" if row["selected"] == "true" else "") for row in rows]
x = np.arange(len(rows))
group_colors = {
    "architecture/features": "#4C78A8",
    "vessel weighting": "#F58518",
    "background penalty": "#54A24B",
}
panels = [
    ("image_correlation", "Image correlation", True),
    ("localization_dice_at_true_volume", "Localization Dice", True),
    ("image_rmse", "Image RMSE", False),
    ("background_rms", "Background RMS", False),
]

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

figure, axes = plt.subplots(2, 2, figsize=(9.0, 5.6))
for axis, (key, title, higher_is_better) in zip(axes.ravel(), panels):
    values = [float(row[key]) for row in rows]
    colors = [group_colors[row["group"]] for row in rows]
    bars = axis.bar(x, values, color=colors)

    for bar, row, value in zip(bars, rows, values):
        if row["selected"] == "true":
            bar.set_edgecolor("black")
            bar.set_linewidth(2.0)
            bar.set_hatch("//")
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}" if value >= 0.01 else f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=5.8,
            rotation=90,
        )

    axis.set_xticks(x, labels, rotation=35, ha="right")
    direction = "higher is better" if higher_is_better else "lower is better"
    axis.set_title(f"{title} ({direction})")
    axis.set_ylim(top=max(values) * 1.16)

figure.suptitle(
    "Deployable homogeneous-baseline ablation sequence "
    "(* and hatching mark the selected experiment)"
)
figure.legend(
    handles=[
        Patch(facecolor=group_colors["architecture/features"], label="architecture/features"),
        Patch(facecolor=group_colors["vessel weighting"], label="vessel weighting"),
        Patch(facecolor=group_colors["background penalty"], label="background penalty"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="selected"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.91),
    ncol=4,
    frameon=False,
)
figure.subplots_adjust(hspace=0.62, wspace=0.28, top=0.82)
figure.savefig(OUTPUT)
plt.close(figure)

print(OUTPUT)
