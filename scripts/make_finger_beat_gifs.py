#!/usr/bin/env python3
"""Render complete simulator beats with anatomy and waveform context.

The GCNM training archives retain one selected phase per anatomy, but their
companion ``*_anatomy.json`` files retain the complete waveform and the full
FingerModel definition.  This script reconstructs those complete beats and
creates one GIF per selected anatomy.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from PIL import Image


TISSUE_COLORS = {
    0: "#EEF1F4",
    1: "#D9A066",
    2: "#E8CE8A",
    3: "#C0556B",
    4: "#C3CBD4",
    5: "#3E8E8A",
    6: "#B7263F",
    7: "#8C7B6B",
    8: "#E08A9B",
}
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
INK = "#13233B"
MUTED = "#5B6B7F"
ARTERIAL = "#C0304A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anatomy-json",
        type=Path,
        default=Path("data/finger_default_anatomical_exact/train_anatomy.json"),
    )
    parser.add_argument(
        "--simulator-root",
        type=Path,
        default=Path("../Finger-Conductivity-Simulator"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/finger_simulator_beats"),
    )
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        help="Explicit anatomy indices; overrides --count and --seed.",
    )
    parser.add_argument("--grid-size", type=int, default=112)
    parser.add_argument("--frame-duration-ms", type=int, default=100)
    parser.add_argument("--dpi", type=int, default=105)
    return parser.parse_args()


def load_simulator(simulator_root: Path):
    source = (simulator_root / "src").resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Simulator source directory not found: {source}")
    sys.path.insert(0, str(source))
    from finger_sim.models import FingerModel, WaveformSpec
    from finger_sim.simulation import simulate_grid

    return FingerModel, WaveformSpec, simulate_grid


def select_indices(total: int, args: argparse.Namespace) -> list[int]:
    if args.indices:
        indices = list(dict.fromkeys(args.indices))
    else:
        if args.count < 1 or args.count > total:
            raise ValueError(f"--count must be in [1, {total}]")
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(total, size=args.count, replace=False).tolist())
    invalid = [index for index in indices if index < 0 or index >= total]
    if invalid:
        raise IndexError(f"Anatomy indices outside [0, {total - 1}]: {invalid}")
    return indices


def _style_spatial_axis(axis, title: str) -> None:
    axis.set_title(title, color=INK, fontsize=12, pad=8)
    axis.set_xlabel("x (mm)", color=MUTED)
    axis.set_ylabel("y (mm)", color=MUTED)
    axis.set_aspect("equal")
    axis.tick_params(colors=MUTED, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#D3DCE3")


def render_beat(
    result,
    model,
    dataset_index: int,
    beat_number: int,
    output_path: Path,
    *,
    frame_duration_ms: int,
    dpi: int,
) -> dict:
    shape = result.grid_shape
    tissue = result.tissue_labels.reshape(shape)
    extent = [
        float(result.grid_x_mm[0]),
        float(result.grid_x_mm[-1]),
        float(result.grid_y_mm[0]),
        float(result.grid_y_mm[-1]),
    ]
    cmap = ListedColormap([TISSUE_COLORS[index] for index in range(9)])
    norm = BoundaryNorm(np.arange(-0.5, 9.5, 1.0), cmap.N)
    inside = (tissue != 0).astype(float)
    peak_delta = max(float(np.nanmax(np.abs(result.delta_sigma))), 1e-8)
    waveform_min = min(-0.04, float(np.nanmin(result.waveform)) - 0.04)
    waveform_max = max(1.04, float(np.nanmax(result.waveform)) + 0.04)

    frames: list[Image.Image] = []
    for frame_index, time_s in enumerate(result.time_s):
        figure = plt.figure(figsize=(13.2, 5.25), facecolor="white")
        grid = figure.add_gridspec(1, 3, width_ratios=[1.02, 1.02, 1.18], wspace=0.34)
        anatomy_axis = figure.add_subplot(grid[0, 0])
        delta_axis = figure.add_subplot(grid[0, 1])
        waveform_axis = figure.add_subplot(grid[0, 2])

        anatomy_axis.imshow(
            tissue,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        anatomy_axis.contour(
            result.grid_x_mm,
            result.grid_y_mm,
            inside,
            levels=[0.5],
            colors=[INK],
            linewidths=1.0,
        )
        _style_spatial_axis(anatomy_axis, "Finger tissue model at 50 kHz")

        delta_frame = result.delta_sigma[frame_index].reshape(shape)
        image = delta_axis.imshow(
            delta_frame,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-peak_delta,
            vmax=peak_delta,
            interpolation="bilinear",
        )
        delta_axis.contour(
            result.grid_x_mm,
            result.grid_y_mm,
            inside,
            levels=[0.5],
            colors=[INK],
            linewidths=1.0,
        )
        _style_spatial_axis(delta_axis, f"Differential conductivity Δσ · t={time_s:.3f} s")
        colorbar = figure.colorbar(image, ax=delta_axis, fraction=0.048, pad=0.035)
        colorbar.set_label("Δσ (S/m)", color=MUTED, fontsize=9)
        colorbar.ax.tick_params(colors=MUTED, labelsize=7)

        waveform_axis.plot(
            result.time_s,
            result.waveform,
            color=ARTERIAL,
            linewidth=2.5,
        )
        waveform_axis.fill_between(
            result.time_s,
            0.0,
            result.waveform,
            color=ARTERIAL,
            alpha=0.10,
        )
        waveform_axis.axvline(time_s, color=INK, linewidth=1.8, linestyle="--")
        waveform_axis.scatter(
            [time_s],
            [result.waveform[frame_index]],
            s=42,
            color=ARTERIAL,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        waveform_axis.set_xlim(float(result.time_s[0]), float(result.time_s[-1]))
        waveform_axis.set_ylim(waveform_min, waveform_max)
        waveform_axis.set_title("Input heartbeat waveform", color=INK, fontsize=12, pad=8)
        waveform_axis.set_xlabel("Time (s)", color=MUTED)
        waveform_axis.set_ylabel("Normalized amplitude", color=MUTED)
        waveform_axis.grid(color="#D3DCE3", linewidth=0.6, alpha=0.8)
        waveform_axis.tick_params(colors=MUTED, labelsize=8)
        for spine in waveform_axis.spines.values():
            spine.set_color("#D3DCE3")

        artery_text = ", ".join(
            f"{2 * artery.radius_x_mm:.2f}×{2 * artery.radius_y_mm:.2f} mm"
            for artery in model.arteries
        )
        figure.suptitle(
            f"Generated beat {beat_number} · training anatomy {dataset_index}",
            x=0.025,
            y=0.985,
            ha="left",
            fontsize=15,
            fontweight="semibold",
            color=INK,
        )
        figure.text(
            0.025,
            0.925,
            f"Finger {model.width_mm:.1f}×{model.height_mm:.1f} mm  |  "
            f"artery outer diameters {artery_text}",
            ha="left",
            color=MUTED,
            fontsize=9,
        )
        handles = [
            Patch(facecolor=TISSUE_COLORS[index], edgecolor="none", label=TISSUE_NAMES[index])
            for index in TISSUE_NAMES
        ]
        figure.legend(
            handles=handles,
            loc="lower center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.39, 0.005),
            fontsize=7.5,
        )
        figure.subplots_adjust(top=0.86, bottom=0.16, left=0.045, right=0.98)

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=dpi, facecolor="white")
        plt.close(figure)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return {
        "dataset_index": dataset_index,
        "gif": str(output_path),
        "frames": len(frames),
        "duration_s": float(result.time_s[-1] + (result.time_s[1] - result.time_s[0])),
        "finger_width_mm": float(model.width_mm),
        "finger_height_mm": float(model.height_mm),
        "artery_count": len(model.arteries),
        "peak_delta_s_m": peak_delta,
    }


def main() -> None:
    args = parse_args()
    anatomy_path = args.anatomy_json.resolve()
    simulator_root = args.simulator_root.resolve()
    output_dir = args.out_dir.resolve()
    records = json.loads(anatomy_path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty list in {anatomy_path}")
    indices = select_indices(len(records), args)
    FingerModel, WaveformSpec, simulate_grid = load_simulator(simulator_root)

    manifest = {
        "source_anatomy_json": str(anatomy_path),
        "available_anatomies": len(records),
        "selection_seed": None if args.indices else args.seed,
        "selected_indices": indices,
        "gifs": [],
    }
    for beat_number, dataset_index in enumerate(indices, start=1):
        record = records[dataset_index]
        model = FingerModel.from_dict(record["simulator_finger_model"])
        waveform = record["waveform"]
        waveform_spec = WaveformSpec(
            kind="custom",
            frames=int(waveform["frames"]),
            duration_s=float(waveform["duration_s"]),
            custom_values=[float(value) for value in waveform["values"]],
            normalize="none",
        )
        result = simulate_grid(model, waveform_spec, size=args.grid_size)
        output_path = output_dir / f"beat_{beat_number:02d}_train_{dataset_index:04d}.gif"
        item = render_beat(
            result,
            model,
            dataset_index,
            beat_number,
            output_path,
            frame_duration_ms=args.frame_duration_ms,
            dpi=args.dpi,
        )
        manifest["gifs"].append(item)
        print(f"Wrote {output_path}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
