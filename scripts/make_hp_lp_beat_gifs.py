#!/usr/bin/env python3
"""Render true absolute anatomy and derived 50-sample HP/LP synthetic beats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_mesh_maps import MeshMappings


FRAMES_PER_BEAT = 50


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/full_band_beats_US120_v2")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rings_b045/US120.yaml")
    )
    parser.add_argument(
        "--atlas-manifest",
        type=Path,
        default=Path(
            "reports/hp_lp_us120_v1/synthetic_diversity_atlas/manifest.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/hp_lp_us120_v1/synthetic_diversity_beat_gifs"),
    )
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def _grid(mappings: MeshMappings, values: np.ndarray) -> np.ndarray:
    return mappings.elem_to_image_grid(np.asarray(values, dtype=np.float64))


def _load_beat(
    root: Path, split: str, anatomy_id: int, beat_id: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    loaded = []
    for component in ("full", "hp", "lp"):
        with np.load(root / component / f"{split}.npz") as source:
            mask = (source["anatomy_id"] == anatomy_id) & (
                source["beat_id"] == beat_id
            )
            indices = np.flatnonzero(mask)
            indices = indices[np.argsort(source["sample_index"][indices])]
            if len(indices) != FRAMES_PER_BEAT or not np.array_equal(
                source["sample_index"][indices], np.arange(FRAMES_PER_BEAT)
            ):
                raise ValueError(
                    f"incomplete {component} beat: {split}/{anatomy_id}/{beat_id}"
                )
            keys = ["sigma", "V", "tissue_labels"]
            if component == "full":
                keys.extend(("sigma_delta_reference", "sigma_resting", "V_delta_reference"))
            loaded.append({key: np.asarray(source[key][indices]) for key in keys})
    return loaded[0], loaded[1], loaded[2]


def _select(records: list[dict], count: int) -> list[dict]:
    if count < 1 or count > len(records):
        raise ValueError(f"count must lie in [1, {len(records)}]")
    # Ensure that the small gallery crosses all available data partitions before
    # filling remaining positions in the atlas' deterministic order.
    selected: list[dict] = []
    for split in ("train", "validation", "test"):
        item = next((record for record in records if record["split"] == split), None)
        if item is not None and item not in selected and len(selected) < count:
            selected.append(item)
    for item in records:
        if item not in selected and len(selected) < count:
            selected.append(item)
    return selected


def _render(
    *,
    mappings: MeshMappings,
    record: dict,
    full: dict[str, np.ndarray],
    hp: dict[str, np.ndarray],
    lp: dict[str, np.ndarray],
    output: Path,
    fps: int,
) -> dict:
    hp_sigma = hp["sigma"].astype(np.float64)
    lp_sigma = lp["sigma"].astype(np.float64)
    full_sigma = full["sigma_delta_reference"].astype(np.float64)
    full_voltage = full["V_delta_reference"].astype(np.float64)
    np.testing.assert_allclose(full_sigma, hp_sigma + lp_sigma, atol=2e-6)
    np.testing.assert_allclose(
        full_voltage,
        hp["V"].astype(np.float64) + lp["V"].astype(np.float64),
        atol=2e-9,
    )
    anatomical_absolute = np.stack(
        [_grid(mappings, row) for row in full["sigma"].astype(np.float64)]
    )
    grids = {
        "HP Δσ": np.stack([_grid(mappings, row) for row in hp_sigma]),
        "LP Δσ": np.stack([_grid(mappings, row) for row in lp_sigma]),
        "Total Δσ": np.stack([_grid(mappings, row) for row in full_sigma]),
    }
    finite_delta = np.concatenate(
        [np.abs(values[np.isfinite(values)]) for values in grids.values()]
    )
    delta_limit = max(float(np.quantile(finite_delta, 0.995)), 1e-8)
    finite_absolute = anatomical_absolute[np.isfinite(anatomical_absolute)]
    absolute_lower, absolute_upper = np.quantile(finite_absolute, (0.005, 0.995))
    sigma_rms = np.sqrt(np.mean(full_sigma * full_sigma, axis=1))
    voltage_rms = np.sqrt(np.mean(full_voltage * full_voltage, axis=1))

    figure = plt.figure(figsize=(15.8, 8.6), dpi=105, facecolor="white")
    layout = figure.add_gridspec(2, 4, height_ratios=(1.0, 0.65), hspace=0.30)
    image_axes = [figure.add_subplot(layout[0, column]) for column in range(4)]
    images = []
    panels = [
        (
            "True simulated anatomical absolute conductivity",
            anatomical_absolute,
            "viridis",
            float(absolute_lower),
            float(absolute_upper),
        ),
        ("HP conductivity change", grids["HP Δσ"], "RdBu_r", -delta_limit, delta_limit),
        ("LP conductivity change", grids["LP Δσ"], "RdBu_r", -delta_limit, delta_limit),
        ("Total conductivity change (HP + LP)", grids["Total Δσ"], "RdBu_r", -delta_limit, delta_limit),
    ]
    for axis, (title, values, cmap, lower, upper) in zip(image_axes, panels):
        shown = axis.imshow(values[0], cmap=cmap, vmin=lower, vmax=upper, origin="upper")
        axis.set_title(title, fontsize=10)
        axis.axis("off")
        figure.colorbar(shown, ax=axis, fraction=0.047, pad=0.02, label="S/m")
        images.append(shown)

    sigma_axis = figure.add_subplot(layout[1, :2])
    voltage_axis = figure.add_subplot(layout[1, 2:])
    samples = np.arange(1, FRAMES_PER_BEAT + 1)
    sigma_axis.plot(samples, sigma_rms, color="#1665d8", linewidth=2)
    voltage_axis.plot(samples, voltage_rms * 1e6, color="#d34b32", linewidth=2)
    sigma_cursor = sigma_axis.axvline(1, color="black", linestyle="--")
    voltage_cursor = voltage_axis.axvline(1, color="black", linestyle="--")
    sigma_axis.set(title="Total Δσ RMS", xlabel="Sample in beat", ylabel="S/m", xlim=(1, 50))
    voltage_axis.set(title="Full referenced voltage RMS", xlabel="Sample in beat", ylabel="µV", xlim=(1, 50))
    for axis in (sigma_axis, voltage_axis):
        axis.grid(alpha=0.25)
    title = figure.suptitle("")

    def update(frame: int):
        for shown, (_, values, _, _, _) in zip(images, panels):
            shown.set_data(values[frame])
        sigma_cursor.set_xdata([frame + 1, frame + 1])
        voltage_cursor.set_xdata([frame + 1, frame + 1])
        title.set_text(
            "US120 synthetic beat · "
            f"{record['split']} · anatomy {record['anatomy_id']} · "
            f"beat {record['beat_id']} · sample {frame + 1}/50"
        )
        return [*images, sigma_cursor, voltage_cursor, title]

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=fps)
    animation.FuncAnimation(
        figure, update, frames=FRAMES_PER_BEAT, interval=1000 / fps, blit=False
    ).save(output, writer=writer)
    plt.close(figure)
    return {
        "split": record["split"],
        "anatomy_id": int(record["anatomy_id"]),
        "beat_id": int(record["beat_id"]),
        "frames": FRAMES_PER_BEAT,
        "delta_scale_s_m": delta_limit,
        "absolute_panel_contract": "stored true simulated sigma_absolute",
        "true_anatomical_absolute_available": True,
        "gif": output.name,
    }


def main() -> None:
    args = _args()
    if args.output_root.exists():
        raise FileExistsError(f"immutable output root already exists: {args.output_root}")
    cfg = GcnmConfig.from_yaml(args.config)
    mappings = MeshMappings(cfg.mappings_h5)
    atlas = json.loads(args.atlas_manifest.read_text(encoding="utf-8"))
    records = []
    for item in _select(atlas["cards"], args.count):
        full, hp, lp = _load_beat(
            args.dataset_root,
            item["split"],
            int(item["anatomy_id"]),
            int(item["beat_id"]),
        )
        name = (
            f"{item['split']}_anatomy{int(item['anatomy_id']):03d}_"
            f"beat{int(item['beat_id']):04d}_50samples.gif"
        )
        records.append(
            _render(
                mappings=mappings,
                record=item,
                full=full,
                hp=hp,
                lp=lp,
                output=args.output_root / name,
                fps=args.fps,
            )
        )
    manifest = {
        "schema": "pvi-gcnm-synthetic-diversity-beat-gifs-v2",
        "dataset_root": str(args.dataset_root.resolve()),
        "source_atlas": str(args.atlas_manifest.resolve()),
        "absolute_field": "true simulator anatomy stored by the full-band v2 archive",
        "gifs": records,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_root), "gifs": len(records)}))


if __name__ == "__main__":
    main()
