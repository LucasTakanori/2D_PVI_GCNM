#!/usr/bin/env python3
"""Render a stratified visual audit of the 1,000-beat HP/LP archive."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_mesh_maps import MeshMappings


FRAMES_PER_BEAT = 50


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("data/hp_lp_beats_US120_v1")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rings_b045/US120.yaml")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/hp_lp_us120_v1/synthetic_diversity_atlas"),
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--validation-count", type=int, default=2)
    parser.add_argument("--test-count", type=int, default=2)
    return parser.parse_args()


def _image(mappings: MeshMappings, values: np.ndarray) -> np.ndarray:
    return mappings.elem_to_image_grid(np.asarray(values, dtype=np.float64))


def _selected_anatomies(
    anatomy_ids: np.ndarray, count: int, rng: np.random.Generator
) -> list[int]:
    unique = np.unique(anatomy_ids)
    if count < 1 or count > len(unique):
        raise ValueError(f"count must lie in [1, {len(unique)}]")
    return sorted(int(value) for value in rng.choice(unique, count, replace=False))


def _one_beat_indices(
    anatomy_ids: np.ndarray,
    beat_ids: np.ndarray,
    sample_indices: np.ndarray,
    anatomy_id: int,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray]:
    available = np.unique(beat_ids[anatomy_ids == anatomy_id])
    beat_id = int(rng.choice(available))
    indices = np.flatnonzero((anatomy_ids == anatomy_id) & (beat_ids == beat_id))
    indices = indices[np.argsort(sample_indices[indices])]
    if len(indices) != FRAMES_PER_BEAT or not np.array_equal(
        sample_indices[indices], np.arange(FRAMES_PER_BEAT)
    ):
        raise ValueError(f"anatomy {anatomy_id}, beat {beat_id} is incomplete")
    return beat_id, indices


def _render_card(
    *,
    mappings: MeshMappings,
    split: str,
    anatomy_id: int,
    beat_id: int,
    hp: dict[str, np.ndarray],
    lp: dict[str, np.ndarray],
    output: Path,
) -> dict:
    hp_sigma = hp["sigma"]
    lp_sigma = lp["sigma"]
    full_sigma = hp_sigma + lp_sigma
    hp_voltage = hp["V"]
    lp_voltage = lp["V"]
    full_voltage = hp_voltage + lp_voltage
    sigma_rms = {
        "HP": np.sqrt(np.mean(hp_sigma * hp_sigma, axis=1)),
        "LP": np.sqrt(np.mean(lp_sigma * lp_sigma, axis=1)),
        "Full": np.sqrt(np.mean(full_sigma * full_sigma, axis=1)),
    }
    peak = int(np.argmax(sigma_rms["Full"]))
    tissue = np.rint(_image(mappings, hp["tissue_labels"][0])).astype(float)
    tissue[~np.isfinite(tissue)] = np.nan
    spatial = np.stack(
        [
            _image(mappings, hp_sigma[peak]),
            _image(mappings, lp_sigma[peak]),
            _image(mappings, full_sigma[peak]),
        ]
    )
    finite = np.abs(spatial[np.isfinite(spatial)])
    limit = max(float(np.quantile(finite, 0.995)), 1e-8)

    figure = plt.figure(figsize=(15.5, 7.8), dpi=120, facecolor="white")
    grid = figure.add_gridspec(2, 4, height_ratios=[1.0, 0.85], hspace=0.34, wspace=0.25)
    tissue_axis = figure.add_subplot(grid[0, 0])
    tissue_axis.imshow(tissue, cmap="tab20", origin="upper", interpolation="nearest")
    tissue_axis.set_title("Finger tissue labels")
    tissue_axis.axis("off")

    for column, (name, image) in enumerate(zip(("HP", "LP", "Full"), spatial), start=1):
        axis = figure.add_subplot(grid[0, column])
        shown = axis.imshow(
            image,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            origin="upper",
        )
        axis.set_title(f"{name} Δσ · peak sample {peak + 1}/50")
        axis.axis("off")
        figure.colorbar(shown, ax=axis, fraction=0.047, pad=0.02, label="S/m")

    voltage_axis = figure.add_subplot(grid[1, :2])
    voltage_limit = max(float(np.quantile(np.abs(full_voltage), 0.995)), 1e-12)
    voltage_image = voltage_axis.imshow(
        full_voltage.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-voltage_limit,
        vmax=voltage_limit,
        origin="lower",
        extent=(1, 50, 1, full_voltage.shape[1]),
    )
    voltage_axis.axvline(peak + 1, color="black", linestyle="--", linewidth=1)
    voltage_axis.set_title("Full referenced differential voltage: 32 channels × 50 samples")
    voltage_axis.set_xlabel("Sample in beat")
    voltage_axis.set_ylabel("Measurement channel")
    figure.colorbar(voltage_image, ax=voltage_axis, fraction=0.025, pad=0.02, label="V")

    waveform_axis = figure.add_subplot(grid[1, 2:])
    samples = np.arange(1, FRAMES_PER_BEAT + 1)
    for name, values in sigma_rms.items():
        waveform_axis.plot(samples, values, linewidth=2, label=name)
    waveform_axis.axvline(peak + 1, color="black", linestyle="--", linewidth=1)
    waveform_axis.set_title("Conductivity-change RMS waveform")
    waveform_axis.set_xlabel("Sample in beat")
    waveform_axis.set_ylabel("RMS Δσ (S/m)")
    waveform_axis.grid(alpha=0.25)
    waveform_axis.legend(frameon=False)

    figure.suptitle(
        f"US120 synthetic diversity audit · {split} · anatomy {anatomy_id} · beat {beat_id}",
        fontsize=15,
        fontweight="semibold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return {
        "split": split,
        "anatomy_id": anatomy_id,
        "beat_id": beat_id,
        "peak_sample": peak,
        "hp_sigma_rms": float(np.sqrt(np.mean(hp_sigma * hp_sigma))),
        "lp_sigma_rms": float(np.sqrt(np.mean(lp_sigma * lp_sigma))),
        "full_sigma_rms": float(np.sqrt(np.mean(full_sigma * full_sigma))),
        "full_voltage_rms": float(np.sqrt(np.mean(full_voltage * full_voltage))),
        "image": output.name,
    }


def main() -> None:
    args = _args()
    if args.output_root.exists():
        raise FileExistsError(f"immutable atlas root already exists: {args.output_root}")
    cfg = GcnmConfig.from_yaml(args.config)
    mappings = MeshMappings(cfg.mappings_h5)
    rng = np.random.default_rng(args.seed)
    counts = {
        "train": args.train_count,
        "validation": args.validation_count,
        "test": args.test_count,
    }
    records = []
    for split, count in counts.items():
        hp_path = args.dataset_root / "hp" / f"{split}.npz"
        lp_path = args.dataset_root / "lp" / f"{split}.npz"
        with np.load(hp_path) as hp_source, np.load(lp_path) as lp_source:
            anatomy_ids = np.asarray(hp_source["anatomy_id"])
            beat_ids = np.asarray(hp_source["beat_id"])
            sample_indices = np.asarray(hp_source["sample_index"])
            np.testing.assert_array_equal(anatomy_ids, lp_source["anatomy_id"])
            np.testing.assert_array_equal(beat_ids, lp_source["beat_id"])
            for anatomy_id in _selected_anatomies(anatomy_ids, count, rng):
                beat_id, indices = _one_beat_indices(
                    anatomy_ids, beat_ids, sample_indices, anatomy_id, rng
                )
                hp = {
                    key: np.asarray(hp_source[key][indices])
                    for key in ("sigma", "V", "tissue_labels")
                }
                lp = {
                    key: np.asarray(lp_source[key][indices])
                    for key in ("sigma", "V", "tissue_labels")
                }
                name = f"{split}_anatomy{anatomy_id:03d}_beat{beat_id:04d}.png"
                records.append(
                    _render_card(
                        mappings=mappings,
                        split=split,
                        anatomy_id=anatomy_id,
                        beat_id=beat_id,
                        hp=hp,
                        lp=lp,
                        output=args.output_root / name,
                    )
                )

    manifest = {
        "schema": "pvi-gcnm-synthetic-diversity-atlas-v1",
        "dataset_root": str(args.dataset_root.resolve()),
        "config": str(args.config.resolve()),
        "seed": args.seed,
        "cards": records,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    cards = "\n".join(
        f'<figure><img src="{html.escape(item["image"])}" loading="lazy">'
        f'<figcaption>{html.escape(item["split"])} · anatomy {item["anatomy_id"]} · '
        f'beat {item["beat_id"]}</figcaption></figure>'
        for item in records
    )
    page = f"""<!doctype html>
<meta charset="utf-8"><title>US120 synthetic diversity atlas</title>
<style>body{{font:16px system-ui;margin:2rem;background:#f4f6f8;color:#17243a}}
main{{max-width:1500px;margin:auto}}figure{{background:white;padding:1rem;border-radius:12px;
box-shadow:0 2px 12px #0001;margin:1.5rem 0}}img{{width:100%;height:auto}}figcaption{{margin-top:.5rem}}
</style><main><h1>US120 1,000-beat synthetic diversity atlas</h1>
<p>Stratified distinct anatomies from train, validation, and exact-nonlinear test.</p>{cards}</main>"""
    (args.output_root / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(args.output_root), "cards": len(records)}))


if __name__ == "__main__":
    main()
