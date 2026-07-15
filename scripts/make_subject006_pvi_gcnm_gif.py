#!/usr/bin/env python3
"""Animate PVI or synthetic GCNM comparisons across beats or holdout phantoms."""

from __future__ import annotations

import argparse
from io import BytesIO
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcnm_pvi.gcnm_mesh_maps import MeshMappings


DEFAULT_EXPERIMENT = "faithful_background025_hom_seed0"

SPLIT_PRESETS = {
    "test": {
        "kind": "cardiac",
        "meta": ROOT / "data/subject006_gcnm_hdf/test.npz",
        "evaluation": "evaluation_real_pvi",
        "split_label": "held-out test PVI (pseudo-reference only)",
        "reference_title": "PVI Newton pseudo-reference",
        "default_out": ROOT
        / "reports/gcnm_pvi_latex/figures/subject006_pvi_gcnm_held_out_test.gif",
    },
    "train": {
        "kind": "cardiac",
        "meta": ROOT / "data/subject006_gcnm_hdf/train.npz",
        "evaluation": "evaluation_real_pvi_train",
        "split_label": "train-split PVI (pseudo-reference only)",
        "reference_title": "PVI Newton pseudo-reference",
        "default_out": ROOT
        / "reports/gcnm_pvi_latex/figures/subject006_pvi_gcnm_train_split.gif",
    },
    "synthetic": {
        "kind": "phantom",
        "evaluation": "evaluation_nonlinear",
        "split_label": "synthetic exact-nonlinear holdout (clean ground truth)",
        "reference_title": "Clean ground truth",
        "default_samples": [0, 1, 2],
        "default_out": ROOT
        / "reports/gcnm_pvi_latex/figures/synthetic_holdout_three_phantoms.gif",
    },
}


def _ordered_unique_periods(meta: dict[str, np.ndarray]) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for period in meta["period_id"]:
        period_id = int(period)
        if period_id not in seen:
            seen.add(period_id)
            ordered.append(period_id)
    return ordered


def _consecutive_periods(meta: dict[str, np.ndarray], count: int) -> list[int]:
    ordered = _ordered_unique_periods(meta)
    for start in range(len(ordered) - count + 1):
        run = ordered[start : start + count]
        if all(run[index + 1] - run[index] == 1 for index in range(count - 1)):
            return run
    raise RuntimeError(
        f"no run of {count} consecutive period_id values found in the split pack"
    )


def _period_indices(period_id: int, meta: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    entries = [
        (int(phase), index)
        for index, (period, phase) in enumerate(zip(meta["period_id"], meta["phase"]))
        if int(period) == period_id
    ]
    return sorted(entries, key=lambda item: item[0])


def _render_frame(
    images: list[np.ndarray],
    *,
    beat_label: str,
    split_label: str,
    panel_titles: list[str],
    footer: str,
    limit: float,
) -> Image.Image:
    if len(images) != len(panel_titles):
        raise ValueError(
            f"received {len(images)} images but {len(panel_titles)} panel titles"
        )
    columns = min(4, len(images))
    rows = math.ceil(len(images) / columns)
    figure, axes_grid = plt.subplots(
        rows,
        columns,
        figsize=(3.0 * columns, 2.65 * rows + 0.55),
        dpi=120,
        squeeze=False,
    )
    axes = list(axes_grid.ravel())
    shown = None
    for axis, image, title in zip(axes, images, panel_titles):
        shown = axis.imshow(image, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower")
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    for axis in axes[len(images) :]:
        axis.axis("off")
    figure.suptitle(f"{beat_label}  |  {split_label}", fontsize=11, y=0.98)
    figure.text(0.5, 0.02, footer, ha="center", fontsize=8, color="0.35")
    figure.subplots_adjust(
        left=0.025,
        right=0.90,
        top=0.91,
        bottom=0.07,
        wspace=0.08,
        hspace=0.18,
    )
    if shown is not None:
        colorbar = figure.colorbar(
            shown,
            ax=axes[: len(images)],
            fraction=0.025,
            pad=0.025,
        )
        colorbar.set_label(r"$\Delta\sigma$ (S/m)", fontsize=8)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", facecolor="white")
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _save_gif(
    frames: list[Image.Image],
    *,
    out_path: Path,
    frame_duration_ms: int,
    summary: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"Wrote {out_path} ({len(frames)} frames, {summary})")


def build_cardiac_gif(
    *,
    predictions_path: Path,
    meta_path: Path,
    split_label: str,
    reference_title: str,
    period_ids: list[int],
    out_path: Path,
    frame_duration_ms: int,
    expected_stages: int | None,
) -> None:
    mapping = MeshMappings(
        ROOT / "data/ring_meshes/subject006_US120/ring_US120_mappings_40.h5",
        img_size=40,
    )
    predictions = np.load(predictions_path)
    meta = np.load(meta_path)

    reference = predictions["truth"]
    gcnm_stages = predictions["gcnm_stages"]
    _validate_stage_count(gcnm_stages, predictions_path, expected_stages)
    panel_titles = [reference_title] + [
        f"GCNM stage {stage}" for stage in range(1, gcnm_stages.shape[0] + 1)
    ]

    for period_id in period_ids:
        if not _period_indices(period_id, meta):
            raise ValueError(f"period {period_id} is not present in {meta_path}")

    selected_indices = [
        sample_index
        for period_id in period_ids
        for _phase, sample_index in _period_indices(period_id, meta)
    ]
    limit = _shared_limit([reference], gcnm_stages, selected_indices)

    frames: list[Image.Image] = []
    for beat_number, period_id in enumerate(period_ids, start=1):
        beat_label = f"Beat {beat_number} (period {period_id})"
        for phase, sample_index in _period_indices(period_id, meta):
            frames.append(
                _render_frame(
                    _panel_images(mapping, [reference], gcnm_stages, sample_index),
                    beat_label=beat_label,
                    split_label=split_label,
                    panel_titles=panel_titles,
                    footer=f"phase {phase}   |   shared scale +/- {limit:.4f} S/m",
                    limit=limit,
                )
            )

    _save_gif(
        frames,
        out_path=out_path,
        frame_duration_ms=frame_duration_ms,
        summary=f"{len(period_ids)} beats x 10 phases, periods {period_ids}",
    )


def build_synthetic_gif(
    *,
    predictions_path: Path,
    split_label: str,
    reference_title: str,
    sample_indices: list[int],
    out_path: Path,
    frame_duration_ms: int,
    hold_frames: int,
    expected_stages: int | None,
) -> None:
    mapping = MeshMappings(
        ROOT / "data/ring_meshes/subject006_US120/ring_US120_mappings_40.h5",
        img_size=40,
    )
    predictions = np.load(predictions_path)
    truth = predictions["truth"]
    old_newton = predictions["old_newton"] if "old_newton" in predictions else None
    gcnm_stages = predictions["gcnm_stages"]
    _validate_stage_count(gcnm_stages, predictions_path, expected_stages)
    references = [truth]
    panel_titles = [reference_title]
    if old_newton is not None and len(old_newton) == len(truth):
        references.append(old_newton)
        panel_titles.append("PVI-style Newton")
    panel_titles.extend(
        f"GCNM stage {stage}" for stage in range(1, gcnm_stages.shape[0] + 1)
    )

    for sample_index in sample_indices:
        if sample_index < 0 or sample_index >= len(truth):
            raise ValueError(
                f"sample {sample_index} is outside holdout range [0, {len(truth) - 1}]"
            )

    limit = _shared_limit(references, gcnm_stages, sample_indices)
    frames: list[Image.Image] = []
    for beat_number, sample_index in enumerate(sample_indices, start=1):
        beat_label = f"Beat {beat_number} (holdout phantom {sample_index})"
        frame = _render_frame(
            _panel_images(mapping, references, gcnm_stages, sample_index),
            beat_label=beat_label,
            split_label=split_label,
            panel_titles=panel_titles,
            footer=f"static holdout phantom   |   shared scale +/- {limit:.4f} S/m",
            limit=limit,
        )
        frames.extend([frame] * hold_frames)

    _save_gif(
        frames,
        out_path=out_path,
        frame_duration_ms=frame_duration_ms,
        summary=(
            f"{len(sample_indices)} holdout phantoms x {hold_frames} hold frames, "
            f"samples {sample_indices}"
        ),
    )


def _panel_images(
    mapping: MeshMappings,
    references: list[np.ndarray],
    gcnm_stages: np.ndarray,
    sample_index: int,
) -> list[np.ndarray]:
    return [
        mapping.elem_to_image_grid(values[sample_index])
        for values in [*references, *gcnm_stages]
    ]


def _shared_limit(
    references: list[np.ndarray],
    gcnm_stages: np.ndarray,
    sample_indices: list[int],
) -> float:
    selected = [values[sample_indices].ravel() for values in references]
    selected.append(gcnm_stages[:, sample_indices].ravel())
    all_values = np.concatenate(selected)
    return max(float(np.quantile(np.abs(all_values), 0.995)), 1e-8)


def _validate_stage_count(
    gcnm_stages: np.ndarray,
    predictions_path: Path,
    expected_stages: int | None,
) -> None:
    actual = int(gcnm_stages.shape[0])
    if actual < 1:
        raise RuntimeError(f"{predictions_path} does not contain any GCNM stages")
    if expected_stages is not None and actual != expected_stages:
        raise RuntimeError(
            f"{predictions_path} contains {actual} GCNM stages; expected "
            f"{expected_stages}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=sorted(SPLIT_PRESETS),
        default="test",
        help="dataset split preset (default: test)",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help="faithful_results experiment folder (default: %(default)s)",
    )
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument(
        "--periods",
        nargs="+",
        type=int,
        default=None,
        help="cardiac periods to animate (cardiac presets only)",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        type=int,
        default=None,
        help="holdout phantom indices to animate (synthetic preset only)",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--frame-ms", type=int, default=180)
    parser.add_argument(
        "--expected-stages",
        type=int,
        default=None,
        help="fail unless the evaluation contains exactly this many GCNM stages",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=10,
        help="frames to hold each static synthetic phantom (default: 10)",
    )
    args = parser.parse_args()

    preset = SPLIT_PRESETS[args.split]
    predictions_path = (
        args.predictions
        or ROOT
        / "data/faithful_results"
        / args.experiment
        / preset["evaluation"]
        / "predictions.npz"
    )
    out_path = args.out or preset["default_out"]
    reference_title = str(preset["reference_title"])

    if preset["kind"] == "phantom":
        sample_indices = (
            list(args.samples)
            if args.samples
            else list(preset.get("default_samples", [0, 1, 2]))
        )
        build_synthetic_gif(
            predictions_path=predictions_path,
            split_label=str(preset["split_label"]),
            reference_title=reference_title,
            sample_indices=sample_indices,
            out_path=out_path,
            frame_duration_ms=args.frame_ms,
            hold_frames=args.hold_frames,
            expected_stages=args.expected_stages,
        )
        return

    meta_path = args.meta or preset["meta"]
    meta = np.load(meta_path)
    period_ids = list(args.periods) if args.periods else _consecutive_periods(meta, 3)
    build_cardiac_gif(
        predictions_path=predictions_path,
        meta_path=meta_path,
        split_label=str(preset["split_label"]),
        reference_title=reference_title,
        period_ids=period_ids,
        out_path=out_path,
        frame_duration_ms=args.frame_ms,
        expected_stages=args.expected_stages,
    )


if __name__ == "__main__":
    main()
