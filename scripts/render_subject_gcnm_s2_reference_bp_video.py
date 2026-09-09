#!/usr/bin/env python3
"""Render a publication-style GCNM stage-2/reference-BP MP4.

The selected BP-artifact manifest identifies the exact held-out window.  The
stage-2 frames are read from the serialized coordinate-direct Parquet row and
the corresponding reference BP periods are read directly from the immutable
source HDF5 session.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as pads


PERIOD_LENGTH = 50


def _fixed_array(table, column: str, shape: tuple[int, ...]) -> np.ndarray:
    values = table[column][0].values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(shape)


def _example_for_session(manifest: dict, session: str) -> dict:
    matches = [row for row in manifest["examples"] if row["session"] == session]
    if len(matches) != 1:
        raise ValueError(f"expected one {session!r} example, found {len(matches)}")
    return matches[0]


def _source_hdf5(coordinate_root: Path, source_name: str) -> Path:
    manifest = json.loads((coordinate_root / "manifest.json").read_text())
    matches = [
        Path(row["source_hdf5"])
        for row in manifest["source_sessions"]
        if row["source_name"] == source_name
    ]
    if len(matches) != 1:
        raise ValueError(f"could not resolve one HDF5 source for {source_name}")
    return matches[0]


def _load_inputs(
    artifact_manifest: Path,
    session: str,
    displayed_beats: int,
    sample_id: str | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    manifest = json.loads(artifact_manifest.read_text())
    if sample_id is None:
        selected_example = _example_for_session(manifest, session)
        sample_id = str(selected_example["sample_id"])
        selection = "artifact median-error example"
    else:
        selection = "explicit held-out sample"
    split_manifest = Path(manifest["split_manifest"])
    assignments = json.loads(split_manifest.read_text())["assignments"]
    data_split = assignments.get(sample_id)
    if data_split is None:
        raise ValueError(f"split manifest omits sample {sample_id}")
    if selection == "explicit held-out sample" and data_split != "test":
        raise ValueError(
            f"explicit sample must be held out, but its split is {data_split!r}"
        )
    coordinate_root = Path(manifest["coordinate_root"])
    table = pads.dataset(
        str(coordinate_root / "shards"), format="parquet"
    ).to_table(
        filter=pads.field("sample_id") == sample_id,
        columns=[
            "sample_id", "subject", "session", "source_name", "mask_start",
            "mask_stop", "s2", "bp_waveform",
        ],
    )
    if table.num_rows != 1:
        raise ValueError(f"expected one Parquet row, found {table.num_rows}")
    row = {name: table[name][0].as_py() for name in (
        "sample_id", "subject", "session", "source_name", "mask_start", "mask_stop"
    )}
    if row["session"] != session:
        raise ValueError(
            f"sample belongs to session {row['session']!r}, not {session!r}"
        )
    s2 = _fixed_array(table, "s2", (1, 40, 40, 5 * PERIOD_LENGTH))[0]
    parquet_target = _fixed_array(table, "bp_waveform", (PERIOD_LENGTH,))

    if displayed_beats < 1 or displayed_beats > 5:
        raise ValueError("displayed_beats must be between one and five")
    first_display_frame = (5 - displayed_beats) * PERIOD_LENGTH
    s2 = s2[..., first_display_frame:]

    source_hdf5 = _source_hdf5(coordinate_root, row["source_name"])
    first_period = int(row["mask_stop"]) - displayed_beats
    last_period = int(row["mask_stop"])
    with h5py.File(source_hdf5, "r") as handle:
        bp = np.asarray(
            handle["data/bp/signal"][
                0,
                first_period * PERIOD_LENGTH : last_period * PERIOD_LENGTH,
            ],
            dtype=np.float32,
        )

    if s2.shape[-1] != bp.size:
        raise ValueError(f"S2/BP frame mismatch: {s2.shape[-1]} versus {bp.size}")
    if not np.array_equal(
        bp[-PERIOD_LENGTH:], parquet_target, equal_nan=True
    ):
        raise ValueError("the final recovered BP period differs from the Parquet target")
    provenance = {
        "artifact_manifest": str(artifact_manifest.resolve()),
        "split_manifest": str(split_manifest.resolve()),
        "data_split": data_split,
        "selection": selection,
        "sample_id": row["sample_id"],
        "subject": row["subject"],
        "session": row["session"],
        "source_name": row["source_name"],
        "source_hdf5": str(source_hdf5.resolve()),
        "mask_period_bounds_zero_based": [
            int(row["mask_start"]),
            int(row["mask_stop"]),
        ],
        "displayed_period_bounds_zero_based": [first_period, last_period],
        "displayed_frames": int(bp.size),
    }
    return s2, bp, provenance


def render_video(
    *,
    artifact_manifest: Path,
    session: str,
    output: Path,
    poster: Path,
    report: Path,
    displayed_beats: int,
    frame_ms: int,
    ffmpeg_path: Path,
    sample_id: str | None,
) -> None:
    s2, bp, provenance = _load_inputs(
        artifact_manifest, session, displayed_beats, sample_id
    )
    conductivity = 1_000.0 * s2
    domain = np.isfinite(conductivity).any(axis=-1)
    x = np.arange(bp.size, dtype=float)

    plt.rcParams.update(
        {
            # Nimbus Sans is the Arial-compatible sans-serif face installed on
            # the cluster and is used explicitly to avoid platform-dependent
            # Matplotlib fallback typography.
            "font.family": "Nimbus Sans",
            "font.size": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "animation.ffmpeg_path": str(ffmpeg_path),
        }
    )
    figure = plt.figure(figsize=(16, 7.2), dpi=120, facecolor="white")
    grid = figure.add_gridspec(
        1, 2, width_ratios=(1.65, 1.0), left=0.075, right=0.97,
        top=0.965, bottom=0.13, wspace=0.15
    )
    bp_axis = figure.add_subplot(grid[0, 0])
    image_axis = figure.add_subplot(grid[0, 1])

    bp_axis.plot(x, bp, color="#172554", linewidth=1.5)
    cursor = bp_axis.axvline(
        0, color="#d1495b", linewidth=0.9, alpha=0.95,
        zorder=4, clip_on=True
    )
    point, = bp_axis.plot(
        [0], [bp[0]], marker="o", markersize=4.0, color="#d1495b",
        markeredgecolor="white", markeredgewidth=0.6, zorder=5
    )
    bp_axis.set_xlim(0, bp.size - 1)
    bp_axis.set_ylim(80, 140)
    bp_axis.set_yticks([80, 110, 140])
    bp_axis.set_xticks([])
    bp_axis.set_ylabel("Blood pressure (mm Hg)")
    bp_axis.spines["top"].set_visible(False)
    bp_axis.spines["right"].set_visible(False)
    bp_axis.spines["bottom"].set_visible(False)
    bp_axis.tick_params(axis="y", length=3, width=0.8)

    quarter_period = PERIOD_LENGTH / 4
    scale_end = bp.size - 1
    scale_start = scale_end - quarter_period
    scale_y = 82.0
    bp_axis.plot(
        [scale_start, scale_end], [scale_y, scale_y],
        color="black", linewidth=1.0, clip_on=True
    )

    image = image_axis.imshow(
        conductivity[..., 0], cmap="RdBu_r", vmin=-5.0, vmax=5.0,
        origin="upper", interpolation="nearest"
    )
    padded_domain = np.pad(domain.astype(float), 1, mode="constant")
    rows = np.arange(-1, domain.shape[0] + 1)
    columns = np.arange(-1, domain.shape[1] + 1)
    image_axis.contour(
        columns, rows, padded_domain, levels=[0.5], colors="black",
        linewidths=2.0
    )
    image_axis.set_aspect("equal")
    image_axis.set_xlim(-1, domain.shape[1])
    image_axis.set_ylim(domain.shape[0], -1)
    image_axis.axis("off")
    colorbar = figure.colorbar(
        image, ax=image_axis, orientation="horizontal", fraction=0.060,
        pad=0.075, ticks=[-5, 0, 5]
    )
    colorbar.set_label("Conductivity change (mS/m)", labelpad=7)
    colorbar.ax.tick_params(length=3, width=0.8)

    def update(frame_index: int) -> None:
        image.set_data(conductivity[..., frame_index])
        cursor.set_xdata([frame_index, frame_index])
        point.set_data([frame_index], [bp[frame_index]])

    output.parent.mkdir(parents=True, exist_ok=True)
    update(0)
    figure.savefig(poster, dpi=120, facecolor="white")
    writer = animation.FFMpegWriter(
        fps=1_000.0 / frame_ms,
        codec="libx264",
        extra_args=[
            "-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ],
        metadata={
            "title": "Reference blood pressure and GCNM stage 2",
            "artist": "PVI-GCNM",
        },
    )
    with writer.saving(figure, str(output), dpi=120):
        for frame_index in range(bp.size):
            update(frame_index)
            writer.grab_frame(facecolor="white")
    plt.close(figure)

    provenance.update(
        {
            "schema": "pvi-gcnm-s2-reference-bp-video-v2",
            "output_mp4": str(output.resolve()),
            "poster_png": str(poster.resolve()),
            "frame_duration_ms": frame_ms,
            "frames_per_period": PERIOD_LENGTH,
            "displayed_periods": displayed_beats,
            "conductivity_units": "mS/m",
            "conductivity_color_limits_mS_per_m": [-5.0, 5.0],
            "conductivity_colorbar_ticks_mS_per_m": [-5, 0, 5],
            "bp_units": "mm Hg",
            "bp_axis_ticks_mm_Hg": [80, 110, 140],
            "bp_axis_limits_mm_Hg": [80, 140],
            "scale_bar_period_fraction": 0.25,
            "scale_bar_label": None,
            "scale_bar_end_caps": False,
            "moving_indicator": "red vertical cursor through waveform point",
            "visible_panel_titles": False,
            "font": "Nimbus Sans (Arial-compatible cluster face), 20 pt",
            "finger_contour": "boundary of the finite 40x40 GCNM raster domain",
            "gcnm_stage2_q99_peak_to_peak_mS_per_m_by_period": [
                float(np.ptp(np.nanquantile(
                    conductivity[..., start : start + PERIOD_LENGTH].reshape(
                        -1, PERIOD_LENGTH
                    ),
                    0.99,
                    axis=0,
                )))
                for start in range(0, conductivity.shape[-1], PERIOD_LENGTH)
            ],
        }
    )
    report.write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--session", default="valsalva")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poster", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--displayed-beats", type=int, default=3)
    parser.add_argument("--frame-ms", type=int, default=90)
    parser.add_argument("--ffmpeg-path", type=Path, required=True)
    parser.add_argument("--sample-id")
    args = parser.parse_args()
    render_video(
        artifact_manifest=args.artifact_manifest,
        session=args.session,
        output=args.output,
        poster=args.poster,
        report=args.report,
        displayed_beats=args.displayed_beats,
        frame_ms=args.frame_ms,
        ffmpeg_path=args.ffmpeg_path,
        sample_id=args.sample_id,
    )


if __name__ == "__main__":
    main()
