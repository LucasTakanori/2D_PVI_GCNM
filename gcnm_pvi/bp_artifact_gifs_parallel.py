"""High-throughput prediction-aligned GIF rendering for future BP runs.

This is intentionally a separate entry point from :mod:`bp_artifact_gifs`.
Existing Slurm jobs keep using the original renderer, while new submissions can
reuse each Matplotlib figure and render independent frame chunks in processes.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from gcnm_pvi import bp_artifact_gifs as legacy


def _frame_chunks(frame_count: int, workers: int) -> list[list[int]]:
    """Return balanced, contiguous frame chunks with no empty work items."""

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    worker_count = min(frame_count, workers)
    quotient, remainder = divmod(frame_count, worker_count)
    chunks: list[list[int]] = []
    start = 0
    for worker_index in range(worker_count):
        size = quotient + (worker_index < remainder)
        chunks.append(list(range(start, start + size)))
        start += size
    return chunks


def _configure_bp_panel(
    axis,
    *,
    output_mode: str,
    prediction: np.ndarray,
    target: np.ndarray,
    waveform: np.ndarray,
):
    """Draw invariant BP content once and return the movable sample marker."""

    legacy._draw_bp_panel(
        axis,
        output_mode=output_mode,
        prediction=prediction,
        target=target,
        waveform=waveform,
        displayed_beat=1,
        sample_in_beat=0,
    )
    if output_mode == "waveform":
        marker = axis.axvline(
            1, color="#64748b", linestyle="--", linewidth=1.3, visible=False
        )
    else:
        marker = axis.axvline(
            1, color="#64748b", linestyle=":", linewidth=1.2, visible=False
        )
    return marker


def _render_frame_chunk(
    offsets: list[int],
    frame_root: str,
    image_panels: list[np.ndarray],
    image_titles: list[str],
    limits: list[float],
    output_mode: str,
    prediction: np.ndarray,
    target: np.ndarray,
    waveform: np.ndarray,
    heading: str,
) -> list[int]:
    """Render one chunk while reusing its figure, axes, and colorbars."""

    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), dpi=95)
    image_artists = []
    first_frame_index = legacy.FIRST_DISPLAY_FRAME + offsets[0]
    for axis, values, title, limit in zip(
        axes.flat[:5], image_panels, image_titles, limits
    ):
        artist = axis.imshow(
            values[..., first_frame_index],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            origin="upper",
        )
        image_artists.append(artist)
        axis.set_title(f"{title}\n±{limit:.2e} S/m", fontsize=9)
        axis.axis("off")
        figure.colorbar(
            artist,
            ax=axis,
            orientation="horizontal",
            fraction=0.055,
            pad=0.055,
        )

    bp_axis = axes.flat[5]
    marker = _configure_bp_panel(
        bp_axis,
        output_mode=output_mode,
        prediction=prediction,
        target=target,
        waveform=waveform,
    )
    title = figure.suptitle("", fontsize=12)
    figure.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.89,
        bottom=0.07,
        hspace=0.30,
        wspace=0.20,
    )

    try:
        for offset in offsets:
            frame_index = legacy.FIRST_DISPLAY_FRAME + offset
            displayed_beat = offset // legacy.FRAMES_PER_BEAT + 1
            sample_in_beat = offset % legacy.FRAMES_PER_BEAT
            for artist, values in zip(image_artists, image_panels):
                artist.set_data(values[..., frame_index])

            show_marker = displayed_beat == legacy.DISPLAY_BEATS
            marker.set_visible(show_marker)
            if show_marker:
                marker.set_xdata([sample_in_beat + 1, sample_in_beat + 1])
            context = (
                "target beat 5/5"
                if show_marker
                else "input context; prediction is for beat 5/5"
            )
            bp_axis.set_title(f"Saved BP output\n{context}", fontsize=9)
            input_beat = (
                legacy.WINDOW_BEATS - legacy.DISPLAY_BEATS + displayed_beat
            )
            title.set_text(
                f"{heading} | input beat {input_beat}/5, "
                f"sample {sample_in_beat + 1}/50"
            )
            figure.savefig(
                Path(frame_root) / f"{offset:03d}.png",
                format="png",
                facecolor="white",
            )
    finally:
        plt.close(figure)
    return offsets


def _render_prediction_gif_parallel(
    output: Path,
    image_panels: list[np.ndarray],
    image_titles: list[str],
    *,
    output_mode: str,
    prediction: np.ndarray,
    target: np.ndarray,
    waveform: np.ndarray,
    heading: str,
    frame_ms: int = 90,
    workers: int,
    scratch_root: Path | None,
    frame_offsets: list[int] | None = None,
) -> None:
    """Render a GIF using reusable figures across a process pool."""

    frame_count = legacy.DISPLAY_BEATS * legacy.FRAMES_PER_BEAT
    ordered_offsets = (
        list(range(frame_count)) if frame_offsets is None else frame_offsets
    )
    if not ordered_offsets:
        raise ValueError("frame_offsets must not be empty")
    if any(offset < 0 or offset >= frame_count for offset in ordered_offsets):
        raise ValueError("frame offset is outside the displayed range")
    index_chunks = _frame_chunks(len(ordered_offsets), workers)
    chunks = [
        [ordered_offsets[index] for index in index_chunk]
        for index_chunk in index_chunks
    ]
    limits = [legacy._panel_limit(panel) for panel in image_panels]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    scratch_dir = None if scratch_root is None else str(Path(scratch_root))
    with tempfile.TemporaryDirectory(
        prefix=f"{output.stem}-frames-", dir=scratch_dir
    ) as frame_root:
        process_context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=len(chunks), mp_context=process_context
        ) as executor:
            futures = [
                executor.submit(
                    _render_frame_chunk,
                    offsets,
                    frame_root,
                    image_panels,
                    image_titles,
                    limits,
                    output_mode,
                    prediction,
                    target,
                    waveform,
                    heading,
                )
                for offsets in chunks
            ]
            for future in futures:
                future.result()

        frames = []
        for offset in ordered_offsets:
            with Image.open(Path(frame_root) / f"{offset:03d}.png") as image:
                frames.append(
                    image.convert("P", palette=Image.Palette.ADAPTIVE)
                )
        temporary = output.with_name(f".{output.stem}.parallel.tmp.gif")
        try:
            frames[0].save(
                temporary,
                save_all=True,
                append_images=frames[1:],
                duration=frame_ms,
                loop=0,
                optimize=False,
            )
            temporary.replace(output)
        finally:
            if temporary.exists():
                temporary.unlink()
            for frame in frames:
                frame.close()


def generate_bp_artifact_gifs_parallel(
    *,
    workers: int,
    scratch_root: Path | None = None,
    **kwargs,
) -> dict:
    """Run the established artifact pipeline with only its renderer replaced."""

    renderer = partial(
        _render_prediction_gif_parallel,
        workers=workers,
        scratch_root=scratch_root,
    )
    original_renderer = legacy._render_prediction_gif
    legacy._render_prediction_gif = renderer
    try:
        return legacy.generate_bp_artifact_gifs(**kwargs)
    finally:
        legacy._render_prediction_gif = original_renderer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-main", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--output-mode", choices=["waveform", "fiducials"], required=True
    )
    parser.add_argument("--coordinate-root", type=Path, required=True)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference-root", type=Path)
    reference.add_argument("--reference-registry", type=Path)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))),
        help="frame-render worker processes (default: SLURM_CPUS_PER_TASK)",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path(os.environ.get("TMPDIR", "/tmp")),
        help="node-local directory for temporary PNG frames",
    )
    args = parser.parse_args()
    report = generate_bp_artifact_gifs_parallel(
        artifact_main=args.artifact_main,
        subject=args.subject,
        output_mode=args.output_mode,
        coordinate_root=args.coordinate_root,
        reference_root=args.reference_root,
        reference_registry=args.reference_registry,
        split_manifest=args.split_manifest,
        workers=args.workers,
        scratch_root=args.scratch_root,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "workers": args.workers,
                "gifs": [item["gif"] for item in report["examples"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
