"""Create prediction-aligned GIFs inside a native pvi_ml artifact tree.

The native pvi_ml folders are left intact.  This module adds only ``gifs/``
and renders the literal serialized Parquet inputs used by the BP experiment:
archived Newton/PVI HP and LP, coordinate GCNM S1 and S2, and dS2/dt.  The
sixth panel shows the saved (not recomputed) BP prediction and reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from gcnm_pvi.coordinate_direct_representation import centered_temporal_difference
from gcnm_pvi.newton_coordinate_hdf5_dataset import read_newton_hdf5_window


FRAMES_PER_BEAT = 50
WINDOW_BEATS = 5
DISPLAY_BEATS = 3
FIRST_DISPLAY_FRAME = (WINDOW_BEATS - DISPLAY_BEATS) * FRAMES_PER_BEAT
SESSION_ORDER = {"baseline": 0, "valsalva": 1, "pressor": 2}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_results(path: Path, output_mode: str) -> tuple[np.ndarray, np.ndarray]:
    output_size = 50 if output_mode == "waveform" else 2
    expected = [
        *[f"pred_{index}" for index in range(1, output_size + 1)],
        *[f"target_{index}" for index in range(1, output_size + 1)],
    ]
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != expected:
            raise ValueError(f"unexpected BP results schema in {path}")
        rows = [[float(value) for value in row] for row in reader]
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 * output_size:
        raise ValueError(f"invalid BP result array shape {values.shape}")
    return values[:, :output_size], values[:, output_size:]


def _ordered_test_rows(coordinate_root: Path, subject: str, split_manifest: Path) -> list[dict]:
    import pyarrow.dataset as pads

    assignments = json.loads(Path(split_manifest).read_text(encoding="utf-8"))["assignments"]
    table = pads.dataset(
        str(Path(coordinate_root) / "shards"), format="parquet"
    ).to_table(filter=pads.field("subject") == subject.lower()).sort_by(
        [
            ("source_order", "ascending"),
            ("mask_start", "ascending"),
            ("mask_stop", "ascending"),
        ]
    )
    rows = table.select(
        [
            "sample_id",
            "subject",
            "session",
            "source_name",
            "mask_start",
            "mask_stop",
        ]
    ).to_pylist()
    missing = [row["sample_id"] for row in rows if row["sample_id"] not in assignments]
    if missing:
        raise ValueError(f"split manifest omits {len(missing)} coordinate samples")
    return [row for row in rows if assignments[row["sample_id"]] == "test"]


def select_median_error_examples(
    metadata_rows: list[dict], predictions: np.ndarray, targets: np.ndarray
) -> list[dict]:
    """Select one deterministic, representative held-out row per session."""

    if len(metadata_rows) != len(predictions) or predictions.shape != targets.shape:
        raise ValueError("test metadata and saved BP results do not align")
    errors = np.mean(np.abs(predictions - targets), axis=1)
    sessions: dict[str, list[dict]] = {}
    for result_index, (row, error) in enumerate(zip(metadata_rows, errors)):
        candidate = {
            **row,
            "result_index": result_index,
            "mean_absolute_error_mmhg": float(error),
        }
        sessions.setdefault(str(row["session"]), []).append(candidate)
    selected = []
    for session in sorted(sessions, key=lambda value: (SESSION_ORDER.get(value, 99), value)):
        candidates = sorted(
            sessions[session],
            key=lambda item: (item["mean_absolute_error_mmhg"], item["sample_id"]),
        )
        selected.append(candidates[(len(candidates) - 1) // 2])
    return selected


def normalize_bp_artifact_layout(artifact_main: Path, subject: str) -> None:
    """Keep pvi_ml's standard directories and place our metadata within them."""

    artifact_main = Path(artifact_main)
    moves = (
        (
            artifact_main / "run_contracts" / f"{subject}.json",
            artifact_main / "configs" / f"{subject}_run_contract.json",
        ),
        (
            artifact_main / "verification" / f"{subject}.json",
            artifact_main / "gifs" / f"{subject}_verification.json",
        ),
    )
    for source, destination in moves:
        if not source.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            if source.read_bytes() != destination.read_bytes():
                raise RuntimeError(f"conflicting auxiliary artifacts: {source}")
            source.unlink()
        else:
            source.replace(destination)
        try:
            source.parent.rmdir()
        except OSError:
            pass


def _read_one(root: Path, sample_id: str, columns: list[str]):
    import pyarrow.dataset as pads

    table = pads.dataset(str(Path(root) / "shards"), format="parquet").to_table(
        filter=pads.field("sample_id") == sample_id,
        columns=["sample_id", *columns],
    )
    if table.num_rows != 1:
        raise ValueError(f"expected one row for {sample_id} in {root}, got {table.num_rows}")
    return table


def _fixed_array(table, column: str, shape: tuple[int, ...]) -> np.ndarray:
    scalar = table[column][0]
    values = scalar.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(shape)


def _panel_limit(values: np.ndarray) -> float:
    shown = values[..., FIRST_DISPLAY_FRAME:]
    finite = np.abs(shown[np.isfinite(shown)])
    return max(float(np.quantile(finite, 0.995)), 1e-8)


def _draw_bp_panel(
    axis,
    *,
    output_mode: str,
    prediction: np.ndarray,
    target: np.ndarray,
    waveform: np.ndarray,
    displayed_beat: int,
    sample_in_beat: int,
) -> None:
    x = np.arange(1, FRAMES_PER_BEAT + 1)
    if output_mode == "waveform":
        axis.plot(x, target, color="#172554", linewidth=2.2, label="BP reference")
        axis.plot(x, prediction, color="#e11d48", linewidth=2.0, label="BP prediction")
        if displayed_beat == DISPLAY_BEATS:
            axis.axvline(sample_in_beat + 1, color="#64748b", linestyle="--", linewidth=1.3)
    else:
        axis.plot(x, waveform, color="#172554", linewidth=2.0, label="BP reference waveform")
        labels = ("DBP", "SBP")
        colors = ("#0891b2", "#e11d48")
        for index, (label, color) in enumerate(zip(labels, colors)):
            axis.axhline(target[index], color=color, linewidth=1.8, alpha=0.85,
                         label=f"reference {label} {target[index]:.1f}")
            axis.axhline(prediction[index], color=color, linewidth=1.8, linestyle="--",
                         label=f"predicted {label} {prediction[index]:.1f}")
        if displayed_beat == DISPLAY_BEATS:
            axis.axvline(sample_in_beat + 1, color="#64748b", linestyle=":", linewidth=1.2)
    axis.set_xlim(1, FRAMES_PER_BEAT)
    all_values = np.concatenate((waveform.ravel(), prediction.ravel(), target.ravel()))
    margin = max(5.0, 0.08 * float(np.ptp(all_values)))
    axis.set_ylim(float(np.min(all_values) - margin), float(np.max(all_values) + margin))
    axis.set_xlabel("Sample in target beat")
    axis.set_ylabel("BP (mmHg)")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", fontsize=7)
    context = "target beat 5/5" if displayed_beat == DISPLAY_BEATS else "input context; prediction is for beat 5/5"
    axis.set_title(f"Saved BP output\n{context}", fontsize=9)


def _render_prediction_gif(
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
) -> None:
    limits = [_panel_limit(panel) for panel in image_panels]
    frames: list[Image.Image] = []
    for offset in range(DISPLAY_BEATS * FRAMES_PER_BEAT):
        frame_index = FIRST_DISPLAY_FRAME + offset
        displayed_beat = offset // FRAMES_PER_BEAT + 1
        sample_in_beat = offset % FRAMES_PER_BEAT
        figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), dpi=95)
        for axis, values, title, limit in zip(
            axes.flat[:5], image_panels, image_titles, limits
        ):
            shown = axis.imshow(
                values[..., frame_index],
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                origin="upper",
            )
            axis.set_title(f"{title}\n±{limit:.2e} S/m", fontsize=9)
            axis.axis("off")
            figure.colorbar(shown, ax=axis, orientation="horizontal", fraction=0.055, pad=0.055)
        _draw_bp_panel(
            axes.flat[5],
            output_mode=output_mode,
            prediction=prediction,
            target=target,
            waveform=waveform,
            displayed_beat=displayed_beat,
            sample_in_beat=sample_in_beat,
        )
        input_beat = WINDOW_BEATS - DISPLAY_BEATS + displayed_beat
        figure.suptitle(
            f"{heading} | input beat {input_beat}/5, sample {sample_in_beat + 1}/50",
            fontsize=12,
        )
        figure.subplots_adjust(left=0.035, right=0.985, top=0.89, bottom=0.07, hspace=0.30, wspace=0.20)
        buffer = BytesIO()
        figure.savefig(buffer, format="png", facecolor="white")
        plt.close(figure)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.gif")
    frames[0].save(
        temporary,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        optimize=False,
    )
    temporary.replace(output)


def generate_bp_artifact_gifs(
    *,
    artifact_main: Path,
    subject: str,
    output_mode: str,
    coordinate_root: Path,
    reference_root: Path | None,
    reference_registry: Path | None = None,
    split_manifest: Path,
) -> dict:
    """Generate one median-error held-out GIF per available session."""

    artifact_main = Path(artifact_main)
    coordinate_root = Path(coordinate_root)
    reference_root = None if reference_root is None else Path(reference_root)
    reference_registry = (
        None if reference_registry is None else Path(reference_registry)
    )
    if (reference_root is None) == (reference_registry is None):
        raise ValueError(
            "artifact GIFs require exactly one Newton reference: Parquet or HDF5 registry"
        )
    split_manifest = Path(split_manifest)
    normalize_bp_artifact_layout(artifact_main, subject)
    result_path = artifact_main / "results" / f"{subject}_results.csv"
    predictions, targets = _load_results(result_path, output_mode)
    test_rows = _ordered_test_rows(coordinate_root, subject, split_manifest)
    selected = select_median_error_examples(test_rows, predictions, targets)
    coordinate_manifest_path = coordinate_root / "manifest.json"
    coordinate_manifest = json.loads(coordinate_manifest_path.read_text(encoding="utf-8"))
    if coordinate_manifest.get("schema") != "pvi-gcnm-coordinate-direct-parquet-v1":
        raise ValueError("artifact GIFs require coordinate-direct Parquet")
    reference_manifest_path = None
    source_by_name: dict[str, Path] = {}
    if reference_root is not None:
        reference_manifest_path = reference_root / "manifest.json"
        reference_manifest = json.loads(
            reference_manifest_path.read_text(encoding="utf-8")
        )
        if (
            reference_manifest.get("schema") != "pvi-reference-parquet-v1"
            or reference_manifest.get("input_mode") != "img"
        ):
            raise ValueError("artifact GIFs require reference-image Parquet")
    else:
        registry = json.loads(reference_registry.read_text(encoding="utf-8"))
        source_by_name = {
            str(row["source_name"]): Path(row["source_hdf5"])
            for row in registry["records"]
            if not row.get("exclusion_reason")
        }

    gifs_root = artifact_main / "gifs"
    gifs_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for example in selected:
        sample_id = example["sample_id"]
        coordinate = _read_one(coordinate_root, sample_id, ["s1", "s2", "bp_waveform"])
        s1 = _fixed_array(coordinate, "s1", (1, 40, 40, 250))[0]
        s2 = _fixed_array(coordinate, "s2", (1, 40, 40, 250))[0]
        d_s2 = centered_temporal_difference(s2)
        coordinate_bp = _fixed_array(coordinate, "bp_waveform", (50,))
        if reference_root is not None:
            reference = _read_one(
                reference_root, sample_id, ["pviHP", "pviLP", "bp_waveform"]
            )
            newton_hp = _fixed_array(reference, "pviHP", (1, 40, 40, 250))[0]
            newton_lp = _fixed_array(reference, "pviLP", (1, 40, 40, 250))[0]
            reference_bp = _fixed_array(reference, "bp_waveform", (50,))
        else:
            source = source_by_name.get(str(example["source_name"]))
            if source is None:
                raise ValueError(
                    f"HDF5 registry omits {example['source_name']}"
                )
            newton_hp, newton_lp, reference_bp = read_newton_hdf5_window(
                source,
                int(example["mask_start"]),
                int(example["mask_stop"]),
            )
        if not np.array_equal(coordinate_bp, reference_bp, equal_nan=True):
            raise ValueError(f"BP waveform mismatch for paired sample {sample_id}")
        result_index = int(example["result_index"])
        output = gifs_root / f"{subject}_{example['session']}_median_test_error.gif"
        _render_prediction_gif(
            output,
            [newton_hp, newton_lp, s1, s2, d_s2],
            ["PVI/Newton HP", "PVI/Newton LP", "GCNM S1", "GCNM S2", "dS2/dt"],
            output_mode=output_mode,
            prediction=predictions[result_index],
            target=targets[result_index],
            waveform=coordinate_bp,
            heading=f"{subject} {example['source_name']} | held-out median-error window",
        )
        reports.append(
            {
                **example,
                "gif": str(output.resolve()),
                "gif_sha256": _sha256(output),
                "prediction": predictions[result_index].tolist(),
                "target": targets[result_index].tolist(),
                "bp_waveform": coordinate_bp.tolist(),
            }
        )
    manifest = {
        "schema": "pvi-ml-gcnm-bp-artifact-gifs-v1",
        "subject": subject,
        "output_mode": output_mode,
        "selection": "lower median mean-absolute-error held-out sample per available session",
        "displayed_input_frames": [FIRST_DISPLAY_FRAME, WINDOW_BEATS * FRAMES_PER_BEAT],
        "displayed_input_beats": [3, 4, 5],
        "bp_prediction_target_beat": 5,
        "panels": ["pvi_newton_hp", "pvi_newton_lp", "s1", "s2", "d_s2_dt", "saved_bp_prediction_and_reference"],
        "results_path": str(result_path.resolve()),
        "results_sha256": _sha256(result_path),
        "coordinate_root": str(coordinate_root.resolve()),
        "coordinate_manifest_sha256": _sha256(coordinate_manifest_path),
        "reference_storage": (
            "parquet" if reference_root is not None else "source_hdf5"
        ),
        "reference_root": (
            None if reference_root is None else str(reference_root.resolve())
        ),
        "reference_manifest_sha256": (
            None
            if reference_manifest_path is None
            else _sha256(reference_manifest_path)
        ),
        "reference_registry": (
            None
            if reference_registry is None
            else str(reference_registry.resolve())
        ),
        "reference_registry_sha256": (
            None if reference_registry is None else _sha256(reference_registry)
        ),
        "split_manifest": str(split_manifest.resolve()),
        "split_manifest_sha256": _sha256(split_manifest),
        "examples": reports,
    }
    manifest_path = gifs_root / f"{subject}_gifs.json"
    temporary_manifest = manifest_path.with_suffix(".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-main", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--output-mode", choices=["waveform", "fiducials"], required=True)
    parser.add_argument("--coordinate-root", type=Path, required=True)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference-root", type=Path)
    reference.add_argument("--reference-registry", type=Path)
    parser.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args()
    report = generate_bp_artifact_gifs(
        artifact_main=args.artifact_main,
        subject=args.subject,
        output_mode=args.output_mode,
        coordinate_root=args.coordinate_root,
        reference_root=args.reference_root,
        reference_registry=args.reference_registry,
        split_manifest=args.split_manifest,
    )
    print(json.dumps({"status": "pass", "gifs": [item["gif"] for item in report["examples"]]}, indent=2))


if __name__ == "__main__":
    main()
