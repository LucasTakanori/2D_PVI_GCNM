#!/usr/bin/env python3
"""Export three experimental beats per subject with projected fine-mesh GCNM.

The exporter supports both the original fixed-checkpoint inference sensitivity
experiment and GCN checkpoints trained using ``F_f(P sigma_c)`` and ``J_f P``.
The selected checkpoint contract is recorded explicitly in every manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, PngImagePlugin

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gcnm_pvi.representations import CoordinateReconstructor, rasterize_frames


PERIOD_LENGTH = 50
BEATS_PER_SUBJECT = 3
IMAGE_SIDE = 40
STIM_CURRENT_A = -0.01
CHANNEL_NAMES = (
    "newton_hp",
    "d_newton_lp_dt",
    "d2_newton_lp_dt2",
    "gcnm_s1_projected_fine",
    "gcnm_s2_projected_fine",
    "d_gcnm_s2_projected_fine_dt",
)
CHANNEL_DIRECTORIES = tuple(
    f"{index:02d}_{name}" for index, name in enumerate(CHANNEL_NAMES, 1)
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_root(repo: Path, ring: str, experiment: str) -> Path:
    if ring == "US120":
        return repo / f"models/differential_US120_1000beats_v1/{experiment}"
    return repo / f"models/differential_main_b045_1000beats_v1/{ring}/{experiment}"


def prepare(args: argparse.Namespace) -> None:
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    source_selection = json.loads(args.source_selection.read_text(encoding="utf-8"))
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    records = {
        (str(row["subject"]).lower(), str(row["source_name"])): row
        for row in registry["records"]
        if not row.get("exclusion_reason")
    }
    subjects = []
    for selected in source_selection["subjects"]:
        subject = str(selected["subject"]).lower()
        window = selected["windows"][0]
        record = records[(subject, str(window["source_name"]))]
        if int(window["mask_stop"]) - int(window["mask_start"]) != 5:
            raise ValueError(f"{subject} selection is not a five-beat window")
        ring = str(selected["mesh"])
        checkpoint_dir = _model_root(
            args.repo_root.resolve(), ring, args.checkpoint_experiment
        )
        checkpoint_paths = [
            checkpoint_dir / f"{args.checkpoint_experiment}_{stage}.pt"
            for stage in range(2)
        ]
        for path in checkpoint_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        subjects.append(
            {
                "subject": subject,
                "ring": ring,
                "session": record["session"],
                "source_name": record["source_name"],
                "source_hdf5": str(Path(record["source_hdf5"]).resolve()),
                "config": str(Path(record["config"]).resolve()),
                "checkpoint_dir": str(checkpoint_dir.resolve()),
                "model_name": args.checkpoint_experiment,
                "checkpoint_sha256": {path.name: _sha256(path) for path in checkpoint_paths},
                "source_period_indices_zero_based": list(
                    range(int(window["mask_start"]), int(window["mask_start"]) + BEATS_PER_SUBJECT)
                ),
                "mask05_window": [int(window["mask_start"]), int(window["mask_stop"])],
            }
        )
    subjects.sort(key=lambda row: int(row["subject"].removeprefix("subject")))
    if len(subjects) != 91:
        raise ValueError(f"expected 91 subjects, found {len(subjects)}")
    output.mkdir(parents=True)
    (output / "_INCOMPLETE").write_text(_utc_now() + "\n", encoding="utf-8")
    trained_projected_fine = args.checkpoint_training_physics == "projected_fine"
    checkpoint_training_physics = (
        "projected fine-mesh F_f(P sigma_c) and chain-rule J_f P"
        if trained_projected_fine
        else "direct coarse-mesh F_c and J_c"
    )
    interpretation = (
        "The GCN checkpoints and experimental inference both use projected fine-mesh "
        "physics, with conductivity updates and graph outputs retained on the coarse mesh."
        if trained_projected_fine
        else (
            "Fixed-checkpoint inference sensitivity experiment. The GCN inputs differ from "
            "their training distribution, so these files cannot be described as models "
            "trained with projected fine-mesh physics."
        )
    )
    manifest = {
        "schema": (
            "pvi-gcnm-figure26-projected-fine-trained-v1"
            if trained_projected_fine
            else "pvi-gcnm-figure26-dual-mesh-sensitivity-v1"
        ),
        "created_utc": _utc_now(),
        "subject_count": len(subjects),
        "beats_per_subject": BEATS_PER_SUBJECT,
        "samples_per_beat": PERIOD_LENGTH,
        "channel_order": list(CHANNEL_NAMES),
        "channel_directories": list(CHANNEL_DIRECTORIES),
        "inference_physics": {
            "conductivity_mapping": "sigma_f = P sigma_c",
            "forward": "F_f(P sigma_c)",
            "jacobian": "J_c = J_f P",
            "update_and_graph_domain": "coarse inverse mesh",
        },
        "checkpoint_experiment": args.checkpoint_experiment,
        "checkpoint_training_physics": checkpoint_training_physics,
        "interpretation": interpretation,
        "selection": (
            "First three consecutive beats from the first deterministic five-beat window "
            "used by the prior 91-subject six-channel audit. Two following samples are "
            "computed but not exported to preserve both centered derivatives at the boundary."
        ),
        "source_selection": str(args.source_selection.resolve()),
        "source_selection_sha256": _sha256(args.source_selection),
        "registry": str(args.registry.resolve()),
        "registry_sha256": _sha256(args.registry),
        "subjects": subjects,
    }
    _write_json(output / "selection_manifest.json", manifest)
    readme_title = (
        "# Figure 26 projected-fine-trained dataset\n\n"
        if trained_projected_fine
        else "# Figure 26 projected-fine-mesh sensitivity dataset\n\n"
    )
    readme_contract = (
        "The saved 64-channel GCN checkpoints and this export both use "
        "`F_f(P sigma_c)` and the chain-rule Jacobian `J_f P`; conductivity updates "
        "and graph outputs remain on the coarse inverse mesh.\n"
        if trained_projected_fine
        else (
            "The saved 64-channel GCN checkpoints were trained using coarse-mesh physics. "
            "Only inference is rewired here to use `F_f(P sigma_c)` and `J_f P`. "
            "Consequently, this is a sensitivity analysis and must not be reported as "
            "dual-mesh training.\n"
        )
    )
    (output / "README.md").write_text(
        readme_title
        + "Each subject contains three complete 50-sample beats and six representation "
        "folders. The exact floating-point tensor is stored in `representations_float32.npy` "
        "with shape `(3, 50, 6, 40, 40)`. PNGs are lossless 800 x 800 nearest-neighbor "
        "renderings for layout work.\n\n"
        + readme_contract,
        encoding="utf-8",
    )
    print(json.dumps({"prepared": str(output), "subjects": len(subjects)}, indent=2))


def _central_difference(values: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    output[..., 1:-1] = (values[..., 2:] - values[..., :-2]) / np.float32(2.0)
    return output


def _load_subject_inputs(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    periods = row["source_period_indices_zero_based"]
    start = int(periods[0]) * PERIOD_LENGTH
    stop = (int(periods[-1]) + 1) * PERIOD_LENGTH + 2
    with h5py.File(row["source_hdf5"], "r") as handle:
        resistance_hp = np.asarray(handle["data/pviHP/resistance"][:, start:stop], dtype=np.float64)
        resistance_lp = np.asarray(handle["data/pviLP/resistance"][:, start:stop], dtype=np.float64)
        newton_hp = np.asarray(handle["data/pviHP/img"][:, :, start:stop], dtype=np.float32)
        newton_lp = np.asarray(handle["data/pviLP/img"][:, :, start:stop], dtype=np.float32)
    frames = BEATS_PER_SUBJECT * PERIOD_LENGTH + 2
    if resistance_hp.shape != (32, frames):
        raise ValueError(f"unexpected resistance shape {resistance_hp.shape}")
    if newton_hp.shape != (IMAGE_SIDE, IMAGE_SIDE, frames):
        raise ValueError(f"unexpected Newton image shape {newton_hp.shape}")
    resistance = resistance_hp + resistance_lp
    saved_voltage = -STIM_CURRENT_A * resistance
    voltage = -(saved_voltage - saved_voltage[:, :1]).T
    return voltage, newton_hp, newton_lp


def _color_lut() -> np.ndarray:
    anchors = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    colors = np.asarray(
        [[5, 48, 97], [69, 117, 180], [247, 247, 247], [214, 96, 77], [103, 0, 31]],
        dtype=np.float64,
    )
    points = np.linspace(0.0, 1.0, 256)
    return np.stack(
        [np.interp(points, anchors, colors[:, channel]) for channel in range(3)], axis=-1
    ).round().astype(np.uint8)


COLOR_LUT = _color_lut()


def _render_png(path: Path, values: np.ndarray, scale: float, metadata_values: dict) -> None:
    finite = np.isfinite(values)
    normalized = np.zeros_like(values, dtype=np.float32)
    normalized[finite] = np.clip(values[finite] / np.float32(scale), -1.0, 1.0)
    indices = np.rint((normalized + 1.0) * 127.5).astype(np.uint8)
    rgb = COLOR_LUT[indices]
    rgb[~finite] = np.asarray([255, 255, 255], dtype=np.uint8)
    image = Image.fromarray(rgb, mode="RGB").resize((800, 800), resample=Image.Resampling.NEAREST)
    metadata = PngImagePlugin.PngInfo()
    for key, value in metadata_values.items():
        metadata.add_text(key, str(value))
    image.save(path, format="PNG", compress_level=9, pnginfo=metadata)


def _subject_row(manifest: dict, args: argparse.Namespace) -> dict:
    if args.subject is not None:
        matches = [row for row in manifest["subjects"] if row["subject"] == args.subject.lower()]
        if len(matches) != 1:
            raise ValueError(f"unknown subject {args.subject}")
        return matches[0]
    return manifest["subjects"][args.subject_index]


def export_subject(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    manifest = json.loads((root / "selection_manifest.json").read_text(encoding="utf-8"))
    row = _subject_row(manifest, args)
    final = root / row["subject"]
    if (final / "_SUCCESS").is_file():
        print(f"reuse complete {final}")
        return
    if final.exists():
        raise FileExistsError(f"partial subject output exists: {final}")
    staging = root / f".{row['subject']}.tmp-{os.getpid()}"
    staging.mkdir()
    try:
        os.environ["GCNM_PHYSICS_WORKERS"] = str(args.physics_workers)
        os.environ["GCNM_INFERENCE_BATCH_SIZE"] = str(args.inference_batch_size)
        os.environ["GCNM_COMPUTE_STAGE2_RESIDUALS"] = "0"
        voltage, newton_hp, newton_lp = _load_subject_inputs(row)
        allow_config_hash_mismatch = (
            manifest["checkpoint_training_physics"] == "direct coarse-mesh F_c and J_c"
        )
        reconstructor = CoordinateReconstructor(
            Path(row["config"]),
            Path(row["checkpoint_dir"]),
            row.get("model_name", "coordinate_direct"),
            device=args.device,
            physics_mesh_mode="projected_fine",
            allow_config_hash_mismatch=allow_config_hash_mismatch,
        )
        if not allow_config_hash_mismatch:
            checkpoint_modes = {
                checkpoint["physics_contract"].get("physics_mesh_mode")
                for checkpoint in reconstructor.checkpoints
            }
            if checkpoint_modes != {"projected_fine"}:
                raise ValueError(
                    "projected-fine-trained export received checkpoints with modes "
                    f"{sorted(str(mode) for mode in checkpoint_modes)}"
                )
        stage_1_elements, stage_2_elements, report = reconstructor.reconstruct(voltage)
        stage_1 = rasterize_frames(reconstructor.mappings, stage_1_elements)
        stage_2 = rasterize_frames(reconstructor.mappings, stage_2_elements)
        channels_with_context = np.stack(
            (
                newton_hp,
                _central_difference(newton_lp),
                _central_difference(_central_difference(newton_lp)),
                stage_1,
                stage_2,
                _central_difference(stage_2),
            ),
            axis=0,
        ).astype(np.float32)
        channels = channels_with_context[..., : BEATS_PER_SUBJECT * PERIOD_LENGTH]
        expected = (6, IMAGE_SIDE, IMAGE_SIDE, BEATS_PER_SUBJECT * PERIOD_LENGTH)
        if channels.shape != expected:
            raise ValueError(f"channel tensor has shape {channels.shape}, expected {expected}")
        exact = channels.reshape(6, 40, 40, 3, 50).transpose(3, 4, 0, 1, 2)
        # Resulting order is (beat, sample, channel, row, column).
        exact = np.ascontiguousarray(exact, dtype=np.float32)
        if exact.shape != (3, 50, 6, 40, 40):
            raise AssertionError(exact.shape)
        array_path = staging / "representations_float32.npy"
        np.save(array_path, exact, allow_pickle=False)

        finite_abs = np.where(np.isfinite(channels), np.abs(channels), np.nan)
        individual_scales = np.nanmax(finite_abs, axis=(1, 2, 3)).astype(float)
        reconstruction_scale = float(np.nanmax(finite_abs[[0, 3, 4]]))
        png_scales = individual_scales.copy()
        png_scales[[0, 3, 4]] = reconstruction_scale
        png_scales[~np.isfinite(png_scales) | (png_scales <= 0)] = 1.0

        for directory in CHANNEL_DIRECTORIES:
            (staging / directory).mkdir()
        for beat in range(BEATS_PER_SUBJECT):
            for sample in range(PERIOD_LENGTH):
                frame = beat * PERIOD_LENGTH + sample
                for channel, directory in enumerate(CHANNEL_DIRECTORIES):
                    _render_png(
                        staging / directory / f"beat_{beat + 1:02d}_sample_{sample + 1:02d}.png",
                        channels[channel, :, :, frame],
                        float(png_scales[channel]),
                        {
                            "subject": row["subject"],
                            "ring": row["ring"],
                            "source_name": row["source_name"],
                            "source_period_zero_based": row["source_period_indices_zero_based"][beat],
                            "beat": beat + 1,
                            "sample": sample + 1,
                            "channel": CHANNEL_NAMES[channel],
                            "symmetric_scale_s_per_m": f"{png_scales[channel]:.17g}",
                            "inference_physics": "F_f(P sigma_c), J_f P",
                            "checkpoint_training_physics": manifest[
                                "checkpoint_training_physics"
                            ],
                        },
                    )
        subject_manifest = {
            **row,
            "schema": (
                "pvi-gcnm-figure26-projected-fine-trained-subject-v1"
                if not allow_config_hash_mismatch
                else "pvi-gcnm-figure26-dual-mesh-subject-v1"
            ),
            "completed_utc": _utc_now(),
            "physics_mesh_mode": reconstructor.physics_mesh_mode,
            "checkpoint_config_sha256": reconstructor.checkpoints[0]["physics_contract"][
                "config_sha256"
            ],
            "current_config_sha256": _sha256(Path(row["config"])),
            "config_hash_match": bool(
                reconstructor.checkpoints[0]["physics_contract"]["config_sha256"]
                == _sha256(Path(row["config"]))
            ),
            "config_hash_override": (
                (
                    "Allowed only for this sensitivity export after independently verifying "
                    "checkpoint inverse-mesh and mapping hashes plus hyper_pvi, lambda_lm, "
                    "and node-sharing connectivity."
                )
                if allow_config_hash_mismatch
                else None
            ),
            "fine_elements": int(len(reconstructor.runtime["mesh_fwd"].elems)),
            "coarse_elements": int(len(reconstructor.runtime["mesh_inv"].elems)),
            "coarse_to_fine_shape": list(reconstructor.mappings.c2f.shape),
            "fine_mesh": str(Path(reconstructor.cfg.mesh_fwd_h5).resolve()),
            "fine_mesh_sha256": _sha256(Path(reconstructor.cfg.mesh_fwd_h5)),
            "inverse_mesh_sha256": _sha256(Path(reconstructor.cfg.mesh_inv_h5)),
            "mappings_sha256": _sha256(Path(reconstructor.cfg.mappings_h5)),
            "array_file": array_path.name,
            "array_shape": list(exact.shape),
            "array_dtype": str(exact.dtype),
            "array_sha256": _sha256(array_path),
            "channel_order": list(CHANNEL_NAMES),
            "channel_directories": list(CHANNEL_DIRECTORIES),
            "individual_data_max_abs_s_per_m": {
                name: float(scale) for name, scale in zip(CHANNEL_NAMES, individual_scales)
            },
            "png_symmetric_scales_s_per_m": {
                name: float(scale) for name, scale in zip(CHANNEL_NAMES, png_scales)
            },
            "shared_reconstruction_png_scale_channels": [
                "newton_hp", "gcnm_s1_projected_fine", "gcnm_s2_projected_fine"
            ],
            "stage_2_residual_diagnostics_computed": bool(
                len(report["stage_2_forward_voltage_rms"])
            ),
            "png_count": BEATS_PER_SUBJECT * PERIOD_LENGTH * len(CHANNEL_NAMES),
        }
        _write_json(staging / "subject_manifest.json", subject_manifest)
        (staging / "_SUCCESS").write_text(_utc_now() + "\n", encoding="utf-8")
        staging.rename(final)
        print(json.dumps({"completed": row["subject"], "output": str(final)}, indent=2))
    except BaseException:
        print(f"incomplete staging retained at {staging}")
        raise


def finalize(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    manifest = json.loads((root / "selection_manifest.json").read_text(encoding="utf-8"))
    completed = []
    for row in manifest["subjects"]:
        subject_root = root / row["subject"]
        if not (subject_root / "_SUCCESS").is_file():
            raise FileNotFoundError(f"incomplete subject: {subject_root}")
        subject_manifest = json.loads(
            (subject_root / "subject_manifest.json").read_text(encoding="utf-8")
        )
        png_count = sum(1 for _ in subject_root.glob("0?_*/*.png"))
        if png_count != 900 or subject_manifest["png_count"] != 900:
            raise ValueError(f"{subject_root} has {png_count} PNGs")
        array = np.load(subject_root / "representations_float32.npy", mmap_mode="r")
        if array.shape != (3, 50, 6, 40, 40) or array.dtype != np.float32:
            raise ValueError(f"{subject_root} has invalid array {array.shape} {array.dtype}")
        completed.append({"subject": row["subject"], "ring": row["ring"], "png_count": png_count})
    validation = {
        "schema": (
            "pvi-gcnm-figure26-projected-fine-trained-validation-v1"
            if manifest.get("checkpoint_training_physics", "").startswith("projected")
            else "pvi-gcnm-figure26-dual-mesh-validation-v1"
        ),
        "status": "pass",
        "completed_utc": _utc_now(),
        "subject_count": len(completed),
        "beat_count": len(completed) * BEATS_PER_SUBJECT,
        "frame_count": len(completed) * BEATS_PER_SUBJECT * PERIOD_LENGTH,
        "png_count": len(completed) * 900,
        "subjects": completed,
    }
    _write_json(root / "validation.json", validation)
    incomplete = root / "_INCOMPLETE"
    if incomplete.exists():
        incomplete.unlink()
    (root / "_SUCCESS").write_text(_utc_now() + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--repo-root", type=Path, required=True)
    prepare_parser.add_argument("--source-selection", type=Path, required=True)
    prepare_parser.add_argument("--registry", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument(
        "--checkpoint-experiment", default="coordinate_direct"
    )
    prepare_parser.add_argument(
        "--checkpoint-training-physics",
        choices=["coarse", "projected_fine"],
        default="coarse",
    )
    prepare_parser.set_defaults(function=prepare)
    export_parser = sub.add_parser("export-subject")
    export_parser.add_argument("--output-root", type=Path, required=True)
    selection = export_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--subject-index", type=int)
    selection.add_argument("--subject")
    export_parser.add_argument("--physics-workers", type=int, default=16)
    export_parser.add_argument("--inference-batch-size", type=int, default=512)
    export_parser.add_argument("--device", default=None)
    export_parser.set_defaults(function=export_subject)
    finalize_parser = sub.add_parser("finalize")
    finalize_parser.add_argument("--output-root", type=Path, required=True)
    finalize_parser.set_defaults(function=finalize)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
