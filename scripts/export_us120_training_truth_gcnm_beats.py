#!/usr/bin/env python3
"""Export matched synthetic truth and trained GCNM stage-2 beat images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.mesh_registry import sha256_file
from gcnm_pvi.representations import CoordinateReconstructor, rasterize_frames


FRAMES_PER_BEAT = 50
TISSUE_COLORS = np.asarray(
    [
        (238, 241, 244),
        (217, 160, 102),
        (232, 206, 138),
        (192, 85, 107),
        (195, 203, 212),
        (62, 142, 138),
        (183, 38, 63),
        (140, 123, 107),
        (224, 138, 155),
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/differential_US120_1000beats_clean_v1/train.npz"),
    )
    parser.add_argument(
        "--source-hp",
        type=Path,
        default=Path("data/hp_lp_beats_US120_v1/hp/train.npz"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rings_b045/US120.yaml"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models/differential_US120_1000beats_v1/coordinate_direct"),
    )
    parser.add_argument("--model-name", default="coordinate_direct")
    parser.add_argument(
        "--physics-mesh-mode",
        choices=("coarse", "projected_fine"),
        default="coarse",
        help=(
            "Physics used to construct both GCNM stage inputs. Use projected_fine "
            "with checkpoints trained from F_f(P sigma_c) and J_f P."
        ),
    )
    parser.add_argument("--ring", default="US120")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("figures/gcnm_training_20_anatomies_US120"),
    )
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--anatomy-id",
        type=int,
        action="append",
        help="Export this anatomy ID instead of selecting anatomies randomly; repeatable.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=900)
    parser.add_argument("--shared-color-limit", type=float, default=None)
    return parser.parse_args()


def complete_beat_indices(
    anatomy_ids: np.ndarray,
    beat_ids: np.ndarray,
    sample_indices: np.ndarray,
    anatomy_id: int,
) -> tuple[int, np.ndarray]:
    available = np.unique(beat_ids[anatomy_ids == anatomy_id])
    if not len(available):
        raise ValueError(f"anatomy {anatomy_id} has no retained beats")
    beat_id = int(available[0])
    indices = np.flatnonzero(
        (anatomy_ids == anatomy_id) & (beat_ids == beat_id)
    )
    indices = indices[np.argsort(sample_indices[indices])]
    if len(indices) != FRAMES_PER_BEAT or not np.array_equal(
        sample_indices[indices], np.arange(FRAMES_PER_BEAT)
    ):
        raise ValueError(f"anatomy {anatomy_id}, beat {beat_id} is incomplete")
    return beat_id, indices


def image_stack(reconstructor: CoordinateReconstructor, values: np.ndarray) -> np.ndarray:
    return np.moveaxis(rasterize_frames(reconstructor.mappings, values), 2, 0)


def map_to_image(grid: np.ndarray, limit: float, size: int) -> Image.Image:
    values = np.asarray(grid, dtype=np.float64)
    finite = np.isfinite(values)
    normalized = np.clip((values + limit) / (2.0 * limit), 0.0, 1.0)
    rgba = np.rint(plt.get_cmap("RdBu_r")(normalized) * 255.0).astype(np.uint8)
    rgba[~finite] = np.asarray([255, 255, 255, 255], dtype=np.uint8)
    image = Image.fromarray(rgba, mode="RGBA").convert("RGB")
    return image.resize((size, size), resample=Image.Resampling.NEAREST)


def tissue_to_image(grid: np.ndarray, size: int) -> Image.Image:
    labels = np.rint(np.nan_to_num(grid, nan=0.0)).astype(np.int64)
    labels = np.clip(labels, 0, len(TISSUE_COLORS) - 1)
    rgb = TISSUE_COLORS[labels]
    return Image.fromarray(rgb, mode="RGB").resize(
        (size, size), resample=Image.Resampling.NEAREST
    )


def fonts() -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, ...]:
    candidates = (
        (
            Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"),
            Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ),
    )
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            return (
                ImageFont.truetype(str(bold), 42),
                ImageFont.truetype(str(bold), 32),
                ImageFont.truetype(str(regular), 24),
            )
    fallback = ImageFont.load_default()
    return fallback, fallback, fallback


def comparison_image(
    truth: Image.Image,
    gcnm: Image.Image,
    *,
    ring: str,
    anatomy_id: int,
    beat_id: int,
    sample_one_based: int,
) -> Image.Image:
    heading_font, label_font, detail_font = fonts()
    margin = 40
    gap = 50
    heading_height = 105
    footer_height = 42
    width = margin * 2 + truth.width + gap + gcnm.width
    height = heading_height + truth.height + footer_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 16),
        f"{ring} training anatomy {anatomy_id} · beat {beat_id} · sample {sample_one_based}/50",
        fill="#15243A",
        font=heading_font,
    )
    left = margin
    right = margin + truth.width + gap
    canvas.paste(truth, (left, heading_height))
    canvas.paste(gcnm, (right, heading_height))
    draw.text((left, 67), "Ground-truth Δσ", fill="#15243A", font=label_font)
    draw.text((right, 67), "GCNM stage 2", fill="#15243A", font=label_font)
    draw.text(
        (margin, heading_height + truth.height + 7),
        "Both images use the shared color scale supplied in the parent folder.",
        fill="#5B6B7F",
        font=detail_font,
    )
    return canvas


def save_colorbar(path: Path, limit: float) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 0.9), dpi=300)
    gradient = np.linspace(-limit, limit, 1024)[None, :]
    axis.imshow(
        gradient,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        extent=(-limit, limit, 0, 1),
    )
    axis.set_yticks([])
    axis.set_xticks(np.linspace(-limit, limit, 5))
    axis.set_xlabel(r"Differential conductivity $\Delta\sigma$ (S/m)")
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def save_waveform_plot(
    path: Path,
    truth: np.ndarray,
    gcnm: np.ndarray,
    anatomy_id: int,
    beat_id: int,
    y_limit: float,
) -> None:
    samples = np.arange(1, FRAMES_PER_BEAT + 1)
    figure, axis = plt.subplots(figsize=(5.2, 3.3), dpi=300)
    axis.plot(samples, truth, color="#B7263F", linewidth=2.1, label="Ground truth")
    axis.plot(samples, gcnm, color="#355F8D", linewidth=2.1, label="GCNM stage 2")
    axis.set_xlim(1, FRAMES_PER_BEAT)
    axis.set_ylim(0, y_limit)
    axis.set_xlabel("Sample in beat")
    axis.set_ylabel(r"RMS $\Delta\sigma$ (S/m)")
    axis.set_title(f"Anatomy {anatomy_id} · beat {beat_id}")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    x = np.asarray(reference, dtype=np.float64).ravel()
    y = np.asarray(estimate, dtype=np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[finite], y[finite])[0, 1])


def install_contract_compatible_validation(
    config_path: Path,
    checkpoint_dir: Path,
    model_name: str,
    physics_mesh_mode: str,
) -> dict[str, object]:
    """Audit checkpoint/config compatibility before reconstruction."""

    cfg = GcnmConfig.from_yaml(config_path)
    checkpoints = [
        torch.load(
            checkpoint_dir / f"{model_name}_{stage}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for stage in range(2)
    ]
    contracts = [checkpoint["physics_contract"] for checkpoint in checkpoints]
    contract = contracts[0]
    for other in contracts[1:]:
        keys = (
            "config_sha256",
            "mesh_forward_sha256",
            "mesh_inverse_sha256",
            "mappings_sha256",
            "baseline_conductivity",
            "hyper_pvi",
            "lambda_lm",
            "lm_step_size",
            "minimum_conductivity",
            "num_elements",
            "num_measurements",
        )
        if any(other.get(key) != contract.get(key) for key in keys):
            raise ValueError("stage-1 and stage-2 physics contracts differ")

    checkpoint_modes = {
        str(item.get("physics_mesh_mode") or "coarse") for item in contracts
    }
    if checkpoint_modes != {physics_mesh_mode}:
        raise ValueError(
            "checkpoint training physics does not match requested reconstruction: "
            f"checkpoint={sorted(checkpoint_modes)}, requested={physics_mesh_mode}"
        )

    current_hash = sha256_file(config_path)
    strict_checks = {
        "mesh_forward_sha256": bool(
            physics_mesh_mode == "coarse"
            or sha256_file(Path(cfg.mesh_fwd_h5))
            == contract.get("mesh_forward_sha256")
        ),
        "mesh_inverse_sha256": bool(
            sha256_file(Path(cfg.mesh_inv_h5)) == contract["mesh_inverse_sha256"]
        ),
        "mappings_sha256": bool(
            sha256_file(Path(cfg.mappings_h5)) == contract["mappings_sha256"]
        ),
        "hyper_pvi": bool(
            np.isclose(float(cfg.hyper_pvi), float(contract["hyper_pvi"]))
        ),
        "lambda_lm": bool(
            np.isclose(float(cfg.lambda_lm), float(contract["lambda_lm"]))
        ),
        "lm_step_size": bool(
            np.isclose(float(cfg.lm_step_size), float(contract["lm_step_size"]))
        ),
        "baseline_conductivity": bool(
            np.isclose(float(contract["baseline_conductivity"]), 0.7)
        ),
        "measurement_contract": contract.get("measurement_contract") == "differential",
        "physics_mesh_mode": checkpoint_modes == {physics_mesh_mode},
    }
    failed = [key for key, passed in strict_checks.items() if not passed]
    if failed:
        raise ValueError(f"checkpoint physics contract mismatch: {failed}")

    return {
        "checkpoint_config_sha256": contract["config_sha256"],
        "current_config_sha256": current_hash,
        "config_hash_match": current_hash == contract["config_sha256"],
        "checkpoint_training_physics": next(iter(checkpoint_modes)),
        "inference_physics": physics_mesh_mode,
        "strict_physics_checks": strict_checks,
    }


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"immutable output folder already exists: {args.output_root}")
    if args.count < 1:
        raise ValueError("count must be positive")

    print(f"loading training data from {args.dataset}", flush=True)
    with np.load(args.dataset) as source:
        required = ("sigma", "V", "anatomy_id", "beat_id", "sample_index")
        missing = [key for key in required if key not in source]
        if missing:
            raise KeyError(f"training archive lacks {missing}")
        anatomy_ids = np.asarray(source["anatomy_id"])
        beat_ids = np.asarray(source["beat_id"])
        sample_indices = np.asarray(source["sample_index"])
        unique_anatomies = np.unique(anatomy_ids)
        if args.anatomy_id:
            selected_anatomies = list(dict.fromkeys(args.anatomy_id))
            missing_anatomies = sorted(set(selected_anatomies) - set(unique_anatomies))
            if missing_anatomies:
                raise ValueError(f"requested anatomy IDs are absent: {missing_anatomies}")
        else:
            if args.count > len(unique_anatomies):
                raise ValueError(
                    f"requested {args.count} anatomies but only {len(unique_anatomies)} exist"
                )
            rng = np.random.default_rng(args.seed)
            selected_anatomies = sorted(
                int(value)
                for value in rng.choice(unique_anatomies, args.count, replace=False)
            )
        selections = []
        all_indices = []
        for anatomy_id in selected_anatomies:
            beat_id, indices = complete_beat_indices(
                anatomy_ids, beat_ids, sample_indices, anatomy_id
            )
            selections.append((anatomy_id, beat_id, indices))
            all_indices.extend(indices.tolist())
        all_indices_array = np.asarray(all_indices, dtype=np.int64)
        truth_elements = np.asarray(source["sigma"][all_indices_array], dtype=np.float32)
        voltage = np.asarray(source["V"][all_indices_array], dtype=np.float32)

    print("loading matching tissue anatomies", flush=True)
    with np.load(args.source_hp) as source_hp:
        np.testing.assert_array_equal(source_hp["anatomy_id"], anatomy_ids)
        np.testing.assert_array_equal(source_hp["beat_id"], beat_ids)
        np.testing.assert_array_equal(source_hp["sample_index"], sample_indices)
        tissue_elements = np.asarray(
            [source_hp["tissue_labels"][indices[0]] for _, _, indices in selections],
            dtype=np.uint8,
        )

    print(
        f"reconstructing {len(voltage)} frames with {args.checkpoint_dir}",
        flush=True,
    )
    contract_audit = install_contract_compatible_validation(
        args.config,
        args.checkpoint_dir,
        args.model_name,
        args.physics_mesh_mode,
    )
    reconstructor = CoordinateReconstructor(
        args.config,
        args.checkpoint_dir,
        args.model_name,
        device=args.device,
        physics_mesh_mode=args.physics_mesh_mode,
        allow_config_hash_mismatch=not bool(contract_audit["config_hash_match"]),
    )
    stage_1_elements, stage_2_elements, residuals = reconstructor.reconstruct(voltage)
    truth_images = image_stack(reconstructor, truth_elements)
    stage_2_images = image_stack(reconstructor, stage_2_elements)
    tissue_images = np.stack(
        [
            reconstructor.mappings.categorical_to_image_grid(
                labels, num_classes=len(TISSUE_COLORS)
            )
            for labels in tissue_elements
        ]
    )

    finite_values = np.concatenate(
        [
            np.abs(truth_images[np.isfinite(truth_images)]),
            np.abs(stage_2_images[np.isfinite(stage_2_images)]),
        ]
    )
    data_driven_limit = max(float(np.quantile(finite_values, 0.995)), 1e-8)
    shared_limit = (
        float(args.shared_color_limit)
        if args.shared_color_limit is not None
        else data_driven_limit
    )
    if shared_limit <= 0:
        raise ValueError("shared color limit must be positive")
    truth_waveforms = np.sqrt(np.mean(truth_elements.astype(np.float64) ** 2, axis=1))
    stage_2_waveforms = np.sqrt(np.mean(stage_2_elements.astype(np.float64) ** 2, axis=1))
    shared_waveform_limit = 1.08 * max(
        float(np.max(truth_waveforms)), float(np.max(stage_2_waveforms))
    )

    args.output_root.mkdir(parents=True)
    save_colorbar(args.output_root / "shared_colorbar.png", shared_limit)
    index_rows = []
    all_waveform_rows = []

    for order, (anatomy_id, beat_id, _indices) in enumerate(selections, start=1):
        first = (order - 1) * FRAMES_PER_BEAT
        stop = first + FRAMES_PER_BEAT
        anatomy_root = args.output_root / f"{order:02d}_anatomy_{anatomy_id:03d}_beat_{beat_id:04d}"
        truth_root = anatomy_root / "ground_truth_delta_sigma"
        gcnm_root = anatomy_root / "gcnm_stage2_delta_sigma"
        comparison_root = anatomy_root / "side_by_side"
        truth_root.mkdir(parents=True)
        gcnm_root.mkdir()
        comparison_root.mkdir()

        tissue_to_image(tissue_images[order - 1], args.image_size).save(
            anatomy_root / "anatomy.png", optimize=True
        )
        beat_truth = truth_waveforms[first:stop]
        beat_gcnm = stage_2_waveforms[first:stop]
        waveform_rows = []
        for local_index in range(FRAMES_PER_BEAT):
            global_index = first + local_index
            sample_one_based = local_index + 1
            filename = f"sample_{sample_one_based:02d}.png"
            truth_image = map_to_image(
                truth_images[global_index], shared_limit, args.image_size
            )
            gcnm_image = map_to_image(
                stage_2_images[global_index], shared_limit, args.image_size
            )
            truth_image.save(truth_root / filename, optimize=True)
            gcnm_image.save(gcnm_root / filename, optimize=True)
            comparison_image(
                truth_image,
                gcnm_image,
                ring=args.ring,
                anatomy_id=anatomy_id,
                beat_id=beat_id,
                sample_one_based=sample_one_based,
            ).save(comparison_root / filename, optimize=True)
            row = {
                "anatomy_id": anatomy_id,
                "beat_id": beat_id,
                "sample_zero_based": local_index,
                "sample_one_based": sample_one_based,
                "ground_truth_rms_delta_sigma_s_per_m": float(beat_truth[local_index]),
                "gcnm_stage2_rms_delta_sigma_s_per_m": float(beat_gcnm[local_index]),
            }
            waveform_rows.append(row)
            all_waveform_rows.append(row)

        with (anatomy_root / "waveform.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(waveform_rows[0]))
            writer.writeheader()
            writer.writerows(waveform_rows)
        save_waveform_plot(
            anatomy_root / "waveform.png",
            beat_truth,
            beat_gcnm,
            anatomy_id,
            beat_id,
            shared_waveform_limit,
        )
        np.savez_compressed(
            anatomy_root / "element_values.npz",
            ground_truth_delta_sigma=truth_elements[first:stop],
            gcnm_stage1_delta_sigma=stage_1_elements[first:stop].astype(np.float32),
            gcnm_stage2_delta_sigma=stage_2_elements[first:stop].astype(np.float32),
            sample_index=np.arange(FRAMES_PER_BEAT, dtype=np.int16),
        )

        truth_rms = max(float(np.sqrt(np.mean(truth_elements[first:stop] ** 2))), 1e-12)
        rmse = float(
            np.sqrt(np.mean((stage_2_elements[first:stop] - truth_elements[first:stop]) ** 2))
        )
        index_rows.append(
            {
                "order": order,
                "ring": args.ring,
                "split": "train",
                "anatomy_id": anatomy_id,
                "beat_id": beat_id,
                "frames": FRAMES_PER_BEAT,
                "gcnm_stage2_nrmse": rmse / truth_rms,
                "gcnm_stage2_correlation": correlation(
                    truth_elements[first:stop], stage_2_elements[first:stop]
                ),
                "folder": anatomy_root.name,
            }
        )
        print(
            f"rendered anatomy {order}/{len(selected_anatomies)}: {anatomy_root.name}",
            flush=True,
        )

    with (args.output_root / "index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    with (args.output_root / "all_waveforms.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_waveform_rows[0]))
        writer.writeheader()
        writer.writerows(all_waveform_rows)

    manifest = {
        "schema": "gcnm-training-truth-stage2-beat-images-v1",
        "ring": args.ring,
        "split": "train",
        "selection": (
            "explicit anatomy IDs selected for two visible rasterized artery lumens and "
            "first retained beat per anatomy"
            if args.anatomy_id
            else "fixed-seed distinct anatomies and first retained beat per anatomy"
        ),
        "selection_seed": args.seed,
        "selected_anatomy_ids": selected_anatomies,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "checkpoint_sha256": {
            "stage_1": sha256_file(args.checkpoint_dir / f"{args.model_name}_0.pt"),
            "stage_2": sha256_file(args.checkpoint_dir / f"{args.model_name}_1.pt"),
        },
        "model": "two-stage coordinate-direct GCNM with 64 hidden channels",
        "model_name": args.model_name,
        "physics_mesh_mode": args.physics_mesh_mode,
        "physics_definition": (
            "F_f(P sigma_c) with chain-rule Jacobian J_f P"
            if args.physics_mesh_mode == "projected_fine"
            else "F_c(sigma_c) with directly computed coarse Jacobian J_c"
        ),
        "exported_output": "literal GCNM stage-2 differential conductivity",
        "anatomies": len(selected_anatomies),
        "frames_per_anatomy": FRAMES_PER_BEAT,
        "image_size_pixels": args.image_size,
        "shared_color_limit_s_per_m": shared_limit,
        "data_driven_995_color_limit_s_per_m": data_driven_limit,
        "shared_waveform_limit_s_per_m": shared_waveform_limit,
        "checkpoint_config_audit": contract_audit,
        "stage_1_forward_residual_mean_v": float(
            np.mean(residuals["stage_1_forward_voltage_rms"])
        ),
        "examples": index_rows,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(args.output_root), "anatomies": len(selected_anatomies)}
        )
    )


if __name__ == "__main__":
    main()
