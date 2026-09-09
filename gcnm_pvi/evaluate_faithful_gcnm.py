#!/usr/bin/env python3
"""Evaluate faithful PVI-GCNM against iterative LM and saved baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from gcnm_pvi.anatomical_phantoms import element_positions
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.dual_mesh_physics import ProjectedFineMeshPhysics
from gcnm_pvi.iterative_physics import (
    diagnostics_summary,
    iterative_lm_reconstruction,
    parallel_dataset_lm_directions,
    parallel_dataset_voltage_residual_rms,
)
from gcnm_pvi.runtime import build_runtime
from gcnm_pvi.train_faithful_gcnm import _make_dataset, _model, _predict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _correlation(prediction: np.ndarray, truth: np.ndarray) -> float:
    if np.std(prediction) < 1e-15 or np.std(truth) < 1e-15:
        return float("nan")
    return float(np.corrcoef(prediction.ravel(), truth.ravel())[0, 1])


def _dice(prediction: np.ndarray, truth: np.ndarray) -> float:
    values = []
    for pred, target in zip(prediction, truth):
        support = np.flatnonzero(np.abs(target) > 1e-8)
        if not len(support):
            continue
        selected = np.argpartition(np.abs(pred), -len(support))[-len(support) :]
        values.append(len(np.intersect1d(support, selected)) / len(support))
    return float(np.mean(values)) if values else float("nan")


def _metrics(prediction: np.ndarray, truth: np.ndarray, mappings) -> dict[str, float]:
    prediction_images = np.stack(
        [mappings.elem_to_image_grid(sample) for sample in prediction]
    )
    truth_images = np.stack([mappings.elem_to_image_grid(sample) for sample in truth])
    finite = np.isfinite(prediction_images) & np.isfinite(truth_images)
    support = np.abs(truth) > 1e-8
    background = ~support
    element_error = prediction - truth
    truth_rms = max(float(np.sqrt(np.mean(truth**2))), 1e-12)
    prediction_rms = float(np.sqrt(np.mean(prediction**2)))
    truth_step_rms = max(float(np.sqrt(np.mean(np.diff(truth, axis=0) ** 2))), 1e-12)
    prediction_step_rms = float(np.sqrt(np.mean(np.diff(prediction, axis=0) ** 2)))
    element_correlations = np.asarray(
        [_correlation(pred, target) for pred, target in zip(prediction, truth)]
    )
    image_correlations = np.asarray(
        [
            _correlation(pred[finite_sample], target[finite_sample])
            for pred, target, finite_sample in zip(
                prediction_images, truth_images, np.isfinite(prediction_images) & np.isfinite(truth_images)
            )
        ]
    )
    return {
        "element_rmse": float(np.sqrt(np.mean(element_error**2))),
        "element_nrmse": float(np.sqrt(np.mean(element_error**2))) / truth_rms,
        "element_mae": float(np.mean(np.abs(element_error))),
        "prediction_rms": prediction_rms,
        "truth_rms": truth_rms,
        "prediction_to_truth_rms": prediction_rms / truth_rms,
        "frame_difference_rms": prediction_step_rms,
        "truth_frame_difference_rms": truth_step_rms,
        "frame_difference_rms_ratio": prediction_step_rms / truth_step_rms,
        "element_correlation": _correlation(prediction, truth),
        "element_correlation_per_sample_mean": float(np.nanmean(element_correlations)),
        "element_correlation_per_sample_median": float(np.nanmedian(element_correlations)),
        "image_rmse": float(
            np.sqrt(np.mean((prediction_images[finite] - truth_images[finite]) ** 2))
        ),
        "image_correlation": _correlation(
            prediction_images[finite], truth_images[finite]
        ),
        "image_correlation_per_sample_mean": float(np.nanmean(image_correlations)),
        "image_correlation_per_sample_median": float(np.nanmedian(image_correlations)),
        "localization_dice_at_true_volume": _dice(prediction, truth),
        "support_sign_accuracy": (
            float(np.mean(np.sign(prediction[support]) == np.sign(truth[support])))
            if np.any(support)
            else float("nan")
        ),
        "background_rms": (
            float(np.sqrt(np.mean(prediction[background] ** 2)))
            if np.any(background)
            else float("nan")
        ),
        "vessel_mean": (
            float(np.mean(prediction[support])) if np.any(support) else float("nan")
        ),
    }


def _newton_relative(
    candidate: dict[str, float],
    newton: dict[str, float],
    *,
    candidate_voltage_residual: float,
    newton_voltage_residual: float,
) -> dict:
    """Describe practical improvement over PVI one-step Newton.

    Perfect synthetic reconstruction is not a hard gate.  The learned image is
    considered useful when it remains dynamic/non-degenerate and improves a
    meaningful subset of morphology, amplitude, localization, sign, and
    forward-consistency measures over the production baseline.
    """

    comparisons = {
        "lower_element_nrmse": candidate["element_nrmse"] < newton["element_nrmse"],
        "higher_image_correlation": candidate["image_correlation"] > newton["image_correlation"],
        "higher_localization_dice": candidate["localization_dice_at_true_volume"]
        > newton["localization_dice_at_true_volume"],
        "higher_support_sign_accuracy": candidate["support_sign_accuracy"]
        > newton["support_sign_accuracy"],
        "closer_rms_amplitude": abs(candidate["prediction_to_truth_rms"] - 1.0)
        < abs(newton["prediction_to_truth_rms"] - 1.0),
        "closer_frame_dynamics": abs(candidate["frame_difference_rms_ratio"] - 1.0)
        < abs(newton["frame_difference_rms_ratio"] - 1.0),
        "lower_nonlinear_voltage_residual": candidate_voltage_residual
        < newton_voltage_residual,
    }
    finite = all(np.isfinite(value) for value in candidate.values())
    nondegenerate = (
        0.05 <= candidate["prediction_to_truth_rms"] <= 20.0
        and 0.05 <= candidate["frame_difference_rms_ratio"] <= 20.0
    )
    improvements = int(sum(comparisons.values()))
    if not finite or not nondegenerate:
        classification = "failed_nonfinite_or_blank_static_explosive_sanity_gate"
    elif improvements >= 4:
        classification = "promising_relative_to_production_newton"
    else:
        classification = "exploratory_not_yet_better_than_production_newton"
    return {
        "classification": classification,
        "improvements": improvements,
        "comparisons": comparisons,
        "candidate_minus_newton": {
            "element_nrmse": candidate["element_nrmse"] - newton["element_nrmse"],
            "image_correlation": candidate["image_correlation"] - newton["image_correlation"],
            "localization_dice": candidate["localization_dice_at_true_volume"]
            - newton["localization_dice_at_true_volume"],
            "support_sign_accuracy": candidate["support_sign_accuracy"]
            - newton["support_sign_accuracy"],
            "prediction_to_truth_rms": candidate["prediction_to_truth_rms"]
            - newton["prediction_to_truth_rms"],
            "frame_difference_rms_ratio": candidate["frame_difference_rms_ratio"]
            - newton["frame_difference_rms_ratio"],
            "nonlinear_voltage_residual_rms": candidate_voltage_residual
            - newton_voltage_residual,
        },
        "sanity": {
            "all_metrics_finite": finite,
            "nonblank_nonexplosive_rms": nondegenerate,
        },
    }


def _save_examples(
    output: Path,
    truth: np.ndarray,
    old_newton: np.ndarray | None,
    lm_stages: np.ndarray,
    gcnm_stages: np.ndarray,
    mappings,
    count: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    columns = [("Truth/reference", truth)]
    if old_newton is not None:
        columns.append(("Saved one-step Newton", old_newton))
    for index, values in enumerate(lm_stages, start=1):
        columns.append((f"Iterative LM {index}", values))
    for index, values in enumerate(gcnm_stages, start=1):
        columns.append((f"Faithful GCNM {index}", values))
    for sample in range(min(count, len(truth))):
        images = [mappings.elem_to_image_grid(values[sample]) for _name, values in columns]
        finite_values = np.concatenate([image[np.isfinite(image)] for image in images])
        limit = max(float(np.quantile(np.abs(finite_values), 0.995)), 1e-8)
        figure, axes = plt.subplots(
            1, len(columns), figsize=(2.35 * len(columns), 2.6), constrained_layout=True
        )
        for axis, ((title, _values), image) in zip(axes, zip(columns, images)):
            artist = axis.imshow(
                image, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower"
            )
            axis.set_title(title, fontsize=8)
            axis.axis("off")
        figure.colorbar(artist, ax=axes, shrink=0.72, label="Delta conductivity")
        figure.savefig(output / f"sample_{sample:03d}.png", dpi=170)
        plt.close(figure)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "subject006_anatomical_gcnm.yaml",
    )
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--save-examples", type=int, default=8)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument(
        "--target-kind",
        choices=["clean", "pvi_pseudo", "real_subject006"],
        default="clean",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=["auto", "saved", "homogeneous"],
        default="auto",
    )
    parser.add_argument("--baseline-conductivity", type=float, default=0.7)
    parser.add_argument(
        "--physics-voltage-sign",
        type=float,
        default=None,
        help="Multiply saved V before physical inversion; defaults to -1 for PVI pseudo-label packs and +1 for synthetic data",
    )
    parser.add_argument(
        "--physics-mesh-mode",
        choices=["auto", "coarse", "projected_fine"],
        default="auto",
        help=(
            "auto follows the checkpoint contract; an explicit override is a "
            "fixed-weight physics sensitivity test and does not change weights"
        ),
    )
    parser.add_argument("--minimum-conductivity", type=float, default=1e-4)
    parser.add_argument(
        "--skip-lm-control",
        action="store_true",
        help="Skip the model-independent LM control when it has already been evaluated on the identical test pack",
    )
    parser.add_argument(
        "--allow-config-hash-mismatch",
        action="store_true",
        help=(
            "permit only the whole-file config hash to differ while still "
            "requiring the forward mesh, inverse mesh, mappings, and numerical "
            "physics contract to match"
        ),
    )
    args = parser.parse_args()

    cfg = GcnmConfig.from_yaml(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    first_checkpoint = torch.load(
        args.models_dir / f"{args.model_name}_0.pt",
        map_location=device,
        weights_only=True,
    )
    contract = first_checkpoint.get("physics_contract", {})
    trained_physics_mesh_mode = contract.get("physics_mesh_mode", "coarse")
    physics_mesh_mode = (
        trained_physics_mesh_mode
        if args.physics_mesh_mode == "auto"
        else args.physics_mesh_mode
    )
    if physics_mesh_mode not in {"coarse", "projected_fine"}:
        raise ValueError(
            f"unsupported checkpoint physics mesh mode {physics_mesh_mode!r}"
        )
    runtime = build_runtime(
        cfg, include_forward=physics_mesh_mode == "projected_fine"
    )
    if physics_mesh_mode == "projected_fine":
        if runtime["mappings"].c2f is None:
            raise ValueError(
                "projected-fine evaluation requires a coarse-to-fine mapping"
            )
        stage_physics = ProjectedFineMeshPhysics(
            runtime["physics_fwd"], runtime["mappings"].c2f
        )
    else:
        stage_physics = runtime["physics_inv"]
    current_hashes = {
        "config_sha256": _sha256(args.config),
        "mesh_forward_sha256": _sha256(Path(cfg.mesh_fwd_h5)),
        "mesh_inverse_sha256": _sha256(Path(cfg.mesh_inv_h5)),
        "mappings_sha256": _sha256(Path(cfg.mappings_h5)),
    }
    for key, actual in current_hashes.items():
        if key in contract and contract[key] != actual:
            if key == "config_sha256" and args.allow_config_hash_mismatch:
                print(
                    "warning: allowing whole-file config hash mismatch; "
                    "mesh, mappings, and numerical physics fields remain validated",
                    flush=True,
                )
                continue
            raise ValueError(
                f"checkpoint physics contract {key} does not match evaluation artifact"
            )
    trained_baseline_mode = contract.get("baseline_mode", "saved")
    baseline_mode = (
        trained_baseline_mode if args.baseline_mode == "auto" else args.baseline_mode
    )
    if baseline_mode != trained_baseline_mode:
        raise ValueError(
            f"evaluation baseline mode {baseline_mode} does not match checkpoint "
            f"contract {trained_baseline_mode}"
        )
    source = np.load(args.test)
    if "sigma" not in source or "V" not in source:
        raise ValueError("test pack must contain sigma and V")
    selection = slice(None, args.max_test)
    truth = np.asarray(source["sigma"][selection], dtype=np.float64)
    measured_saved = np.asarray(source["V"][selection], dtype=np.float64)
    voltage_sign = (
        float(args.physics_voltage_sign)
        if args.physics_voltage_sign is not None
        else -1.0 if args.target_kind in {"pvi_pseudo", "real_subject006"} else 1.0
    )
    measured = voltage_sign * measured_saved
    old_newton = (
        np.asarray(source["newton"][selection], dtype=np.float64)
        if "newton" in source
        else truth.copy() if args.target_kind == "pvi_pseudo" else None
    )
    if baseline_mode == "saved":
        if "sigma_baseline" not in source:
            raise ValueError(
                "checkpoint requires saved per-sample baselines, but the test pack "
                "does not contain sigma_baseline; use a model trained with "
                "baseline_mode=homogeneous for real PVI inference"
            )
        baselines = np.asarray(source["sigma_baseline"][selection], dtype=np.float64)
    else:
        trained_conductivity = float(
            contract.get("baseline_conductivity", args.baseline_conductivity)
        )
        if not np.isclose(trained_conductivity, args.baseline_conductivity):
            raise ValueError(
                f"baseline conductivity {args.baseline_conductivity} does not match "
                f"checkpoint contract {trained_conductivity}"
            )
        baselines = np.full_like(truth, args.baseline_conductivity)
    iterations = args.iterations or cfg.iterations
    positions = element_positions(runtime["mesh_inv"]).astype(np.float32)
    current = np.zeros_like(truth)
    baseline_voltages = None
    gcnm_stages, gcnm_directions = [], []
    gcnm_physics, gcnm_voltage = [], []
    for iteration in range(iterations):
        current = (
            np.maximum(baselines + current, float(args.minimum_conductivity)) - baselines
        )
        direction, diagnostics, baseline_voltages = parallel_dataset_lm_directions(
            stage_physics,
            baselines,
            current,
            measured,
            regularizer=runtime["mappings"].laplace,
            hyper_pvi=cfg.hyper_pvi,
            lambda_lm=cfg.lambda_lm,
            step_size=cfg.lm_step_size,
            minimum_conductivity=args.minimum_conductivity,
            baseline_voltages=baseline_voltages,
            progress_label=f"faithful evaluation stage {iteration + 1}",
        )
        checkpoint = torch.load(
            args.models_dir / f"{args.model_name}_{iteration}.pt",
            map_location=device,
            weights_only=True,
        )
        if checkpoint.get("physics_contract", contract) != contract:
            raise ValueError("stage checkpoints have inconsistent physics contracts")
        expected_contract = {
            "hyper_pvi": cfg.hyper_pvi,
            "lambda_lm": cfg.lambda_lm,
            "lm_step_size": cfg.lm_step_size,
            "minimum_conductivity": args.minimum_conductivity,
            "num_elements": int(truth.shape[1]),
            "num_measurements": int(measured.shape[1]),
        }
        for key, expected in expected_contract.items():
            if key in contract and not np.isclose(contract[key], expected):
                raise ValueError(
                    f"checkpoint physics contract {key}={contract[key]} does not "
                    f"match evaluation value {expected}"
                )
        use_coordinates = bool(
            checkpoint.get(
                "use_coordinates",
                all(name in checkpoint["feature_order"] for name in ("x", "y", "radius")),
            )
        )
        use_voltage_mlp = bool(checkpoint.get("use_voltage_mlp", False))
        dataset = _make_dataset(
            truth,
            current,
            direction,
            positions,
            runtime["edge_index"],
            scale=float(checkpoint["scale"]),
            use_coordinates=use_coordinates,
            positive_weight=0.0,
            voltage=measured if use_voltage_mlp else None,
            voltage_scale=float(checkpoint.get("voltage_scale", 1.0)),
        )
        model = _model(
            checkpoint["output_mode"],
            checkpoint["channels"],
            int(checkpoint["in_channels"]),
            use_voltage_mlp=use_voltage_mlp,
            measurements=int(measured.shape[1]),
            voltage_latent=int(checkpoint.get("voltage_latent", 64)),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        current = _predict(model, dataset, float(checkpoint["scale"]))
        residuals, baseline_voltages, final_clips = parallel_dataset_voltage_residual_rms(
            stage_physics,
            baselines,
            current,
            measured,
            minimum_conductivity=args.minimum_conductivity,
            baseline_voltages=baseline_voltages,
        )
        gcnm_directions.append(direction.copy())
        gcnm_stages.append(current.copy())
        physics_report = diagnostics_summary(diagnostics)
        physics_report["post_gcn_voltage_residual_rms_mean"] = float(np.mean(residuals))
        physics_report["post_gcn_clipped_elements_total"] = int(final_clips)
        gcnm_physics.append(physics_report)
        gcnm_voltage.append(residuals)

    if args.skip_lm_control:
        lm_stages = np.empty((0, len(truth), truth.shape[1]), dtype=np.float64)
        lm_diagnostics = []
    else:
        lm_stages, lm_diagnostics = iterative_lm_reconstruction(
            stage_physics,
            baselines,
            measured,
            iterations=iterations,
            regularizer=runtime["mappings"].laplace,
            hyper_pvi=cfg.hyper_pvi,
            lambda_lm=cfg.lambda_lm,
            step_size=cfg.lm_step_size,
            minimum_conductivity=args.minimum_conductivity,
        )
    lm_voltage = []
    lm_physics = []
    lm_baselines = None
    for iteration in range(len(lm_stages)):
        residuals, lm_baselines, clips = parallel_dataset_voltage_residual_rms(
            stage_physics,
            baselines,
            lm_stages[iteration],
            measured,
            minimum_conductivity=args.minimum_conductivity,
            baseline_voltages=lm_baselines,
        )
        stage_report = diagnostics_summary(lm_diagnostics[iteration])
        stage_report["post_lm_voltage_residual_rms_mean"] = float(np.mean(residuals))
        stage_report["post_lm_clipped_elements_total"] = int(clips)
        lm_physics.append(stage_report)
        lm_voltage.append(residuals)

    gcnm_stages_array = np.stack(gcnm_stages)
    real_without_truth = args.target_kind == "real_subject006"
    saved_newton_metrics = None
    saved_newton_voltage_residual = None
    saved_newton_clipped_elements = None
    if old_newton is not None and not real_without_truth:
        saved_newton_metrics = _metrics(old_newton, truth, runtime["mappings"])
        newton_residuals, _newton_baselines, saved_newton_clipped_elements = (
            parallel_dataset_voltage_residual_rms(
                stage_physics,
                baselines,
                old_newton,
                measured,
                minimum_conductivity=args.minimum_conductivity,
            )
        )
        saved_newton_voltage_residual = float(np.mean(newton_residuals))
    iterative_records = []
    for index, values in enumerate(lm_stages):
        record = {"stage": index + 1, "physics": lm_physics[index]}
        if not real_without_truth:
            record["metrics"] = _metrics(values, truth, runtime["mappings"])
        iterative_records.append(record)
    learned_records = []
    for index, values in enumerate(gcnm_stages_array):
        record = {"stage": index + 1, "physics": gcnm_physics[index]}
        if not real_without_truth:
            record["metrics"] = _metrics(values, truth, runtime["mappings"])
            if (
                saved_newton_metrics is not None
                and saved_newton_voltage_residual is not None
            ):
                record["relative_to_saved_one_step_newton"] = _newton_relative(
                    record["metrics"],
                    saved_newton_metrics,
                    candidate_voltage_residual=float(np.mean(gcnm_voltage[index])),
                    newton_voltage_residual=saved_newton_voltage_residual,
                )
        learned_records.append(record)
    report = {
        "method": "faithful differential PVI-GCNM with iterative-LM control",
        "target_kind": args.target_kind,
        "target_warning": (
            "clean synthetic conductivity ground truth"
            if args.target_kind == "clean"
            else "no anatomical ground truth; PVI reconstruction is display-only"
            if real_without_truth
            else "PVI production Newton pseudo-label; not anatomical ground truth"
        ),
        "samples": int(len(truth)),
        "per_stage_physics_recomputed": True,
        "physics_mesh_mode": physics_mesh_mode,
        "trained_physics_mesh_mode": trained_physics_mesh_mode,
        "fixed_weight_physics_mesh_override": (
            physics_mesh_mode != trained_physics_mesh_mode
        ),
        "iterative_lm_control_executed": not args.skip_lm_control,
        "baseline_source": (
            "anatomical baseline supplied by the evaluation pack"
            if baseline_mode == "saved"
            else f"homogeneous {args.baseline_conductivity} S/m inference baseline"
        ),
        "baseline_mode": baseline_mode,
        "verified_artifact_hashes": current_hashes,
        "saved_to_physics_voltage_sign": voltage_sign,
        "voltage_sign_reason": (
            "real PVI pack follows the MATLAB display convention; physical conductivity inversion uses the opposite sign"
            if args.target_kind in {"pvi_pseudo", "real_subject006"} and voltage_sign == -1.0
            else "saved voltage already follows the physical synthetic forward-model convention"
        ),
        "saved_newton": (
            {
                "metrics": saved_newton_metrics,
                "nonlinear_voltage_residual_rms_mean": saved_newton_voltage_residual,
                "clipped_elements_total": int(saved_newton_clipped_elements),
                "role": "production PVI one-step Newton baseline; never a label",
            }
            if saved_newton_metrics is not None
            else None
        ),
        "iterative_lm": iterative_records,
        "faithful_gcnm": learned_records,
        "architecture_gate_policy": {
            "objective": (
                "rich signed dynamic representations that improve materially on "
                "production one-step Newton; perfect synthetic recovery is not required"
            ),
            "hard_sanity_only": (
                "finite output and nonblank/nonstatic/nonexplosive RMS ratios in [0.05, 20]"
            ),
            "promising_rule": (
                "pass sanity and improve at least four of seven Newton-relative metrics"
            ),
            "absolute_correlation_or_nrmse_threshold_is_hard_gate": False,
        },
    }
    if real_without_truth:
        report["metrics_policy"] = (
            "No image correlation, Dice, RMSE, or background RMS is computed "
            "against PVI because it is not anatomical ground truth."
        )
        report["primary_metric"] = "nonlinear differential voltage residual RMS"
        report["pvi_reconstruction_role"] = "display-only pseudo-reference"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.out_dir / "predictions.npz",
        truth=truth.astype(np.float32),
        old_newton=(
            old_newton.astype(np.float32)
            if old_newton is not None
            else np.empty((0, truth.shape[1]), dtype=np.float32)
        ),
        lm_stages=lm_stages.astype(np.float32),
        gcnm_directions=np.stack(gcnm_directions).astype(np.float32),
        gcnm_stages=gcnm_stages_array.astype(np.float32),
        voltage_residuals_lm=(
            np.stack(lm_voltage).astype(np.float32)
            if lm_voltage
            else np.empty((0, len(truth)), dtype=np.float32)
        ),
        voltage_residuals_gcnm=np.stack(gcnm_voltage).astype(np.float32),
    )
    _save_examples(
        args.out_dir / "examples",
        truth,
        old_newton,
        lm_stages,
        gcnm_stages_array,
        runtime["mappings"],
        args.save_examples,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
