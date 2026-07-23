#!/usr/bin/env python3
"""Evaluate the core-guided GCNM with calibration and a zero-safe voltage gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gcnm_pvi.anatomical_phantoms import element_positions
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.core_guided_model import CoreGuidedGCNMStage
from gcnm_pvi.direction_anchored_model import DirectionAnchoredCoreStage
from gcnm_pvi.core_guided_physics import (
    build_fixed_linear_operator,
    linear_amplitude_calibration,
    normalized_context_direction,
)
from gcnm_pvi.evaluate_faithful_gcnm import _metrics
from gcnm_pvi.generate_multisubject_beat_dataset import _rank_one_template
from gcnm_pvi.iterative_physics import (
    dataset_lm_directions,
    dataset_voltage_residual_rms,
)
from gcnm_pvi.runtime import build_runtime
from gcnm_pvi.train_core_guided_gcnm import (
    _graphs,
    _predict,
    _vessel_targets,
)
from gcnm_pvi.train_voltage_vessel_gcnm import _anatomy_targets


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def _beat_templates(
    voltage: np.ndarray,
    period_id: np.ndarray | None,
    group_id: np.ndarray | None = None,
) -> np.ndarray:
    if period_id is None:
        return np.asarray(voltage, dtype=np.float64).copy()
    templates = np.zeros_like(voltage, dtype=np.float64)
    period_templates = {}
    for period in np.unique(period_id):
        selected = np.flatnonzero(period_id == period)
        period_templates[int(period)] = _rank_one_template(voltage[selected])
    if group_id is None:
        for period, template in period_templates.items():
            templates[period_id == period] = template
        return templates
    # Finger geometry is fixed inside an acquisition trial. The first right
    # singular vector of all unit beat templates suppresses beat-specific
    # noise without using conductivity labels or PVI images.
    for group in np.unique(group_id):
        selected = np.flatnonzero(group_id == group)
        periods = np.unique(period_id[selected])
        stack = np.stack([period_templates[int(period)] for period in periods])
        norms = np.linalg.norm(stack, axis=1)
        normalized = stack / np.maximum(norms[:, None], 1e-12)
        _u, _s, vh = np.linalg.svd(normalized, full_matrices=False)
        group_template = vh[0] * max(float(np.median(norms)), 1e-12)
        templates[selected] = group_template
    return templates


def _hard_vessel_masks(parameters: np.ndarray, positions: np.ndarray) -> np.ndarray:
    _combined, soft, _centers = _vessel_targets(parameters, positions)
    threshold = np.exp(-0.5 / 1.25**2)
    return soft >= threshold


def _object_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    predicted_centers: np.ndarray,
    parameters: np.ndarray,
    positions: np.ndarray,
    tissue_labels: np.ndarray,
) -> dict[str, float]:
    dice, center_error, amplitude_error = [], [], []
    halo_error, outside_values, mass_ratio = [], [], []
    for index in range(len(prediction)):
        vessel_masks = _hard_vessel_masks(parameters[index], positions)
        core = np.any(vessel_masks, axis=1)
        count = int(np.count_nonzero(core))
        if count:
            selected = np.argpartition(np.abs(prediction[index]), -count)[-count:]
            dice.append(
                2.0 * len(np.intersect1d(np.flatnonzero(core), selected))
                / (count + len(selected))
            )
        target_centers = parameters[index, :, :2]
        predicted = predicted_centers[index]
        direct = np.mean(np.linalg.norm(predicted - target_centers, axis=1))
        swapped = np.mean(
            np.linalg.norm(predicted[::-1] - target_centers, axis=1)
        )
        center_error.append(min(direct, swapped))
        for slot in range(2):
            mask = vessel_masks[:, slot]
            if np.any(mask):
                amplitude_error.append(
                    abs(
                        float(np.mean(prediction[index, mask]))
                        - float(np.mean(truth[index, mask]))
                    )
                )
        muscle = (tissue_labels[index] == 3) & ~core
        outside = (tissue_labels[index] != 3) & ~core
        if np.any(muscle):
            halo_error.extend((prediction[index, muscle] - truth[index, muscle]) ** 2)
        if np.any(outside):
            outside_values.extend(prediction[index, outside] ** 2)
        denominator = np.sum(np.abs(truth[index]))
        if denominator > 1e-10:
            mass_ratio.append(np.sum(np.abs(prediction[index])) / denominator)
    return {
        "core_dice_at_true_core_volume": float(np.mean(dice)) if dice else 0.0,
        "matched_centroid_error_normalized": float(np.mean(center_error)),
        "per_vessel_core_amplitude_mae_s_m": float(np.mean(amplitude_error)),
        "muscle_halo_rmse_s_m": float(np.sqrt(np.mean(halo_error))) if halo_error else 0.0,
        "outside_tissue_rms_s_m": float(np.sqrt(np.mean(outside_values))) if outside_values else 0.0,
        "absolute_mass_ratio_median": float(np.median(mass_ratio)) if mass_ratio else 0.0,
    }


def _temporal_center_stability(
    centers: np.ndarray,
    period_id: np.ndarray | None,
) -> dict[str, float] | None:
    if period_id is None:
        return None
    deviations = []
    for period in np.unique(period_id):
        values = centers[period_id == period]
        values = np.stack([sample[np.argsort(sample[:, 0])] for sample in values])
        deviations.append(float(np.mean(np.std(values, axis=0))))
    return {
        "mean_within_beat_center_std_normalized": float(np.mean(deviations)),
        "maximum_within_beat_center_std_normalized": float(np.max(deviations)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--target-kind", choices=["clean", "real_subject006"], required=True
    )
    parser.add_argument("--anatomy", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--minimum-conductivity", type=float, default=1e-4)
    parser.add_argument(
        "--stage1-maximum-amplitude-scale",
        type=float,
        default=None,
        help=(
            "optional inference-only override for the stage-1 voltage-space "
            "scale cap; defaults to the checkpoint contract"
        ),
    )
    parser.add_argument(
        "--stage2-maximum-amplitude-scale",
        type=float,
        default=None,
        help=(
            "optional inference-only override for the final voltage-space "
            "scale cap; defaults to the checkpoint contract"
        ),
    )
    args = parser.parse_args()

    cfg = GcnmConfig.from_yaml(args.config)
    runtime = build_runtime(cfg, include_forward=False)
    checkpoint_1 = torch.load(args.stage1, map_location="cpu", weights_only=False)
    checkpoint_2 = torch.load(args.stage2, map_location="cpu", weights_only=False)
    if checkpoint_1["contract"] != checkpoint_2["contract"]:
        raise ValueError("stage checkpoint contracts differ")
    contract = checkpoint_1["contract"]
    supported_architectures = {
        "core_guided_dense_two_stage_v1",
        "direction_anchored_core_residual_two_stage_v1",
    }
    if contract["architecture"] not in supported_architectures:
        raise ValueError("checkpoint is not a supported core-guided GCNM")
    checkpoint_scale = float(contract["maximum_amplitude_scale"])
    stage1_scale_limit = (
        checkpoint_scale
        if args.stage1_maximum_amplitude_scale is None
        else float(args.stage1_maximum_amplitude_scale)
    )
    stage2_scale_limit = (
        checkpoint_scale
        if args.stage2_maximum_amplitude_scale is None
        else float(args.stage2_maximum_amplitude_scale)
    )
    if stage1_scale_limit <= 0 or stage2_scale_limit <= 0:
        raise ValueError("amplitude-scale limits must be positive")
    if contract["architecture"] == "direction_anchored_core_residual_two_stage_v1":
        stage1 = DirectionAnchoredCoreStage(
            accumulate_current=False,
            correction_limit=float(contract["stage1_limit"]),
        ).float()
        stage2 = DirectionAnchoredCoreStage(
            accumulate_current=True,
            correction_limit=float(contract["stage2_limit"]),
        ).float()
    else:
        stage1 = CoreGuidedGCNMStage(
            residual=False,
            correction_limit=float(contract["stage1_limit"]),
        ).float()
        stage2 = CoreGuidedGCNMStage(
            residual=True,
            correction_limit=float(contract["stage2_limit"]),
        ).float()
    stage1.load_state_dict(checkpoint_1["state_dict"])
    stage2.load_state_dict(checkpoint_2["state_dict"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stage1.to(device)
    stage2.to(device)

    with np.load(args.test) as source:
        limit = len(source["V"]) if args.max_samples is None else min(
            len(source["V"]), args.max_samples
        )
        voltage_saved = np.asarray(source["V"][:limit], dtype=np.float64)
        display_reference = np.asarray(source["sigma"][:limit], dtype=np.float64)
        old_newton = (
            np.asarray(source["newton"][:limit], dtype=np.float64)
            if "newton" in source
            else display_reference.copy()
        )
        tissue_labels = (
            np.asarray(source["tissue_labels"][:limit], dtype=np.uint8)
            if "tissue_labels" in source
            else None
        )
        period_id = (
            np.asarray(source["period_id"][:limit])
            if "period_id" in source
            else np.asarray(source["beat_id"][:limit])
            if "beat_id" in source
            else None
        )
        trial_id = (
            np.asarray(source["trial_index"][:limit])
            if "trial_index" in source
            else None
        )
        saved_template = (
            np.asarray(source["V_template"][:limit], dtype=np.float64)
            if "V_template" in source
            else None
        )
    sign = -1.0 if args.target_kind == "real_subject006" else 1.0
    measured = sign * voltage_saved
    template_voltage = (
        sign * saved_template
        if saved_template is not None
        else _beat_templates(measured, period_id, trial_id)
    )
    elements = display_reference.shape[1]
    baseline = np.full_like(
        display_reference, float(contract["baseline_conductivity"])
    )
    baseline_vector = baseline[0]
    fixed = build_fixed_linear_operator(
        runtime["physics_inv"],
        baseline_vector,
        regularizer=runtime["mappings"].laplace,
        hyper_pvi=cfg.hyper_pvi,
        lambda_lm=cfg.lambda_lm,
    )
    direction_1 = fixed.directions(measured)
    context = normalized_context_direction(fixed.directions(template_voltage))
    positions = element_positions(runtime["mesh_inv"]).astype(np.float32)
    parameters = None
    if args.target_kind == "clean":
        if args.anatomy is None:
            raise ValueError("clean evaluation requires --anatomy")
        parameters = _anatomy_targets(
            args.anatomy,
            float(contract["conductivity_scale"]),
            limit,
            include_diffusion=False,
        )
    zero = np.zeros_like(display_reference)
    graphs_1 = _graphs(
        display_reference, zero, direction_1, context, measured,
        tissue_labels, parameters, positions, runtime["edge_index"],
        conductivity_scale=float(contract["conductivity_scale"]),
        voltage_scale=float(contract["voltage_scale"]),
    )
    stage1_raw, centers_1, core_probability_1 = _predict(
        stage1, graphs_1, float(contract["conductivity_scale"]),
        batch_size=cfg.batch_size,
    )
    stage1_map, scale_1, linear_residual_1 = linear_amplitude_calibration(
        stage1_raw,
        measured,
        fixed.jacobian,
        maximum_scale=stage1_scale_limit,
    )
    baseline_voltages = np.broadcast_to(
        fixed.baseline_voltage[None, :], measured.shape
    )
    direction_2, _diagnostics_2, _ = dataset_lm_directions(
        runtime["physics_inv"], baseline, stage1_map, measured,
        regularizer=runtime["mappings"].laplace,
        hyper_pvi=cfg.hyper_pvi, lambda_lm=cfg.lambda_lm,
        step_size=cfg.lm_step_size,
        minimum_conductivity=args.minimum_conductivity,
        baseline_voltages=baseline_voltages,
        progress_label="core-guided stage 2 evaluation",
    )
    graphs_2 = _graphs(
        display_reference, stage1_map, direction_2, context, measured,
        tissue_labels, parameters, positions, runtime["edge_index"],
        conductivity_scale=float(contract["conductivity_scale"]),
        voltage_scale=float(contract["voltage_scale"]),
    )
    stage2_raw, centers_2, core_probability_2 = _predict(
        stage2, graphs_2, float(contract["conductivity_scale"]),
        batch_size=cfg.batch_size,
    )
    stage2_map, scale_2, linear_residual_2 = linear_amplitude_calibration(
        stage2_raw,
        measured,
        fixed.jacobian,
        maximum_scale=stage2_scale_limit,
    )
    residual_1, _, clips_1 = dataset_voltage_residual_rms(
        runtime["physics_inv"], baseline, stage1_map, measured,
        minimum_conductivity=args.minimum_conductivity,
        baseline_voltages=baseline_voltages,
    )
    residual_2, _, clips_2 = dataset_voltage_residual_rms(
        runtime["physics_inv"], baseline, stage2_map, measured,
        minimum_conductivity=args.minimum_conductivity,
        baseline_voltages=baseline_voltages,
    )
    initial_residual = np.sqrt(np.mean(measured**2, axis=1))
    candidates = np.stack((initial_residual, residual_1, residual_2), axis=1)
    selected_index = np.argmin(candidates, axis=1)
    selected = np.zeros_like(stage1_map)
    selected[selected_index == 1] = stage1_map[selected_index == 1]
    selected[selected_index == 2] = stage2_map[selected_index == 2]
    selected_residual = np.min(candidates, axis=1)
    selection_fraction = {
        "initial_zero": float(np.mean(selected_index == 0)),
        "stage_1": float(np.mean(selected_index == 1)),
        "stage_2": float(np.mean(selected_index == 2)),
    }
    physics = {
        "amplitude_calibration": {
            "checkpoint_scale_limit": checkpoint_scale,
            "stage_1_scale_limit": stage1_scale_limit,
            "stage_2_scale_limit": stage2_scale_limit,
            "override_is_inference_only": bool(
                args.stage1_maximum_amplitude_scale is not None
                or args.stage2_maximum_amplitude_scale is not None
            ),
        },
        "initial_voltage_rms": _summary(initial_residual),
        "stage_1_voltage_residual_rms": _summary(residual_1),
        "stage_2_voltage_residual_rms": _summary(residual_2),
        "selected_voltage_residual_rms": _summary(selected_residual),
        "linear_calibration_residual_stage_1": _summary(linear_residual_1),
        "linear_calibration_residual_stage_2": _summary(linear_residual_2),
        "amplitude_scale_stage_1": _summary(scale_1),
        "amplitude_scale_stage_2": _summary(scale_2),
        "selection_fraction": selection_fraction,
        "stage_1_better_than_initial_fraction": float(np.mean(residual_1 < initial_residual)),
        "stage_2_better_than_initial_fraction": float(np.mean(residual_2 < initial_residual)),
        "stage_1_clipped_elements": int(clips_1),
        "stage_2_clipped_elements": int(clips_2),
    }
    if args.target_kind == "clean":
        stage_metrics = {}
        for name, prediction, centers in (
            ("stage_1", stage1_map, centers_1),
            ("stage_2", stage2_map, centers_2),
            ("voltage_selected", selected, np.where(
                (selected_index == 2)[:, None, None], centers_2, centers_1
            )),
        ):
            stage_metrics[name] = {
                **_metrics(prediction, display_reference, runtime["mappings"]),
                **_object_metrics(
                    prediction,
                    display_reference,
                    centers,
                    parameters,
                    positions,
                    tissue_labels,
                ),
            }
        report = {
            "method": contract["architecture"],
            "target_kind": "clean signed synthetic conductivity ground truth",
            "samples": limit,
            "architecture": contract["architecture"],
            "metrics": stage_metrics,
            "physics": physics,
        }
    else:
        report = {
            "method": contract["architecture"],
            "target_kind": "real subject 006",
            "samples": limit,
            "architecture": contract["architecture"],
            "metrics_policy": (
                "No image metric is computed against PVI because PVI is a "
                "display-only pseudo-reference, not anatomical ground truth."
            ),
            "primary_metric": "nonlinear differential voltage residual RMS",
            "pvi_reconstruction_role": "display-only pseudo-reference",
            "localization_context": (
                "label-free trial-level SVD of beat voltage templates"
                if trial_id is not None
                else "label-free rank-one beat voltage template"
            ),
            "physics": physics,
            "temporal_center_stability": {
                "stage_1": _temporal_center_stability(centers_1, period_id),
                "stage_2": _temporal_center_stability(centers_2, period_id),
            },
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.out_dir / "predictions.npz",
        truth=display_reference.astype(np.float32),
        old_newton=old_newton.astype(np.float32),
        lm_stages=np.empty((0, limit, elements), dtype=np.float32),
        gcnm_directions=np.stack((direction_1, direction_2)).astype(np.float32),
        gcnm_stages=np.stack((stage1_map, stage2_map)).astype(np.float32),
        gcnm_stages_raw=np.stack((stage1_raw, stage2_raw)).astype(np.float32),
        selected=selected.astype(np.float32),
        selected_stage=selected_index.astype(np.int8),
        voltage_residuals_gcnm=np.stack((residual_1, residual_2)).astype(np.float32),
        voltage_residuals_selected=selected_residual.astype(np.float32),
        amplitude_scales=np.stack((scale_1, scale_2)).astype(np.float32),
        vessel_centers_stage_1=centers_1.astype(np.float32),
        vessel_centers_stage_2=centers_2.astype(np.float32),
        core_probability_stage_1=core_probability_1.astype(np.float32),
        core_probability_stage_2=core_probability_2.astype(np.float32),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
