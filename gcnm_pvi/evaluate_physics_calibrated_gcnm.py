#!/usr/bin/env python3
"""Evaluate the two-stage physics-calibrated core-guided residual GCNM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gcnm_pvi.anatomical_phantoms import element_positions
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.core_guided_physics import (
    build_fixed_linear_operator,
    normalized_context_direction,
)
from gcnm_pvi.evaluate_core_guided_gcnm import (
    _object_metrics,
    _temporal_center_stability,
)
from gcnm_pvi.evaluate_faithful_gcnm import _metrics
from gcnm_pvi.iterative_physics import dataset_voltage_residual_rms
from gcnm_pvi.physics_calibrated_model import PhysicsCalibratedCoreStage
from gcnm_pvi.physics_calibrated_physics import (
    nonlinear_stage_physics,
    stage_physics_summary,
)
from gcnm_pvi.runtime import build_runtime
from gcnm_pvi.train_physics_calibrated_gcnm import (
    _fixed_stage_physics,
    _predict_calibrated,
    _stage_graphs,
    _trial_templates,
)
from gcnm_pvi.train_voltage_vessel_gcnm import _anatomy_targets


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def _activation(values: np.ndarray) -> dict[str, float]:
    per_sample = np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2, axis=1))
    return {
        "mean_spatial_rms_s_m": float(np.mean(per_sample)),
        "median_spatial_rms_s_m": float(np.median(per_sample)),
        "maximum_spatial_rms_s_m": float(np.max(per_sample)),
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
    args = parser.parse_args()

    cfg = GcnmConfig.from_yaml(args.config)
    runtime = build_runtime(cfg, include_forward=False)
    checkpoint_1 = torch.load(args.stage1, map_location="cpu", weights_only=False)
    checkpoint_2 = torch.load(args.stage2, map_location="cpu", weights_only=False)
    if checkpoint_1["contract"] != checkpoint_2["contract"]:
        raise ValueError("stage checkpoint contracts differ")
    contract = checkpoint_1["contract"]
    if contract["architecture"] != "physics_calibrated_core_residual_two_stage_v1":
        raise ValueError("checkpoint is not a physics-calibrated two-stage GCNM")
    stage1 = PhysicsCalibratedCoreStage(
        correction_limit=float(contract["stage1_limit"])
    ).float()
    stage2 = PhysicsCalibratedCoreStage(
        correction_limit=float(contract["stage2_limit"])
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
    if saved_template is not None:
        template_voltage = sign * saved_template
    elif period_id is not None and trial_id is not None:
        template_voltage = _trial_templates(measured, period_id, trial_id)
    else:
        template_voltage = measured.copy()

    elements = display_reference.shape[1]
    baseline = np.full_like(
        display_reference, float(contract["baseline_conductivity"])
    )
    fixed = build_fixed_linear_operator(
        runtime["physics_inv"],
        baseline[0],
        regularizer=runtime["mappings"].laplace,
        hyper_pvi=cfg.hyper_pvi,
        lambda_lm=cfg.lambda_lm,
    )
    positions = element_positions(runtime["mesh_inv"]).astype(np.float32)
    parameters = None
    supervised = args.target_kind == "clean"
    if supervised:
        if args.anatomy is None:
            raise ValueError("clean evaluation requires --anatomy")
        parameters = _anatomy_targets(
            args.anatomy,
            float(contract["conductivity_scale"]),
            limit,
            include_diffusion=False,
        )
    zero = np.zeros_like(display_reference)
    physics_1 = _fixed_stage_physics(fixed, zero, measured)
    context = normalized_context_direction(
        fixed.directions(template_voltage)
    )
    graphs_1 = _stage_graphs(
        truth=display_reference if supervised else zero,
        current=zero,
        physics=physics_1,
        context=context,
        measured=measured,
        tissue_labels=tissue_labels,
        vessel_parameters=parameters,
        positions=positions,
        edge_index=runtime["edge_index"],
        conductivity_scale=float(contract["conductivity_scale"]),
        voltage_scale=float(contract["voltage_scale"]),
        supervised=supervised,
    )
    stage1_map, amplitude_1, linear_ratio_1, centers_1, core_probability_1 = (
        _predict_calibrated(
            stage1,
            graphs_1,
            conductivity_scale=float(contract["conductivity_scale"]),
            maximum_amplitude=float(contract["maximum_amplitude_s_m"]),
            batch_size=cfg.batch_size,
        )
    )
    baseline_voltages = np.broadcast_to(
        fixed.baseline_voltage[None, :], measured.shape
    )
    physics_2 = nonlinear_stage_physics(
        runtime["physics_inv"],
        baseline,
        stage1_map,
        measured,
        regularizer=runtime["mappings"].laplace,
        hyper_pvi=cfg.hyper_pvi,
        lambda_lm=cfg.lambda_lm,
        step_size=cfg.lm_step_size,
        minimum_conductivity=args.minimum_conductivity,
        baseline_voltages=baseline_voltages,
        progress_label="calibrated stage 2 evaluation",
    )
    graphs_2 = _stage_graphs(
        truth=display_reference if supervised else zero,
        current=stage1_map,
        physics=physics_2,
        context=context,
        measured=measured,
        tissue_labels=tissue_labels,
        vessel_parameters=parameters,
        positions=positions,
        edge_index=runtime["edge_index"],
        conductivity_scale=float(contract["conductivity_scale"]),
        voltage_scale=float(contract["voltage_scale"]),
        supervised=supervised,
    )
    stage2_map, amplitude_2, linear_ratio_2, centers_2, core_probability_2 = (
        _predict_calibrated(
            stage2,
            graphs_2,
            conductivity_scale=float(contract["conductivity_scale"]),
            maximum_amplitude=float(contract["maximum_amplitude_s_m"]),
            batch_size=cfg.batch_size,
        )
    )
    residual_1, _, clips_1 = dataset_voltage_residual_rms(
        runtime["physics_inv"],
        baseline,
        stage1_map,
        measured,
        minimum_conductivity=args.minimum_conductivity,
        baseline_voltages=baseline_voltages,
    )
    residual_2, _, clips_2 = dataset_voltage_residual_rms(
        runtime["physics_inv"],
        baseline,
        stage2_map,
        measured,
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
    physics_report = {
        "initial_voltage_rms": _summary(initial_residual),
        "stage_1_voltage_residual_rms": _summary(residual_1),
        "stage_2_voltage_residual_rms": _summary(residual_2),
        "selected_voltage_residual_rms": _summary(selected_residual),
        "selected_residual_to_measured_mean_ratio": float(
            np.mean(selected_residual) / np.mean(initial_residual)
        ),
        "stage_1_linear_residual_ratio": _summary(linear_ratio_1),
        "stage_2_linear_remaining_residual_ratio": _summary(linear_ratio_2),
        "stage_1_amplitude_s_m": _summary(amplitude_1),
        "stage_2_correction_amplitude_s_m": _summary(amplitude_2),
        "stage_2_input_physics": stage_physics_summary(physics_2),
        "selection_fraction": {
            "initial_zero": float(np.mean(selected_index == 0)),
            "stage_1": float(np.mean(selected_index == 1)),
            "stage_2": float(np.mean(selected_index == 2)),
        },
        "stage_1_better_than_initial_fraction": float(
            np.mean(residual_1 < initial_residual)
        ),
        "stage_2_better_than_initial_fraction": float(
            np.mean(residual_2 < initial_residual)
        ),
        "stage_1_clipped_elements": int(clips_1),
        "stage_2_clipped_elements": int(clips_2),
    }
    activation = {
        "pvi_pseudo_reference": _activation(display_reference),
        "stage_1": _activation(stage1_map),
        "stage_2": _activation(stage2_map),
        "selected": _activation(selected),
    }
    pvi_activation = activation["pvi_pseudo_reference"]["mean_spatial_rms_s_m"]
    activation["stage_1_magnitude_fraction_of_pvi"] = float(
        activation["stage_1"]["mean_spatial_rms_s_m"]
        / max(pvi_activation, 1e-12)
    )
    activation["stage_2_magnitude_fraction_of_pvi"] = float(
        activation["stage_2"]["mean_spatial_rms_s_m"]
        / max(pvi_activation, 1e-12)
    )
    activation["selected_magnitude_fraction_of_pvi"] = float(
        activation["selected"]["mean_spatial_rms_s_m"]
        / max(pvi_activation, 1e-12)
    )

    if supervised:
        metrics = {}
        for name, prediction, centers in (
            ("stage_1", stage1_map, centers_1),
            ("stage_2", stage2_map, centers_2),
            (
                "voltage_selected",
                selected,
                np.where((selected_index == 2)[:, None, None], centers_2, centers_1),
            ),
        ):
            metrics[name] = {
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
            "method": "two-stage physics-calibrated core-guided residual GCNM",
            "target_kind": "clean signed synthetic conductivity ground truth",
            "samples": limit,
            "architecture": contract["architecture"],
            "metrics": metrics,
            "physics": physics_report,
            "activation": activation,
        }
    else:
        report = {
            "method": "two-stage physics-calibrated core-guided residual GCNM",
            "target_kind": "real subject 006",
            "samples": limit,
            "architecture": contract["architecture"],
            "metrics_policy": (
                "No image metric is computed against PVI because PVI is a "
                "display-only pseudo-reference, not anatomical ground truth."
            ),
            "primary_metric": "nonlinear differential voltage residual RMS",
            "pvi_reconstruction_role": "display-only magnitude reference",
            "physics": physics_report,
            "activation": activation,
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
        gcnm_directions=np.stack(
            (physics_1.directions, physics_2.directions)
        ).astype(np.float32),
        gcnm_stages=np.stack((stage1_map, stage2_map)).astype(np.float32),
        selected=selected.astype(np.float32),
        selected_stage=selected_index.astype(np.int8),
        voltage_residuals_gcnm=np.stack((residual_1, residual_2)).astype(
            np.float32
        ),
        voltage_residuals_selected=selected_residual.astype(np.float32),
        amplitude_scales=np.stack((amplitude_1, amplitude_2)).astype(np.float32),
        linear_residual_ratios=np.stack((linear_ratio_1, linear_ratio_2)).astype(
            np.float32
        ),
        vessel_centers_stage_1=centers_1.astype(np.float32),
        vessel_centers_stage_2=centers_2.astype(np.float32),
        core_probability_stage_1=core_probability_1.astype(np.float32),
        core_probability_stage_2=core_probability_2.astype(np.float32),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
