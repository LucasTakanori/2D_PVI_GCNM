#!/usr/bin/env python3
"""Evaluate the voltage-conditioned vessel model with a voltage-RMS gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gcnm_pvi.anatomical_phantoms import element_positions
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.core_guided_physics import build_fixed_linear_operator
from gcnm_pvi.evaluate_core_guided_gcnm import (
    _beat_templates,
    _temporal_center_stability,
)
from gcnm_pvi.evaluate_faithful_gcnm import _metrics
from gcnm_pvi.generalized_vessel_model import (
    BeatMeanPoolVesselLocalizer,
    BeatSpatialDiffusionLocalizer,
    BeatVesselParameterRefiner,
)
from gcnm_pvi.iterative_physics import (
    dataset_lm_directions,
    dataset_voltage_residual_rms,
)
from gcnm_pvi.runtime import build_runtime
from gcnm_pvi.vessel_runtime import (
    make_vessel_graphs as _graphs,
    predict_vessel_graphs as _predict,
    predict_vessel_graphs_with_parameters as _predict_with_parameters,
)
from gcnm_pvi.voltage_vessel_model import (
    SpatialAttentionVesselLocalizer,
    SpatialAttentionVesselDiffusionLocalizer,
    SpatialVesselDiffusionRefiner,
    SpatialVesselParameterRefiner,
    VoltageConditionedPhysicsRefiner,
    VoltageConditionedVesselLocalizer,
)


def _voltage_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--localizer", type=Path, required=True)
    parser.add_argument("--refiner", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--target-kind", choices=["clean", "real_subject006"], required=True
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--minimum-conductivity", type=float, default=1e-4)
    args = parser.parse_args()

    cfg = GcnmConfig.from_yaml(args.config)
    runtime = build_runtime(cfg, include_forward=False)
    localizer_checkpoint = torch.load(
        args.localizer, map_location="cpu", weights_only=False
    )
    refiner_checkpoint = torch.load(
        args.refiner, map_location="cpu", weights_only=False
    )
    if localizer_checkpoint["contract"] != refiner_checkpoint["contract"]:
        raise ValueError("localizer and refiner checkpoint contracts differ")
    contract = localizer_checkpoint["contract"]
    architecture = contract.get("architecture", "mean_pool_dense_refiner")
    generalized = architecture in {"beat_voltage_slots", "beat_diffusion_slots"}
    diffusion_architecture = architecture in {
        "diffusion_slots",
        "beat_diffusion_slots",
    }
    parameter_architecture = architecture in {
        "spatial_slots",
        "diffusion_slots",
        "beat_voltage_slots",
        "beat_diffusion_slots",
    }
    minimum_axis = float(contract.get("minimum_vessel_axis", 0.05))
    maximum_axis = float(contract.get("maximum_vessel_axis", 0.23))
    localizer = (
        BeatSpatialDiffusionLocalizer(
            minimum_axis=minimum_axis, maximum_axis=maximum_axis
        )
        if architecture == "beat_diffusion_slots"
        else BeatMeanPoolVesselLocalizer(
            minimum_axis=minimum_axis, maximum_axis=maximum_axis
        )
        if architecture == "beat_voltage_slots"
        else SpatialAttentionVesselDiffusionLocalizer(
            minimum_axis=minimum_axis, maximum_axis=maximum_axis
        )
        if architecture == "diffusion_slots"
        else SpatialAttentionVesselLocalizer(
            minimum_axis=minimum_axis, maximum_axis=maximum_axis
        )
        if architecture == "spatial_slots"
        else VoltageConditionedVesselLocalizer(
            minimum_axis=minimum_axis, maximum_axis=maximum_axis
        )
    ).float()
    localizer.load_state_dict(localizer_checkpoint["state_dict"])
    refiner = (
        BeatVesselParameterRefiner(
            diffusion=architecture == "beat_diffusion_slots",
            minimum_axis=minimum_axis,
            maximum_axis=maximum_axis,
        )
        if generalized
        else SpatialVesselDiffusionRefiner(
            minimum_axis=minimum_axis, maximum_axis=maximum_axis
        )
        if architecture == "diffusion_slots"
        else SpatialVesselParameterRefiner(
            minimum_axis=minimum_axis, maximum_axis=maximum_axis
        )
        if architecture == "spatial_slots"
        else VoltageConditionedPhysicsRefiner()
    ).float()
    refiner.load_state_dict(refiner_checkpoint["state_dict"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    localizer.to(device)
    refiner.to(device)

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
        saved_baselines = (
            np.asarray(source["sigma_baseline"][:limit], dtype=np.float64)
            if "sigma_baseline" in source
            else None
        )
        saved_tissue_labels = (
            np.asarray(source["tissue_labels"][:limit], dtype=np.uint8)
            if "tissue_labels" in source
            else None
        )
        saved_template = (
            np.asarray(source["V_template"][:limit], dtype=np.float64)
            if "V_template" in source
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
    voltage_sign = -1.0 if args.target_kind == "real_subject006" else 1.0
    measured = voltage_sign * voltage_saved
    template_voltage = (
        voltage_sign * saved_template
        if saved_template is not None
        else _beat_templates(measured, period_id, trial_id)
        if generalized and period_id is not None
        else measured.copy()
    )
    baseline_mode = contract.get("baseline_mode", "homogeneous")
    baseline_conductivity = float(contract["baseline_conductivity"])
    if baseline_mode == "saved":
        if saved_baselines is None:
            raise ValueError(
                "checkpoint requires an anatomical sigma_baseline, but the test "
                "pack does not contain it"
            )
        baselines = saved_baselines
    else:
        baselines = np.full_like(display_reference, baseline_conductivity)
    if saved_tissue_labels is not None:
        tissue_labels = saved_tissue_labels
    elif diffusion_architecture and saved_baselines is not None:
        # The real evaluation pack uses the unaugmented default simulator prior.
        # Its muscle conductivity is 0.352 S/m at 50 kHz. This is a prior mask,
        # not subject-specific anatomical ground truth.
        tissue_labels = np.zeros_like(saved_baselines, dtype=np.uint8)
        tissue_labels[np.isclose(saved_baselines, 0.352, rtol=0.05, atol=0.005)] = 3
    else:
        tissue_labels = None
    positions = element_positions(runtime["mesh_inv"]).astype(np.float32)
    zero = np.zeros_like(display_reference)
    dummy_parameters = np.zeros(
        (limit, 2, 8 if diffusion_architecture else 6), dtype=np.float32
    )

    context_direction = None
    if generalized:
        fixed = build_fixed_linear_operator(
            runtime["physics_inv"],
            baselines[0],
            regularizer=runtime["mappings"].laplace,
            hyper_pvi=cfg.hyper_pvi,
            lambda_lm=cfg.lambda_lm,
        )
        direction_1 = fixed.directions(measured)
        context_direction = fixed.directions(template_voltage)
        baseline_voltages = np.broadcast_to(
            fixed.baseline_voltage[None, :], measured.shape
        )
    else:
        direction_1, _diagnostics_1, baseline_voltages = dataset_lm_directions(
            runtime["physics_inv"], baselines, zero, measured,
            regularizer=runtime["mappings"].laplace, hyper_pvi=cfg.hyper_pvi,
            lambda_lm=cfg.lambda_lm, step_size=cfg.lm_step_size,
            minimum_conductivity=args.minimum_conductivity,
            progress_label="stage 1 evaluation",
        )
    graphs_1 = _graphs(
        display_reference, zero, direction_1, measured, dummy_parameters,
        positions, runtime["edge_index"],
        conductivity_scale=float(contract["conductivity_scale"]),
        voltage_scale=float(contract["voltage_scale"]),
        tissue_labels=tissue_labels,
        voltage_template=template_voltage,
        context_direction=context_direction,
        voltage_clean=measured,
        beat_id=period_id,
        voltage_input_mode=(
            "beat_normalized" if generalized else "global_scale"
        ),
        voltage_rms_reference=float(
            contract.get("voltage_rms_reference", contract["voltage_scale"])
        ),
    )
    if parameter_architecture:
        stage_1, stage_1_parameters = _predict_with_parameters(
            localizer, graphs_1, float(contract["conductivity_scale"])
        )
    else:
        stage_1 = _predict(
            localizer, graphs_1, float(contract["conductivity_scale"])
        )
        stage_1_parameters = np.empty((0, 2, 6), dtype=np.float32)
    residual_1, _, clips_1 = dataset_voltage_residual_rms(
        runtime["physics_inv"], baselines, stage_1, measured,
        minimum_conductivity=args.minimum_conductivity,
        baseline_voltages=baseline_voltages,
    )

    direction_2, _diagnostics_2, _ = dataset_lm_directions(
        runtime["physics_inv"], baselines, stage_1, measured,
        regularizer=runtime["mappings"].laplace, hyper_pvi=cfg.hyper_pvi,
        lambda_lm=cfg.lambda_lm, step_size=cfg.lm_step_size,
        minimum_conductivity=args.minimum_conductivity,
        baseline_voltages=baseline_voltages,
        progress_label="stage 2 evaluation",
    )
    graphs_2 = _graphs(
        display_reference, stage_1, direction_2, measured, dummy_parameters,
        positions, runtime["edge_index"],
        conductivity_scale=float(contract["conductivity_scale"]),
        voltage_scale=float(contract["voltage_scale"]),
        initial_parameters=(
            stage_1_parameters
            if parameter_architecture
            else None
        ),
        tissue_labels=tissue_labels,
        voltage_template=template_voltage,
        context_direction=context_direction,
        voltage_clean=measured,
        beat_id=period_id,
        voltage_input_mode=(
            "beat_normalized" if generalized else "global_scale"
        ),
        voltage_rms_reference=float(
            contract.get("voltage_rms_reference", contract["voltage_scale"])
        ),
    )
    if parameter_architecture:
        stage_2, stage_2_parameters = _predict_with_parameters(
            refiner, graphs_2, float(contract["conductivity_scale"])
        )
    else:
        stage_2 = _predict(refiner, graphs_2, float(contract["conductivity_scale"]))
        stage_2_parameters = np.empty((0, 2, 6), dtype=np.float32)
    residual_2, _, clips_2 = dataset_voltage_residual_rms(
        runtime["physics_inv"], baselines, stage_2, measured,
        minimum_conductivity=args.minimum_conductivity,
        baseline_voltages=baseline_voltages,
    )
    initial_residual = np.sqrt(np.mean(measured**2, axis=1))
    candidates = np.stack((initial_residual, residual_1, residual_2), axis=1)
    selected_stage = np.argmin(candidates, axis=1)
    selected = np.zeros_like(stage_1)
    selected[selected_stage == 1] = stage_1[selected_stage == 1]
    selected[selected_stage == 2] = stage_2[selected_stage == 2]
    selected_residual = np.min(candidates, axis=1)
    accept_stage_2 = residual_2 <= residual_1

    localization_variation = (
        {
            "definition": (
                "within-beat standard deviation of the two x-sorted vessel "
                "centers, in normalized mesh coordinates"
            ),
            "stage_1": _temporal_center_stability(
                stage_1_parameters[:, :, :2], period_id
            ),
            "stage_2": _temporal_center_stability(
                stage_2_parameters[:, :, :2], period_id
            ),
        }
        if parameter_architecture and period_id is not None
        else None
    )

    physics = {
        "initial_voltage_rms": _voltage_summary(initial_residual),
        "stage_1_voltage_residual_rms": _voltage_summary(residual_1),
        "stage_2_voltage_residual_rms": _voltage_summary(residual_2),
        "selected_voltage_residual_rms": _voltage_summary(selected_residual),
        "stage_2_acceptance_fraction": float(np.mean(accept_stage_2)),
        "zero_safe_selection_fraction": {
            "initial_zero": float(np.mean(selected_stage == 0)),
            "stage_1": float(np.mean(selected_stage == 1)),
            "stage_2": float(np.mean(selected_stage == 2)),
        },
        "stage_1_clipped_elements": int(clips_1),
        "stage_2_clipped_elements": int(clips_2),
        "stage_1_negative_absolute_mass_fraction": float(
            np.sum(np.clip(-stage_1, 0.0, None))
            / max(np.sum(np.abs(stage_1)), 1e-30)
        ),
        "stage_2_negative_absolute_mass_fraction": float(
            np.sum(np.clip(-stage_2, 0.0, None))
            / max(np.sum(np.abs(stage_2)), 1e-30)
        ),
    }
    if args.target_kind == "clean":
        report = {
            "method": "voltage-conditioned two-vessel localizer plus gated physics refiner",
            "target_kind": "clean synthetic conductivity ground truth",
            "samples": limit,
            "architecture": architecture,
            "baseline_mode": baseline_mode,
            "metrics": {
                "stage_1": _metrics(stage_1, display_reference, runtime["mappings"]),
                "stage_2": _metrics(stage_2, display_reference, runtime["mappings"]),
                "voltage_rms_selected": _metrics(
                    selected, display_reference, runtime["mappings"]
                ),
            },
            "physics": physics,
            "vessel_localization_variation": localization_variation,
        }
    else:
        report = {
            "method": "voltage-conditioned two-vessel localizer plus gated physics refiner",
            "target_kind": "real subject 006",
            "samples": limit,
            "architecture": architecture,
            "baseline_mode": baseline_mode,
            "metrics_policy": (
                "No image correlation, Dice, or background RMS is computed against "
                "the PVI reconstruction because it is not anatomical ground truth."
            ),
            "primary_metric": "nonlinear differential voltage residual RMS",
            "pvi_reconstruction_role": "display-only pseudo-reference",
            "physics": physics,
            "vessel_localization_variation": localization_variation,
            # Compatibility alias for older analysis scripts.
            "temporal_center_stability": (
                None
                if localization_variation is None
                else {
                    "stage_1": localization_variation["stage_1"],
                    "stage_2": localization_variation["stage_2"],
                }
            ),
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.out_dir / "predictions.npz",
        truth=display_reference.astype(np.float32),
        old_newton=old_newton.astype(np.float32),
        lm_stages=np.empty((0, limit, display_reference.shape[1]), dtype=np.float32),
        gcnm_directions=np.stack((direction_1, direction_2)).astype(np.float32),
        gcnm_stages=np.stack((stage_1, stage_2)).astype(np.float32),
        selected=selected.astype(np.float32),
        selected_stage=selected_stage.astype(np.int8),
        voltage_residuals_gcnm=np.stack((residual_1, residual_2)).astype(np.float32),
        voltage_residuals_selected=selected_residual.astype(np.float32),
        stage_2_accepted=accept_stage_2,
        vessel_parameters_stage_1=stage_1_parameters.astype(np.float32),
        vessel_parameters_stage_2=stage_2_parameters.astype(np.float32),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
