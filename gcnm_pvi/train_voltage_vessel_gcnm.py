#!/usr/bin/env python3
"""Train a voltage-conditioned two-vessel localizer plus physics refiner."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gcnm_pvi.anatomical_phantoms import element_positions
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.core_guided_physics import build_fixed_linear_operator
from gcnm_pvi.generalized_vessel_model import (
    BeatMeanPoolVesselLocalizer,
    BeatSpatialDiffusionLocalizer,
    BeatVesselParameterRefiner,
)
from gcnm_pvi.iterative_physics import (
    LowRankRegularizedSolver,
    diagnostics_summary,
    parallel_dataset_lm_directions,
)
from gcnm_pvi.runtime import build_runtime
from gcnm_pvi.voltage_vessel_model import (
    SpatialAttentionVesselLocalizer,
    SpatialAttentionVesselDiffusionLocalizer,
    SpatialVesselDiffusionRefiner,
    SpatialVesselParameterRefiner,
    VoltageConditionedPhysicsRefiner,
    VoltageConditionedVesselLocalizer,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_arrays(path: Path, limit: int | None = None) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        required = ("sigma", "sigma_baseline", "V")
        missing = [key for key in required if key not in source]
        if missing:
            raise KeyError(f"{path} is missing {missing}")
        arrays = {key: np.asarray(source[key], dtype=np.float64) for key in required}
        for key in ("V_clean", "V_template"):
            if key in source:
                arrays[key] = np.asarray(source[key], dtype=np.float64)
        for key in ("beat_id", "phase_index"):
            if key in source:
                arrays[key] = np.asarray(source[key])
        if "tissue_labels" in source:
            arrays["tissue_labels"] = np.asarray(source["tissue_labels"], dtype=np.uint8)
    if limit is not None:
        arrays = {key: value[:limit] for key, value in arrays.items()}
    return arrays


def _anatomy_targets(
    path: Path,
    scale: float,
    limit: int | None,
    *,
    include_diffusion: bool = False,
) -> np.ndarray:
    records = json.loads(path.read_text(encoding="utf-8"))
    if limit is not None:
        records = records[:limit]
    output = []
    for record in records:
        vessels = record["vessels"]
        amplitudes = record["vessel_delta"]
        if len(vessels) != 2:
            raise ValueError(f"{path} contains a sample without exactly two vessels")
        rotation = float(record["rotation"])
        cosine, sine = math.cos(rotation), math.sin(rotation)
        sample = []
        simulator_model = record.get("simulator_finger_model", {})
        simulator_arteries = simulator_model.get("arteries", [])
        diffusion_fraction = float(
            simulator_model.get("muscle_diffusion_fraction", 0.0)
        )
        diffusion_length_mm = float(
            simulator_model.get("muscle_diffusion_length_mm", 1.0)
        )
        for vessel_index, (vessel, amplitude) in enumerate(zip(vessels, amplitudes)):
            center_x = cosine * vessel["center_x"] - sine * vessel["center_y"]
            center_y = sine * vessel["center_x"] + cosine * vessel["center_y"]
            angle = float(vessel["angle"]) + rotation
            angle = (angle + math.pi) % (2.0 * math.pi) - math.pi
            parameters = [
                center_x,
                center_y,
                float(vessel["axis_a"]),
                float(vessel["axis_b"]),
                angle,
                float(amplitude) / scale,
            ]
            if include_diffusion:
                if vessel_index >= len(simulator_arteries):
                    raise ValueError(
                        f"{path} record lacks simulator artery {vessel_index}"
                    )
                artery = simulator_arteries[vessel_index]
                lumen_radius_mm = (
                    math.sqrt(
                        float(artery["radius_x_mm"])
                        * float(artery["radius_y_mm"])
                    )
                    * float(artery.get("lumen_fraction", 1.0))
                )
                lumen_radius_normalized = math.sqrt(
                    float(vessel["axis_a"]) * float(vessel["axis_b"])
                )
                normalized_per_mm = lumen_radius_normalized / max(
                    lumen_radius_mm, 1e-8
                )
                parameters.extend(
                    [
                        diffusion_fraction,
                        diffusion_length_mm * normalized_per_mm,
                    ]
                )
            sample.append(parameters)
        output.append(sample)
    return np.asarray(output, dtype=np.float32)


def _graphs(
    truth: np.ndarray,
    current: np.ndarray,
    direction: np.ndarray,
    voltage: np.ndarray,
    vessel_parameters: np.ndarray,
    positions: np.ndarray,
    edge_index,
    *,
    conductivity_scale: float,
    voltage_scale: float,
    initial_parameters: np.ndarray | None = None,
    tissue_labels: np.ndarray | None = None,
    voltage_template: np.ndarray | None = None,
    context_direction: np.ndarray | None = None,
    voltage_clean: np.ndarray | None = None,
    beat_id: np.ndarray | None = None,
    voltage_input_mode: str = "global_scale",
    voltage_rms_reference: float | None = None,
) -> list[Data]:
    graphs = []
    for index in range(len(truth)):
        target = truth[index] / conductivity_scale
        if voltage_input_mode == "beat_normalized":
            template = (
                voltage[index]
                if voltage_template is None
                else voltage_template[index]
            )
            phase_rms = max(float(np.sqrt(np.mean(voltage[index] ** 2))), 1e-12)
            template_rms = max(float(np.sqrt(np.mean(template**2))), 1e-12)
            reference = max(float(voltage_rms_reference or 1.0), 1e-12)
            normalized_voltage = np.clip(voltage[index] / phase_rms, -8.0, 8.0)
            normalized_template = np.clip(template / template_rms, -8.0, 8.0)
            log_rms = float(np.clip(np.log(phase_rms / reference), -5.0, 5.0))
            context = (
                direction[index]
                if context_direction is None
                else context_direction[index]
            )
            context_scale = max(
                float(np.percentile(np.abs(context), 95.0)), 1e-12
            )
            normalized_context = np.clip(np.abs(context) / context_scale, 0.0, 5.0)
            features = np.column_stack(
                (
                    current[index] / conductivity_scale,
                    direction[index] / conductivity_scale,
                    normalized_context,
                    positions[:, 0],
                    positions[:, 1],
                )
            )
            geometry_features = np.column_stack(
                (normalized_context, positions[:, 0], positions[:, 1])
            )
        else:
            normalized_voltage = voltage[index] / voltage_scale
            normalized_template = normalized_voltage
            log_rms = 0.0
            features = np.column_stack(
                (
                    current[index] / conductivity_scale,
                    direction[index] / conductivity_scale,
                    positions[:, 0],
                    positions[:, 1],
                )
            )
            geometry_features = np.column_stack(
                (np.abs(direction[index]) / conductivity_scale, positions)
            )
        graph = Data(
            x=torch.tensor(features, dtype=torch.float32),
            geometry_x=torch.tensor(geometry_features, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(target[:, None], dtype=torch.float32),
            voltage=torch.tensor(
                normalized_voltage[None, :], dtype=torch.float32
            ),
            voltage_template=torch.tensor(
                normalized_template[None, :], dtype=torch.float32
            ),
            voltage_log_rms=torch.tensor([[log_rms]], dtype=torch.float32),
            voltage_clean=torch.tensor(
                (
                    voltage[index]
                    if voltage_clean is None
                    else voltage_clean[index]
                )[None, :],
                dtype=torch.float32,
            ),
            beat_id=torch.tensor(
                [index if beat_id is None else int(beat_id[index])],
                dtype=torch.long,
            ),
            vessel_parameters=torch.tensor(
                vessel_parameters[index][None, :, :], dtype=torch.float32
            ),
            muscle_gate=torch.tensor(
                (
                    (tissue_labels[index] == 3).astype(np.float32)
                    if tissue_labels is not None
                    else np.ones(len(target), dtype=np.float32)
                )[:, None],
                dtype=torch.float32,
            ),
        )
        if initial_parameters is not None:
            graph.initial_parameters = torch.tensor(
                initial_parameters[index][None, :, :], dtype=torch.float32
            )
        graphs.append(graph)
    return graphs


def _support_dice_loss(prediction: torch.Tensor, batch) -> torch.Tensor:
    losses = []
    for start, stop in zip(batch.ptr[:-1], batch.ptr[1:]):
        pred = prediction[start:stop, 0]
        target = batch.y[start:stop, 0]
        support = torch.abs(target) > 1e-8
        if not torch.any(support):
            losses.append(torch.mean(torch.abs(pred)))
            continue
        reference = torch.abs(target[support]).median().detach().clamp_min(1e-4)
        probability = torch.sigmoid(
            (torch.abs(pred) - 0.25 * reference) / (0.1 * reference)
        )
        truth = support.to(pred.dtype)
        intersection = torch.sum(probability * truth)
        losses.append(
            1.0
            - (2.0 * intersection + 1e-6)
            / (torch.sum(probability) + torch.sum(truth) + 1e-6)
        )
    return torch.stack(losses).mean()


def _correlation_loss(prediction: torch.Tensor, batch) -> torch.Tensor:
    """One minus Pearson correlation, averaged independently per graph."""
    losses = []
    for start, stop in zip(batch.ptr[:-1], batch.ptr[1:]):
        pred = prediction[start:stop, 0]
        target = batch.y[start:stop, 0]
        pred_centered = pred - torch.mean(pred)
        target_centered = target - torch.mean(target)
        if torch.linalg.vector_norm(target_centered) < 1e-8:
            losses.append(pred.new_tensor(0.0))
            continue
        denominator = (
            torch.linalg.vector_norm(pred_centered)
            * torch.linalg.vector_norm(target_centered)
        ).clamp_min(1e-8)
        correlation = torch.sum(pred_centered * target_centered) / denominator
        losses.append(1.0 - correlation)
    return torch.stack(losses).mean()


def _pair_cost(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    center = torch.mean((prediction[:, 0:2] - target[:, 0:2]) ** 2, dim=1)
    axes = torch.mean((prediction[:, 2:4] - target[:, 2:4]) ** 2, dim=1)
    # Ellipse orientation is pi-periodic, hence cos(2 delta-angle).
    angle = 1.0 - torch.cos(2.0 * (prediction[:, 4] - target[:, 4]))
    amplitude = (prediction[:, 5] - target[:, 5]) ** 2
    cost = 4.0 * center + 2.0 * axes + 0.1 * angle + amplitude
    if prediction.shape[1] >= 8 and target.shape[1] >= 8:
        diffusion_fraction = (prediction[:, 6] - target[:, 6]) ** 2
        diffusion_length = (prediction[:, 7] - target[:, 7]) ** 2
        cost = cost + 0.5 * diffusion_fraction + 2.0 * diffusion_length
    return cost


def _slot_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    direct = _pair_cost(prediction[:, 0], target[:, 0]) + _pair_cost(
        prediction[:, 1], target[:, 1]
    )
    swapped = _pair_cost(prediction[:, 0], target[:, 1]) + _pair_cost(
        prediction[:, 1], target[:, 0]
    )
    return torch.minimum(direct, swapped).mean()


def _loss(
    model,
    batch,
    *,
    localizer: bool,
    positive_weight: float,
    background_weight: float,
    dice_weight: float,
    slot_weight: float,
    separation_weight: float,
    attention_weight: float,
    minimum_center_separation: float,
    correlation_weight: float = 0.0,
    physics_weight: float = 0.0,
    physics_jacobian: torch.Tensor | None = None,
    conductivity_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if localizer:
        output = model(batch, return_parameters=True)
        prediction, parameters, _masks = output[:3]
        attention = output[3] if len(output) > 3 else None
    else:
        prediction = model(batch)
        parameters = None
        attention = None
    support = torch.abs(batch.y) > 1e-8
    weights = 1.0 + float(positive_weight) * support.to(batch.y.dtype)
    map_loss = torch.mean(weights * (prediction - batch.y) ** 2)
    background_loss = torch.mean(prediction[~support] ** 2)
    dice_loss = _support_dice_loss(prediction, batch)
    correlation_loss = _correlation_loss(prediction, batch)
    total = (
        map_loss
        + float(background_weight) * background_loss
        + float(dice_weight) * dice_loss
        + float(correlation_weight) * correlation_loss
    )
    slot_loss = prediction.new_tensor(0.0)
    separation_loss = prediction.new_tensor(0.0)
    attention_loss = prediction.new_tensor(0.0)
    physics_loss = prediction.new_tensor(0.0)
    if parameters is not None:
        parameter_count = int(parameters.shape[-1])
        target_parameters = batch.vessel_parameters.reshape(
            -1, 2, parameter_count
        )
        slot_loss = _slot_loss(parameters, target_parameters)
        center_distance = torch.linalg.vector_norm(
            parameters[:, 0, 0:2] - parameters[:, 1, 0:2], dim=1
        )
        separation_loss = torch.mean(
            torch.relu(float(minimum_center_separation) - center_distance) ** 2
        )
        total = (
            total
            + float(slot_weight) * slot_loss
            + float(separation_weight) * separation_loss
        )
    if attention is not None:
        similarities = []
        for start, stop in zip(batch.ptr[:-1], batch.ptr[1:]):
            first = attention[start:stop, 0]
            second = attention[start:stop, 1]
            similarity = torch.sum(first * second) / (
                torch.linalg.vector_norm(first)
                * torch.linalg.vector_norm(second)
                + 1e-8
            )
            similarities.append(similarity)
        attention_loss = torch.stack(similarities).mean()
        total = total + float(attention_weight) * attention_loss
    if float(physics_weight) > 0.0:
        if physics_jacobian is None:
            raise ValueError("physics_weight requires a fixed Jacobian")
        losses = []
        targets = batch.voltage_clean.reshape(-1, physics_jacobian.shape[0])
        for graph_index, (start, stop) in enumerate(zip(batch.ptr[:-1], batch.ptr[1:])):
            sigma = prediction[start:stop, 0] * float(conductivity_scale)
            predicted_voltage = physics_jacobian @ sigma
            target_voltage = targets[graph_index]
            scale = torch.sqrt(torch.mean(target_voltage**2)).clamp_min(1e-8)
            losses.append(torch.mean(((predicted_voltage - target_voltage) / scale) ** 2))
        physics_loss = torch.stack(losses).mean()
        total = total + float(physics_weight) * physics_loss
    return total, {
        "map": float(map_loss.detach()),
        "background": float(background_loss.detach()),
        "dice": float(dice_loss.detach()),
        "correlation": float(correlation_loss.detach()),
        "slot": float(slot_loss.detach()),
        "separation": float(separation_loss.detach()),
        "attention": float(attention_loss.detach()),
        "physics": float(physics_loss.detach()),
    }


def _epoch(model, loader, optimizer=None, **loss_kwargs) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    totals: dict[str, float] = {}
    total_loss, graphs = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        if training:
            optimizer.zero_grad()
        loss, parts = _loss(model, batch, **loss_kwargs)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        count = int(batch.num_graphs)
        total_loss += float(loss.detach()) * count
        graphs += count
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value * count
    return total_loss / max(graphs, 1), {
        key: value / max(graphs, 1) for key, value in totals.items()
    }


def _fit(
    model,
    train_graphs,
    validation_graphs,
    cfg,
    *,
    epochs: int,
    label: str,
    loss_kwargs: dict,
) -> tuple[torch.nn.Module, list[dict], int, float]:
    train_loader = DataLoader(train_graphs, batch_size=cfg.batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_graphs, batch_size=cfg.batch_size, shuffle=False
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss, best_epoch, patience = float("inf"), 0, cfg.patience
    history = []
    for epoch in range(epochs):
        train_loss, train_parts = _epoch(
            model, train_loader, optimizer=optimizer, **loss_kwargs
        )
        with torch.no_grad():
            validation_loss, validation_parts = _epoch(
                model, validation_loader, **loss_kwargs
            )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_components": train_parts,
            "validation_components": validation_parts,
        }
        history.append(record)
        print(
            f"{label} epoch {epoch}: train={train_loss:.6e} "
            f"validation={validation_loss:.6e}",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = cfg.patience
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_loss


def _predict(model, graphs, scale: float) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    with torch.no_grad():
        for graph in graphs:
            predictions.append(model(graph.to(device)).cpu().numpy().ravel() * scale)
    return np.stack(predictions)


def _predict_with_parameters(model, graphs, scale: float) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    device = next(model.parameters()).device
    predictions, parameters = [], []
    with torch.no_grad():
        for graph in graphs:
            output = model(graph.to(device), return_parameters=True)
            predictions.append(output[0].cpu().numpy().ravel() * scale)
            parameters.append(output[1].cpu().numpy()[0])
    return np.stack(predictions), np.stack(parameters)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--train-anatomy", type=Path, required=True)
    parser.add_argument("--validation-anatomy", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--baseline-conductivity", type=float, default=0.7)
    parser.add_argument(
        "--baseline-mode",
        choices=["saved", "homogeneous"],
        default="homogeneous",
    )
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--background-weight", type=float, default=0.25)
    parser.add_argument("--dice-weight", type=float, default=0.05)
    parser.add_argument("--slot-weight", type=float, default=0.2)
    parser.add_argument("--separation-weight", type=float, default=0.1)
    parser.add_argument("--attention-weight", type=float, default=0.05)
    parser.add_argument("--correlation-weight", type=float, default=0.0)
    parser.add_argument("--physics-weight", type=float, default=0.0)
    parser.add_argument("--minimum-center-separation", type=float, default=0.25)
    parser.add_argument("--minimum-vessel-axis", type=float, default=0.05)
    parser.add_argument("--maximum-vessel-axis", type=float, default=0.23)
    parser.add_argument(
        "--conductivity-scale-mode",
        choices=["global_p995", "positive_p995", "maximum"],
        default="global_p995",
    )
    parser.add_argument(
        "--architecture",
        choices=[
            "mean_pool_dense_refiner",
            "spatial_slots",
            "diffusion_slots",
            "beat_voltage_slots",
            "beat_diffusion_slots",
        ],
        default="mean_pool_dense_refiner",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        args.models_dir / f"{args.model_name}_localizer.pt",
        args.models_dir / f"{args.model_name}_refiner.pt",
    ]
    report_path = args.results_dir / f"{args.model_name}_training_report.json"
    if not args.allow_overwrite and any(path.exists() for path in paths + [report_path]):
        raise FileExistsError("experiment outputs already exist; choose a new model name")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = GcnmConfig.from_yaml(args.config)
    runtime = build_runtime(cfg, include_forward=False)
    train = _load_arrays(args.train, args.max_train)
    validation = _load_arrays(args.validation, args.max_validation)
    magnitude = np.abs(train["sigma"])
    if args.conductivity_scale_mode == "positive_p995":
        positive = magnitude[magnitude > 1e-8]
        conductivity_scale = max(float(np.percentile(positive, 99.5)), 1e-6)
    elif args.conductivity_scale_mode == "maximum":
        conductivity_scale = max(float(np.max(magnitude)), 1e-6)
    else:
        conductivity_scale = max(float(np.percentile(magnitude, 99.5)), 1e-6)
    voltage_scale = max(float(np.percentile(np.abs(train["V"]), 99.5)), 1e-12)
    generalized = args.architecture in {
        "beat_voltage_slots",
        "beat_diffusion_slots",
    }
    diffusion_architecture = args.architecture in {
        "diffusion_slots",
        "beat_diffusion_slots",
    }
    parameter_architecture = args.architecture in {
        "spatial_slots",
        "diffusion_slots",
        "beat_voltage_slots",
        "beat_diffusion_slots",
    }
    if generalized and "V_template" not in train:
        raise KeyError("beat-normalized training requires V_template in the dataset")
    if generalized and args.baseline_mode != "homogeneous":
        raise ValueError("generalized beat models require the homogeneous inference baseline")
    train_voltage_rms = np.sqrt(np.mean(train["V"] ** 2, axis=1))
    voltage_rms_reference = max(float(np.median(train_voltage_rms)), 1e-12)
    train_parameters = _anatomy_targets(
        args.train_anatomy,
        conductivity_scale,
        args.max_train,
        include_diffusion=diffusion_architecture,
    )
    validation_parameters = _anatomy_targets(
        args.validation_anatomy,
        conductivity_scale,
        args.max_validation,
        include_diffusion=diffusion_architecture,
    )
    if len(train_parameters) != len(train["sigma"]):
        raise ValueError("training anatomy and array counts differ")
    if len(validation_parameters) != len(validation["sigma"]):
        raise ValueError("validation anatomy and array counts differ")

    if args.baseline_mode == "saved":
        baseline_train = train["sigma_baseline"].copy()
        baseline_validation = validation["sigma_baseline"].copy()
    else:
        baseline_train = np.full_like(train["sigma"], args.baseline_conductivity)
        baseline_validation = np.full_like(
            validation["sigma"], args.baseline_conductivity
        )
    positions = element_positions(runtime["mesh_inv"]).astype(np.float32)
    zeros_train = np.zeros_like(train["sigma"])
    zeros_validation = np.zeros_like(validation["sigma"])
    start = time.time()
    fixed = None
    context_train = None
    context_validation = None
    if generalized:
        fixed = build_fixed_linear_operator(
            runtime["physics_inv"],
            regularizer=runtime["mappings"].laplace,
            hyper_pvi=cfg.hyper_pvi,
            lambda_lm=cfg.lambda_lm,
            baseline=baseline_train[0],
        )
        direction_train = fixed.directions(train["V"])
        direction_validation = fixed.directions(validation["V"])
        context_train = fixed.directions(train["V_template"])
        context_validation = fixed.directions(
            validation.get("V_template", validation["V"])
        )
        base_voltage_train = np.broadcast_to(
            fixed.baseline_voltage[None, :], train["V"].shape
        )
        base_voltage_validation = np.broadcast_to(
            fixed.baseline_voltage[None, :], validation["V"].shape
        )
        diagnostics_train_1 = None
        diagnostics_validation_1 = None
        stage1_physics = {
            "operator": "fixed homogeneous LM direction",
            "train": {
                "samples": len(train["V"]),
                "voltage_residual_rms_mean": float(
                    np.mean(np.sqrt(np.mean(train["V"] ** 2, axis=1)))
                ),
                "step_rms_mean": float(np.mean(np.sqrt(np.mean(direction_train**2, axis=1)))),
            },
            "validation": {
                "samples": len(validation["V"]),
                "voltage_residual_rms_mean": float(
                    np.mean(np.sqrt(np.mean(validation["V"] ** 2, axis=1)))
                ),
                "step_rms_mean": float(
                    np.mean(np.sqrt(np.mean(direction_validation**2, axis=1)))
                ),
            },
        }
    else:
        direction_train, diagnostics_train_1, base_voltage_train = parallel_dataset_lm_directions(
            runtime["physics_inv"],
            baseline_train,
            zeros_train,
            train["V"],
            regularizer=runtime["mappings"].laplace,
            hyper_pvi=cfg.hyper_pvi,
            lambda_lm=cfg.lambda_lm,
            step_size=cfg.lm_step_size,
            baseline_voltages=None,
            progress_label="vessel localizer train",
        )
        direction_validation, diagnostics_validation_1, base_voltage_validation = (
            parallel_dataset_lm_directions(
                runtime["physics_inv"],
                baseline_validation,
                zeros_validation,
                validation["V"],
                regularizer=runtime["mappings"].laplace,
                hyper_pvi=cfg.hyper_pvi,
                lambda_lm=cfg.lambda_lm,
                step_size=cfg.lm_step_size,
                baseline_voltages=None,
                progress_label="vessel localizer validation",
            )
        )
        stage1_physics = {
            "train": diagnostics_summary(diagnostics_train_1),
            "validation": diagnostics_summary(diagnostics_validation_1),
        }
    train_graphs_1 = _graphs(
        train["sigma"], zeros_train, direction_train, train["V"], train_parameters,
        positions, runtime["edge_index"], conductivity_scale=conductivity_scale,
        voltage_scale=voltage_scale,
        tissue_labels=train.get("tissue_labels"),
        voltage_template=train.get("V_template"),
        context_direction=context_train,
        voltage_clean=train.get("V_clean"),
        beat_id=train.get("beat_id"),
        voltage_input_mode="beat_normalized" if generalized else "global_scale",
        voltage_rms_reference=voltage_rms_reference,
    )
    validation_graphs_1 = _graphs(
        validation["sigma"], zeros_validation, direction_validation,
        validation["V"], validation_parameters, positions, runtime["edge_index"],
        conductivity_scale=conductivity_scale, voltage_scale=voltage_scale,
        tissue_labels=validation.get("tissue_labels"),
        voltage_template=validation.get("V_template"),
        context_direction=context_validation,
        voltage_clean=validation.get("V_clean"),
        beat_id=validation.get("beat_id"),
        voltage_input_mode="beat_normalized" if generalized else "global_scale",
        voltage_rms_reference=voltage_rms_reference,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs or cfg.max_epochs
    common_loss = {
        "positive_weight": args.positive_weight,
        "background_weight": args.background_weight,
        "dice_weight": args.dice_weight,
        "slot_weight": args.slot_weight,
        "separation_weight": args.separation_weight,
        "attention_weight": args.attention_weight,
        "correlation_weight": args.correlation_weight,
        "minimum_center_separation": args.minimum_center_separation,
        "physics_weight": args.physics_weight,
        "conductivity_scale": conductivity_scale,
    }
    physics_jacobian = None
    if args.physics_weight > 0.0:
        if fixed is None:
            fixed = build_fixed_linear_operator(
                runtime["physics_inv"],
                baseline_train[0],
                regularizer=runtime["mappings"].laplace,
                hyper_pvi=cfg.hyper_pvi,
                lambda_lm=cfg.lambda_lm,
            )
        physics_jacobian = torch.tensor(
            fixed.jacobian, dtype=torch.float32, device=device
        )
    runtime_loss = {**common_loss, "physics_jacobian": physics_jacobian}
    localizer = (
        BeatSpatialDiffusionLocalizer(
            minimum_axis=args.minimum_vessel_axis,
            maximum_axis=args.maximum_vessel_axis,
        )
        if args.architecture == "beat_diffusion_slots"
        else BeatMeanPoolVesselLocalizer(
            minimum_axis=args.minimum_vessel_axis,
            maximum_axis=args.maximum_vessel_axis,
        )
        if args.architecture == "beat_voltage_slots"
        else SpatialAttentionVesselDiffusionLocalizer(
            minimum_axis=args.minimum_vessel_axis,
            maximum_axis=args.maximum_vessel_axis,
        )
        if args.architecture == "diffusion_slots"
        else SpatialAttentionVesselLocalizer(
            minimum_axis=args.minimum_vessel_axis,
            maximum_axis=args.maximum_vessel_axis,
        )
        if args.architecture == "spatial_slots"
        else VoltageConditionedVesselLocalizer(
            minimum_axis=args.minimum_vessel_axis,
            maximum_axis=args.maximum_vessel_axis,
        )
    ).to(device)
    localizer, history_1, best_epoch_1, best_loss_1 = _fit(
        localizer,
        train_graphs_1,
        validation_graphs_1,
        cfg,
        epochs=epochs,
        label="stage 1 localizer",
        loss_kwargs={"localizer": True, **runtime_loss},
    )
    if parameter_architecture:
        stage1_train, stage1_parameters_train = _predict_with_parameters(
            localizer, train_graphs_1, conductivity_scale
        )
        stage1_validation, stage1_parameters_validation = _predict_with_parameters(
            localizer, validation_graphs_1, conductivity_scale
        )
    else:
        stage1_train = _predict(localizer, train_graphs_1, conductivity_scale)
        stage1_validation = _predict(
            localizer, validation_graphs_1, conductivity_scale
        )
        stage1_parameters_train = None
        stage1_parameters_validation = None

    stage_2_solver = LowRankRegularizedSolver(
        runtime["mappings"].laplace,
        runtime["mappings"].num_elements,
        hyper_pvi=cfg.hyper_pvi,
        lambda_lm=cfg.lambda_lm,
    )
    direction_train_2, diagnostics_train_2, _ = parallel_dataset_lm_directions(
        runtime["physics_inv"], baseline_train, stage1_train, train["V"],
        regularizer=runtime["mappings"].laplace, hyper_pvi=cfg.hyper_pvi,
        lambda_lm=cfg.lambda_lm, step_size=cfg.lm_step_size,
        baseline_voltages=base_voltage_train, progress_label="physics refiner train",
        system_solver=stage_2_solver,
    )
    direction_validation_2, diagnostics_validation_2, _ = parallel_dataset_lm_directions(
        runtime["physics_inv"], baseline_validation, stage1_validation,
        validation["V"], regularizer=runtime["mappings"].laplace,
        hyper_pvi=cfg.hyper_pvi, lambda_lm=cfg.lambda_lm,
        step_size=cfg.lm_step_size, baseline_voltages=base_voltage_validation,
        progress_label="physics refiner validation", system_solver=stage_2_solver,
    )
    train_graphs_2 = _graphs(
        train["sigma"], stage1_train, direction_train_2, train["V"], train_parameters,
        positions, runtime["edge_index"], conductivity_scale=conductivity_scale,
        voltage_scale=voltage_scale, initial_parameters=stage1_parameters_train,
        tissue_labels=train.get("tissue_labels"),
        voltage_template=train.get("V_template"),
        context_direction=context_train,
        voltage_clean=train.get("V_clean"),
        beat_id=train.get("beat_id"),
        voltage_input_mode="beat_normalized" if generalized else "global_scale",
        voltage_rms_reference=voltage_rms_reference,
    )
    validation_graphs_2 = _graphs(
        validation["sigma"], stage1_validation, direction_validation_2,
        validation["V"], validation_parameters, positions, runtime["edge_index"],
        conductivity_scale=conductivity_scale, voltage_scale=voltage_scale,
        initial_parameters=stage1_parameters_validation,
        tissue_labels=validation.get("tissue_labels"),
        voltage_template=validation.get("V_template"),
        context_direction=context_validation,
        voltage_clean=validation.get("V_clean"),
        beat_id=validation.get("beat_id"),
        voltage_input_mode="beat_normalized" if generalized else "global_scale",
        voltage_rms_reference=voltage_rms_reference,
    )
    refiner = (
        BeatVesselParameterRefiner(
            diffusion=args.architecture == "beat_diffusion_slots",
            minimum_axis=args.minimum_vessel_axis,
            maximum_axis=args.maximum_vessel_axis,
        )
        if generalized
        else SpatialVesselDiffusionRefiner(
            minimum_axis=args.minimum_vessel_axis,
            maximum_axis=args.maximum_vessel_axis,
        )
        if args.architecture == "diffusion_slots"
        else SpatialVesselParameterRefiner(
            minimum_axis=args.minimum_vessel_axis,
            maximum_axis=args.maximum_vessel_axis,
        )
        if args.architecture == "spatial_slots"
        else VoltageConditionedPhysicsRefiner()
    ).to(device)
    refiner, history_2, best_epoch_2, best_loss_2 = _fit(
        refiner,
        train_graphs_2,
        validation_graphs_2,
        cfg,
        epochs=epochs,
        label="stage 2 refiner",
        loss_kwargs={
            "localizer": parameter_architecture,
            **runtime_loss,
        },
    )

    contract = {
        "model_name": args.model_name,
        "measurements": int(train["V"].shape[1]),
        "elements": int(train["sigma"].shape[1]),
        "conductivity_scale": conductivity_scale,
        "conductivity_scale_mode": args.conductivity_scale_mode,
        "voltage_scale": voltage_scale,
        "voltage_input_mode": (
            "per_sample_unit_rms_plus_log_rms_and_beat_template"
            if generalized
            else "global_scale"
        ),
        "voltage_rms_reference": voltage_rms_reference,
        "beat_context_required": generalized,
        "signed_amplitude": generalized,
        "stage2_parameter_refiner": generalized,
        "stage2_raw_lm_added": False,
        "baseline_mode": args.baseline_mode,
        "baseline_conductivity": args.baseline_conductivity,
        "minimum_center_separation": args.minimum_center_separation,
        "minimum_vessel_axis": args.minimum_vessel_axis,
        "maximum_vessel_axis": args.maximum_vessel_axis,
        "architecture": args.architecture,
        "config_sha256": _sha256(args.config),
        "mesh_inverse_sha256": _sha256(Path(cfg.mesh_inv_h5)),
        "mappings_sha256": _sha256(Path(cfg.mappings_h5)),
        "train_dataset_sha256": _sha256(args.train),
        "validation_dataset_sha256": _sha256(args.validation),
    }
    torch.save(
        {
            "model_kind": (
                "beat_normalized_signed_diffusion_localizer"
                if args.architecture == "beat_diffusion_slots"
                else "beat_normalized_signed_vessel_localizer"
                if args.architecture == "beat_voltage_slots"
                else "spatial_attention_two_vessel_diffusion_localizer"
                if args.architecture == "diffusion_slots"
                else "spatial_attention_two_vessel_localizer"
                if args.architecture == "spatial_slots"
                else "voltage_conditioned_two_vessel_localizer"
            ),
            "state_dict": localizer.cpu().state_dict(),
            "contract": contract,
        },
        paths[0],
    )
    torch.save(
        {
            "model_kind": (
                "beat_normalized_signed_diffusion_parameter_refiner"
                if args.architecture == "beat_diffusion_slots"
                else "beat_normalized_signed_vessel_parameter_refiner"
                if args.architecture == "beat_voltage_slots"
                else "spatial_vessel_diffusion_refiner"
                if args.architecture == "diffusion_slots"
                else "spatial_vessel_parameter_refiner"
                if args.architecture == "spatial_slots"
                else "voltage_conditioned_physics_refiner"
            ),
            "state_dict": refiner.cpu().state_dict(),
            "contract": contract,
        },
        paths[1],
    )
    report = {
        "method": "voltage-conditioned two-vessel localizer plus gated physics refiner",
        "target": "clean synthetic vascular differential conductivity",
        "pvi_images_used_as_labels": False,
        "architecture": args.architecture,
        "train_samples": len(train["sigma"]),
        "validation_samples": len(validation["sigma"]),
        "contract": contract,
        "loss": common_loss,
        "stage_1": {
            "best_epoch": best_epoch_1,
            "best_validation_loss": best_loss_1,
            "physics": stage1_physics,
            "history": history_1,
        },
        "stage_2": {
            "best_epoch": best_epoch_2,
            "best_validation_loss": best_loss_2,
            "physics": {
                "train": diagnostics_summary(diagnostics_train_2),
                "validation": diagnostics_summary(diagnostics_validation_2),
            },
            "history": history_2,
        },
        "elapsed_seconds": time.time() - start,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
