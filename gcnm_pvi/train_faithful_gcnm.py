#!/usr/bin/env python3
"""Train faithful PVI-GCNM stages with newly recomputed physics per stage."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gcnm_pvi.anatomical_phantoms import element_positions
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_model import (
    GCNBlock,
    PhysicsProposalResidualGCNBlock,
    ShallowPhysicsResidualGCNBlock,
)
from gcnm_pvi.iterative_physics import dataset_lm_directions, diagnostics_summary
from gcnm_pvi.runtime import build_runtime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, limit: int | None = None) -> dict[str, np.ndarray]:
    source = np.load(path)
    required = ("sigma", "sigma_baseline", "V")
    missing = [key for key in required if key not in source]
    if missing:
        raise ValueError(f"{path} is missing required arrays: {missing}")
    values = {key: np.asarray(source[key], dtype=np.float64) for key in required}
    if limit is not None:
        values = {key: value[:limit] for key, value in values.items()}
    return values


def _make_dataset(
    truth: np.ndarray,
    current: np.ndarray,
    direction: np.ndarray,
    positions: np.ndarray,
    edge_index,
    *,
    scale: float,
    use_coordinates: bool,
    positive_weight: float,
) -> list[Data]:
    dataset: list[Data] = []
    for index in range(len(truth)):
        target = truth[index] / scale
        feature_columns = [current[index] / scale, direction[index] / scale]
        if use_coordinates:
            feature_columns.extend(positions.T)
        features = np.column_stack(feature_columns)
        magnitude = np.abs(target)
        nonzero = magnitude[magnitude > 0]
        reference = float(np.median(nonzero)) if nonzero.size else 1.0
        weights = 1.0 + positive_weight * np.clip(
            magnitude / max(reference, 1e-8), 0.0, 2.0
        )
        dataset.append(
            Data(
                edge_index=edge_index,
                x=torch.tensor(features, dtype=torch.float32),
                y=torch.tensor(target[:, None], dtype=torch.float32),
                weights=torch.tensor(weights[:, None], dtype=torch.float32),
                background=torch.tensor((magnitude <= 1e-8)[:, None]),
            )
        )
    return dataset


def _model(mode: str, channels: list[int], in_channels: int):
    if mode == "direct":
        return GCNBlock(channels, in_channels=in_channels).float()
    if mode == "proposal_residual":
        return PhysicsProposalResidualGCNBlock(
            channels,
            in_channels=in_channels,
        ).float()
    if mode == "shallow_residual":
        return ShallowPhysicsResidualGCNBlock(
            channels,
            in_channels=in_channels,
        ).float()
    raise ValueError(f"unknown output mode {mode}")


def _soft_support_dice_loss(prediction: torch.Tensor, batch) -> torch.Tensor:
    """Differentiable vessel-support Dice, averaged per graph.

    The adaptive threshold is derived only from each clean training target.
    This loss is never used as evidence on real subject data.
    """
    losses = []
    boundaries = batch.ptr if hasattr(batch, "ptr") else [0, len(prediction)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        pred = prediction[start:stop, 0]
        target = batch.y[start:stop, 0]
        support = torch.abs(target) > 1e-8
        if not torch.any(support):
            continue
        reference = torch.median(torch.abs(target[support])).detach().clamp_min(1e-6)
        probability = torch.sigmoid(
            (torch.relu(pred) - 0.25 * reference) / (0.10 * reference + 1e-6)
        )
        truth = support.to(probability.dtype)
        intersection = torch.sum(probability * truth)
        dice = (2.0 * intersection + 1e-6) / (
            torch.sum(probability) + torch.sum(truth) + 1e-6
        )
        losses.append(1.0 - dice)
    if not losses:
        return prediction.new_tensor(0.0)
    return torch.stack(losses).mean()


def _loss(
    model,
    batch,
    background_weight: float,
    dice_weight: float = 0.0,
    hard_background_weight: float = 0.0,
    hard_background_fraction: float = 0.05,
) -> torch.Tensor:
    prediction = model(batch)
    loss = torch.mean(batch.weights * (prediction - batch.y) ** 2)
    if background_weight > 0:
        mask = batch.background.bool()
        if torch.any(mask):
            loss = loss + float(background_weight) * torch.mean(prediction[mask] ** 2)
            if hard_background_weight > 0:
                background_error = prediction[mask] ** 2
                count = max(
                    1,
                    int(np.ceil(float(hard_background_fraction) * background_error.numel())),
                )
                hardest = torch.topk(background_error, k=count, largest=True).values
                loss = loss + float(hard_background_weight) * torch.mean(hardest)
    if dice_weight > 0:
        loss = loss + float(dice_weight) * _soft_support_dice_loss(prediction, batch)
    return loss


def _run_epoch(
    model,
    loader,
    background_weight: float,
    dice_weight: float = 0.0,
    hard_background_weight: float = 0.0,
    hard_background_fraction: float = 0.05,
    optimizer=None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total, graphs = 0.0, 0
    device = next(model.parameters()).device
    for batch in loader:
        batch = batch.to(device)
        if training:
            optimizer.zero_grad()
        loss = _loss(
            model,
            batch,
            background_weight,
            dice_weight,
            hard_background_weight,
            hard_background_fraction,
        )
        if training:
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * batch.num_graphs
        graphs += batch.num_graphs
    return total / max(graphs, 1)


def _checkpoint_score(
    model,
    loader,
    *,
    mode: str,
    validation_loss: float,
    background_coefficient: float,
    dice_coefficient: float,
) -> tuple[float, dict[str, float]]:
    if mode == "loss":
        return validation_loss, {"validation_loss": validation_loss}
    device = next(model.parameters()).device
    predictions, targets, backgrounds = [], [], []
    dice_values: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction_batch = model(batch).cpu().numpy().ravel()
            target_batch = batch.y.cpu().numpy().ravel()
            background_batch = batch.background.cpu().numpy().ravel().astype(bool)
            predictions.append(prediction_batch)
            targets.append(target_batch)
            backgrounds.append(background_batch)
            boundaries = batch.ptr.cpu().numpy()
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                support = np.flatnonzero(~background_batch[start:stop])
                if not len(support):
                    continue
                local_prediction = prediction_batch[start:stop]
                selected = np.argpartition(np.abs(local_prediction), -len(support))[
                    -len(support) :
                ]
                dice_values.append(
                    len(np.intersect1d(support, selected)) / len(support)
                )
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    background = np.concatenate(backgrounds)
    rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
    truth_rms = max(float(np.sqrt(np.mean(target**2))), 1e-12)
    nrmse = rmse / truth_rms
    bg_rms = float(np.sqrt(np.mean(prediction[background] ** 2)))
    dice = float(np.mean(dice_values)) if dice_values else 0.0
    score = nrmse + background_coefficient * bg_rms - dice_coefficient * dice
    return score, {
        "validation_loss": validation_loss,
        "nrmse": nrmse,
        "background_rms_normalized": bg_rms,
        "dice_global": dice,
        "composite_score": score,
    }


def _predict(model, dataset: list[Data], scale: float) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    output = []
    with torch.no_grad():
        for sample in dataset:
            output.append(model(sample.to(device)).squeeze().cpu().numpy() * scale)
    return np.stack(output)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "subject006_anatomical_gcnm.yaml",
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--output-mode",
        choices=["direct", "proposal_residual", "shallow_residual"],
        default="direct",
    )
    parser.add_argument("--use-coordinates", action="store_true")
    parser.add_argument("--positive-weight", type=float, default=0.0)
    parser.add_argument("--background-weight", type=float, default=0.0)
    parser.add_argument("--dice-weight", type=float, default=0.0)
    parser.add_argument("--hard-background-weight", type=float, default=0.0)
    parser.add_argument("--hard-background-fraction", type=float, default=0.05)
    parser.add_argument("--checkpoint-mode", choices=["loss", "composite"], default="loss")
    parser.add_argument("--checkpoint-background-coefficient", type=float, default=1.0)
    parser.add_argument("--checkpoint-dice-coefficient", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--minimum-conductivity", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--baseline-mode", choices=["saved", "homogeneous"], default="homogeneous"
    )
    parser.add_argument("--baseline-conductivity", type=float, default=0.7)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    cfg = GcnmConfig.from_yaml(args.config)
    seed = cfg.seed if args.seed is None else int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    runtime = build_runtime(cfg, include_forward=False)
    train = _load(args.train, args.max_train)
    validation = _load(args.validation, args.max_validation)
    if args.baseline_mode == "homogeneous":
        train["sigma_baseline"] = np.full_like(
            train["sigma_baseline"], args.baseline_conductivity
        )
        validation["sigma_baseline"] = np.full_like(
            validation["sigma_baseline"], args.baseline_conductivity
        )
    positions = element_positions(runtime["mesh_inv"]).astype(np.float32)
    scale = max(float(np.percentile(np.abs(train["sigma"]), 99.5)), 1e-6)
    iterations = args.iterations or cfg.iterations
    epochs = args.epochs or cfg.max_epochs
    in_channels = 5 if args.use_coordinates else 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    current_train = np.zeros_like(train["sigma"])
    current_validation = np.zeros_like(validation["sigma"])
    baseline_train = None
    baseline_validation = None
    existing_checkpoints = list(args.models_dir.glob(f"{args.model_name}_*.pt"))
    existing_report = args.results_dir / f"{args.model_name}_training_report.json"
    if not args.allow_overwrite and (existing_checkpoints or existing_report.exists()):
        raise FileExistsError(
            f"experiment {args.model_name} already has outputs; use a new EXPERIMENT "
            "name or pass --allow-overwrite explicitly"
        )
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    artifact_hashes = {
        "config_sha256": _sha256(args.config),
        "mesh_inverse_sha256": _sha256(Path(cfg.mesh_inv_h5)),
        "mappings_sha256": _sha256(Path(cfg.mappings_h5)),
        "train_dataset_sha256": _sha256(args.train),
        "validation_dataset_sha256": _sha256(args.validation),
    }

    history: list[list[dict]] = []
    physics_reports: list[dict] = []
    start_time = time.time()
    for iteration in range(iterations):
        current_train = (
            np.maximum(
                train["sigma_baseline"] + current_train,
                float(args.minimum_conductivity),
            )
            - train["sigma_baseline"]
        )
        current_validation = (
            np.maximum(
                validation["sigma_baseline"] + current_validation,
                float(args.minimum_conductivity),
            )
            - validation["sigma_baseline"]
        )
        direction_train, diag_train, baseline_train = dataset_lm_directions(
            runtime["physics_inv"],
            train["sigma_baseline"],
            current_train,
            train["V"],
            regularizer=runtime["mappings"].laplace,
            hyper_pvi=cfg.hyper_pvi,
            lambda_lm=cfg.lambda_lm,
            step_size=cfg.lm_step_size,
            minimum_conductivity=args.minimum_conductivity,
            baseline_voltages=baseline_train,
            progress_label=f"stage {iteration + 1} train",
        )
        direction_validation, diag_validation, baseline_validation = dataset_lm_directions(
            runtime["physics_inv"],
            validation["sigma_baseline"],
            current_validation,
            validation["V"],
            regularizer=runtime["mappings"].laplace,
            hyper_pvi=cfg.hyper_pvi,
            lambda_lm=cfg.lambda_lm,
            step_size=cfg.lm_step_size,
            minimum_conductivity=args.minimum_conductivity,
            baseline_voltages=baseline_validation,
            progress_label=f"stage {iteration + 1} validation",
        )
        physics_reports.append(
            {
                "stage": iteration + 1,
                "train": diagnostics_summary(diag_train),
                "validation": diagnostics_summary(diag_validation),
            }
        )
        train_set = _make_dataset(
            train["sigma"], current_train, direction_train, positions, runtime["edge_index"],
            scale=scale, use_coordinates=args.use_coordinates,
            positive_weight=args.positive_weight,
        )
        validation_set = _make_dataset(
            validation["sigma"], current_validation, direction_validation,
            positions, runtime["edge_index"], scale=scale,
            use_coordinates=args.use_coordinates, positive_weight=args.positive_weight,
        )
        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
        validation_loader = DataLoader(
            validation_set, batch_size=cfg.batch_size, shuffle=False
        )
        model = _model(args.output_mode, cfg.channels, in_channels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        best_state = copy.deepcopy(model.state_dict())
        best_score = float("inf")
        best_epoch = 0
        patience = cfg.patience
        iteration_history: list[dict] = []
        for epoch in range(epochs):
            train_loss = _run_epoch(
                model,
                train_loader,
                args.background_weight,
                args.dice_weight,
                args.hard_background_weight,
                args.hard_background_fraction,
                optimizer,
            )
            with torch.no_grad():
                validation_loss = _run_epoch(
                    model,
                    validation_loader,
                    args.background_weight,
                    args.dice_weight,
                    args.hard_background_weight,
                    args.hard_background_fraction,
                )
            score, score_metrics = _checkpoint_score(
                model,
                validation_loader,
                mode=args.checkpoint_mode,
                validation_loss=validation_loss,
                background_coefficient=args.checkpoint_background_coefficient,
                dice_coefficient=args.checkpoint_dice_coefficient,
            )
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                **score_metrics,
            }
            iteration_history.append(record)
            print(
                f"stage {iteration + 1} epoch {epoch}: train={train_loss:.6e} "
                f"validation={validation_loss:.6e} selection={score:.6e}",
                flush=True,
            )
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience = cfg.patience
            else:
                patience -= 1
                if patience <= 0:
                    break
        model.load_state_dict(best_state)
        current_train = _predict(model, train_set, scale)
        current_validation = _predict(model, validation_set, scale)
        feature_order = ["current", "per_stage_lm_direction"]
        if args.use_coordinates:
            feature_order.extend(["x", "y", "radius"])
        checkpoint = {
            "state_dict": model.cpu().state_dict(),
            "channels": cfg.channels,
            "in_channels": in_channels,
            "scale": scale,
            "iteration": iteration,
            "feature_order": feature_order,
            "output_mode": args.output_mode,
            "loss_contract": {
                "positive_weight": args.positive_weight,
                "background_weight": args.background_weight,
                "dice_weight": args.dice_weight,
                "hard_background_weight": args.hard_background_weight,
                "hard_background_fraction": args.hard_background_fraction,
                "dice_supervision": "clean synthetic targets only",
            },
            "physics": "nonlinear differential F/J/LM recomputed at every stage",
            "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "physics_contract": {
                "baseline_mode": args.baseline_mode,
                "baseline_conductivity": args.baseline_conductivity,
                "hyper_pvi": cfg.hyper_pvi,
                "lambda_lm": cfg.lambda_lm,
                "lm_step_size": cfg.lm_step_size,
                "minimum_conductivity": args.minimum_conductivity,
                "num_elements": int(train["sigma"].shape[1]),
                "num_measurements": int(train["V"].shape[1]),
                "regularizer": "projected_laplacian",
                "voltage_weighting": "none",
                **artifact_hashes,
            },
        }
        checkpoint_path = args.models_dir / f"{args.model_name}_{iteration}.pt"
        torch.save(checkpoint, checkpoint_path)
        history.append(iteration_history)
        print(f"saved {checkpoint_path}", flush=True)

    report = {
        "method": "faithful differential PVI-GCNM",
        "target": "clean synthetic vascular delta conductivity",
        "pvi_images_used_as_labels": False,
        "per_stage_physics_recomputed": True,
        "model_name": args.model_name,
        "train_samples": int(len(train["sigma"])),
        "validation_samples": int(len(validation["sigma"])),
        "iterations": iterations,
        "epochs_requested": epochs,
        "scale": scale,
        "output_mode": args.output_mode,
        "use_coordinates": args.use_coordinates,
        "positive_weight": args.positive_weight,
        "background_weight": args.background_weight,
        "dice_weight": args.dice_weight,
        "hard_background_weight": args.hard_background_weight,
        "hard_background_fraction": args.hard_background_fraction,
        "checkpoint_mode": args.checkpoint_mode,
        "random_seed": seed,
        "baseline_mode": args.baseline_mode,
        "baseline_conductivity": args.baseline_conductivity,
        "physics_stages": physics_reports,
        "artifact_hashes": artifact_hashes,
        "history": history,
        "elapsed_seconds": time.time() - start_time,
    }
    output = args.results_dir / f"{args.model_name}_training_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
