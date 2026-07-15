#!/usr/bin/env python3
"""Train residual GCNM stages from clean synthetic conductivity truth."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gcnm_pvi.anatomical_phantoms import element_positions
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_model import ResidualGCNBlock
from gcnm_pvi.runtime import build_runtime


def _load(path: Path, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    truth = np.asarray(data["sigma"], dtype=np.float64)
    newton = np.asarray(data["newton"], dtype=np.float64)
    if limit:
        truth, newton = truth[:limit], newton[:limit]
    return truth, newton


def _make_dataset(
    truth: np.ndarray,
    newton: np.ndarray,
    current: np.ndarray,
    positions: np.ndarray,
    edge_index,
    scale: float,
    positive_weight: float,
) -> list[Data]:
    dataset = []
    for index in range(len(truth)):
        target = truth[index] / scale
        update = newton[index] / scale
        state = current[index] / scale
        features = np.column_stack([state, update, positions])
        magnitude = np.abs(target)
        nonzero = magnitude[magnitude > 0]
        reference = float(np.median(nonzero)) if nonzero.size else 1.0
        weights = 1.0 + positive_weight * np.clip(magnitude / max(reference, 1e-8), 0.0, 2.0)
        dataset.append(
            Data(
                edge_index=edge_index,
                x=torch.tensor(features, dtype=torch.float32),
                y=torch.tensor(target[:, None], dtype=torch.float32),
                weights=torch.tensor(weights[:, None], dtype=torch.float32),
            )
        )
    return dataset


def _loss(model, batch) -> torch.Tensor:
    prediction = model(batch)
    return torch.mean(batch.weights * (prediction - batch.y) ** 2)


def _run_epoch(model, loader, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total, graphs = 0.0, 0
    device = next(model.parameters()).device
    for batch in loader:
        batch = batch.to(device)
        if training:
            optimizer.zero_grad()
        loss = _loss(model, batch)
        if training:
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * batch.num_graphs
        graphs += batch.num_graphs
    return total / max(graphs, 1)


def _predict(model, dataset, scale: float) -> np.ndarray:
    model.eval()
    output = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for sample in dataset:
            output.append(model(sample.to(device)).squeeze().cpu().numpy() * scale)
    return np.stack(output)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs" / "subject006_anatomical_gcnm.yaml")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--positive-weight", type=float, default=8.0)
    args = parser.parse_args()

    cfg = GcnmConfig.from_yaml(args.config)
    runtime = build_runtime(cfg, include_forward=False)
    train_truth, train_newton = _load(args.train, args.max_train)
    val_truth, val_newton = _load(args.validation, args.max_validation)
    positions = element_positions(runtime["mesh_inv"])
    scale = float(np.percentile(np.abs(train_truth), 99.5))
    scale = max(scale, 1e-6)
    iterations = args.iterations or cfg.iterations
    epochs = args.epochs or cfg.max_epochs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    edge_index = runtime["edge_index"]
    positions = positions.astype(np.float32)
    current_train = np.zeros_like(train_truth)
    current_val = np.zeros_like(val_truth)
    cfg_models = Path(cfg.models_dir)
    cfg_data = Path(cfg.data_dir)
    cfg_models.mkdir(parents=True, exist_ok=True)
    cfg_data.mkdir(parents=True, exist_ok=True)

    history = []
    start_time = time.time()
    for iteration in range(iterations):
        train_set = _make_dataset(
            train_truth,
            train_newton,
            current_train,
            positions,
            edge_index,
            scale,
            args.positive_weight,
        )
        val_set = _make_dataset(
            val_truth,
            val_newton,
            current_val,
            positions,
            edge_index,
            scale,
            args.positive_weight,
        )
        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
        model = ResidualGCNBlock(cfg.channels, in_channels=5).float().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        best_state = copy.deepcopy(model.state_dict())
        best_loss = float("inf")
        patience = cfg.patience
        iteration_history = []
        for epoch in range(epochs):
            train_loss = _run_epoch(model, train_loader, optimizer)
            with torch.no_grad():
                validation_loss = _run_epoch(model, val_loader)
            iteration_history.append([train_loss, validation_loss])
            print(
                f"iteration {iteration} epoch {epoch}: "
                f"train={train_loss:.6e} validation={validation_loss:.6e}",
                flush=True,
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                patience = cfg.patience
            else:
                patience -= 1
                if patience <= 0:
                    break
        model.load_state_dict(best_state)
        current_train = _predict(model, train_set, scale)
        current_val = _predict(model, val_set, scale)
        checkpoint = {
            "state_dict": model.cpu().state_dict(),
            "channels": cfg.channels,
            "in_channels": 5,
            "scale": scale,
            "iteration": iteration,
            "feature_order": ["current", "newton", "x", "y", "radius"],
        }
        path = cfg_models / f"{cfg.model_name}_{iteration}.pt"
        torch.save(checkpoint, path)
        history.append(iteration_history)
        print(f"saved {path}", flush=True)

    report = {
        "target": "clean synthetic vascular delta conductivity",
        "pvi_images_used_as_labels": False,
        "train_samples": int(len(train_truth)),
        "validation_samples": int(len(val_truth)),
        "iterations": iterations,
        "epochs_requested": epochs,
        "scale": scale,
        "positive_weight": args.positive_weight,
        "history": history,
        "elapsed_seconds": time.time() - start_time,
    }
    output = cfg_data / f"{cfg.model_name}_training_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
