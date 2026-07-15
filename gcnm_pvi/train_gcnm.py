#!/usr/bin/env python3
"""GCNM training on PVI 8-electrode meshes."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_model import (
    GCNBlock,
    applyModel,
    computeLMUpdates,
    initializeDataset,
    trainModel,
)
from gcnm_pvi.gcnm_phantoms import generate_dataset, load_dataset
from gcnm_pvi.runtime import build_runtime


def main():
    p = argparse.ArgumentParser(description="Train GCNM on PVI meshes")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "finger_pvi08.yaml")
    p.add_argument("--samples-dir", type=Path, default=None, help="Prebuilt NPZ dataset")
    p.add_argument("--validation-samples", type=Path, default=None)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-validation-samples", type=int, default=None)
    p.add_argument(
        "--fixed-production-update",
        action="store_true",
        help="Use the exact one-step PVI update as the second GCN feature",
    )
    args = p.parse_args()

    cfg = GcnmConfig.from_yaml(args.config)
    rt = build_runtime(cfg)
    physics_fwd = rt["physics_fwd"]
    physics_inv = rt["physics_inv"]
    edge_index = rt["edge_index"]

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    validation_count = None
    if args.samples_dir and Path(args.samples_dir).exists():
        sigma_array, V_array = load_dataset(args.samples_dir)
        if args.max_train_samples:
            sigma_array = sigma_array[: args.max_train_samples]
            V_array = V_array[: args.max_train_samples]
        if args.validation_samples:
            sigma_validation, V_validation = load_dataset(args.validation_samples)
            if args.max_validation_samples:
                sigma_validation = sigma_validation[: args.max_validation_samples]
                V_validation = V_validation[: args.max_validation_samples]
            validation_count = len(sigma_validation)
            sigma_array = np.concatenate([sigma_array, sigma_validation], axis=0)
            V_array = np.concatenate([V_array, V_validation], axis=0)
    else:
        rng = np.random.default_rng(cfg.seed)
        sigma_array, V_array = generate_dataset(
            physics_fwd,
            rt["mesh_inv"],
            cfg.n_samples,
            rng,
            mode=cfg.phantom_mode,
            noise_scale=cfg.noise_scale,
            out_dir=cfg.samples_dir,
        )

    differential = cfg.imaging_mode == "differential"
    if differential and not args.fixed_production_update and hasattr(physics_inv, "calibrate"):
        physics_inv.calibrate(V_array[0])

    N_samples, N_nodes = sigma_array.shape
    TRUTHS = torch.zeros((N_samples, N_nodes))
    PREDICTIONS = torch.zeros((1 + cfg.iterations, N_samples, N_nodes))
    UPDATES = torch.zeros((cfg.iterations, N_samples, N_nodes))
    LOSS_TR = torch.zeros((cfg.iterations, cfg.max_epochs))
    LOSS_VA = torch.zeros((cfg.iterations, cfg.max_epochs))

    dataset = initializeDataset(
        sigma_array,
        V_array,
        edge_index,
        initial_sigma=0.0 if differential else 1.0,
    )
    fixed_updates = None
    if args.fixed_production_update:
        # HDF packs store the exact production one-step output as their
        # pseudo-label, so it is also the fixed Newton feature.  Avoid
        # rebuilding the same inverse operator during distillation training.
        fixed_updates = np.asarray(sigma_array, dtype=np.float64)
    for i in range(N_samples):
        TRUTHS[i, :] = dataset[i].y.squeeze()
        PREDICTIONS[0, i, :] = dataset[i].x.squeeze()

    if cfg.shared_model:
        model = GCNBlock(cfg.channels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.data_dir, exist_ok=True)

    start_time = time.time()
    for k in range(cfg.iterations):
        print("Starting iteration", k)
        if fixed_updates is None:
            dataset = computeLMUpdates(
                dataset,
                physics_inv,
                lambda_lm=cfg.lambda_lm,
                hyper_pvi=cfg.hyper_pvi if cfg.use_laplace else 0.0,
                differential=differential,
            )
        else:
            for i, data in enumerate(dataset):
                update = torch.tensor(fixed_updates[i], dtype=torch.float64).unsqueeze(1)
                data.x = torch.cat((data.x[:, :1], update), dim=1)
        for i in range(N_samples):
            UPDATES[k, i, :] = dataset[i].x[:, 1]

        for i in range(len(dataset)):
            dataset[i].x = dataset[i].x.to(device)
            dataset[i].y = dataset[i].y.to(device)
            dataset[i].edge_index = dataset[i].edge_index.to(device)

        if not cfg.shared_model:
            model = GCNBlock(cfg.channels).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        split = (
            (len(dataset) - validation_count) / len(dataset)
            if validation_count is not None
            else cfg.split
        )
        model, LOSS_TR[k, :], LOSS_VA[k, :] = trainModel(
            model,
            dataset,
            optimizer,
            split,
            cfg.batch_size,
            cfg.max_epochs,
            cfg.patience,
            start_time,
        )
        dataset, PREDICTIONS[k + 1, :, :] = applyModel(model, dataset)

        for i in range(len(dataset)):
            dataset[i].x = dataset[i].x.to("cpu")
            dataset[i].y = dataset[i].y.to("cpu")
            dataset[i].edge_index = dataset[i].edge_index.to("cpu")

        save_name = os.path.join(cfg.models_dir, f"{cfg.model_name}_{k}.pt")
        torch.save(model.to("cpu").state_dict(), save_name)
        model = model.to(device)
        print("Saved model as:", save_name)

    total_time = time.time() - start_time
    np.savez_compressed(
        os.path.join(cfg.data_dir, f"{cfg.model_name}_training_output.npz"),
        model_name=cfg.model_name,
        iterations=cfg.iterations,
        channels=np.array(cfg.channels),
        shared_model=cfg.shared_model,
        lambda_lm=cfg.lambda_lm,
        imaging_mode=cfg.imaging_mode,
        fixed_production_update=args.fixed_production_update,
        TRUTHS=TRUTHS.numpy(),
        PREDICTIONS=PREDICTIONS.numpy(),
        UPDATES=UPDATES.numpy(),
        LOSS_TR=LOSS_TR.numpy(),
        LOSS_VA=LOSS_VA.numpy(),
        total_time=total_time,
    )
    print(f"Training complete in {total_time:.1f}s")


if __name__ == "__main__":
    main()
