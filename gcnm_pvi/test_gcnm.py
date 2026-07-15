#!/usr/bin/env python3
"""GCNM evaluation vs N-step LM baseline + image metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_image import image_mse, save_image_triplet
from gcnm_pvi.gcnm_model import GCNBlock, applyModel, computeLMUpdates, initializeDataset
from gcnm_pvi.gcnm_phantoms import generate_dataset
from gcnm_pvi.runtime import build_runtime


def re_sigma_l1(pred, true):
    return np.linalg.norm(pred - true, 1) / np.linalg.norm(true, 1)


def re_voltage_l2(physics, sigma, vmeas):
    U = physics.solve(sigma)
    return np.linalg.norm(U - vmeas) / np.linalg.norm(vmeas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "finger_pvi08.yaml")
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--save-images", type=int, default=3, help="Save this many image triplets")
    args = p.parse_args()

    cfg = GcnmConfig.from_yaml(args.config)
    rt = build_runtime(cfg)
    physics_fwd = rt["physics_fwd"]
    physics_inv = rt["physics_inv"]
    mappings = rt["mappings"]
    edge_index = rt["edge_index"]
    differential = cfg.imaging_mode == "differential"

    rng = np.random.default_rng(args.seed)
    sigma_array, V_array = generate_dataset(
        physics_fwd,
        rt["mesh_inv"],
        args.n_test,
        rng,
        mode=cfg.phantom_mode,
        noise_scale=cfg.noise_scale,
    )

    if differential:
        physics_inv.calibrate(V_array[0])

    dataset = initializeDataset(sigma_array, V_array, edge_index)
    N_nodes = physics_inv.num_elems
    PRED_GCNM = np.zeros((1 + cfg.iterations, args.n_test, N_nodes))
    for i in range(args.n_test):
        PRED_GCNM[0, i, :] = dataset[i].x.squeeze().numpy()

    for k in range(cfg.iterations):
        print("GCNM inference, iteration", k)
        dataset = computeLMUpdates(
            dataset,
            physics_inv,
            lambda_lm=cfg.lambda_lm,
            hyper_pvi=cfg.hyper_pvi if cfg.use_laplace else 0.0,
            differential=differential,
        )
        model = GCNBlock(cfg.channels)
        model.load_state_dict(
            torch.load(
                Path(cfg.models_dir) / f"{cfg.model_name}_{k}.pt",
                weights_only=True,
            )
        )
        dataset, preds = applyModel(model, dataset)
        PRED_GCNM[k + 1, :, :] = preds.numpy()

    PRED_LM = np.zeros((1 + cfg.iterations, args.n_test, N_nodes))
    PRED_LM[0] = 1.0
    for i in range(args.n_test):
        sigma = np.ones(N_nodes)
        for k in range(cfg.iterations):
            if differential:
                delta, _ = physics_inv.lm_update_differential(
                    V_array[i],
                    lambda_lm=cfg.lambda_lm,
                    hyper_pvi=cfg.hyper_pvi if cfg.use_laplace else 0.0,
                )
            else:
                delta, _ = physics_inv.lm_update(
                    sigma,
                    V_array[i],
                    lambda_lm=cfg.lambda_lm,
                    hyper_pvi=cfg.hyper_pvi if cfg.use_laplace else 0.0,
                )
            sigma = sigma + cfg.lm_step_size * delta
            PRED_LM[k + 1, i, :] = sigma

    print("\nPer-iteration mean RE_sigma_l1:")
    print(f"{'iter':>4} | {'GCNM':>8} | {'LM baseline':>11}")
    for k in range(1 + cfg.iterations):
        re_g = np.mean([re_sigma_l1(PRED_GCNM[k, i], sigma_array[i]) for i in range(args.n_test)])
        re_l = np.mean([re_sigma_l1(PRED_LM[k, i], sigma_array[i]) for i in range(args.n_test)])
        print(f"{k:>4} | {re_g:8.4f} | {re_l:11.4f}")

    final_g = PRED_GCNM[-1]
    final_l = PRED_LM[-1]
    mse_g = np.mean((final_g - sigma_array) ** 2)
    mse_l = np.mean((final_l - sigma_array) ** 2)
    img_mse_g = np.mean([image_mse(final_g[i], sigma_array[i], mappings) for i in range(args.n_test)])
    img_mse_l = np.mean([image_mse(final_l[i], sigma_array[i], mappings) for i in range(args.n_test)])
    rev_g = np.mean([re_voltage_l2(physics_inv, final_g[i], V_array[i]) for i in range(args.n_test)])
    rev_l = np.mean([re_voltage_l2(physics_inv, final_l[i], V_array[i]) for i in range(args.n_test)])

    print(f"\nFinal element MSE: GCNM {mse_g:.6f} | LM {mse_l:.6f}")
    print(f"Final image MSE:   GCNM {img_mse_g:.6f} | LM {img_mse_l:.6f}")
    print(f"Final RE_voltage:  GCNM {rev_g:.6f} | LM {rev_l:.6f}")

    out_dir = Path(cfg.data_dir) / "test_images"
    for i in range(min(args.save_images, args.n_test)):
        save_image_triplet(
            out_dir / f"sample_{i:03d}.png",
            sigma_array[i],
            final_g[i],
            mappings,
            title=f"GCNM sample {i}",
        )

    np.savez_compressed(
        Path(cfg.data_dir) / f"{cfg.model_name}_testing_output.npz",
        TRUTHS=sigma_array,
        V=V_array,
        PRED_GCNM=PRED_GCNM,
        PRED_LM=PRED_LM,
    )
    print("Saved testing output.")


if __name__ == "__main__":
    main()
