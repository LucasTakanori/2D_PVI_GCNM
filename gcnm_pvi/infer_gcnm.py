#!/usr/bin/env python3
"""Run GCNM inference on a ScioSpec session -> conductivity images."""

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
from gcnm_pvi.gcnm_image import save_image_triplet
from gcnm_pvi.gcnm_model import GCNBlock, applyModel, computeLMUpdates, initializeDataset
from gcnm_pvi.pvi_preprocess import hp_lp_split, to_real_vmeas
from gcnm_pvi.runtime import build_runtime
from gcnm_pvi.sciospec_reader import load_eit_session, session_to_vmeas


def infer_frame_series(cfg: GcnmConfig, vmeas: np.ndarray, rt: dict) -> np.ndarray:
    """vmeas (M,T) -> sigma (K,T) via trained GCNM."""
    physics_inv = rt["physics_inv"]
    mappings = rt["mappings"]
    edge_index = rt["edge_index"]
    differential = cfg.imaging_mode == "differential"

    T = vmeas.shape[1]
    sigma_out = np.zeros((physics_inv.num_elems, T), dtype=np.float64)

    for t in range(T):
        vt = vmeas[:, t]
        sigma_array = np.ones((1, physics_inv.num_elems))
        V_array = vt.reshape(1, -1)
        dataset = initializeDataset(sigma_array, V_array, edge_index)

        if differential:
            physics_inv.calibrate(vt)

        for k in range(cfg.iterations):
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
        sigma_out[:, t] = preds.numpy().ravel()

    return sigma_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "finger_pvi08.yaml")
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fs", type=float, default=50.0)
    p.add_argument("--max-frames", type=int, default=100)
    p.add_argument("--save-every", type=int, default=25)
    args = p.parse_args()

    cfg = GcnmConfig.from_yaml(args.config)
    rt = build_runtime(cfg)
    mappings = rt["mappings"]

    session = load_eit_session(args.session_dir)
    vmeas = session_to_vmeas(session, rt["elec_configs"])
    hp, _ = hp_lp_split(vmeas, args.fs, cfg.filter_cutoff_hz)
    vmeas = to_real_vmeas(hp)[:, : args.max_frames]

    sigma = infer_frame_series(cfg, vmeas, rt)
    img = mappings.elem_to_image_grid(sigma)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vmeas.npy", vmeas)
    np.save(out / "sigma_gcnm.npy", sigma)
    np.save(out / "img_gcnm.npy", img)

    for t in range(0, sigma.shape[1], args.save_every):
        save_image_triplet(
            out / "frames" / f"frame_{t:06d}.png",
            sigma[:, t],
            sigma[:, t],
            mappings,
            title=f"GCNM t={t}",
        )

    print(f"Inference done: {sigma.shape[1]} frames -> {out}")


if __name__ == "__main__":
    main()
