#!/usr/bin/env python3
"""Evaluate the subject006 GCNM distillation smoke model on a held-out trial pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_model import GCNBlock, applyModel, initializeDataset
from gcnm_pvi.gcnm_phantoms import load_dataset
from gcnm_pvi.runtime import build_runtime


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs" / "subject006_pvi08_smoke.yaml")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=root / "data" / "subject006_smoke" / "heldout_report.json",
    )
    args = parser.parse_args()

    cfg = GcnmConfig.from_yaml(args.config)
    runtime = build_runtime(cfg, include_forward=False)
    truth, voltage = load_dataset(args.samples)
    dataset = initializeDataset(
        truth,
        voltage,
        runtime["edge_index"],
        initial_sigma=0.0,
    )
    for index, data in enumerate(dataset):
        update = torch.tensor(truth[index], dtype=torch.float64).unsqueeze(1)
        data.x = torch.cat((data.x, update), dim=1)

    model = GCNBlock(cfg.channels)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    _, prediction = applyModel(model, dataset)
    prediction = prediction.numpy()

    truth_img = np.stack([runtime["mappings"].elem_to_image_grid(row) for row in truth])
    pred_img = np.stack([runtime["mappings"].elem_to_image_grid(row) for row in prediction])
    valid = np.isfinite(truth_img) & np.isfinite(pred_img)
    image_error = pred_img[valid] - truth_img[valid]
    element_error = prediction - truth
    report = {
        "samples": int(len(truth)),
        "held_out_trials": np.unique(np.load(args.samples)["trial_index"]).astype(int).tolist(),
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "element_mse": float(np.mean(element_error**2)),
        "element_mae": float(np.mean(np.abs(element_error))),
        "element_correlation": _corr(prediction, truth),
        "zero_initialization_mse": float(np.mean(truth**2)),
        "image_mse": float(np.mean(image_error**2)),
        "image_correlation": _corr(pred_img[valid], truth_img[valid]),
        "scope": (
            "Engineering distillation smoke test. The fixed Newton feature and "
            "target are the same PVI pseudo-label, so this does not demonstrate "
            "improvement over the production inverse."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
