#!/usr/bin/env python3
"""Compare clean-truth GCNM reconstructions with the noisy Newton baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

from gcnm_pvi.anatomical_phantoms import element_positions
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_model import ResidualGCNBlock
from gcnm_pvi.runtime import build_runtime


def _correlation(prediction: np.ndarray, truth: np.ndarray) -> float:
    if np.std(prediction) < 1e-15 or np.std(truth) < 1e-15:
        return float("nan")
    return float(np.corrcoef(prediction.ravel(), truth.ravel())[0, 1])


def _localization_dice(prediction: np.ndarray, truth: np.ndarray) -> float:
    scores = []
    for pred, target in zip(prediction, truth):
        support = np.flatnonzero(np.abs(target) > 1e-8)
        if not len(support):
            continue
        selected = np.argpartition(np.abs(pred), -len(support))[-len(support) :]
        overlap = len(np.intersect1d(support, selected))
        scores.append(overlap / len(support))
    return float(np.mean(scores))


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    error = prediction - truth
    support = np.abs(truth) > 1e-8
    background = ~support
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "correlation": _correlation(prediction, truth),
        "localization_dice_at_true_volume": _localization_dice(prediction, truth),
        "background_rms": float(np.sqrt(np.mean(prediction[background] ** 2))),
        "vessel_mean": float(np.mean(prediction[support])),
    }


def _map_images(values: np.ndarray, mappings) -> np.ndarray:
    return np.stack([mappings.elem_to_image_grid(sample) for sample in values])


def _save_examples(
    out_dir: Path,
    truth_img: np.ndarray,
    newton_img: np.ndarray,
    prediction_img: np.ndarray,
    count: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(min(count, len(truth_img))):
        finite = np.isfinite(truth_img[index])
        limit = float(np.nanmax(np.abs(truth_img[index])))
        figure, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
        for axis, image, title in zip(
            axes,
            [truth_img[index], newton_img[index], prediction_img[index]],
            ["Clean conductivity truth", "Noisy physical Newton", "GCNM"],
        ):
            shown = image.copy()
            shown[~finite] = np.nan
            artist = axis.imshow(shown, cmap="RdBu_r", vmin=-limit, vmax=limit)
            axis.set_title(title)
            axis.axis("off")
        figure.colorbar(artist, ax=axes, shrink=0.75, label="Δ conductivity")
        figure.savefig(out_dir / f"sample_{index:03d}.png", dpi=160)
        plt.close(figure)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs" / "subject006_anatomical_gcnm.yaml")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--save-examples", type=int, default=4)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data" / "subject006_anatomical_results" / "evaluation",
    )
    args = parser.parse_args()

    cfg = GcnmConfig.from_yaml(args.config)
    runtime = build_runtime(cfg, include_forward=False)
    source = np.load(args.test)
    truth = np.asarray(source["sigma"], dtype=np.float64)
    newton = np.asarray(source["newton"], dtype=np.float64)
    positions = element_positions(runtime["mesh_inv"])
    current = np.zeros_like(truth)
    model_dir = args.models_dir or Path(cfg.models_dir)
    iterations = args.iterations or cfg.iterations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    stages = []
    for iteration in range(iterations):
        checkpoint = torch.load(
            model_dir / f"{cfg.model_name}_{iteration}.pt",
            map_location=device,
            weights_only=True,
        )
        scale = float(checkpoint["scale"])
        model = ResidualGCNBlock(
            checkpoint["channels"],
            in_channels=int(checkpoint["in_channels"]),
        ).float().to(device)
        model.load_state_dict(checkpoint["state_dict"])
        predictions = []
        model.eval()
        with torch.no_grad():
            for sample in range(len(truth)):
                features = np.column_stack(
                    [current[sample] / scale, newton[sample] / scale, positions]
                )
                data = Data(
                    x=torch.tensor(features, dtype=torch.float32, device=device),
                    edge_index=runtime["edge_index"].to(device),
                )
                predictions.append(model(data).squeeze().cpu().numpy() * scale)
        current = np.stack(predictions)
        stages.append(current.copy())

    truth_img = _map_images(truth, runtime["mappings"])
    newton_img = _map_images(newton, runtime["mappings"])
    prediction_img = _map_images(current, runtime["mappings"])
    finite = np.isfinite(truth_img) & np.isfinite(newton_img) & np.isfinite(prediction_img)
    report = {
        "target": "clean synthetic vascular conductivity; no PVI image labels",
        "samples": int(len(truth)),
        "newton": _metrics(newton, truth),
        "gcnm": _metrics(current, truth),
        "image": {
            "newton_rmse": float(np.sqrt(np.mean((newton_img[finite] - truth_img[finite]) ** 2))),
            "gcnm_rmse": float(np.sqrt(np.mean((prediction_img[finite] - truth_img[finite]) ** 2))),
            "newton_correlation": _correlation(newton_img[finite], truth_img[finite]),
            "gcnm_correlation": _correlation(prediction_img[finite], truth_img[finite]),
        },
        "iterations": iterations,
    }
    report["improvement"] = {
        "element_rmse_fraction": report["gcnm"]["rmse"] / report["newton"]["rmse"],
        "image_rmse_fraction": report["image"]["gcnm_rmse"] / report["image"]["newton_rmse"],
        "background_rms_fraction": report["gcnm"]["background_rms"] / report["newton"]["background_rms"],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.out_dir / "predictions.npz",
        truth=truth.astype(np.float32),
        newton=newton.astype(np.float32),
        gcnm=current.astype(np.float32),
        stages=np.stack(stages).astype(np.float32),
    )
    _save_examples(
        args.out_dir / "examples",
        truth_img,
        newton_img,
        prediction_img,
        args.save_examples,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
