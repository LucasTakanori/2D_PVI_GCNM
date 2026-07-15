#!/usr/bin/env python3
"""Compare GCNM / Newton images against fundational_pvi HDF5 reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcnm_pvi.validate_session import _load_h5_tensor, _match_image_grid


def compare_images(
    pred_img: np.ndarray,
    h5_path: Path,
    component: str = "pviHP",
    grid_method: str = "zoom",
    max_frames: int | None = None,
) -> dict:
    ref = _load_h5_tensor(h5_path, component, "img")
    t = min(pred_img.shape[-1], ref.shape[-1])
    if max_frames:
        t = min(t, max_frames)
    pred_c = pred_img[..., :t]
    ref_c = ref[..., :t]
    pred_m, grid_info = _match_image_grid(pred_c, ref_c, method=grid_method)
    mse = float(np.mean((pred_m - ref_c) ** 2))
    return {"mse": mse, "frames": t, "grid": grid_info}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-img-npy", type=Path, required=True, help="(H,W,T) prediction")
    p.add_argument("--h5", type=Path, required=True, help="Reference masked HDF5")
    p.add_argument("--component", default="pviHP")
    p.add_argument("--grid-method", default="zoom", choices=["zoom"])
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    pred = np.load(args.pred_img_npy)
    out = compare_images(
        pred,
        args.h5,
        component=args.component,
        grid_method=args.grid_method,
        max_frames=args.max_frames,
    )
    print(f"Compared {out['frames']} frames vs {args.h5}")
    print(f"Grid: {out['grid']}")
    print(f"Image MSE vs HDF5 {args.component}/img: {out['mse']:.6e}")


if __name__ == "__main__":
    main()
