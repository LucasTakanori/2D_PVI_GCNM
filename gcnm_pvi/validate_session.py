"""Validate production export against fundational_pvi HDF5 reference."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import zoom


def _load_h5_tensor(h5_path: Path, component: str, field: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        return np.asarray(f[f"data/{component}/{field}"][()], dtype=np.float64)


def _match_image_grid(pred: np.ndarray, ref: np.ndarray, method: str = "zoom") -> tuple[np.ndarray, dict]:
    """Bring pred (H,W,T) to ref spatial shape."""
    hp, wp, _ = pred.shape
    hr, wr, _ = ref.shape
    info = {"pred_shape": list(pred.shape), "ref_shape": list(ref.shape), "method": method}
    if (hp, wp) == (hr, wr):
        return pred, info
    if method == "zoom":
        zy = hr / hp
        zx = wr / wp
        matched = zoom(pred, (zy, zx, 1.0), order=1)
        info["zoom"] = [zy, zx]
        return matched, info
    raise ValueError(f"Cannot match grids {pred.shape} vs {ref.shape} with method={method}")


def validate_against_hdf5(
    export_dir: Path,
    h5_path: Path,
    *,
    component: str = "pviHP",
    max_frames: int | None = 500,
    img_grid_method: str = "zoom",
) -> dict:
    """Compare export npy files vs HDF5. Returns report dict."""
    export_dir = Path(export_dir)
    h5_path = Path(h5_path)
    report: dict = {"export_dir": str(export_dir), "h5": str(h5_path), "component": component}

    pred_img = np.load(export_dir / "img.npy")
    pred_r = np.load(export_dir / "resistance.npy")
    ref_img = _load_h5_tensor(h5_path, component, "img")
    ref_r = _load_h5_tensor(h5_path, component, "resistance")

    t_pred = pred_img.shape[-1]
    t_ref = ref_img.shape[-1]
    t_use = min(t_pred, t_ref, max_frames or min(t_pred, t_ref))
    report["time"] = {
        "pred_frames": int(t_pred),
        "h5_frames": int(t_ref),
        "compared_frames": int(t_use),
        "warning": (
            "HDF5 time axis is cardiac-period interpolated; export uses raw frame index. "
            "Metrics are indicative only until alignment pipeline is mirrored."
        ),
    }

    pred_img_c = pred_img[..., :t_use]
    ref_img_c = ref_img[..., :t_use]
    pred_matched, grid_info = _match_image_grid(pred_img_c, ref_img_c, method=img_grid_method)
    report["grid"] = grid_info

    pred_valid = np.isfinite(pred_matched)
    ref_valid = np.isfinite(ref_img_c)
    jointly_valid = pred_valid & ref_valid
    if not jointly_valid.any():
        raise ValueError("Prediction and reference have no jointly finite image pixels")
    diff = pred_matched[jointly_valid] - ref_img_c[jointly_valid]
    img_mse = float(np.mean(diff ** 2))
    img_mae = float(np.mean(np.abs(diff)))
    mask_intersection = int(np.count_nonzero(pred_valid[..., 0] & ref_valid[..., 0]))
    mask_union = int(np.count_nonzero(pred_valid[..., 0] | ref_valid[..., 0]))
    report["img"] = {
        "mse": img_mse,
        "mae": img_mae,
        "jointly_finite_values": int(np.count_nonzero(jointly_valid)),
        "mask_iou": float(mask_intersection / mask_union) if mask_union else float("nan"),
    }

    pred_r_c = pred_r[:, :t_use]
    ref_r_c = ref_r[:, :t_use]
    per_ch_corr = []
    for ch in range(min(pred_r_c.shape[0], ref_r_c.shape[0])):
        a = pred_r_c[ch]
        b = ref_r_c[ch]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            per_ch_corr.append(float("nan"))
        else:
            per_ch_corr.append(float(np.corrcoef(a, b)[0, 1]))
    report["resistance"] = {
        "mse": float(np.mean((pred_r_c - ref_r_c) ** 2)),
        "mean_channel_corr": float(np.nanmean(per_ch_corr)),
        "per_channel_corr": per_ch_corr,
    }

    report["pass"] = {
        "img_grid_ok": bool(pred_matched.shape[:2] == ref_img_c.shape[:2]),
        "img_mse_finite": bool(np.isfinite(img_mse)),
        "resistance_corr_ok": bool(np.nanmean(per_ch_corr) > 0.5),
    }
    return report


def main():
    import argparse
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    p = argparse.ArgumentParser(description="Validate production export vs HDF5")
    p.add_argument("--export-dir", type=Path, required=True)
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--component", default="pviHP")
    p.add_argument("--max-frames", type=int, default=500)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    report = validate_against_hdf5(
        args.export_dir,
        args.h5,
        component=args.component,
        max_frames=args.max_frames,
    )
    text = json.dumps(report, indent=2)
    print(text)
    out = args.out_json or (args.export_dir / "validation_report.json")
    Path(out).write_text(text, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
