#!/usr/bin/env python3
"""Export one ScioSpec bioz session with production-like PVI preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_differential import reconstruct_newton_series
from gcnm_pvi.gcnm_mesh_maps import MeshMappings
from gcnm_pvi.production_preprocess import hp_lp_movmean, process_vmeas_production
from gcnm_pvi.runtime import build_runtime
from gcnm_pvi.sciospec_reader import load_eit_session, read_eit_frame, session_to_vmeas


def main():
    p = argparse.ArgumentParser(description="Production-like export from .eit session")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "finger_pvi08_production.yaml")
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fs", type=float, default=50.0)
    p.add_argument("--max-frames", type=int, default=None, help="None = all frames in folder")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--skip-hp-lp-on-img", action="store_true", help="Skip movmean on img stack")
    args = p.parse_args()

    cfg = GcnmConfig.from_yaml(args.config)
    rt = build_runtime(cfg, include_forward=False)
    physics = rt["physics_inv"]
    mappings = rt["mappings"]

    session = load_eit_session(args.session_dir, max_frames=args.max_frames, stride=args.stride)
    vmeas_raw = session_to_vmeas(session, rt["elec_configs"])

    frame0 = read_eit_frame(Path(args.session_dir) / f"{session.frames[0].name}.eit")
    raw_stim_current = float(frame0.stim_current)
    stim_current = float(cfg.stim_current)
    if not np.isclose(abs(stim_current), abs(raw_stim_current), rtol=1e-6, atol=1e-12):
        raise ValueError(
            f"Config stim_current={stim_current} A does not match raw magnitude "
            f"{raw_stim_current} A"
        )

    proc = process_vmeas_production(
        vmeas_raw.astype(np.complex128),
        fs=args.fs,
        stim_current=stim_current,
        filter_cutoff_hz=cfg.filter_cutoff_hz,
        hp_lp_window=cfg.hp_lp_window,
    )

    vmeas_f = proc["pvi"]["vmeas"]
    sigma_all = reconstruct_newton_series(
        physics,
        np.real(vmeas_f),
        hyper_pvi=cfg.hyper_pvi,
        regularizer=mappings.laplace,
    )
    img_pvi = mappings.elem_to_image_grid(sigma_all)

    out_hp = proc["pviHP"]
    out_lp = proc["pviLP"]
    if not args.skip_hp_lp_on_img:
        img_hp, img_lp = hp_lp_movmean(img_pvi, window=cfg.hp_lp_window, time_axis=2)
        out_hp = dict(out_hp)
        out_lp = dict(out_lp)
        out_hp["img"] = img_hp
        out_lp["img"] = img_lp
    else:
        out_hp = dict(out_hp)
        out_hp["img"] = img_pvi

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "vmeas_raw.npy", vmeas_raw)
    np.save(out / "vmeas.npy", vmeas_f)
    np.save(out / "sigma_elem.npy", sigma_all)
    np.save(out / "img_pvi.npy", img_pvi)
    np.save(out / "img.npy", out_hp["img"])
    np.save(out / "resistance.npy", out_hp["resistance"])
    np.save(out / "reactance.npy", out_hp["reactance"])

    meta = {
        "session_dir": str(args.session_dir),
        "frames": int(vmeas_f.shape[1]),
        "fs": args.fs,
        "stim_current": stim_current,
        "raw_stim_current": raw_stim_current,
        "meas_pattern": list(cfg.meas_pattern),
        "hyper_pvi": cfg.hyper_pvi,
        "inverse_method": "MATLAB production one-step Newton (unweighted J, projected Laplacian, first-frame differential)",
        "img_size": mappings.img_size,
        "mappings_h5": str(cfg.mappings_h5),
        "preprocessing": "production_preprocess (idx0+5Hz LP + movmean HP/LP w=100)",
        "note": "Timeline is raw frame index, not HDF5 cardiac-period interpolation.",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Exported {vmeas_f.shape[1]} frames to {out}")


if __name__ == "__main__":
    main()
