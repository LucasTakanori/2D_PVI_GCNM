#!/usr/bin/env python3
"""Export GCNM training packs from ScioSpec session or vmeas.npy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_differential import reconstruct_newton_series
from gcnm_pvi.gcnm_mesh_maps import MeshMappings
from gcnm_pvi.pvi_preprocess import hp_lp_split, to_real_vmeas
from gcnm_pvi.runtime import build_runtime
from gcnm_pvi.sciospec_reader import export_vmeas_npy, load_eit_session, session_to_vmeas


def main():
    p = argparse.ArgumentParser(description="Build GCNM NPZ dataset from raw or vmeas")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "finger_pvi08.yaml")
    p.add_argument("--session-dir", type=Path, default=None, help="Folder of .eit files")
    p.add_argument("--vmeas-npy", type=Path, default=None, help="Precomputed (M,T) vmeas")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fs", type=float, default=50.0)
    p.add_argument("--component", choices=["raw", "hp", "lp"], default="hp")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()

    cfg = GcnmConfig.from_yaml(args.config)
    rt = build_runtime(cfg)
    physics = rt["physics_inv"]
    mappings = rt["mappings"]

    if args.vmeas_npy:
        vmeas = np.load(args.vmeas_npy)
    elif args.session_dir:
        session = load_eit_session(
            args.session_dir, max_frames=args.max_frames, stride=args.stride
        )
        vmeas = session_to_vmeas(session, rt["elec_configs"])
    else:
        raise SystemExit("Provide --session-dir or --vmeas-npy")

    if args.component == "hp":
        hp, _ = hp_lp_split(vmeas, args.fs, cfg.filter_cutoff_hz)
        vmeas = to_real_vmeas(hp)
    elif args.component == "lp":
        _, lp = hp_lp_split(vmeas, args.fs, cfg.filter_cutoff_hz)
        vmeas = to_real_vmeas(lp)
    else:
        vmeas = to_real_vmeas(vmeas)

    if args.vmeas_npy:
        if args.max_frames:
            vmeas = vmeas[:, : args.max_frames]
        if args.stride > 1:
            vmeas = vmeas[:, :: args.stride]

    sigma_all = reconstruct_newton_series(
        physics,
        vmeas,
        hyper_pvi=cfg.hyper_pvi,
        regularizer=mappings.laplace,
    )
    img_all = mappings.elem_to_image_grid(sigma_all)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vmeas.npy", vmeas)
    np.save(out / "sigma_elem.npy", sigma_all)
    np.save(out / "img.npy", img_all)

    T = vmeas.shape[1]
    (out / "npz").mkdir(parents=True, exist_ok=True)
    for t in range(T):
        np.savez_compressed(
            out / "npz" / f"frame_{t:06d}.npz",
            V=vmeas[:, t],
            sigma=sigma_all[:, t],
            img=img_all[:, :, t],
        )

    meta = out / "README.txt"
    meta.write_text(
        f"GCNM export\nframes={T}\nvmeas shape={vmeas.shape}\n"
        f"sigma shape={sigma_all.shape}\nimg shape={img_all.shape}\n",
        encoding="utf-8",
    )
    print(f"Exported {T} frames to {out}")


if __name__ == "__main__":
    main()
