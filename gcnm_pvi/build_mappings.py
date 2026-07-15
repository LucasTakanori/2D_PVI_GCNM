#!/usr/bin/env python3
"""Build 40×40 (or N×N) m2i mapping HDF5 from inverse mesh."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.fem2d_mesh2img import fem2d_mesh2img, save_m2i_hdf5
from gcnm_pvi.runtime import _normalize_mesh, load_meshes


def main():
    p = argparse.ArgumentParser(description="Build m2i mapping at target image size")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "finger_pvi08_production.yaml")
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--out", type=Path, default=None, help="Output mappings .h5")
    args = p.parse_args()

    cfg = GcnmConfig.from_yaml(args.config)
    img_size = args.img_size or cfg.img_size
    _, mesh_inv, _, _, _ = load_meshes(cfg)
    mesh_inv = _normalize_mesh(mesh_inv)

    m2i = fem2d_mesh2img(mesh_inv, img_size=img_size)
    out = args.out or cfg.mappings_h5
    if out is None:
        raise SystemExit("Set mappings_h5 in config or pass --out")

    laplace_src = cfg.mappings_h5 if cfg.mappings_h5 and Path(cfg.mappings_h5).is_file() else None
    save_m2i_hdf5(m2i, out, laplace_src=laplace_src)
    print(f"Wrote {out}  (m2i {m2i.shape}, img_size={img_size})")


if __name__ == "__main__":
    main()
