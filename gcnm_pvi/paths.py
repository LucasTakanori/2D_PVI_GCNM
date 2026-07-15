"""Resolve PVI solver on PYTHONPATH and project directories."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from gcnm_pvi import pvi_compat  # noqa: F401  (patch np.math before pvi imports)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_pvi_solver_root() -> Path:
    env = os.environ.get("PVI_SOLVER_ROOT")
    if env:
        return Path(env)
    return (
        repo_root().parent
        / "Peripheral-Vascular-Impedance-Imaging"
        / "python_port"
        / "pvi_solver"
    )


def ensure_pvi_solver_on_path(pvi_solver_root: Path | None = None) -> Path:
    root = Path(pvi_solver_root or default_pvi_solver_root()).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PVI solver directory not found: {root}")
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def default_mesh_bundle(pvi_solver_root: Path | None = None) -> dict[str, Path]:
    root = Path(pvi_solver_root or default_pvi_solver_root())
    data = root / "_data" / "_mesh08_r64"
    return {
        "mesh_fwd_h5": data / "tank2d_mdl_64_refined.h5",
        "mesh_inv_h5": data / "tank2d_mdl_64.h5",
        "mappings_h5": data / "tank2d_mdl_64_mappings.h5",
    }
