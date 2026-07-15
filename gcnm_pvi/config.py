"""Load YAML/JSON run configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gcnm_pvi.paths import default_mesh_bundle, default_pvi_solver_root, repo_root


@dataclass
class GcnmConfig:
    pvi_solver_root: Path = field(default_factory=default_pvi_solver_root)
    mesh_fwd_h5: Path | None = None
    mesh_inv_h5: Path | None = None
    mappings_h5: Path | None = None

    num_elecs: int = 8
    stim_current: float = 1e-3
    stim_pattern: tuple[int, int] = (0, 3)
    meas_pattern: tuple[int, int] = (0, 1)

    img_size: int = 32
    imaging_mode: str = "differential"  # absolute | differential
    lambda_lm: float = 0.1
    hyper_pvi: float = 3e-4
    use_laplace: bool = True
    lm_step_size: float = 1.0
    connectivity: str = "node"

    n_samples: int = 500
    noise_scale: float = 0.005
    seed: int = 0
    phantom_mode: str = "inclusion"  # inclusion | perturbation

    iterations: int = 10
    channels: list[int] = field(default_factory=lambda: [250, 250, 250])
    split: float = 0.8
    learning_rate: float = 0.002
    batch_size: int = 10
    max_epochs: int = 10000
    patience: int = 200
    shared_model: bool = False

    model_name: str = "gcnm_pvi08"
    samples_dir: str | None = None
    models_dir: str = "models"
    data_dir: str = "data"

    filter_cutoff_hz: float = 5.0
    filter_pass: str = "low"
    hp_lp_window: int = 100

    @classmethod
    def from_yaml(cls, path: Path | str) -> "GcnmConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls.from_dict(raw, base_dir=path.parent)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], base_dir: Path | None = None) -> "GcnmConfig":
        base_dir = base_dir or repo_root()
        cfg = cls()
        if "pvi_solver_root" in raw:
            p = Path(raw["pvi_solver_root"])
            cfg.pvi_solver_root = p if p.is_absolute() else (base_dir / p).resolve()
        bundle = default_mesh_bundle(cfg.pvi_solver_root)
        for key in ("mesh_fwd_h5", "mesh_inv_h5", "mappings_h5"):
            if key in raw:
                p = Path(raw[key])
                setattr(cfg, key, p if p.is_absolute() else (base_dir / p).resolve())
            elif getattr(cfg, key) is None:
                setattr(cfg, key, bundle[key.replace("_h5", "").replace("mesh_", "mesh_") + "_h5" if False else key])
        # fill defaults from bundle when unset
        if cfg.mesh_fwd_h5 is None:
            cfg.mesh_fwd_h5 = bundle["mesh_fwd_h5"]
        if cfg.mesh_inv_h5 is None:
            cfg.mesh_inv_h5 = bundle["mesh_inv_h5"]
        if cfg.mappings_h5 is None:
            cfg.mappings_h5 = bundle["mappings_h5"]

        simple_map = {
            "num_elecs": int,
            "stim_current": float,
            "img_size": int,
            "imaging_mode": str,
            "lambda_lm": float,
            "hyper_pvi": float,
            "use_laplace": bool,
            "lm_step_size": float,
            "connectivity": str,
            "n_samples": int,
            "noise_scale": float,
            "seed": int,
            "phantom_mode": str,
            "iterations": int,
            "split": float,
            "learning_rate": float,
            "batch_size": int,
            "max_epochs": int,
            "patience": int,
            "shared_model": bool,
            "model_name": str,
            "samples_dir": str,
            "models_dir": str,
            "data_dir": str,
            "filter_cutoff_hz": float,
            "filter_pass": str,
            "hp_lp_window": int,
        }
        for k, caster in simple_map.items():
            if k in raw and raw[k] is not None:
                setattr(cfg, k, caster(raw[k]))
        if "stim_pattern" in raw:
            cfg.stim_pattern = tuple(raw["stim_pattern"])
        if "meas_pattern" in raw:
            cfg.meas_pattern = tuple(raw["meas_pattern"])
        if "channels" in raw:
            cfg.channels = list(raw["channels"])
        if cfg.samples_dir is None:
            cfg.samples_dir = f"samples/{cfg.model_name}"
        return cfg
