"""Shared runtime: load meshes, mappings, physics, and GCNM graph."""

from __future__ import annotations

import copy

import numpy as np

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_graph import build_edge_index
from gcnm_pvi.gcnm_mesh_maps import MeshMappings
from gcnm_pvi.gcnm_physics import PviDifferentialPhysics, PviPhysics
from gcnm_pvi.paths import ensure_pvi_solver_on_path


def _normalize_mesh(mesh):
    """HDF5 loads scalars as length-1 arrays; numpy 2 breaks CEM assembly."""
    for elec in mesh.elecs:
        z = np.asarray(elec.impedance).ravel()
        elec.impedance = float(z[0]) if z.size else float(elec.impedance)
    return mesh


def load_meshes(cfg: GcnmConfig):
    ensure_pvi_solver_on_path(cfg.pvi_solver_root)
    from pvi_mesh2d import load_mesh_hdf5  # noqa: E402
    from pvi_configs import PviElecConfigs  # noqa: E402

    mesh_fwd = _normalize_mesh(load_mesh_hdf5(str(cfg.mesh_fwd_h5)))
    mesh_inv = _normalize_mesh(load_mesh_hdf5(str(cfg.mesh_inv_h5)))
    elec_configs = PviElecConfigs(
        num_elecs=cfg.num_elecs,
        stim_current=cfg.stim_current,
        stim_pattern=cfg.stim_pattern,
        meas_pattern=cfg.meas_pattern,
    )
    if elec_configs.num_meas_total != 32:
        raise ValueError(f"Expected 32 channels, got {elec_configs.num_meas_total}")
    mappings = MeshMappings(cfg.mappings_h5, img_size=cfg.img_size)
    rtr = mappings.rtr() if cfg.use_laplace else None
    return mesh_fwd, mesh_inv, elec_configs, mappings, rtr


def build_physics(
    cfg: GcnmConfig,
    mesh,
    rtr,
    differential: bool | None = None,
    *,
    forward_backend: str = "dense",
):
    ensure_pvi_solver_on_path(cfg.pvi_solver_root)
    from pvi_configs import PviElecConfigs  # noqa: E402

    elec_configs = PviElecConfigs(
        num_elecs=cfg.num_elecs,
        stim_current=cfg.stim_current,
        stim_pattern=cfg.stim_pattern,
        meas_pattern=cfg.meas_pattern,
    )
    use_diff = cfg.imaging_mode == "differential" if differential is None else differential
    cls = PviDifferentialPhysics if use_diff else PviPhysics
    return cls(mesh, elec_configs, rtr=rtr, backend=forward_backend)


def build_runtime(
    cfg: GcnmConfig,
    *,
    include_forward: bool = True,
    forward_backend: str = "dense",
):
    mesh_fwd, mesh_inv, elec_configs, mappings, rtr = load_meshes(cfg)
    physics_fwd = (
        PviPhysics(
            copy.deepcopy(mesh_fwd),
            elec_configs,
            rtr=None,
            backend=forward_backend,
        )
        if include_forward
        else None
    )
    use_diff = cfg.imaging_mode == "differential"
    physics_inv = build_physics(
        cfg,
        mesh_inv,
        rtr,
        differential=use_diff,
        forward_backend=forward_backend,
    )
    edge_index = build_edge_index(mesh_inv, connectivity=cfg.connectivity)
    return {
        "mesh_fwd": mesh_fwd,
        "mesh_inv": mesh_inv,
        "elec_configs": elec_configs,
        "mappings": mappings,
        "physics_fwd": physics_fwd,
        "physics_inv": physics_inv,
        "edge_index": edge_index,
        "rtr": rtr,
    }
