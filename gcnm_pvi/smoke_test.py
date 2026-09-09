#!/usr/bin/env python3
"""End-to-end smoke test (tiny mesh, no HDF5 files required)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcnm_pvi.electrode_protocol import default_elec_configs
from gcnm_pvi.gcnm_graph import build_edge_index
from gcnm_pvi.gcnm_model import (
    GCNBlock,
    applyModel,
    computeLMUpdates,
    initializeDataset,
    trainModel,
)
from gcnm_pvi.gcnm_phantoms import generate_dataset
from gcnm_pvi.gcnm_physics import PviDifferentialPhysics, PviPhysics
from gcnm_pvi.paths import ensure_pvi_solver_on_path
from gcnm_pvi.tiny_mesh import make_tiny_circle_mesh

N_SAMPLES = 12
ITERATIONS = 2
CHANNELS = [32, 32]
LAMBDA_LM = 0.1


def main():
    ensure_pvi_solver_on_path()
    elec_configs = default_elec_configs()
    print(f"[1/6] protocol OK: {elec_configs.num_meas_total} channels")

    mesh = make_tiny_circle_mesh()
    physics = PviPhysics(mesh, elec_configs)
    V_hom, J = physics.forward_and_jacobian(np.ones(physics.num_elems))
    rank = np.linalg.matrix_rank(J)
    assert V_hom.shape == (32,) and J.shape == (32, physics.num_elems)
    print(f"[2/6] forward+Jacobian OK, rank(J)={rank}")

    rng = np.random.default_rng(0)
    sigma_array, V_array = generate_dataset(
        physics, mesh, N_SAMPLES, rng, mode="perturbation", n_inclusions_range=(1, 2)
    )
    delta, _ = physics.lm_update(np.ones(physics.num_elems), V_array[0], lambda_lm=LAMBDA_LM)
    assert delta.shape == (physics.num_elems,) and np.all(np.isfinite(delta))
    print(f"[3/6] LM update OK, |delta| mean={np.mean(np.abs(delta)):.4f}")

    diff = PviDifferentialPhysics(mesh, elec_configs)
    diff.calibrate(V_array[0])
    dres = diff.differential_residual(V_array[1])
    assert dres.shape == (32,)
    print(f"[4/6] differential calibrate OK, |dV| mean={np.mean(np.abs(dres)):.4f}")

    edge_index = build_edge_index(mesh, connectivity="node")
    dataset = initializeDataset(sigma_array, V_array, edge_index)

    def mean_re(ds):
        return np.mean(
            [
                np.linalg.norm(d.x[:, 0].numpy() - d.y.squeeze().numpy(), 1)
                / np.linalg.norm(d.y.squeeze().numpy(), 1)
                for d in ds
            ]
        )

    re_init = mean_re(dataset)
    start = time.time()
    models = []
    for k in range(ITERATIONS):
        dataset = computeLMUpdates(dataset, physics, lambda_lm=LAMBDA_LM)
        model = GCNBlock(CHANNELS)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model, _, _ = trainModel(
            model, dataset, optimizer, split=0.75, batch_size=4,
            max_epochs=40, patience=15, start_time=start,
        )
        dataset, _ = applyModel(model, dataset)
        models.append(model)
        print(f"      iteration {k}: mean RE_sigma = {mean_re(dataset):.4f}")

    re_final = mean_re(dataset)
    assert re_final < re_init, f"RE did not improve: {re_init:.4f} -> {re_final:.4f}"
    print(f"[5/6] learning OK: RE {re_init:.4f} -> {re_final:.4f}")

    dataset = computeLMUpdates(dataset, physics, lambda_lm=LAMBDA_LM)
    torch.save(models[-1].state_dict(), "/tmp/smoke_gcnm_model.pt")
    model2 = GCNBlock(CHANNELS)
    model2.load_state_dict(torch.load("/tmp/smoke_gcnm_model.pt", weights_only=True))
    model2.eval()
    models[-1].eval()
    with torch.no_grad():
        a = models[-1](dataset[0])
        b = model2(dataset[0])
    assert torch.equal(a, b)
    print("[6/6] save/load roundtrip OK")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
