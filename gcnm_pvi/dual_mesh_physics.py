"""Project fine-mesh PVI physics onto a coarse inverse parameterization."""

from __future__ import annotations

import numpy as np
from scipy import sparse


class ProjectedFineMeshPhysics:
    """Expose fine-mesh forward physics in coarse inverse coordinates.

    If ``P`` maps coarse element conductivities to the fine mesh, this adapter
    evaluates ``F_f(P sigma_c)`` and returns the chain-rule Jacobian
    ``J_f(P sigma_c) P``.  The returned directions therefore remain defined on
    the coarse graph while every forward solution and reciprocal-field
    Jacobian is computed on the distinct fine mesh.
    """

    def __init__(self, fine_physics, coarse_to_fine: sparse.spmatrix) -> None:
        mapping = sparse.csr_matrix(coarse_to_fine, dtype=np.float64)
        if mapping.ndim != 2:
            raise ValueError("coarse-to-fine mapping must be a matrix")
        if mapping.shape[0] != int(fine_physics.num_elems):
            raise ValueError(
                "coarse-to-fine rows must match fine elements: "
                f"{mapping.shape[0]} != {fine_physics.num_elems}"
            )
        row_sums = np.asarray(mapping.sum(axis=1)).ravel()
        if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(
                "every fine element must be represented by a unit-sum coarse mapping"
            )
        self.fine_physics = fine_physics
        self.coarse_to_fine = mapping
        self.num_elems = int(mapping.shape[1])
        self.num_meas = int(fine_physics.num_meas)

    def to_fine(self, sigma_coarse: np.ndarray) -> np.ndarray:
        conductivity = np.asarray(sigma_coarse, dtype=np.float64).ravel()
        if conductivity.shape != (self.num_elems,):
            raise ValueError(
                f"coarse conductivity has shape {conductivity.shape}, "
                f"expected {(self.num_elems,)}"
            )
        return np.asarray(self.coarse_to_fine @ conductivity).ravel()

    def solve(self, sigma_coarse: np.ndarray) -> np.ndarray:
        return self.fine_physics.solve(self.to_fine(sigma_coarse))

    def forward_and_jacobian(
        self, sigma_coarse: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        voltage, jacobian_fine = self.fine_physics.forward_and_jacobian(
            self.to_fine(sigma_coarse)
        )
        jacobian_fine = np.asarray(jacobian_fine, dtype=np.float64)
        expected = (self.num_meas, self.coarse_to_fine.shape[0])
        if jacobian_fine.shape != expected:
            raise ValueError(
                f"fine Jacobian has shape {jacobian_fine.shape}, expected {expected}"
            )
        # Sparse-left multiplication avoids NumPy's object-array behavior for
        # dense @ sparse on some SciPy versions.
        jacobian_coarse = np.asarray(
            self.coarse_to_fine.T @ jacobian_fine.T
        ).T
        return np.asarray(voltage, dtype=np.float64), jacobian_coarse
