"""One-step Newton reconstruction matching the MATLAB production PVI solver."""

from __future__ import annotations

import numpy as np
from scipy import linalg, sparse

from gcnm_pvi.gcnm_physics import PviDifferentialPhysics


def reconstruct_newton_series(
    physics: PviDifferentialPhysics,
    vmeas: np.ndarray,
    hyper_pvi: float = 3e-4,
    use_complex: bool = True,
    regularizer: np.ndarray | sparse.spmatrix | None = None,
    vmeas_ref: np.ndarray | None = None,
    sigma_init: float = 0.7,
) -> np.ndarray:
    """Run the production MATLAB one-step Newton inverse on a time series.

    This is algebraically equivalent to ``pvi_inv_make.m`` followed by the
    external-packages version of ``pvi_inv_solve.m``.  That implementation
    computes an unweighted Jacobian on the coarse inverse mesh at 0.7 S/m,
    regularizes it with ``hyper_pvi**2 * Rrec``, and finally subtracts the
    reconstruction at the first voltage frame.  The forward-model calibration
    and fine-mesh initial voltage therefore cancel from the returned
    differential image, leaving::

        sigma_reconst = -D1 @ (vmeas - vmeas_ref)
        D1 = solve(J.T @ J + hyper_pvi**2 * Rrec, J.T)

    Parameters
    ----------
    physics:
        Physics object on the coarse inverse mesh.
    vmeas:
        Measured voltage matrix with shape ``(M, T)``.
    regularizer:
        MATLAB ``Rrec = (c2f.T @ Rfwd @ c2f) / 2``.  For exported ring
        bundles this is ``MeshMappings.laplace``.  It is intentionally *not*
        ``laplace.T @ laplace``.
    vmeas_ref:
        Reference voltage vector.  Defaults to the first column of ``vmeas``,
        exactly as in the production session path.

    Returns
    -------
    np.ndarray
        Differential element conductivity with shape ``(K, T)``.  With the
        default reference, column zero is exactly zero.
    """
    vm = np.asarray(vmeas)
    if vm.ndim == 1:
        vm = vm[:, None]
    if use_complex:
        vm = np.real(vm)
    vm = np.asarray(vm, dtype=np.float64)

    if regularizer is None:
        raise ValueError(
            "production reconstruction requires the projected Laplacian "
            "(pass mappings.laplace), not physics.rtr"
        )
    reg = (
        regularizer.toarray()
        if sparse.issparse(regularizer)
        else np.asarray(regularizer)
    )
    if reg.shape != (physics.num_elems, physics.num_elems):
        raise ValueError(
            f"regularizer shape {reg.shape} does not match "
            f"{physics.num_elems} inverse elements"
        )

    inverse_matrix = production_reconstruction_matrix(
        physics,
        hyper_pvi=hyper_pvi,
        regularizer=reg,
        sigma_init=sigma_init,
    )

    reference = vm[:, 0] if vmeas_ref is None else np.real(np.asarray(vmeas_ref)).ravel()
    if reference.shape != (vm.shape[0],):
        raise ValueError(
            f"vmeas_ref shape {reference.shape} does not match {vm.shape[0]} channels"
        )
    return inverse_matrix @ (vm - reference[:, None])


def production_reconstruction_matrix(
    physics: PviDifferentialPhysics,
    *,
    hyper_pvi: float,
    regularizer: np.ndarray | sparse.spmatrix,
    sigma_init: float = 0.7,
) -> np.ndarray:
    """Return the fixed production map from differential voltage to Δσ."""
    reg = (
        regularizer.toarray()
        if sparse.issparse(regularizer)
        else np.asarray(regularizer)
    )
    sigma0 = np.full(physics.num_elems, float(sigma_init), dtype=np.float64)
    forward = physics._forward(sigma0)
    jacobian = np.asarray(physics.jacobian_from_forward(forward), dtype=np.float64)
    system = jacobian.T @ jacobian + float(hyper_pvi) ** 2 * reg
    d1 = linalg.solve(
        system,
        jacobian.T,
        assume_a="pos",
        check_finite=False,
    )
    return -d1
