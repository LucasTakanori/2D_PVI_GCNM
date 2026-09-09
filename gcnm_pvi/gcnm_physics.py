"""PVI forward physics + LM updates for GCNM (absolute and differential modes)."""

from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse.linalg import splu

from gcnm_pvi.paths import ensure_pvi_solver_on_path

ensure_pvi_solver_on_path()
from pvi_forward import PviForward  # noqa: E402


class PviPhysics:
    """Voltage-normalized LM step on a PVI mesh (GCNM-compatible)."""

    def __init__(
        self,
        mesh,
        elec_configs,
        rtr: np.ndarray | None = None,
        *,
        backend: str = "dense",
    ):
        if backend not in {"dense", "sparse"}:
            raise ValueError(
                f"unknown PVI forward backend {backend!r}; expected 'dense' or 'sparse'"
            )
        self.mesh = mesh
        self.elec_configs = elec_configs
        self.num_meas = elec_configs.num_meas_total
        self.num_elems = len(mesh.elems)
        self.rtr = rtr
        self.backend = backend
        self._dyad_cache: np.ndarray | None = None
        self._prepare_forward_cache()

        v_hom = self.solve(np.ones(self.num_elems))
        self._w = 1.0 / np.maximum(np.abs(v_hom), 1e-12)

    def _prepare_forward_cache(self) -> None:
        """Cache every FEM term that does not depend on conductivity.

        The upstream Python port rebuilt simplex dyads, complete-electrode
        blocks, stimulation matrices, and scatter indices for every frame and
        factorized the same stiffness matrix separately for direct and
        reciprocal fields.  This cache changes only execution order: the FEM
        equations and float64 inputs are unchanged.
        """

        prototype = PviForward(mesh=self.mesh, elec_configs=self.elec_configs)
        nodes = np.asarray(self.mesh.nodes, dtype=np.float64)
        elements = np.asarray(self.mesh.elems, dtype=np.int64)
        num_nodes = len(nodes)
        num_elecs = len(self.mesh.elecs)
        self._dyad_cache = prototype._simpdyad(nodes, elements)
        dyads_by_element = np.moveaxis(self._dyad_cache, 2, 0)
        row_nodes = np.repeat(elements[:, :, None], elements.shape[1], axis=2)
        col_nodes = np.repeat(elements[:, None, :], elements.shape[1], axis=1)
        self._core_flat_indices = (
            row_nodes * num_nodes + col_nodes
        ).reshape(-1)
        self._core_rows = row_nodes.reshape(-1)
        self._core_cols = col_nodes.reshape(-1)
        self._core_dyads = dyads_by_element.reshape(len(elements), -1)

        ground = int(self.mesh.elecs[0].nodes[0])
        self._num_nodes = num_nodes
        self._num_elecs = num_elecs
        self._ground_node = ground
        ae, aq, ad = prototype._make_matrix_cem(self.mesh)
        constant = np.block([[ae, aq], [aq.T, ad]]).astype(np.float64, copy=False)
        constant[:, ground] = 0.0
        constant[ground, :] = 0.0
        constant[ground, ground] = 1.0
        constant_rows, constant_cols = np.nonzero(constant)
        self._constant_rows = constant_rows.astype(np.int64, copy=False)
        self._constant_cols = constant_cols.astype(np.int64, copy=False)
        self._constant_values = constant[constant_rows, constant_cols]
        # Sparse instances keep only the exact entries extracted from the
        # historical dense CEM construction; the one-time dense allocation is
        # released before any frame solve.
        self._constant_stiffness: np.ndarray | None = (
            constant if self.backend == "dense" else None
        )
        self._grounded_core = (self._core_rows == ground) | (
            self._core_cols == ground
        )
        self._sparse_rows = np.concatenate(
            (self._constant_rows, self._core_rows)
        )
        self._sparse_cols = np.concatenate(
            (self._constant_cols, self._core_cols)
        )

        # These arrays describe mesh topology and CEM terms only. Marking them
        # read-only makes a prepared pattern safe to share while every physics
        # lane still creates its own numeric matrix and LU factorization.
        for pattern in (
            self._core_flat_indices,
            self._core_rows,
            self._core_cols,
            self._constant_rows,
            self._constant_cols,
            self._constant_values,
            self._grounded_core,
            self._sparse_rows,
            self._sparse_cols,
            self._core_dyads,
            self._dyad_cache,
        ):
            pattern.flags.writeable = False

        zero_nodes = np.zeros((num_nodes, num_elecs), dtype=np.float64)
        current = float(self.elec_configs.stim_config.current)
        direct = np.vstack(
            (zero_nodes, np.asarray(self.elec_configs.stim_config.matrix, dtype=float))
        ) * current
        reciprocal = np.vstack(
            (zero_nodes, np.asarray(self.elec_configs.meas_config.matrix.T, dtype=float))
        ) * current
        direct[ground, :] = 0.0
        reciprocal[ground, :] = 0.0
        self._combined_rhs = np.column_stack((direct, reciprocal))
        self._measurement_extractor = np.asarray(
            self.elec_configs.potential_config.extractor, dtype=np.float64
        )

    def _solve_dense(self, conductivity: np.ndarray) -> np.ndarray:
        """Reference dense assembly and solve (the historical default)."""
        if self._constant_stiffness is None:
            raise RuntimeError("dense CEM cache is unavailable for sparse backend")
        core_values = (conductivity[:, None] * self._core_dyads).reshape(-1)
        core = np.bincount(
            self._core_flat_indices,
            weights=core_values,
            minlength=self._num_nodes * self._num_nodes,
        ).reshape(self._num_nodes, self._num_nodes)
        stiffness = self._constant_stiffness.copy()
        stiffness[: self._num_nodes, : self._num_nodes] += core
        # Ground entries in the cached constant matrix must dominate any core
        # contributions assembled at the grounded node.
        stiffness[:, self._ground_node] = 0.0
        stiffness[self._ground_node, :] = 0.0
        stiffness[self._ground_node, self._ground_node] = 1.0
        return linalg.solve(
            stiffness,
            self._combined_rhs,
            assume_a="gen",
            check_finite=False,
            overwrite_a=True,
        )

    def _assemble_sparse_stiffness(
        self, conductivity: np.ndarray
    ) -> sparse.csc_matrix:
        """Assemble the identical CEM system from cached COO contributions."""
        total_size = self._num_nodes + self._num_elecs
        core_values = (conductivity[:, None] * self._core_dyads).reshape(-1)

        # Keep all nine cached FEM positions per triangle. Contributions on
        # the grounded row or column are numerically zeroed before assembly;
        # the cached CEM data contains the sole unit ground diagonal.
        core_values[self._grounded_core] = 0.0
        values = np.concatenate((self._constant_values, core_values))
        stiffness = sparse.coo_matrix(
            (values, (self._sparse_rows, self._sparse_cols)),
            shape=(total_size, total_size),
            dtype=np.float64,
        ).tocsc()
        stiffness.sum_duplicates()
        stiffness.eliminate_zeros()
        return stiffness

    def _solve_sparse(self, conductivity: np.ndarray) -> np.ndarray:
        """Sparse exact solve with a lane-local CSC matrix and SuperLU."""
        stiffness = self._assemble_sparse_stiffness(conductivity)
        factor = splu(stiffness)
        return factor.solve(self._combined_rhs)

    def _forward(self, sigma: np.ndarray) -> PviForward:
        conductivity = np.asarray(sigma, dtype=np.float64).ravel()
        if len(conductivity) != self.num_elems:
            raise ValueError(
                f"conductivity has {len(conductivity)} elements, expected {self.num_elems}"
            )
        self.mesh.elems_data = conductivity
        fwd = PviForward(mesh=self.mesh, elec_configs=self.elec_configs)
        solutions = (
            self._solve_dense(conductivity)
            if self.backend == "dense"
            else self._solve_sparse(conductivity)
        )
        direct = solutions[:, : self._num_elecs]
        reciprocal = solutions[:, self._num_elecs :]
        voltage = direct[: self._num_nodes]
        voltage_reci = reciprocal[: self._num_nodes]
        voltage_elecs = direct[self._num_nodes :]
        per_stim = self.elec_configs.num_meas_per_stim
        measured = np.empty((self.num_meas, 1), dtype=np.float64)
        for stimulus in range(self._num_elecs):
            rows = np.arange(per_stim) + per_stim * stimulus
            measured[rows, 0] = (
                self._measurement_extractor[stimulus]
                @ voltage_elecs[:, stimulus]
            )
        fwd.stim_current = float(self.elec_configs.stim_config.current)
        fwd.run_reci = True
        fwd.has_results = True
        fwd.results = fwd.results_class(
            voltage=voltage,
            voltage_reci=voltage_reci,
            voltage_elecs=voltage_elecs,
            vmeas=measured,
        )
        return fwd

    def solve(self, sigma: np.ndarray) -> np.ndarray:
        return self._forward(sigma).results.vmeas.ravel()

    def forward_and_jacobian(self, sigma: np.ndarray):
        fwd = self._forward(sigma)
        V = fwd.results.vmeas.ravel()
        J = self.jacobian_from_forward(fwd)
        return V, J

    def jacobian_from_forward(self, forward: PviForward) -> np.ndarray:
        """Vectorized equivalent of ``PviForward.compute_jacobian``.

        Simplex geometry depends only on the mesh, so it is cached once per
        physics object.  The original nested Python element loop is expressed
        as one batched contraction without changing the reciprocity formula.
        """
        elements = np.asarray(self.mesh.elems, dtype=np.int64)
        if self._dyad_cache is None:
            self._dyad_cache = forward._simpdyad(self.mesh.nodes, elements)
        voltage = np.asarray(forward.results.voltage)[: len(self.mesh.nodes), :]
        reciprocal = np.asarray(forward.results.voltage_reci)[: len(self.mesh.nodes), :]
        pair_index = self.elec_configs.pair_idx
        per_stim = self.elec_configs.num_meas_per_stim
        jacobian = np.empty((self.num_meas, self.num_elems), dtype=np.float64)
        reciprocal_elements = reciprocal[elements]
        for stimulus in range(len(self.mesh.elecs)):
            selected = pair_index[:, stimulus].astype(bool)
            drive = voltage[elements, stimulus]
            measure = reciprocal_elements[:, :, selected]
            values = np.einsum(
                "ki,ijk,kjm->km",
                drive,
                self._dyad_cache,
                measure,
                optimize=True,
            )
            rows = np.arange(per_stim) + per_stim * stimulus
            jacobian[rows, :] = values.T
        return -jacobian / float(forward.stim_current)

    def lm_update(
        self,
        sigma: np.ndarray,
        vmeas: np.ndarray,
        lambda_lm: float = 0.1,
        hyper_pvi: float = 0.0,
    ):
        """One LM / GN step. Returns (delta_sigma, V_at_sigma)."""
        V, J = self.forward_and_jacobian(sigma)
        Jn = self._w[:, None] * J
        rn = self._w * (V - np.asarray(vmeas, dtype=np.float64).ravel())

        JtJ = Jn.T @ Jn
        if hyper_pvi > 0 and self.rtr is not None:
            JtJ = JtJ + (hyper_pvi ** 2) * self.rtr

        delta = -np.linalg.solve(
            JtJ + lambda_lm * np.eye(JtJ.shape[0]), Jn.T @ rn
        )
        return delta, V


class PviDifferentialPhysics(PviPhysics):
    """PVI-faithful differential imaging (reference frame + calibration)."""

    def __init__(
        self,
        mesh,
        elec_configs,
        rtr: np.ndarray | None = None,
        *,
        backend: str = "dense",
    ):
        super().__init__(mesh, elec_configs, rtr=rtr, backend=backend)
        self.sigma_init: np.ndarray | None = None
        self.sigma_ref: np.ndarray | None = None
        self.V_init: np.ndarray | None = None
        self.V_ref: np.ndarray | None = None
        self.dV_alpha: np.ndarray | None = None
        self._J0: np.ndarray | None = None
        self._fixed_linear_cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}

    def calibrate(
        self,
        vmeas_ref: np.ndarray,
        sigma_init: float | np.ndarray | None = None,
    ) -> None:
        if sigma_init is None:
            sigma_init = 0.7 * np.ones(self.num_elems)
        sigma_init = np.asarray(sigma_init, dtype=np.float64).ravel()
        self.sigma_init = sigma_init.copy()

        F1 = self._forward(sigma_init)
        V1 = F1.results.vmeas.ravel()
        vref = np.real(np.asarray(vmeas_ref, dtype=np.float64).ravel())

        denom = np.dot(vref, V1)
        sigma_alpha = np.dot(V1, V1) / denom if abs(denom) > 1e-30 else 1.0
        self.sigma_ref = sigma_alpha * sigma_init
        self.V_ref = V1.copy()
        self.V_init = V1.copy()
        self.dV_alpha = vref - self.V_ref
        self._J0 = self.jacobian_from_forward(F1)
        self._fixed_linear_cache.clear()

    def differential_residual(self, vmeas_t: np.ndarray) -> np.ndarray:
        if self.V_init is None or self.dV_alpha is None:
            raise RuntimeError("call calibrate() first")
        vt = np.real(np.asarray(vmeas_t, dtype=np.float64).ravel())
        return (vt - self.V_init) - self.dV_alpha

    def lm_update_differential(
        self,
        vmeas_t: np.ndarray,
        lambda_lm: float = 0.1,
        hyper_pvi: float = 3e-4,
        use_fixed_jacobian: bool = True,
    ):
        if self.sigma_init is None or self.sigma_ref is None:
            raise RuntimeError("call calibrate() first")

        dV = self.differential_residual(vmeas_t)
        rn = self._w * (-dV)

        if use_fixed_jacobian and self._J0 is not None:
            key = (float(lambda_lm), float(hyper_pvi))
            cached = self._fixed_linear_cache.get(key)
            if cached is None:
                Jn = self._w[:, None] * self._J0
                reg = (hyper_pvi ** 2) * self.rtr if hyper_pvi > 0 and self.rtr is not None else None
                A = Jn.T @ Jn + lambda_lm * np.eye(Jn.shape[1])
                if reg is not None:
                    A = A + reg
                D1 = np.linalg.solve(A, Jn.T)
                correction = (
                    np.linalg.solve(A, reg @ (self.sigma_ref - self.sigma_init))
                    if reg is not None
                    else np.zeros(self.num_elems, dtype=np.float64)
                )
                cached = (D1, correction)
                self._fixed_linear_cache[key] = cached
            D1, correction = cached
            delta = D1 @ rn + correction
        else:
            J = self.forward_and_jacobian(self.sigma_init)[1]
            Jn = self._w[:, None] * J
            reg = (hyper_pvi ** 2) * self.rtr if hyper_pvi > 0 and self.rtr is not None else None
            A = Jn.T @ Jn + lambda_lm * np.eye(Jn.shape[1])
            if reg is not None:
                A = A + reg
            rhs = Jn.T @ rn
            if reg is not None:
                rhs = rhs + reg @ (self.sigma_ref - self.sigma_init)
            delta = np.linalg.solve(A, rhs)

        return delta, self.V_init.copy()

    def newton_step_production(self, vmeas_t: np.ndarray, hyper_pvi: float = 3e-4) -> np.ndarray:
        """Legacy normalized differential step; not the strict MATLAB inverse.

        New production exports must use
        :func:`gcnm_pvi.gcnm_differential.reconstruct_newton_series`, which
        requires the projected Laplacian and uses an unweighted Jacobian.
        """
        delta, _ = self.lm_update_differential(
            vmeas_t, lambda_lm=0.0, hyper_pvi=hyper_pvi, use_fixed_jacobian=True
        )
        return self.sigma_init - delta
