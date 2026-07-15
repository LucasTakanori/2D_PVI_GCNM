"""PVI forward physics + LM updates for GCNM (absolute and differential modes)."""

from __future__ import annotations

import numpy as np

from gcnm_pvi.paths import ensure_pvi_solver_on_path

ensure_pvi_solver_on_path()
from pvi_forward import PviForward  # noqa: E402


class PviPhysics:
    """Voltage-normalized LM step on a PVI mesh (GCNM-compatible)."""

    def __init__(self, mesh, elec_configs, rtr: np.ndarray | None = None):
        self.mesh = mesh
        self.elec_configs = elec_configs
        self.num_meas = elec_configs.num_meas_total
        self.num_elems = len(mesh.elems)
        self.rtr = rtr

        v_hom = self.solve(np.ones(self.num_elems))
        self._w = 1.0 / np.maximum(np.abs(v_hom), 1e-12)

    def _forward(self, sigma: np.ndarray) -> PviForward:
        self.mesh.elems_data = np.asarray(sigma, dtype=np.float64).ravel()
        fwd = PviForward(mesh=self.mesh, elec_configs=self.elec_configs)
        fwd.make()
        fwd.solve()
        return fwd

    def solve(self, sigma: np.ndarray) -> np.ndarray:
        return self._forward(sigma).results.vmeas.ravel()

    def forward_and_jacobian(self, sigma: np.ndarray):
        fwd = self._forward(sigma)
        V = fwd.results.vmeas.ravel()
        J = fwd.compute_jacobian()
        return V, J

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

    def __init__(self, mesh, elec_configs, rtr: np.ndarray | None = None):
        super().__init__(mesh, elec_configs, rtr=rtr)
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
        self._J0 = F1.compute_jacobian()
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
