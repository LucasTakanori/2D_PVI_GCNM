"""Unit tests for faithful per-stage nonlinear PVI physics."""

from __future__ import annotations

import unittest

import numpy as np

from gcnm_pvi.iterative_physics import (
    differential_lm_step,
    iterative_lm_reconstruction,
)


class FakeNonlinearPhysics:
    """Small separable nonlinear forward map with an analytic Jacobian."""

    def __init__(self) -> None:
        self.linear = np.array([[1.0, 0.3], [0.2, 1.2], [0.5, -0.4]])
        self.quadratic = np.array([[0.15, 0.05], [0.02, 0.12], [0.08, 0.04]])
        self.forward_states: list[np.ndarray] = []
        self.jacobian_states: list[np.ndarray] = []

    def solve(self, sigma: np.ndarray) -> np.ndarray:
        state = np.asarray(sigma, dtype=np.float64)
        self.forward_states.append(state.copy())
        return self.linear @ state + 0.5 * self.quadratic @ (state**2)

    def forward_and_jacobian(self, sigma: np.ndarray):
        state = np.asarray(sigma, dtype=np.float64)
        self.forward_states.append(state.copy())
        self.jacobian_states.append(state.copy())
        voltage = self.linear @ state + 0.5 * self.quadratic @ (state**2)
        jacobian = self.linear + self.quadratic * state[None, :]
        return voltage, jacobian


class IterativePhysicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = np.array([0.7, 0.8])
        self.regularizer = np.eye(2)

    def _measured(self, physics: FakeNonlinearPhysics, delta: np.ndarray) -> np.ndarray:
        return physics.solve(self.baseline + delta) - physics.solve(self.baseline)

    def test_zero_residual_produces_zero_direction(self) -> None:
        physics = FakeNonlinearPhysics()
        current = np.array([0.04, -0.02])
        measured = self._measured(physics, current)
        direction, diagnostic, _ = differential_lm_step(
            physics,
            self.baseline,
            current,
            measured,
            regularizer=self.regularizer,
            hyper_pvi=0.0,
            lambda_lm=1e-3,
        )
        np.testing.assert_allclose(direction, 0.0, atol=1e-11)
        self.assertLess(diagnostic.voltage_residual_rms, 1e-12)

    def test_positive_inclusion_has_positive_first_update(self) -> None:
        physics = FakeNonlinearPhysics()
        measured = self._measured(physics, np.array([0.05, 0.02]))
        direction, _diagnostic, _ = differential_lm_step(
            physics,
            self.baseline,
            np.zeros(2),
            measured,
            regularizer=self.regularizer,
            hyper_pvi=0.0,
            lambda_lm=1e-3,
        )
        self.assertTrue(np.all(direction > 0))

    def test_one_step_decreases_voltage_residual(self) -> None:
        physics = FakeNonlinearPhysics()
        truth = np.array([0.08, -0.03])
        measured = self._measured(physics, truth)
        initial_rms = float(np.sqrt(np.mean(measured**2)))
        direction, _diagnostic, _ = differential_lm_step(
            physics,
            self.baseline,
            np.zeros(2),
            measured,
            regularizer=self.regularizer,
            hyper_pvi=0.0,
            lambda_lm=1e-2,
        )
        residual = self._measured(physics, direction) - measured
        self.assertLess(float(np.sqrt(np.mean(residual**2))), initial_rms)

    def test_stage_two_recomputes_at_stage_one_state(self) -> None:
        physics = FakeNonlinearPhysics()
        truth = np.array([[0.12, -0.07]])
        measured = np.stack([self._measured(physics, truth[0])])
        stages, _diagnostics = iterative_lm_reconstruction(
            physics,
            self.baseline[None, :],
            measured,
            iterations=2,
            regularizer=self.regularizer,
            hyper_pvi=0.0,
            lambda_lm=0.05,
        )
        self.assertEqual(stages.shape, (2, 1, 2))
        self.assertEqual(len(physics.jacobian_states), 2)
        np.testing.assert_allclose(physics.jacobian_states[0], self.baseline)
        np.testing.assert_allclose(
            physics.jacobian_states[1], self.baseline + stages[0, 0]
        )
        self.assertFalse(np.allclose(stages[1, 0], 2.0 * stages[0, 0]))


if __name__ == "__main__":
    unittest.main()
