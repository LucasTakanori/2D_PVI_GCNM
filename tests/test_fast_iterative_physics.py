import numpy as np
from scipy import linalg, sparse

from gcnm_pvi.iterative_physics import (
    FixedZeroCurrentLMSolver,
    LowRankRegularizedSolver,
    absolute_lm_step,
    dataset_absolute_lm_directions,
    dataset_lm_directions,
    parallel_dataset_absolute_lm_directions,
    parallel_dataset_lm_directions,
)


def test_low_rank_lm_solver_matches_dense_laplacian_system():
    rng = np.random.default_rng(4)
    elements, measurements = 80, 12
    difference = sparse.diags(
        [-np.ones(elements - 1), np.ones(elements - 1)], [0, 1],
        shape=(elements - 1, elements),
    )
    regularizer = (difference.T @ difference).tocsr()
    jacobian = rng.normal(size=(measurements, elements))
    residual = rng.normal(size=measurements)
    hyper = 5e-4
    expected = linalg.solve(
        jacobian.T @ jacobian + hyper**2 * regularizer.toarray(),
        -(jacobian.T @ residual),
        assume_a="sym",
        check_finite=False,
    )

    solver = LowRankRegularizedSolver(
        regularizer, elements, hyper_pvi=hyper, lambda_lm=0.0
    )
    actual = solver.solve(jacobian, residual)

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)


def test_low_rank_lm_solver_falls_back_to_robust_svd_driver(monkeypatch):
    import gcnm_pvi.iterative_physics as module

    rng = np.random.default_rng(41)
    elements, measurements = 40, 8
    jacobian = rng.normal(size=(measurements, elements))
    residual = rng.normal(size=measurements)
    solver = LowRankRegularizedSolver(
        None, elements, hyper_pvi=0.0, lambda_lm=1e-4
    )
    expected = solver.solve(jacobian, residual)
    original = module.linalg.svd
    calls = []

    def fail_fast_driver(*args, **kwargs):
        calls.append(kwargs.get("lapack_driver"))
        if kwargs.get("lapack_driver") == "gesdd":
            raise module.linalg.LinAlgError("synthetic convergence failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module.linalg, "svd", fail_fast_driver)
    actual = solver.solve(jacobian, residual)
    assert calls[:2] == ["gesdd", "gesvd"]
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=1e-11)


class _LinearPhysics:
    def __init__(self, jacobian):
        self.jacobian = np.asarray(jacobian)

    def solve(self, conductivity):
        return self.jacobian @ np.asarray(conductivity)

    def forward_and_jacobian(self, conductivity):
        return self.solve(conductivity), self.jacobian


def test_absolute_lm_uses_absolute_forward_voltage():
    jacobian = np.array([[2.0, 0.0], [0.0, 4.0]])
    physics = _LinearPhysics(jacobian)
    current = np.array([0.7, 0.7])
    truth = np.array([0.62, 0.91])
    direction, diagnostics, predicted = absolute_lm_step(
        physics,
        current,
        physics.solve(truth),
        regularizer=None,
        hyper_pvi=0.0,
        lambda_lm=0.0,
    )
    np.testing.assert_allclose(current + direction, truth)
    np.testing.assert_allclose(predicted, physics.solve(current))
    assert diagnostics.voltage_residual_rms > 0


def test_fixed_first_absolute_solver_subtracts_homogeneous_forward_voltage():
    jacobian = np.array([[3.0, 0.0], [0.0, 5.0]])
    physics = _LinearPhysics(jacobian)
    baseline = np.array([0.7, 0.7])
    truth = np.array([[0.65, 0.82], [0.74, 0.61]])
    solver = FixedZeroCurrentLMSolver(
        physics,
        baseline,
        regularizer=None,
        hyper_pvi=0.0,
        lambda_lm=0.0,
    )
    directions, diagnostics, predicted = solver.solve_many_absolute(
        truth @ jacobian.T
    )
    np.testing.assert_allclose(baseline[None, :] + directions, truth)
    np.testing.assert_allclose(
        predicted, np.broadcast_to(physics.solve(baseline), predicted.shape)
    )
    assert len(diagnostics) == len(truth)


def test_parallel_low_rank_directions_preserve_serial_order_and_values(monkeypatch):
    monkeypatch.delenv("GCNM_PHYSICS_WORKERS", raising=False)
    rng = np.random.default_rng(12)
    samples, elements, measurements = 11, 30, 8
    jacobian = rng.normal(size=(measurements, elements))
    difference = sparse.diags(
        [-np.ones(elements - 1), np.ones(elements - 1)], [0, 1],
        shape=(elements - 1, elements),
    )
    regularizer = (difference.T @ difference).tocsr()
    baseline = np.full((samples, elements), 0.7)
    current = rng.normal(scale=0.01, size=(samples, elements))
    measured = rng.normal(scale=0.02, size=(samples, measurements))
    physics = _LinearPhysics(jacobian)
    baseline_voltage = np.stack([physics.solve(item) for item in baseline])
    solver = LowRankRegularizedSolver(
        regularizer, elements, hyper_pvi=5e-4, lambda_lm=0.0
    )
    serial = dataset_lm_directions(
        physics, baseline, current, measured,
        regularizer=regularizer, hyper_pvi=5e-4, lambda_lm=0.0,
        baseline_voltages=baseline_voltage, system_solver=solver,
    )
    parallel = parallel_dataset_lm_directions(
        physics, baseline, current, measured,
        regularizer=regularizer, hyper_pvi=5e-4, lambda_lm=0.0,
        baseline_voltages=baseline_voltage, system_solver=solver, workers=3,
    )
    np.testing.assert_allclose(parallel[0], serial[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(parallel[2], serial[2], rtol=0, atol=0)
    assert parallel[1] == serial[1]


def test_process_low_rank_directions_preserve_serial_order_and_values(monkeypatch):
    monkeypatch.setenv("GCNM_PHYSICS_EXECUTOR", "process")
    monkeypatch.delenv("GCNM_PHYSICS_WORKERS", raising=False)
    rng = np.random.default_rng(120)
    samples, elements, measurements = 8, 24, 6
    jacobian = rng.normal(size=(measurements, elements))
    difference = sparse.diags(
        [-np.ones(elements - 1), np.ones(elements - 1)], [0, 1],
        shape=(elements - 1, elements),
    )
    regularizer = (difference.T @ difference).tocsr()
    baseline = np.full((samples, elements), 0.7)
    current = rng.normal(scale=0.01, size=(samples, elements))
    measured = rng.normal(scale=0.02, size=(samples, measurements))
    physics = _LinearPhysics(jacobian)
    baseline_voltage = np.stack([physics.solve(item) for item in baseline])
    solver = LowRankRegularizedSolver(
        regularizer, elements, hyper_pvi=5e-4, lambda_lm=0.0
    )
    serial = dataset_lm_directions(
        physics, baseline, current, measured,
        regularizer=regularizer, hyper_pvi=5e-4, lambda_lm=0.0,
        baseline_voltages=baseline_voltage, system_solver=solver,
    )
    parallel = parallel_dataset_lm_directions(
        physics, baseline, current, measured,
        regularizer=regularizer, hyper_pvi=5e-4, lambda_lm=0.0,
        baseline_voltages=baseline_voltage, system_solver=solver, workers=3,
    )
    np.testing.assert_allclose(parallel[0], serial[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(parallel[2], serial[2], rtol=0, atol=0)
    assert parallel[1] == serial[1]


def test_parallel_absolute_directions_preserve_serial_order(monkeypatch):
    monkeypatch.delenv("GCNM_PHYSICS_WORKERS", raising=False)
    rng = np.random.default_rng(44)
    samples, elements = 9, 6
    jacobian = rng.normal(size=(elements, elements))
    jacobian += 3.0 * np.eye(elements)
    physics = _LinearPhysics(jacobian)
    current = rng.uniform(0.5, 0.9, size=(samples, elements))
    truth = rng.uniform(0.4, 1.0, size=(samples, elements))
    measured = truth @ jacobian.T
    serial = dataset_absolute_lm_directions(
        physics,
        current,
        measured,
        regularizer=None,
        hyper_pvi=0.0,
        lambda_lm=1e-5,
    )
    parallel = parallel_dataset_absolute_lm_directions(
        physics,
        current,
        measured,
        regularizer=None,
        hyper_pvi=0.0,
        lambda_lm=1e-5,
        workers=3,
    )
    np.testing.assert_allclose(parallel[0], serial[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(parallel[2], serial[2], rtol=1e-12, atol=1e-12)
    assert parallel[1] == serial[1]
