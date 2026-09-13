from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy import linalg, sparse

from gcnm_pvi.iterative_physics import (
    FixedZeroCurrentLMSolver,
    LM_SOLVER_IMPLEMENTATION,
    LowRankRegularizedSolver,
    absolute_lm_step,
    dataset_absolute_lm_directions,
    dataset_lm_directions,
    parallel_dataset_absolute_lm_directions,
    parallel_dataset_lm_directions,
)


def test_solver_implementation_identifier_is_stable():
    assert LM_SOLVER_IMPLEMENTATION == "measurement_svd_v1"


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
    augmented = solver._augmented_solve(jacobian, residual)

    # The dense normal equation is itself less accurate than the equivalent
    # augmented least-squares formulation at this condition number.
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(actual, augmented, rtol=2e-10, atol=2e-12)


def _legacy_schur_solve(solver, jacobian, residual):
    """Former subtractive Schur implementation, retained as a regression foil."""

    positive_jacobian = jacobian @ solver.positive_vectors
    inverse_sqrt = np.sqrt(solver.positive_inverse)
    whitened = positive_jacobian * inverse_sqrt[None, :]
    _left, singular, vh = linalg.svd(whitened, full_matrices=False)

    def positive_solve(rhs):
        values = np.atleast_2d(rhs).T if np.ndim(rhs) == 1 else rhs
        whitened_rhs = inverse_sqrt[:, None] * values
        coefficients = vh @ whitened_rhs
        orthogonal = whitened_rhs - vh.T @ coefficients
        retained = coefficients / (1.0 + singular[:, None] ** 2)
        result = inverse_sqrt[:, None] * (orthogonal + vh.T @ retained)
        for _ in range(6):
            equation_residual = values - (
                solver.positive_values[:, None] * result
                + positive_jacobian.T @ (positive_jacobian @ result)
            )
            whitened_rhs = inverse_sqrt[:, None] * equation_residual
            coefficients = vh @ whitened_rhs
            orthogonal = whitened_rhs - vh.T @ coefficients
            retained = coefficients / (1.0 + singular[:, None] ** 2)
            result += inverse_sqrt[:, None] * (orthogonal + vh.T @ retained)
        return result

    positive_rhs = -(positive_jacobian.T @ residual)
    null_jacobian = jacobian @ solver.null_vectors
    coupling = positive_jacobian.T @ null_jacobian
    solved = positive_solve(np.column_stack((positive_rhs, coupling)))
    positive_base, positive_coupling = solved[:, 0], solved[:, 1:]
    schur = null_jacobian.T @ null_jacobian - coupling.T @ positive_coupling
    null_rhs = -(null_jacobian.T @ residual)
    null_solution = linalg.solve(
        schur, null_rhs - coupling.T @ positive_base, assume_a="sym"
    )
    return (
        solver.positive_vectors
        @ (positive_base - positive_coupling @ null_solution)
        + solver.null_vectors @ null_solution
    )


def test_measurement_space_solver_avoids_subtractive_schur_failure():
    rng = np.random.default_rng(1)
    elements, measurements = 40, 8
    regularizer = np.diag(np.r_[np.ones(elements - 1), 0.0])
    jacobian = np.empty((measurements, elements))
    jacobian[:, :-1] = 1e6 * rng.normal(size=(measurements, elements - 1))
    jacobian[:, -1] = rng.normal(size=measurements)
    residual = rng.normal(size=measurements)
    solver = LowRankRegularizedSolver(
        regularizer, elements, hyper_pvi=1.0, lambda_lm=0.0
    )

    expected = solver._augmented_solve(jacobian, residual)
    measurement_space = solver._measurement_space_solve(jacobian, residual)
    actual = solver.solve(jacobian, residual)
    try:
        legacy = _legacy_schur_solve(solver, jacobian, residual)
    except linalg.LinAlgError:
        legacy = None

    np.testing.assert_allclose(
        measurement_space, expected, rtol=5e-10, atol=2e-12
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-10, atol=2e-12)
    assert legacy is None or np.linalg.norm(legacy - expected) > np.linalg.norm(expected)


def test_low_rank_solver_rejects_unobservable_regularizer_null_mode():
    elements = 12
    regularizer = np.diag(np.r_[np.ones(elements - 1), 0.0])
    jacobian = np.eye(5, elements)
    solver = LowRankRegularizedSolver(
        regularizer, elements, hyper_pvi=1.0, lambda_lm=0.0
    )

    with pytest.raises(linalg.LinAlgError, match="null mode|rank deficient"):
        solver.solve(jacobian, np.ones(5))


def test_low_rank_solver_returns_zero_for_zero_jacobian_without_null_mode():
    elements, measurements = 9, 4
    solver = LowRankRegularizedSolver(
        np.eye(elements), elements, hyper_pvi=5e-4, lambda_lm=0.0
    )

    actual = solver.solve(np.zeros((measurements, elements)), np.arange(measurements))

    np.testing.assert_array_equal(actual, np.zeros(elements))


def test_measurement_complement_matches_augmented_oracle_when_measurements_exceed_elements():
    rng = np.random.default_rng(23)
    elements, measurements = 5, 12
    jacobian = rng.normal(size=(measurements, elements))
    residual = rng.normal(size=measurements)
    solver = LowRankRegularizedSolver(
        np.eye(elements), elements, hyper_pvi=0.2, lambda_lm=0.0
    )

    expected = solver._augmented_solve(jacobian, residual)
    measurement_space = solver._measurement_space_solve(jacobian, residual)
    actual = solver.solve(jacobian, residual)

    np.testing.assert_allclose(measurement_space, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_low_rank_solver_is_safe_for_concurrent_calls():
    rng = np.random.default_rng(12)
    elements, measurements = 30, 9
    difference = sparse.diags(
        [-np.ones(elements - 1), np.ones(elements - 1)], [0, 1],
        shape=(elements - 1, elements),
    )
    solver = LowRankRegularizedSolver(
        difference.T @ difference,
        elements,
        hyper_pvi=5e-4,
        lambda_lm=0.0,
    )
    jobs = [
        (rng.normal(size=(measurements, elements)), rng.normal(size=measurements))
        for _ in range(12)
    ]
    expected = [solver.solve(jacobian, residual) for jacobian, residual in jobs]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda args: solver.solve(*args), jobs))

    for concurrent, serial in zip(actual, expected):
        np.testing.assert_array_equal(concurrent, serial)


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


@pytest.mark.fork_isolation
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
