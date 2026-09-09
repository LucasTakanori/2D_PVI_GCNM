import numpy as np
from scipy import sparse

from gcnm_pvi.dual_mesh_physics import ProjectedFineMeshPhysics


class _LinearFinePhysics:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64)
        self.num_meas, self.num_elems = self.matrix.shape

    def solve(self, conductivity: np.ndarray) -> np.ndarray:
        return self.matrix @ np.asarray(conductivity)

    def forward_and_jacobian(self, conductivity: np.ndarray):
        return self.solve(conductivity), self.matrix.copy()


def test_projected_fine_physics_applies_chain_rule():
    forward = np.asarray([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 1.0]])
    coarse_to_fine = sparse.csr_matrix(
        np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    )
    physics = ProjectedFineMeshPhysics(
        _LinearFinePhysics(forward), coarse_to_fine
    )
    conductivity = np.asarray([0.7, 1.2])

    voltage, jacobian = physics.forward_and_jacobian(conductivity)

    np.testing.assert_allclose(voltage, forward @ coarse_to_fine @ conductivity)
    np.testing.assert_allclose(jacobian, forward @ coarse_to_fine)
    assert jacobian.shape == (2, 2)


def test_projected_fine_physics_rejects_nonpartition_mapping():
    forward = _LinearFinePhysics(np.eye(2))
    invalid = sparse.csr_matrix(np.asarray([[0.5], [1.0]]))
    try:
        ProjectedFineMeshPhysics(forward, invalid)
    except ValueError as error:
        assert "unit-sum" in str(error)
    else:
        raise AssertionError("invalid mapping was accepted")
