import copy

import numpy as np

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_physics import PviPhysics
from gcnm_pvi.paths import ensure_pvi_solver_on_path
from gcnm_pvi.runtime import load_meshes


def test_cached_forward_matches_upstream_pvi_equations():
    config = GcnmConfig.from_yaml("configs/rings_b045/US120.yaml")
    ensure_pvi_solver_on_path(config.pvi_solver_root)
    from pvi_forward import PviForward

    _forward_mesh, inverse_mesh, electrode_configs, _mappings, _rtr = load_meshes(
        config
    )
    conductivity = np.full(len(inverse_mesh.elems), 0.7, dtype=np.float64)
    conductivity += np.linspace(-0.003, 0.003, len(conductivity))

    reference_mesh = copy.deepcopy(inverse_mesh)
    reference_mesh.elems_data = conductivity.copy()
    reference = PviForward(reference_mesh, electrode_configs)
    reference.make()
    reference.solve()
    reference_jacobian = reference.compute_jacobian()

    optimized = PviPhysics(copy.deepcopy(inverse_mesh), electrode_configs)
    optimized_forward = optimized._forward(conductivity)
    optimized_jacobian = optimized.jacobian_from_forward(optimized_forward)

    np.testing.assert_allclose(
        optimized_forward.results.vmeas,
        reference.results.vmeas,
        rtol=0.0,
        # Sparse factorization order can change the final rounding bit while
        # preserving the upstream equations.  Keep this at float64 precision
        # instead of requiring bitwise-identical solver output.
        atol=1e-15,
    )
    np.testing.assert_allclose(
        optimized_jacobian,
        reference_jacobian,
        rtol=2e-11,
        atol=1e-16,
    )
