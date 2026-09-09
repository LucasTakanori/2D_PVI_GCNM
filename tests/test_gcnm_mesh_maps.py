from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from gcnm_pvi.gcnm_mesh_maps import MeshMappings


def test_categorical_rasterization_uses_maximum_overlap_not_mean_label() -> None:
    mappings = MeshMappings.__new__(MeshMappings)
    mappings.img_size = 2
    mappings.m2i = csr_matrix(
        np.asarray(
            [
                [0.30, 0.70, 0.00],
                [0.60, 0.40, 0.00],
                [0.00, 0.00, 0.00],
                [0.00, 0.00, 1.00],
            ]
        )
    )
    labels = np.asarray([3, 5, 4], dtype=np.uint8)

    continuous_average = mappings.elem_to_image(labels)
    assert np.rint(continuous_average[0]) == 4

    categorical = mappings.categorical_to_image_grid(labels, num_classes=9)

    np.testing.assert_array_equal(
        categorical,
        np.asarray([[5, 0], [3, 4]], dtype=np.uint8),
    )
