# tiny_mesh.py
"""Small in-memory 8-el circle mesh for smoke tests only."""

import numpy as np
from scipy.spatial import Delaunay

from gcnm_pvi.paths import ensure_pvi_solver_on_path

ensure_pvi_solver_on_path()
from pvi_mesh2d import PviElec2D, PviMesh2DBase  # noqa: E402


def make_tiny_circle_mesh(num_boundary=32, num_elecs=8, radius=1.0, contact_impedance=0.01):
    assert num_boundary % num_elecs == 0
    span = num_boundary // num_elecs

    t = np.linspace(0, 2 * np.pi, num_boundary, endpoint=False)
    pts = [np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)]
    for frac, n in [(0.72, 20), (0.45, 12), (0.2, 6)]:
        tt = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.5 * np.pi / n
        pts.append(
            np.stack([frac * radius * np.cos(tt), frac * radius * np.sin(tt)], axis=1)
        )
    pts.append(np.zeros((1, 2)))
    nodes = np.vstack(pts)

    elems = Delaunay(nodes).simplices.astype(np.int64)
    v1 = nodes[elems[:, 1]] - nodes[elems[:, 0]]
    v2 = nodes[elems[:, 2]] - nodes[elems[:, 0]]
    cw = (v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) < 0
    elems[cw] = elems[cw][:, [0, 2, 1]]

    links = np.stack(
        [np.arange(num_boundary), (np.arange(num_boundary) + 1) % num_boundary], axis=1
    )

    elecs = []
    for i in range(num_elecs):
        link_idx = np.arange(span * i, span * i + span - 1)
        node_idx = np.arange(span * i, span * i + span)
        elecs.append(
            PviElec2D(
                impedance=contact_impedance,
                nodes=node_idx,
                elems=np.array([]),
                links=link_idx,
            )
        )

    mesh = PviMesh2DBase()
    mesh.name = "tiny_circle_smoketest"
    mesh.nodes = nodes
    mesh.elems = elems
    mesh.links = links
    mesh.nodes_idx_boundary = np.arange(num_boundary)
    mesh.elems_idx_boundary = np.array([])
    mesh.links_idx_boundary = np.arange(num_boundary)
    mesh.elems_data = np.ones(len(elems))
    mesh.elecs = elecs
    return mesh
