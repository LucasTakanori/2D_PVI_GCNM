# gcnm_graph.py
"""Build GCNM edge_index from PviMesh2D element adjacency."""

import numpy as np
import scipy.sparse as sp
import torch


def build_edge_index(mesh, connectivity: str = "node") -> torch.Tensor:
    """Element-adjacency edge_index [2, n_graph_edges] for torch_geometric."""
    elems = np.asarray(mesh.elems, dtype=np.int64)
    num_elems = elems.shape[0]
    num_nodes = int(elems.max()) + 1

    rows = np.repeat(np.arange(num_elems), elems.shape[1])
    cols = elems.ravel()
    vals = np.ones(rows.shape[0], dtype=np.int8)
    cNE = sp.csr_matrix((vals, (rows, cols)), shape=(num_elems, num_nodes))
    C = (cNE @ cNE.T).tocoo()

    if connectivity == "node":
        keep = (C.row != C.col) & (C.data >= 1)
    elif connectivity == "edge":
        keep = (C.row != C.col) & (C.data == 2)
    else:
        raise ValueError("connectivity must be 'node' or 'edge'")

    edge_index = np.vstack([C.row[keep], C.col[keep]])
    return torch.tensor(edge_index, dtype=torch.long)
