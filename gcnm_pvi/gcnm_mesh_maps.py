"""Load FEM mapping matrices (m2i, laplace, c2f) from PVI HDF5 exports."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def load_sparse_matrix(h5_path: Path | str, name: str) -> csr_matrix:
    """Load mat2D/{name} COO stored by export_matrix_hdf5.m (1-based indices)."""
    with h5py.File(h5_path, "r") as f:
        group = f[f"mat2D/{name}"]
        rows = group["rows"][:] - 1
        cols = group["cols"][:] - 1
        values = group["values"][:]
        size = tuple(group["size"][:].flatten())
    return coo_matrix((values.transpose(), (cols, rows)), shape=size).tocsr()


class MeshMappings:
    def __init__(self, mappings_h5: Path | str, img_size: int | None = None):
        self.path = Path(mappings_h5)
        self.m2i = load_sparse_matrix(self.path, "m2i")
        self.laplace = load_sparse_matrix(self.path, "laplace")
        try:
            self.c2f = load_sparse_matrix(self.path, "c2f")
        except KeyError:
            self.c2f = None
        n_pix = self.m2i.shape[0]
        side = int(round(np.sqrt(n_pix)))
        if side * side != n_pix:
            raise ValueError(f"m2i rows {n_pix} is not a perfect square")
        self.img_size = img_size or side
        if self.img_size != side:
            raise ValueError(f"img_size {self.img_size} != sqrt(m2i rows)={side}")

    @property
    def num_pixels(self) -> int:
        return self.m2i.shape[0]

    @property
    def num_elements(self) -> int:
        return self.m2i.shape[1]

    def elem_to_image(self, sigma_elem: np.ndarray) -> np.ndarray:
        """Map element conductivity (K,) or (K,T) to flat image pixels."""
        s = np.asarray(sigma_elem, dtype=np.float64)
        if s.ndim == 1:
            return np.asarray(self.m2i @ s).ravel()
        return np.asarray(self.m2i @ s)

    def elem_to_image_grid(self, sigma_elem: np.ndarray) -> np.ndarray:
        """Return (H, W) or (H, W, T) image."""
        flat = self.elem_to_image(sigma_elem)
        n = self.img_size
        if flat.ndim == 1:
            return flat.reshape(n, n, order="F")
        return flat.reshape(n, n, flat.shape[1], order="F")

    def rtr(self) -> np.ndarray:
        """Regularization matrix R^T R on inverse mesh elements."""
        R = self.laplace
        return (R.T @ R).toarray()
