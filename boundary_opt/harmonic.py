"""Harmonic extension on a triangle mesh with one boundary loop."""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg

from .mesh import FloatArray, Mesh, boundary_loop, cotangent_stiffness


class HarmonicField:
    """Prefactorized harmonic solve and its adjoint."""

    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh
        self.boundary_vertices = boundary_loop(mesh.faces)
        interior = np.ones(len(mesh.vertices), dtype=bool)
        interior[self.boundary_vertices] = False
        self.interior_vertices = np.flatnonzero(interior).astype(np.int64)
        if len(self.interior_vertices) == 0:
            raise ValueError("mesh has no interior vertices")

        stiffness = cotangent_stiffness(mesh)
        interior_block = stiffness[self.interior_vertices][
            :, self.interior_vertices
        ].tocsc()
        self._coupling = stiffness[self.interior_vertices][
            :, self.boundary_vertices
        ].tocsr()
        try:
            self._factorization = scipy.sparse.linalg.splu(interior_block)
        except RuntimeError as error:
            raise ValueError(
                "mesh interior is not connected to the boundary"
            ) from error

    def solve(self, boundary_values: FloatArray) -> FloatArray:
        """Extend one value per ordered boundary vertex harmonically."""
        boundary_values = np.asarray(boundary_values, dtype=np.float64).reshape(-1)
        if boundary_values.shape != self.boundary_vertices.shape:
            raise ValueError(
                "boundary_values must contain one value per boundary vertex"
            )
        values = np.empty(len(self.mesh.vertices), dtype=np.float64)
        values[self.boundary_vertices] = boundary_values
        values[self.interior_vertices] = self._factorization.solve(
            -(self._coupling @ boundary_values)
        )
        return values

    def solve_adjoint(self, sensitivity: FloatArray) -> FloatArray:
        """Map a full-field sensitivity back to boundary values."""
        sensitivity = np.asarray(sensitivity, dtype=np.float64).reshape(-1)
        if sensitivity.shape != (len(self.mesh.vertices),):
            raise ValueError("sensitivity must contain one value per vertex")
        interior_adjoint = self._factorization.solve(
            sensitivity[self.interior_vertices], trans="T"
        )
        return np.asarray(
            sensitivity[self.boundary_vertices] - self._coupling.T @ interior_adjoint,
            dtype=np.float64,
        )
