"""Finite-element geometry used by harmonic boundary optimization."""

from __future__ import annotations

import numpy as np
import scipy.sparse

from geometry import Mesh
from geometry.mesh import FloatArray


def cotangent_stiffness(mesh: Mesh) -> scipy.sparse.csr_matrix:
    """Positive-semidefinite cotangent stiffness matrix."""
    triangles = mesh.vertices[mesh.faces]
    edge01 = triangles[:, 1] - triangles[:, 0]
    edge02 = triangles[:, 2] - triangles[:, 0]
    double_area = np.linalg.norm(np.cross(edge01, edge02), axis=1)
    scale = np.linalg.norm(edge01, axis=1) * np.linalg.norm(edge02, axis=1)
    if np.any(double_area <= np.finfo(np.float64).eps * scale * 16.0):
        raise ValueError("mesh contains a degenerate triangle")

    cot0 = np.einsum("ij,ij->i", edge01, edge02) / double_area
    edge10 = triangles[:, 0] - triangles[:, 1]
    edge12 = triangles[:, 2] - triangles[:, 1]
    cot1 = np.einsum("ij,ij->i", edge10, edge12) / double_area
    edge20 = triangles[:, 0] - triangles[:, 2]
    edge21 = triangles[:, 1] - triangles[:, 2]
    cot2 = np.einsum("ij,ij->i", edge20, edge21) / double_area

    starts = np.concatenate((mesh.faces[:, 1], mesh.faces[:, 2], mesh.faces[:, 0]))
    ends = np.concatenate((mesh.faces[:, 2], mesh.faces[:, 0], mesh.faces[:, 1]))
    weights = 0.5 * np.concatenate((cot0, cot1, cot2))
    rows = np.concatenate((starts, starts, ends, ends))
    columns = np.concatenate((starts, ends, starts, ends))
    data = np.concatenate((weights, -weights, -weights, weights))
    return scipy.sparse.coo_matrix(
        (data, (rows, columns)), shape=(len(mesh.vertices), len(mesh.vertices))
    ).tocsr()


def face_gradient_basis(mesh: Mesh) -> tuple[FloatArray, FloatArray]:
    """Return face areas and gradients of each triangle's barycentric basis."""
    triangles = mesh.vertices[mesh.faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    double_area = np.linalg.norm(cross, axis=1)
    scale = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1) * np.linalg.norm(
        triangles[:, 2] - triangles[:, 0], axis=1
    )
    if np.any(double_area <= np.finfo(np.float64).eps * scale * 16.0):
        raise ValueError("mesh contains a degenerate triangle")
    normals = cross / double_area[:, None]
    basis = (
        np.stack(
            (
                np.cross(normals, triangles[:, 2] - triangles[:, 1]),
                np.cross(normals, triangles[:, 0] - triangles[:, 2]),
                np.cross(normals, triangles[:, 1] - triangles[:, 0]),
            ),
            axis=1,
        )
        / double_area[:, None, None]
    )
    return 0.5 * double_area, basis
