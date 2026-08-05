"""Triangle-mesh I/O, boundary topology, and intrinsic FEM geometry."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Mesh:
    vertices: FloatArray
    faces: IntArray

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (V, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if not np.isfinite(self.vertices).all():
            raise ValueError("vertices contain NaN or infinite values")
        if len(self.vertices) == 0 or len(self.faces) == 0:
            raise ValueError("mesh must contain vertices and faces")
        if self.faces.min() < 0 or self.faces.max() >= len(self.vertices):
            raise ValueError("faces reference a vertex outside the mesh")


def load_obj(path: str | Path) -> Mesh:
    """Load OBJ positions and polygon faces, triangulating polygons by a fan."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    source = Path(path)
    for line in source.read_text(encoding="utf-8", errors="strict").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "v":
            if len(fields) < 4:
                raise ValueError(f"invalid vertex in {source}")
            vertices.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "f":
            if len(fields) < 4:
                raise ValueError(f"face has fewer than three vertices in {source}")
            polygon: list[int] = []
            for token in fields[1:]:
                raw = int(token.split("/", 1)[0])
                index = raw - 1 if raw > 0 else len(vertices) + raw
                polygon.append(index)
            faces.extend(
                (polygon[0], polygon[index], polygon[index + 1])
                for index in range(1, len(polygon) - 1)
            )
    return Mesh(
        np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    )


def boundary_loop(faces: IntArray) -> IntArray:
    """Return the sole manifold boundary loop, in cyclic vertex order."""
    counts: Counter[tuple[int, int]] = Counter()
    for triangle in np.asarray(faces, dtype=np.int64):
        for start, end in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            counts[tuple(sorted((start, end)))] += 1
    if any(count > 2 for count in counts.values()):
        raise ValueError("mesh is not edge-manifold")
    boundary_edges = [edge for edge, count in counts.items() if count == 1]
    if not boundary_edges:
        raise ValueError("mesh has no boundary")

    adjacency: defaultdict[int, list[int]] = defaultdict(list)
    for start, end in boundary_edges:
        adjacency[start].append(end)
        adjacency[end].append(start)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("mesh boundary is not a collection of manifold loops")

    start = min(adjacency)
    loop = [start]
    previous = -1
    current = start
    while True:
        first, second = adjacency[current]
        following = first if first != previous else second
        if following == start:
            break
        if following in loop:
            raise ValueError("boundary contains a repeated vertex before closing")
        loop.append(following)
        previous, current = current, following
    if len(loop) != len(adjacency):
        raise ValueError("optimizer requires exactly one boundary loop")
    return np.asarray(loop, dtype=np.int64)


def boundary_arclength(vertices: FloatArray, loop: IntArray) -> FloatArray:
    """Normalized cumulative arc length at each ordered boundary vertex."""
    following = np.roll(loop, -1)
    lengths = np.linalg.norm(vertices[following] - vertices[loop], axis=1)
    perimeter = float(lengths.sum())
    if not np.isfinite(perimeter) or perimeter <= 0.0 or np.any(lengths <= 0.0):
        raise ValueError("boundary loop contains a zero or invalid edge")
    return np.concatenate(([0.0], np.cumsum(lengths[:-1]))) / perimeter


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
    matrix = scipy.sparse.coo_matrix(
        (data, (rows, columns)), shape=(len(mesh.vertices), len(mesh.vertices))
    )
    return matrix.tocsr()


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
