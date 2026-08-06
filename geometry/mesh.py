"""Triangle-mesh data, OBJ input, normalization, and boundary topology."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Mesh:
    vertices: FloatArray
    faces: IntArray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (V, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if not np.isfinite(vertices).all():
            raise ValueError("vertices contain NaN or infinite values")
        if len(vertices) == 0:
            raise ValueError("mesh must contain vertices")
        if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("faces reference a vertex outside the mesh")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)


def normalize_mesh(mesh: Mesh, scale: float) -> Mesh:
    """Center a mesh and set its longest bounding-box side to ``scale``."""
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("normalization scale must be positive and finite")
    lower = mesh.vertices.min(axis=0)
    upper = mesh.vertices.max(axis=0)
    extent = float((upper - lower).max())
    if extent <= 0.0:
        raise ValueError("mesh has zero spatial extent")
    return Mesh(scale * (mesh.vertices - 0.5 * (lower + upper)) / extent, mesh.faces)


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
            polygon = []
            for token in fields[1:]:
                raw = int(token.split("/", 1)[0])
                polygon.append(raw - 1 if raw > 0 else len(vertices) + raw)
            faces.extend(
                (polygon[0], polygon[index], polygon[index + 1])
                for index in range(1, len(polygon) - 1)
            )
    return Mesh(np.asarray(vertices), np.asarray(faces))


def boundary_loop(faces: IntArray) -> IntArray:
    """Return the sole manifold boundary loop in cyclic vertex order."""
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
        raise ValueError("mesh must have exactly one boundary loop")
    return np.asarray(loop, dtype=np.int64)
