from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from knitting._peeling.surface import UINT32_MAX, SurfacePoint

Edge = tuple[int, int]


@dataclass(frozen=True, slots=True)
class KnittingGraph:
    points: np.ndarray
    course_edges: np.ndarray
    wale_edges: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        course_edges = np.asarray(self.course_edges, dtype=np.int64).reshape((-1, 2))
        wale_edges = np.asarray(self.wale_edges, dtype=np.int64).reshape((-1, 2))
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("graph points must have shape (n, 3)")
        for edges in (course_edges, wale_edges):
            if len(edges) and (edges.min() < 0 or edges.max() >= len(points)):
                raise ValueError("graph edge references an unknown stitch")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "course_edges", course_edges)
        object.__setattr__(self, "wale_edges", wale_edges)

    @property
    def course_count(self) -> int:
        labels = component_labels(len(self.points), self.course_edges)
        return int(labels.max(initial=-1)) + 1

    @property
    def increase_count(self) -> int:
        counts = np.bincount(self.wale_edges[:, 0], minlength=len(self.points))
        return int(np.count_nonzero(counts == 2))

    @property
    def decrease_count(self) -> int:
        counts = np.bincount(self.wale_edges[:, 1], minlength=len(self.points))
        return int(np.count_nonzero(counts == 2))


@dataclass
class RowColVertex:
    at: SurfacePoint
    row_in: int = UINT32_MAX
    row_out: int = UINT32_MAX
    col_in: list[int] = field(default_factory=lambda: [UINT32_MAX, UINT32_MAX])
    col_out: list[int] = field(default_factory=lambda: [UINT32_MAX, UINT32_MAX])


@dataclass
class RowColGraph:
    vertices: list[RowColVertex] = field(default_factory=list)

    def vertex_count(self) -> int:
        return len(self.vertices)

    def vertex_at(self, index: int) -> RowColVertex:
        self._require_vertex(index)
        return self.vertices[index]

    def column_parents(self, index: int) -> tuple[int, ...]:
        return tuple(parent for parent in self.vertex_at(index).col_in if parent != UINT32_MAX)

    def column_children(self, index: int) -> tuple[int, ...]:
        return tuple(child for child in self.vertex_at(index).col_out if child != UINT32_MAX)

    def row_edges(self) -> tuple[Edge, ...]:
        return tuple(
            (source, int(vertex.row_out))
            for source, vertex in enumerate(self.vertices)
            if vertex.row_out != UINT32_MAX
        )

    def column_edges(self) -> tuple[Edge, ...]:
        return tuple(
            (source, int(target))
            for source, vertex in enumerate(self.vertices)
            for target in vertex.col_out
            if target != UINT32_MAX
        )

    def surface_positions(self, mesh_vertices: np.ndarray) -> np.ndarray:
        if not self.vertices:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray(
            [vertex.at.interpolate(mesh_vertices) for vertex in self.vertices], dtype=np.float64
        )

    def add_vertex(self, vertex: RowColVertex) -> int:
        index = len(self.vertices)
        self.vertices.append(vertex)
        return index

    def link_row(self, source: int, target: int) -> None:
        self._require_vertex(source)
        self._require_vertex(target)
        if source == target:
            raise ValueError("row/column graph cannot create a self row link")
        if self.vertices[source].row_out not in (UINT32_MAX, target):
            raise ValueError("row/column graph vertex already has a different row_out")
        if self.vertices[target].row_in not in (UINT32_MAX, source):
            raise ValueError("row/column graph vertex already has a different row_in")
        self.vertices[source].row_out = target
        self.vertices[target].row_in = source

    def link_column(self, source: int, target: int) -> None:
        self._require_vertex(source)
        self._require_vertex(target)
        if source == target:
            raise ValueError("row/column graph cannot create a self column link")
        _require_slot_capacity(self.vertices[source].col_out, target, "outgoing column")
        _require_slot_capacity(self.vertices[target].col_in, source, "incoming column")
        _add_unique_slot(self.vertices[source].col_out, target, "outgoing column")
        _add_unique_slot(self.vertices[target].col_in, source, "incoming column")

    def validate(self) -> None:
        for index, vertex in enumerate(self.vertices):
            if vertex.row_in == index or vertex.row_out == index:
                raise ValueError(f"graph vertex {index} has a self row link")
            for label, slots in (("column_in", vertex.col_in), ("column_out", vertex.col_out)):
                if len(slots) != 2:
                    raise ValueError(f"graph vertex {index} {label} must contain exactly two slots")
                if slots[0] == UINT32_MAX and slots[1] != UINT32_MAX:
                    raise ValueError(f"graph vertex {index} {label} slots are not left-compressed")
                active = [neighbor for neighbor in slots if neighbor != UINT32_MAX]
                if len(active) != len(set(active)):
                    raise ValueError(f"graph vertex {index} {label} contains a duplicate link")
                if index in active:
                    raise ValueError(f"graph vertex {index} has a self column link")
            if vertex.row_in != UINT32_MAX:
                self._require_vertex(vertex.row_in)
                if self.vertices[vertex.row_in].row_out != index:
                    raise ValueError("row/column graph has a non-reciprocal row_in link")
            if vertex.row_out != UINT32_MAX:
                self._require_vertex(vertex.row_out)
                if self.vertices[vertex.row_out].row_in != index:
                    raise ValueError("row/column graph has a non-reciprocal row_out link")
            for parent in vertex.col_in:
                if parent == UINT32_MAX:
                    continue
                self._require_vertex(parent)
                if index not in self.vertices[parent].col_out:
                    raise ValueError("row/column graph has a non-reciprocal incoming column link")
            for child in vertex.col_out:
                if child == UINT32_MAX:
                    continue
                self._require_vertex(child)
                if index not in self.vertices[child].col_in:
                    raise ValueError("row/column graph has a non-reciprocal outgoing column link")

    def _require_vertex(self, index: int) -> None:
        if index == UINT32_MAX or index < 0 or index >= len(self.vertices):
            raise ValueError("row/column graph vertex index is out of range")


def _require_slot_capacity(slots: list[int], value: int, label: str) -> None:
    if slots[0] == value or slots[1] == value:
        return
    if slots[0] == UINT32_MAX or slots[1] == UINT32_MAX:
        return
    raise ValueError(f"row/column graph vertex has too many {label} links")


def _add_unique_slot(slots: list[int], value: int, label: str) -> None:
    if slots[0] == value or slots[1] == value:
        return
    if slots[0] == UINT32_MAX:
        slots[0] = value
        return
    if slots[1] == UINT32_MAX:
        slots[1] = value
        return
    raise ValueError(f"row/column graph vertex has too many {label} links")


def component_labels(vertex_count: int, edges: np.ndarray) -> np.ndarray:
    """Label connected components of an undirected edge graph."""
    adjacency: list[list[int]] = [[] for _ in range(vertex_count)]
    for first, second in np.asarray(edges, dtype=np.int64).reshape((-1, 2)):
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))

    labels = np.full(vertex_count, -1, dtype=np.int64)
    for seed in range(vertex_count):
        if labels[seed] != -1:
            continue
        label = int(labels.max(initial=-1)) + 1
        labels[seed] = label
        todo = [seed]
        while todo:
            current = todo.pop()
            for neighbor in adjacency[current]:
                if labels[neighbor] == -1:
                    labels[neighbor] = label
                    todo.append(neighbor)
    return labels
