from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from knitting._peeling.surface.point import UINT32_MAX, SurfacePoint

WEIGHT_SUM = 1024
T = TypeVar("T")


@dataclass(frozen=True)
class QuantizedSurfacePoint:
    simplex: tuple[int, int, int]
    weights: tuple[int, int, int]

    @staticmethod
    def from_surface_point(vertex: SurfacePoint) -> QuantizedSurfacePoint:
        x = float(vertex.weights[0])
        xy = x + float(vertex.weights[1])
        xyz = xy + float(vertex.weights[2])
        ix = _round_half_away_from_zero(WEIGHT_SUM * x / xyz)
        ixy = _round_half_away_from_zero(WEIGHT_SUM * xy / xyz)
        ixyz = _round_half_away_from_zero(WEIGHT_SUM)
        return QuantizedSurfacePoint.simplify(
            QuantizedSurfacePoint(vertex.simplex, (ix, ixy - ix, ixyz - ixy))
        )

    @staticmethod
    def simplify(vertex: QuantizedSurfacePoint) -> QuantizedSurfacePoint:
        simplex = [0, UINT32_MAX, UINT32_MAX]
        weights = [WEIGHT_SUM, 0, 0]
        out = 0
        for index, weight in zip(vertex.simplex, vertex.weights, strict=True):
            if weight != 0:
                simplex[out] = index
                weights[out] = weight
                out += 1
        return QuantizedSurfacePoint(tuple(simplex), tuple(weights))

    def weights_on(self, simplex: tuple[int, int, int]) -> tuple[int, int, int]:
        out = [0, 0, 0]
        for index, weight in zip(self.simplex, self.weights, strict=True):
            if index == UINT32_MAX:
                continue
            out[simplex.index(index)] = weight
        return tuple(out)

    def to_surface_point(self) -> SurfacePoint:
        return SurfacePoint(
            self.simplex,
            tuple(weight / float(WEIGHT_SUM) for weight in self.weights),
        )

    @staticmethod
    def common_simplex(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
        return SurfacePoint.common_simplex(a, b)


@dataclass
class TriangleSegment(Generic[T]):
    """A directed segment inside one triangle, carrying trim traversal state."""

    first: int
    second: int
    value: T


class TriangleSplit(Generic[T]):
    """Triangle-local segment arrangement used to cut a mesh along surface chains."""

    def __init__(
        self,
        reverse_value: Callable[[T], T] | None = None,
        combine_values: Callable[[T, T], T] | None = None,
        split_value: Callable[[T], tuple[T, T]] | None = None,
    ) -> None:
        self.vertices: list[QuantizedSurfacePoint] = []
        self.simplex_vertices: dict[tuple[int, int, int], list[int]] = {}
        self.simplex_edges: dict[tuple[int, int, int], list[TriangleSegment[T]]] = {}
        self.reverse_value = reverse_value or (lambda value: value)
        self.combine_values = combine_values or (lambda _target, incoming: incoming)
        self.split_value = split_value or (lambda value: (value, value))

    def add_vertex(self, vertex: SurfacePoint | QuantizedSurfacePoint) -> int:
        if isinstance(vertex, SurfacePoint):
            integer_vertex = QuantizedSurfacePoint.from_surface_point(vertex)
        else:
            integer_vertex = QuantizedSurfacePoint.simplify(vertex)

        verts = self.simplex_vertices.setdefault(integer_vertex.simplex, [])
        edges = self.simplex_edges.setdefault(integer_vertex.simplex, [])
        for index in verts:
            if self.vertices[index] == integer_vertex:
                return index

        index = len(self.vertices)
        self.vertices.append(integer_vertex)
        verts.append(index)

        old_size = len(edges)
        for edge_index in range(old_size):
            edge = edges[edge_index]
            if self._point_in_segment_vertices(
                integer_vertex, self.vertices[edge.first], self.vertices[edge.second]
            ):
                second_half = TriangleSegment(index, edge.second, edge.value)
                edge.second = index
                edges.append(second_half)
        return index

    def add_edge(self, a: int | SurfacePoint, b: int | SurfacePoint, value: T) -> None:
        if isinstance(a, SurfacePoint):
            a = self.add_vertex(a)
        if isinstance(b, SurfacePoint):
            b = self.add_vertex(b)
        self._add_edge_indices(a, b, value)

    def _add_edge_indices(self, ai: int, bi: int, value: T) -> None:
        if ai == bi:
            return

        a_vertex = self.vertices[ai]
        b_vertex = self.vertices[bi]
        common = QuantizedSurfacePoint.common_simplex(a_vertex.simplex, b_vertex.simplex)
        a = _ivec2(a_vertex.weights_on(common))
        b = _ivec2(b_vertex.weights_on(common))

        for vi in list(self.simplex_vertices.setdefault(common, [])):
            v = _ivec2(self.vertices[vi].weights_on(common))
            if _point_in_segment(v, a, b):
                first_value, second_value = self.split_value(value)
                self._add_edge_indices(ai, vi, first_value)
                self._add_edge_indices(vi, bi, second_value)
                return

        edges = self.simplex_edges.setdefault(common, [])
        edge_index = 0
        while edge_index < len(edges):
            edge = edges[edge_index]
            if edge.first == ai and edge.second == bi:
                edge.value = self.combine_values(edge.value, value)
                return
            if edge.first == bi and edge.second == ai:
                edge.value = self.combine_values(edge.value, self.reverse_value(value))
                return

            a2 = _ivec2(self.vertices[edge.first].weights_on(common))
            b2 = _ivec2(self.vertices[edge.second].weights_on(common))

            if _segments_intersect(a, b, a2, b2):
                point = _rounded_intersection(a, b, a2, b2)
                point3 = (point[0], point[1], WEIGHT_SUM - point[0] - point[1])
                point_index = self.add_vertex(QuantizedSurfacePoint(common, point3))

                old_first = edge.first
                old_second = edge.second
                old_value = edge.value
                del edges[edge_index]

                old_first_value, old_second_value = self.split_value(old_value)
                self._add_edge_indices(old_first, point_index, old_first_value)
                self._add_edge_indices(point_index, old_second, old_second_value)

                first_value, second_value = self.split_value(value)
                self._add_edge_indices(ai, point_index, first_value)
                self._add_edge_indices(point_index, bi, second_value)
                return
            edge_index += 1

        edges.append(TriangleSegment(ai, bi, value))

    def split_triangles(
        self,
        vertex_count: int,
        faces,
        *,
        debug_faces: set[int] | None = None,
    ) -> tuple[list[SurfacePoint], list[tuple[int, int, int]], list[int]]:
        split_vertices = [SurfacePoint.on_vertex(index) for index in range(vertex_count)]
        point_to_split_vertex: list[int] = []
        for vertex in self.vertices:
            if vertex.simplex[1] == UINT32_MAX:
                point_to_split_vertex.append(vertex.simplex[0])
            else:
                point_to_split_vertex.append(len(split_vertices))
                split_vertices.append(vertex.to_surface_point())

        split_faces: list[tuple[int, int, int]] = []

        for face_index, face in enumerate(faces):
            split_face_start = len(split_faces)
            simplex = list(map(int, face))
            need_flip = False
            if simplex[0] > simplex[1]:
                simplex[0], simplex[1] = simplex[1], simplex[0]
                need_flip = not need_flip
            if simplex[1] > simplex[2]:
                simplex[1], simplex[2] = simplex[2], simplex[1]
                need_flip = not need_flip
            if simplex[0] > simplex[1]:
                simplex[0], simplex[1] = simplex[1], simplex[0]
                need_flip = not need_flip
            simplex_tuple = tuple(simplex)

            source_vertices: list[int] = []
            coords: list[tuple[int, int]] = []
            half_edges: list[tuple[int, int]] = []
            split_to_coord: dict[int, int] = {}

            def ref_vertex(
                split_vertex: int,
                weights_on: tuple[int, int, int],
                split_to_coord: dict[int, int] = split_to_coord,
                coords: list[tuple[int, int]] = coords,
                source_vertices: list[int] = source_vertices,
            ) -> int:
                if split_vertex in split_to_coord:
                    return split_to_coord[split_vertex]
                coord = len(coords)
                split_to_coord[split_vertex] = coord
                coords.append(_ivec2(weights_on))
                source_vertices.append(split_vertex)
                return coord

            for edge in self.simplex_edges.get(simplex_tuple, []):
                a = ref_vertex(
                    point_to_split_vertex[edge.first],
                    self.vertices[edge.first].weights_on(simplex_tuple),
                )
                b = ref_vertex(
                    point_to_split_vertex[edge.second],
                    self.vertices[edge.second].weights_on(simplex_tuple),
                )
                half_edges.append((a, b))
                half_edges.append((b, a))

            def do_side(
                ifrom: int,
                ito: int,
                simplex_tuple: tuple[int, int, int] = simplex_tuple,
                half_edges: list[tuple[int, int]] = half_edges,
            ) -> None:
                edge_simplex = (
                    min(simplex_tuple[ifrom], simplex_tuple[ito]),
                    max(simplex_tuple[ifrom], simplex_tuple[ito]),
                    UINT32_MAX,
                )
                from_weights = [0, 0, 0]
                from_weights[ifrom] = WEIGHT_SUM
                to_weights = [0, 0, 0]
                to_weights[ito] = WEIGHT_SUM
                from_coord = ref_vertex(simplex_tuple[ifrom], tuple(from_weights))
                to_coord = ref_vertex(simplex_tuple[ito], tuple(to_weights))

                weight_to_vertex = {0: from_coord, WEIGHT_SUM: to_coord}
                for sv in self.simplex_vertices.get(edge_simplex, []):
                    weights = self.vertices[sv].weights_on(simplex_tuple)
                    weight_to_vertex[weights[ito]] = ref_vertex(point_to_split_vertex[sv], weights)

                ordered = sorted(weight_to_vertex.items())
                previous = ordered[0][1]
                for _, current in ordered[1:]:
                    half_edges.append((previous, current))
                    previous = current

            do_side(0, 1)
            do_side(1, 2)
            do_side(2, 0)

            if debug_faces and face_index in debug_faces:
                print(
                    f"split face {face_index} original={tuple(map(int, face))} "
                    f"sorted={simplex_tuple} need_flip={need_flip}"
                )
                print("  coords/source vertices:")
                for coord_index, (coord, source) in enumerate(
                    zip(coords, source_vertices, strict=True)
                ):
                    print(f"    {coord_index}: coord={coord} split={source}")
                print("  half edges:")
                for edge in half_edges:
                    print(f"    {edge}")

            while half_edges:
                loop = [half_edges[0][0], half_edges[0][1]]

                while True:
                    from_coord = coords[loop[-2]]
                    at_coord = coords[loop[-1]]
                    candidates = [
                        (idx, edge[1]) for idx, edge in enumerate(half_edges) if edge[0] == loop[-1]
                    ]
                    if not candidates:
                        raise ValueError("ran out of half-edges while triangulating triangle split")

                    best_index = -1
                    best_to = -1
                    best_d = (0, 0)
                    best_quad = 4
                    for idx, candidate_to in candidates:
                        next_coord = coords[candidate_to]
                        d = (
                            (next_coord[0] - at_coord[0]) * (at_coord[0] - from_coord[0])
                            + (next_coord[1] - at_coord[1]) * (at_coord[1] - from_coord[1]),
                            (next_coord[0] - at_coord[0]) * -(at_coord[1] - from_coord[1])
                            + (next_coord[1] - at_coord[1]) * (at_coord[0] - from_coord[0]),
                        )
                        if d == (0, 0):
                            continue
                        if d[0] <= 0 and d[1] > 0:
                            quad = 0
                            d = (d[1], -d[0])
                        elif d[0] > 0 and d[1] >= 0:
                            quad = 1
                        elif d[0] >= 0 and d[1] < 0:
                            quad = 2
                            d = (-d[1], d[0])
                        elif d[0] < 0 and d[1] <= 0:
                            quad = 3
                            d = (-d[0], -d[1])
                        else:
                            raise AssertionError("unhandled quadrant")

                        if quad < best_quad or (
                            quad == best_quad and d[1] * best_d[0] > best_d[1] * d[0]
                        ):
                            best_index = idx
                            best_to = candidate_to
                            best_quad = quad
                            best_d = d

                    if best_index == -1:
                        raise ValueError("failed to choose next half-edge")
                    best_edge = half_edges[best_index]
                    del half_edges[best_index]
                    if best_edge[0] == loop[0] and best_edge[1] == loop[1]:
                        break
                    loop.append(best_to)

                if loop[0] != loop[-1]:
                    raise ValueError("triangle split loop did not close")
                loop.pop()
                if len(loop) < 3:
                    raise ValueError("triangle split produced a degenerate loop")

                remain = list(loop)
                while len(remain) >= 3:
                    found = False
                    for i in range(len(remain)):
                        prev_i = i - 1 if i > 0 else len(remain) - 1
                        next_i = i + 1 if i + 1 < len(remain) else 0
                        prev = coords[remain[prev_i]]
                        at = coords[remain[i]]
                        next_coord = coords[remain[next_i]]

                        perp_dot = -(next_coord[1] - at[1]) * (prev[0] - at[0]) + (
                            next_coord[0] - at[0]
                        ) * (prev[1] - at[1])
                        if perp_dot <= 0:
                            continue
                        inside = False
                        if len(remain) > 3:
                            for j in range(len(remain)):
                                if remain[j] in {remain[prev_i], remain[i], remain[next_i]}:
                                    continue
                                point = coords[remain[j]]
                                right_dot = -(next_coord[1] - at[1]) * (point[0] - at[0]) + (
                                    next_coord[0] - at[0]
                                ) * (point[1] - at[1])
                                if right_dot < 0:
                                    continue
                                left_dot = -(at[1] - prev[1]) * (point[0] - prev[0]) + (
                                    at[0] - prev[0]
                                ) * (point[1] - prev[1])
                                if left_dot < 0:
                                    continue
                                top_dot = -(prev[1] - next_coord[1]) * (
                                    point[0] - next_coord[0]
                                ) + (prev[0] - next_coord[0]) * (point[1] - next_coord[1])
                                if top_dot < 0:
                                    continue
                                inside = True
                                break
                        if not inside:
                            tri = (
                                source_vertices[remain[prev_i]],
                                source_vertices[remain[i]],
                                source_vertices[remain[next_i]],
                            )
                            if need_flip:
                                tri = (tri[0], tri[2], tri[1])
                            split_faces.append(tri)
                            del remain[i]
                            found = True
                            break
                    if not found:
                        raise ValueError("failed to find an ear while triangulating triangle split")

            if debug_faces and face_index in debug_faces:
                print("  produced:")
                for tri in split_faces[split_face_start:]:
                    print(f"    {tri}")

        return split_vertices, split_faces, point_to_split_vertex

    def _point_in_segment_vertices(
        self,
        point: QuantizedSurfacePoint,
        a: QuantizedSurfacePoint,
        b: QuantizedSurfacePoint,
    ) -> bool:
        common = QuantizedSurfacePoint.common_simplex(
            point.simplex, QuantizedSurfacePoint.common_simplex(a.simplex, b.simplex)
        )
        return _point_in_segment(
            _ivec2(point.weights_on(common)),
            _ivec2(a.weights_on(common)),
            _ivec2(b.weights_on(common)),
        )


def _round_half_away_from_zero(value: float) -> int:
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def _ivec2(weights: tuple[int, int, int]) -> tuple[int, int]:
    return (weights[0], weights[1])


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] - b[0], a[1] - b[1])


def _point_in_segment(point: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> bool:
    ab = _sub(b, a)
    ap = _sub(point, a)
    perp = ap[0] * -ab[1] + ap[1] * ab[0]
    if perp != 0:
        return False
    along = ap[0] * ab[0] + ap[1] * ab[1]
    if along <= 0:
        return False
    limit = ab[0] * ab[0] + ab[1] * ab[1]
    return along < limit


def _segments_intersect(
    a: tuple[int, int], b: tuple[int, int], a2: tuple[int, int], b2: tuple[int, int]
) -> bool:
    if a == b or a2 == b2:
        return False

    ba = _sub(b, a)
    a2a = _sub(a2, a)
    b2a = _sub(b2, a)
    perp_a2 = a2a[0] * -ba[1] + a2a[1] * ba[0]
    perp_b2 = b2a[0] * -ba[1] + b2a[1] * ba[0]

    if perp_a2 == 0 and perp_b2 == 0:
        along_a2 = a2a[0] * ba[0] + a2a[1] * ba[1]
        along_b2 = b2a[0] * ba[0] + b2a[1] * ba[1]
        limit = ba[0] * ba[0] + ba[1] * ba[1]
        if along_a2 <= 0 and along_b2 <= 0:
            return False
        return along_a2 < limit or along_b2 < limit

    if perp_a2 <= 0 and perp_b2 <= 0:
        return False
    if perp_a2 >= 0 and perp_b2 >= 0:
        return False

    b2a2 = _sub(b2, a2)
    aa2 = _sub(a, a2)
    ba2 = _sub(b, a2)
    perp_a = aa2[0] * -b2a2[1] + aa2[1] * b2a2[0]
    perp_b = ba2[0] * -b2a2[1] + ba2[1] * b2a2[0]

    if perp_a <= 0 and perp_b <= 0:
        return False
    return perp_a < 0 or perp_b < 0


def _rounded_intersection(
    a: tuple[int, int], b: tuple[int, int], a2: tuple[int, int], b2: tuple[int, int]
) -> tuple[int, int]:
    ba = _sub(b, a)
    a2a = _sub(a2, a)
    b2a = _sub(b2, a)
    perp_a2 = a2a[0] * -ba[1] + a2a[1] * ba[0]
    perp_b2 = b2a[0] * -ba[1] + b2a[1] * ba[0]
    if perp_a2 == perp_b2:
        raise ValueError("cannot compute rounded intersection for parallel edges")
    t = float(0 - perp_a2) / float(perp_b2 - perp_a2)
    return (
        _round_half_away_from_zero((b[0] - a[0]) * t + float(a[0])),
        _round_half_away_from_zero((b[1] - a[1]) * t + float(a[1])),
    )
