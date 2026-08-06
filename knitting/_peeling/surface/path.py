from __future__ import annotations

import math
from collections import defaultdict
from heapq import heappop, heappush
from itertools import pairwise

import numpy as np

from geometry import Mesh
from knitting._peeling.surface.point import UINT32_MAX, SurfacePoint


def find_surface_path(
    max_spacing: float,
    mesh: Mesh,
    source: SurfacePoint,
    target: SurfacePoint,
) -> list[SurfacePoint]:
    if not math.isfinite(max_spacing) or max_spacing <= 0.0:
        raise ValueError("max_spacing must be positive and finite")
    if source == target:
        return [source]

    vertices = np.asarray(mesh.vertices, dtype=float)
    edges = _ordered_edges(mesh.faces)
    adjacency: list[list[int]] = [[] for _ in range(len(vertices))]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    target_index = target.simplex[0]
    distances = [math.inf] * len(vertices)
    heap: list[tuple[float, int, float, int, float]] = []

    def queue(vertex: int, distance: float) -> None:
        distance = float(distance)
        distances[vertex] = distance
        heuristic = float(np.linalg.norm(vertices[target_index] - vertices[vertex]))
        _heap_push(heap, vertex, distance, heuristic)

    source_position = source.interpolate(vertices)
    target_position = target.interpolate(vertices)
    queue(
        source.simplex[0],
        float(np.linalg.norm(source_position - vertices[source.simplex[0]])),
    )
    while heap:
        _, _, _, vertex, distance = heappop(heap)
        if distance > distances[vertex]:
            continue
        if vertex == target_index:
            break
        for neighbor in adjacency[vertex]:
            next_distance = float(distance + np.linalg.norm(vertices[neighbor] - vertices[vertex]))
            if next_distance < distances[neighbor]:
                queue(neighbor, next_distance)

    bound = float(
        distances[target_index] + np.linalg.norm(target_position - vertices[target_index])
    )
    bound2 = float(bound * bound)

    trimmed_vertices: list[np.ndarray] = []
    trimmed_faces: list[tuple[int, int, int]] = []
    to_trimmed = [UINT32_MAX] * len(vertices)
    from_trimmed: list[int] = []

    def vertex_to_trimmed(vertex: int) -> int:
        if to_trimmed[vertex] == UINT32_MAX:
            to_trimmed[vertex] = len(trimmed_vertices)
            from_trimmed.append(vertex)
            trimmed_vertices.append(vertices[vertex])
        return to_trimmed[vertex]

    for face in mesh.faces:
        x, y, z = map(int, face)
        face_vertices = np.asarray([vertices[x], vertices[y], vertices[z]], dtype=float)
        lower = np.minimum(face_vertices[0], np.minimum(face_vertices[1], face_vertices[2]))
        upper = np.maximum(face_vertices[0], np.maximum(face_vertices[1], face_vertices[2]))
        source_delta = np.maximum(lower, np.minimum(upper, source_position)) - source_position
        target_delta = np.maximum(lower, np.minimum(upper, target_position)) - target_position
        if float(_length2(source_delta) + _length2(target_delta)) < bound2:
            trimmed_faces.append((vertex_to_trimmed(x), vertex_to_trimmed(y), vertex_to_trimmed(z)))

    trimmed_source = _remap_surface_point(source, vertex_to_trimmed)
    trimmed_target = _remap_surface_point(target, vertex_to_trimmed)
    trimmed = Mesh(
        np.asarray(trimmed_vertices, dtype=float), np.asarray(trimmed_faces, dtype=np.int64)
    )
    trimmed_path = _find_path_on_trimmed_surface(
        max_spacing, trimmed, trimmed_source, trimmed_target
    )

    path: list[SurfacePoint] = []
    for vertex in trimmed_path:
        simplex = list(vertex.simplex)
        for index, source_index in enumerate(simplex):
            if source_index != UINT32_MAX:
                simplex[index] = from_trimmed[source_index]
        path.append(SurfacePoint.canonicalize(tuple(simplex), vertex.weights))

    sampled: list[SurfacePoint] = []
    for current, next_point in pairwise(path):
        sampled.append(current)
        length = float(
            np.linalg.norm(next_point.interpolate(vertices) - current.interpolate(vertices))
        )
        segment_count = max(1, math.ceil(length / max_spacing))
        sampled.extend(
            SurfacePoint.mix(current, next_point, index / float(segment_count))
            for index in range(1, segment_count)
        )
    sampled.append(path[-1])
    return sampled


def _find_path_on_trimmed_surface(
    max_spacing: float,
    mesh: Mesh,
    source: SurfacePoint,
    target: SurfacePoint,
) -> list[SurfacePoint]:
    loc_vertices: list[SurfacePoint] = []
    loc_positions: list[np.ndarray] = []
    vertices = np.asarray(mesh.vertices, dtype=float)

    vertex_locs: list[int] = []
    for vertex_index in range(len(vertices)):
        vertex_locs.append(len(loc_vertices))
        loc_vertices.append(SurfacePoint.on_vertex(vertex_index))
        loc_positions.append(vertices[vertex_index])

    edge_to_triangles: dict[tuple[int, int], list[int]] = defaultdict(list)
    simplex_to_triangle: dict[tuple[int, int, int], int] = {}
    edge_keys: set[tuple[int, int]] = set()

    for triangle_index, face in enumerate(mesh.faces):
        x, y, z = map(int, face)
        for a, b in ((x, y), (y, z), (z, x)):
            if a > b:
                a, b = b, a
            edge_to_triangles[(a, b)].append(triangle_index)
            edge_keys.add((a, b))
        simplex = tuple(sorted((x, y, z)))
        simplex_to_triangle[simplex] = triangle_index

    edge_locs: dict[tuple[int, int], tuple[int, int]] = {}
    max_spacing = float(max_spacing)
    for a, b in sorted(edge_keys):
        length = float(np.linalg.norm(vertices[b] - vertices[a]))
        segment_count = max(1, math.ceil(length / max_spacing))
        count = segment_count - 1
        begin = len(loc_vertices)
        end = begin + count
        edge_locs[(a, b)] = (begin, end)
        for index in range(1, segment_count):
            loc_vertices.append(SurfacePoint.on_edge(a, b, index / float(segment_count)))
            loc_positions.append(loc_vertices[-1].interpolate(vertices))

    loc_triangles: list[list[int]] = [[] for _ in loc_vertices]
    triangle_adjacent_locs: list[list[int]] = [[] for _ in mesh.faces]
    for triangle_index, face in enumerate(mesh.faces):
        x, y, z = map(int, face)

        for a, b in ((x, y), (y, z), (z, x)):
            if a > b:
                a, b = b, a
            begin, end = edge_locs[(a, b)]
            for loc in range(begin, end):
                loc_triangles[loc].append(triangle_index)
                triangle_adjacent_locs[triangle_index].append(loc)

        for vertex in (x, y, z):
            loc = vertex_locs[vertex]
            loc_triangles[loc].append(triangle_index)
            triangle_adjacent_locs[triangle_index].append(loc)

    def add_surface_point(vertex: SurfacePoint) -> int:
        if vertex.simplex[1] == UINT32_MAX:
            return vertex_locs[vertex.simplex[0]]

        loc = len(loc_vertices)
        loc_vertices.append(vertex)
        loc_positions.append(vertex.interpolate(vertices))
        loc_triangles.append([])

        if vertex.simplex[2] == UINT32_MAX:
            edge = (vertex.simplex[0], vertex.simplex[1])
            if edge[0] > edge[1]:
                edge = (edge[1], edge[0])
            triangles = edge_to_triangles.get(edge)
            if not triangles:
                raise RuntimeError("surface path endpoint edge is not in the mesh")
            for triangle_index in triangles:
                loc_triangles[loc].append(triangle_index)
                triangle_adjacent_locs[triangle_index].append(loc)
        else:
            simplex = tuple(sorted(vertex.simplex))
            triangle_index = simplex_to_triangle.get(simplex)
            if triangle_index is None:
                raise RuntimeError("surface path endpoint simplex is not in the mesh")
            loc_triangles[loc].append(triangle_index)
            triangle_adjacent_locs[triangle_index].append(loc)
        return loc

    source_loc = add_surface_point(source)
    target_loc = add_surface_point(target)
    target_position = loc_positions[target_loc]

    distances = [math.inf] * len(loc_vertices)
    previous = [UINT32_MAX] * len(loc_vertices)
    heap: list[tuple[float, int, float, int, float]] = []

    def queue(loc: int, distance: float, from_loc: int) -> None:
        distance = float(distance)
        distances[loc] = distance
        previous[loc] = from_loc
        heuristic = float(np.linalg.norm(target_position - loc_positions[loc]))
        _heap_push(heap, loc, distance, heuristic)

    queue(source_loc, 0.0, UINT32_MAX)
    while heap:
        _, _, _, loc, distance = heappop(heap)
        if distance > distances[loc]:
            continue
        if loc == target_loc:
            break
        for triangle_index in loc_triangles[loc]:
            for neighbor in triangle_adjacent_locs[triangle_index]:
                if neighbor == loc:
                    continue
                new_distance = float(
                    distance + np.linalg.norm(loc_positions[neighbor] - loc_positions[loc])
                )
                if new_distance < distances[neighbor]:
                    queue(neighbor, new_distance, loc)

    if previous[target_loc] == UINT32_MAX:
        raise RuntimeError("surface path requested between disconnected vertices")

    path_indices: list[int] = []
    loc = target_loc
    while loc != UINT32_MAX:
        path_indices.append(loc)
        loc = previous[loc]
    path_indices.reverse()
    return [loc_vertices[index] for index in path_indices]


def _ordered_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for face in faces:
        x, y, z = map(int, face)
        for a, b in ((x, y), (y, z), (z, x)):
            if a > b:
                a, b = b, a
            edges.add((a, b))
    return sorted(edges)


def _heap_push(
    heap: list[tuple[float, int, float, int, float]], loc: int, distance: float, heuristic: float
) -> None:
    priority = float(heuristic + distance)
    heappush(heap, (priority, -loc, -distance, loc, distance))


def _length2(vector: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=float)
    return float(np.dot(vector, vector))


def _remap_surface_point(vertex: SurfacePoint, remap) -> SurfacePoint:
    simplex = list(vertex.simplex)
    for index, value in enumerate(simplex):
        if value != UINT32_MAX:
            simplex[index] = remap(value)
    return SurfacePoint.canonicalize(tuple(simplex), vertex.weights)
