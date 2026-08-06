"""Piecewise-linear level chains on triangle meshes."""

from __future__ import annotations

import numpy as np

from knitting._peeling.surface import SurfacePoint


def extract_level_chains(
    faces: np.ndarray,
    values: np.ndarray,
    level: float,
) -> list[list[SurfacePoint]]:
    """Extract piecewise-linear ``values == level`` chains from triangles."""
    faces = np.asarray(faces, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    tolerance = 1.0e-12 * max(1.0, abs(float(level)), float(np.max(np.abs(values))))
    points: dict[tuple[int, int], tuple[tuple[int, int], tuple[float, float]]] = {}
    segments: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    directed_segments: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    def crossing(first: int, second: int):
        first_value = float(values[first] - level)
        second_value = float(values[second] - level)
        first_on = abs(first_value) <= tolerance
        second_on = abs(second_value) <= tolerance
        if first_on:
            key = (first, first)
            points[key] = (key, (1.0, 0.0))
            return key
        if second_on:
            key = (second, second)
            points[key] = (key, (1.0, 0.0))
            return key
        if first_value * second_value >= 0.0:
            return None
        a, b = sorted((first, second))
        amount = float((level - values[a]) / (values[b] - values[a]))
        key = (a, b)
        points[key] = (key, (1.0 - amount, amount))
        return key

    for face in faces:
        intersections: list[tuple[int, int]] = []
        for first, second in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            first_on = abs(float(values[first] - level)) <= tolerance
            second_on = abs(float(values[second] - level)) <= tolerance
            if first_on and second_on:
                for vertex in (first, second):
                    key = (vertex, vertex)
                    points[key] = (key, (1.0, 0.0))
                    if key not in intersections:
                        intersections.append(key)
                continue
            key = crossing(first, second)
            if key is not None and key not in intersections:
                intersections.append(key)
        if len(intersections) == 2:
            first, second = intersections
            local = {
                int(face[0]): np.asarray((0.0, 0.0)),
                int(face[1]): np.asarray((1.0, 0.0)),
                int(face[2]): np.asarray((0.0, 1.0)),
            }

            def local_position(key: tuple[int, int]) -> np.ndarray:
                edge_vertices, weights = points[key]
                return weights[0] * local[edge_vertices[0]] + weights[1] * local[
                    edge_vertices[1]
                ]

            high = int(face[np.argmax(values[face])])
            first_position = local_position(first)
            direction = local_position(second) - first_position
            toward_high = local[high] - first_position
            # Oriented level chains keep larger scalar values on their left.
            if direction[0] * toward_high[1] - direction[1] * toward_high[0] < 0.0:
                first, second = second, first
            directed_segments.add((first, second))
            a, b = sorted((first, second))
            if a != b:
                segments.add((a, b))

    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {
        key: set() for key in points
    }
    for first, second in segments:
        adjacency[first].add(second)
        adjacency[second].add(first)

    unused = set(segments)

    def edge(first: tuple[int, int], second: tuple[int, int]):
        return tuple(sorted((first, second)))

    def walk(start: tuple[int, int]) -> list[tuple[int, int]]:
        chain = [start]
        current = start
        while True:
            candidates = sorted(
                neighbor
                for neighbor in adjacency[current]
                if edge(current, neighbor) in unused
            )
            if not candidates:
                return chain
            following = candidates[0]
            unused.remove(edge(current, following))
            chain.append(following)
            current = following
            if current == start:
                return chain

    chains: list[list[tuple[int, int]]] = []
    for start in sorted(key for key, neighbors in adjacency.items() if len(neighbors) == 1):
        if any(edge(start, neighbor) in unused for neighbor in adjacency[start]):
            chains.append(walk(start))
    while unused:
        chains.append(walk(min(unused)[0]))

    for chain in chains:
        if len(chain) > 1 and (chain[0], chain[1]) not in directed_segments:
            chain.reverse()

    return [
        [
            (
                SurfacePoint.on_vertex(edge_vertices[0])
                if edge_vertices[0] == edge_vertices[1]
                else SurfacePoint.on_edge(
                    edge_vertices[0], edge_vertices[1], weights[1]
                )
            )
            for edge_vertices, weights in (points[key] for key in chain)
        ]
        for chain in chains
        if len(chain) >= 2
    ]
