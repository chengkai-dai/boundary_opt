from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cmp_to_key
from itertools import pairwise

import numpy as np

from geometry import Mesh
from knitting._peeling.surface._triangle_split import TriangleSplit
from knitting._peeling.surface.point import UINT32_MAX, SurfacePoint


class EdgeType(Enum):
    INITIAL = "initial"
    REVERSE = "reverse"
    COMBINE = "combine"
    SPLIT_FIRST = "split_first"
    SPLIT_SECOND = "split_second"


@dataclass(frozen=True)
class EdgeRecord:
    edge_type: EdgeType
    a: int
    b: int


@dataclass(frozen=True)
class TrimValue:
    total: int
    edge: int


@dataclass(frozen=True)
class SurfaceTrimResult:
    mesh: Mesh
    vertex_sources: list[SurfacePoint]
    left_chains: list[list[int]]
    right_chains: list[list[int]]


def trim_surface(
    mesh: Mesh,
    left_of: list[list[SurfacePoint]],
    right_of: list[list[SurfacePoint]] | None = None,
    *,
    debug: bool = False,
    debug_faces: set[int] | None = None,
) -> SurfaceTrimResult:
    right_of = right_of or []
    edge_records: list[EdgeRecord] = []

    def reverse_value(value: TrimValue) -> TrimValue:
        edge_records.append(EdgeRecord(EdgeType.REVERSE, value.edge, value.edge))
        return TrimValue(-value.total, len(edge_records) - 1)

    def combine_values(target: TrimValue, incoming: TrimValue) -> TrimValue:
        edge_records.append(EdgeRecord(EdgeType.COMBINE, target.edge, incoming.edge))
        return TrimValue(target.total + incoming.total, len(edge_records) - 1)

    def split_value(value: TrimValue) -> tuple[TrimValue, TrimValue]:
        edge_records.append(EdgeRecord(EdgeType.SPLIT_FIRST, value.edge, value.edge))
        first = TrimValue(value.total, len(edge_records) - 1)
        edge_records.append(EdgeRecord(EdgeType.SPLIT_SECOND, value.edge, value.edge))
        second = TrimValue(value.total, len(edge_records) - 1)
        return first, second

    triangle_split: TriangleSplit[TrimValue] = TriangleSplit(
        reverse_value=reverse_value,
        combine_values=combine_values,
        split_value=split_value,
    )
    empty_edges: set[tuple[int, int]] = set()

    fresh_id = 0
    for chain in left_of:
        previous = triangle_split.add_vertex(chain[0])
        previous_id = fresh_id
        fresh_id += 1
        for current_ev in chain[1:]:
            current = triangle_split.add_vertex(current_ev)
            current_id = fresh_id
            fresh_id += 1
            if previous == current:
                empty_edges.add((previous_id, current_id))
            edge_records.append(EdgeRecord(EdgeType.INITIAL, previous_id, current_id))
            triangle_split.add_edge(previous, current, TrimValue(1, len(edge_records) - 1))
            previous = current
            previous_id = current_id

    for chain in right_of:
        previous = triangle_split.add_vertex(chain[0])
        previous_id = fresh_id
        fresh_id += 1
        for current_ev in chain[1:]:
            current = triangle_split.add_vertex(current_ev)
            current_id = fresh_id
            fresh_id += 1
            if previous == current:
                empty_edges.add((previous_id, current_id))
            edge_records.append(EdgeRecord(EdgeType.INITIAL, current_id, previous_id))
            triangle_split.add_edge(current, previous, TrimValue(1 << 8, len(edge_records) - 1))
            previous = current
            previous_id = current_id

    if debug:
        total_simplex_edges = sum(len(edges) for edges in triangle_split.simplex_edges.values())
        total_chain_edges = sum(max(0, len(chain) - 1) for chain in left_of) + sum(
            max(0, len(chain) - 1) for chain in right_of
        )
        print(f"triangle split has {len(triangle_split.vertices)} vertices.")
        print(f"triangle split has {len(triangle_split.simplex_vertices)} simplices with vertices.")
        print(
            f"triangle split has {len(triangle_split.simplex_edges)} simplices with edges "
            f"({total_simplex_edges} edges from {total_chain_edges} chain edges)."
        )

    left_split_chains, right_split_chains = _read_chain_vertices_from_edge_records(
        triangle_split,
        edge_records,
        empty_edges,
        left_of,
        right_of,
    )

    for chain in left_split_chains:
        _cleanup_chain(mesh, triangle_split, chain, 1)
    for chain in right_split_chains:
        _cleanup_chain(mesh, triangle_split, chain, -(1 << 8))

    split_vertices, split_faces, triangle_split_to_split = triangle_split.split_triangles(
        len(mesh.vertices), mesh.faces, debug_faces=debug_faces
    )
    if debug:
        print(f"Split mesh has {len(split_faces)} triangles on {len(split_vertices)} vertices.")

    edge_values: dict[tuple[int, int], int] = {}
    for edges in triangle_split.simplex_edges.values():
        for edge in edges:
            a = triangle_split_to_split[edge.first]
            b = triangle_split_to_split[edge.second]
            if (a, b) in edge_values or (b, a) in edge_values:
                raise ValueError(f"duplicate trim edge value for split edge {(a, b)}")
            edge_values[(a, b)] = edge.value.total
            edge_values[(b, a)] = -edge.value.total

    edge_to_face: dict[tuple[int, int], int] = {}
    for face_index, face in enumerate(split_faces):
        x, y, z = face
        for edge in ((x, y), (y, z), (z, x)):
            if edge in edge_to_face:
                raise ValueError("split mesh has duplicate directed edge")
            edge_to_face[edge] = face_index

    unvisited = None
    values: list[int | None] = [unvisited] * len(split_faces)
    keep = [False] * len(split_faces)

    for seed in range(len(split_faces)):
        if values[seed] is not None:
            continue
        component = [seed]
        values[seed] = (128 << 8) | 128
        component_index = 0
        while component_index < len(component):
            face_index = component[component_index]
            value = values[face_index]
            assert value is not None
            x, y, z = split_faces[face_index]

            def over(
                a: int,
                b: int,
                value: int = value,
                component: list[int] = component,
                face_index: int = face_index,
            ) -> None:
                opposite = edge_to_face.get((b, a))
                if opposite is None:
                    return
                new_value = value + edge_values.get((b, a), 0)
                if values[opposite] is None:
                    values[opposite] = new_value
                    component.append(opposite)
                elif values[opposite] != new_value:
                    if debug_faces:
                        opposite_face = split_faces[opposite]
                        print(
                            f"mismatch faces: {face_index}="
                            f"{tuple(map(int, split_faces[face_index]))}, "
                            f"{opposite}={tuple(map(int, opposite_face))}"
                        )
                        for label, face_vertices in (
                            ("from", split_faces[face_index]),
                            ("to", opposite_face),
                        ):
                            print(f"{label} face edges:")
                            for ea, eb in (
                                (int(face_vertices[0]), int(face_vertices[1])),
                                (int(face_vertices[1]), int(face_vertices[2])),
                                (int(face_vertices[2]), int(face_vertices[0])),
                            ):
                                print(
                                    f"  {(ea, eb)} delta="
                                    f"{edge_values.get((ea, eb), 0)} reverse_delta="
                                    f"{edge_values.get((eb, ea), 0)}"
                                )
                        for vertex_index in sorted(
                            {int(v) for v in split_faces[face_index]}
                            | {int(v) for v in opposite_face}
                        ):
                            print(f"split vertex {vertex_index}: {split_vertices[vertex_index]}")
                            incident = [
                                (edge, delta)
                                for edge, delta in edge_values.items()
                                if delta != 0
                                and (edge[0] == vertex_index or edge[1] == vertex_index)
                            ]
                            for edge, delta in sorted(incident):
                                print(f"    incident {edge} delta={delta}")
                    raise ValueError(
                        "trim traversal produced inconsistent face potentials "
                        f"between faces {face_index} and {opposite}"
                    )

            over(x, y)
            over(y, z)
            over(z, x)
            component_index += 1

        max_right = 128
        max_left = 128
        for face_index in component:
            value = values[face_index]
            assert value is not None
            right = value >> 8
            left = value & 0xFF
            max_right = max(max_right, right)
            max_left = max(max_left, left)
        keep_value = (max_right << 8) | max_left
        for face_index in component:
            keep[face_index] = values[face_index] == keep_value

    split_to_clipped = [UINT32_MAX] * len(split_vertices)
    vertex_sources: list[SurfacePoint] = []
    clipped_to_split: list[int] = []

    def use_vertex(split_vertex: int) -> int:
        if split_to_clipped[split_vertex] == UINT32_MAX:
            split_to_clipped[split_vertex] = len(vertex_sources)
            vertex_sources.append(split_vertices[split_vertex])
            clipped_to_split.append(split_vertex)
        return split_to_clipped[split_vertex]

    clipped_faces: list[tuple[int, int, int]] = []
    for face_index, face in enumerate(split_faces):
        if not keep[face_index]:
            continue
        clipped_faces.append(tuple(use_vertex(v) for v in face))

    clipped_positions = np.asarray(
        [vertex.interpolate(mesh.vertices) for vertex in vertex_sources], dtype=float
    )
    clipped_mesh = Mesh(vertices=clipped_positions, faces=np.asarray(clipped_faces, dtype=np.int64))

    return SurfaceTrimResult(
        mesh=clipped_mesh,
        vertex_sources=vertex_sources,
        left_chains=[
            _transform_chain(chain, triangle_split_to_split, split_to_clipped)
            for chain in left_split_chains
        ],
        right_chains=[
            _transform_chain(chain, triangle_split_to_split, split_to_clipped)
            for chain in right_split_chains
        ],
    )


@dataclass(frozen=True)
class _Source:
    a: int
    b: int
    numerator: int
    denominator: int


@dataclass(frozen=True)
class _SubEdge:
    a: int
    b: int
    numerator: int
    denominator: int


def _read_chain_vertices_from_edge_records(
    triangle_split: TriangleSplit[TrimValue],
    edge_records: list[EdgeRecord],
    empty_edges: set[tuple[int, int]],
    left_of: list[list[SurfacePoint]],
    right_of: list[list[SurfacePoint]],
) -> tuple[list[list[int]], list[list[int]]]:
    sources: list[list[_Source]] = []
    for edge_index, edge in enumerate(edge_records):
        if edge.edge_type == EdgeType.INITIAL:
            sources.append([_Source(edge.a, edge.b, 1, 2)])
        elif edge.edge_type == EdgeType.REVERSE:
            sources.append(
                [
                    _Source(
                        source.b,
                        source.a,
                        source.denominator - source.numerator,
                        source.denominator,
                    )
                    for source in sources[edge.a]
                ]
            )
        elif edge.edge_type == EdgeType.COMBINE:
            sources.append(
                [
                    _Source(source.a, source.b, source.numerator, source.denominator)
                    for source in sources[edge.a]
                ]
                + [
                    _Source(source.a, source.b, source.numerator, source.denominator)
                    for source in sources[edge.b]
                ]
            )
        elif edge.edge_type == EdgeType.SPLIT_FIRST:
            sources.append(
                [
                    _Source(source.a, source.b, source.numerator * 2 - 1, source.denominator * 2)
                    for source in sources[edge.a]
                ]
            )
        elif edge.edge_type == EdgeType.SPLIT_SECOND:
            sources.append(
                [
                    _Source(source.a, source.b, source.numerator * 2 + 1, source.denominator * 2)
                    for source in sources[edge.a]
                ]
            )
        else:
            raise AssertionError(edge.edge_type)
        if len(sources) != edge_index + 1:
            raise AssertionError("edge source table lost sync")

    edge_subedges: dict[tuple[int, int], list[_SubEdge]] = {}
    for edges in triangle_split.simplex_edges.values():
        for segment in edges:
            for source in sources[segment.value.edge]:
                if source.a < source.b:
                    edge_subedges.setdefault((source.a, source.b), []).append(
                        _SubEdge(
                            segment.first, segment.second, source.numerator, source.denominator
                        )
                    )
                else:
                    edge_subedges.setdefault((source.b, source.a), []).append(
                        _SubEdge(
                            segment.second,
                            segment.first,
                            source.denominator - source.numerator,
                            source.denominator,
                        )
                    )

    for subedges in edge_subedges.values():
        subedges.sort(key=cmp_to_key(_compare_subedges))
        for first, second in pairwise(subedges):
            if first.b != second.a:
                raise ValueError("edge subedges are not connected")

    fresh_id = 0
    left_split_chains: list[list[int]] = []
    for chain in left_of:
        split_chain: list[int] = []
        previous_id = fresh_id
        fresh_id += 1
        for _ in chain[1:]:
            current_id = fresh_id
            fresh_id += 1
            _append_subedges(split_chain, edge_subedges, empty_edges, previous_id, current_id)
            previous_id = current_id
        left_split_chains.append(split_chain)

    right_split_chains: list[list[int]] = []
    for chain in right_of:
        split_chain = []
        previous_id = fresh_id
        fresh_id += 1
        for _ in chain[1:]:
            current_id = fresh_id
            fresh_id += 1
            _append_subedges(split_chain, edge_subedges, empty_edges, previous_id, current_id)
            previous_id = current_id
        right_split_chains.append(split_chain)

    return left_split_chains, right_split_chains


def _append_subedges(
    split_chain: list[int],
    edge_subedges: dict[tuple[int, int], list[_SubEdge]],
    empty_edges: set[tuple[int, int]],
    previous_id: int,
    current_id: int,
) -> None:
    key = (previous_id, current_id)
    subedges = edge_subedges.get(key)
    if key in empty_edges:
        if subedges is not None:
            raise ValueError("empty edge unexpectedly has subedges")
        return
    if subedges is None:
        raise ValueError(f"missing subedges for source edge {key}")
    for subedge in subedges:
        if not split_chain:
            split_chain.append(subedge.a)
        elif split_chain[-1] != subedge.a:
            raise ValueError("source subedges do not continue current chain")
        split_chain.append(subedge.b)


def _compare_subedges(a: _SubEdge, b: _SubEdge) -> int:
    lhs = a.numerator * b.denominator
    rhs = b.numerator * a.denominator
    return (lhs > rhs) - (lhs < rhs)


def _transform_chain(
    split_chain: list[int], triangle_split_to_split: list[int], split_to_clipped: list[int]
) -> list[int]:
    out: list[int] = []
    for vertex in split_chain:
        split_vertex = triangle_split_to_split[vertex]
        clipped_vertex = split_to_clipped[split_vertex]
        out.append(clipped_vertex)
    return out


def _cleanup_chain(
    mesh: Mesh,
    triangle_split: TriangleSplit[TrimValue],
    split_chain: list[int],
    value: int,
) -> None:
    if not split_chain:
        return

    def remove_from_triangle_split(first: int, last: int) -> None:
        for i in range(first, last):
            a_index = split_chain[i]
            b_index = split_chain[i + 1]
            a = triangle_split.vertices[a_index]
            b = triangle_split.vertices[b_index]
            common = SurfacePoint.common_simplex(a.simplex, b.simplex)
            edges = triangle_split.simplex_edges.get(common)
            if edges is None:
                raise ValueError("cleanup could not find triangle split edges")
            found = False
            for edge in edges:
                if edge.first == a_index and edge.second == b_index:
                    edge.value = TrimValue(edge.value.total - value, edge.value.edge)
                    found = True
                    break
                if edge.second == a_index and edge.first == b_index:
                    edge.value = TrimValue(edge.value.total + value, edge.value.edge)
                    found = True
                    break
            if not found:
                raise ValueError("cleanup could not find chain edge in triangle split")

    while True:
        lengths = [0.0]
        for previous, current in pairwise(split_chain):
            a = triangle_split.vertices[previous].to_surface_point().interpolate(mesh.vertices)
            b = triangle_split.vertices[current].to_surface_point().interpolate(mesh.vertices)
            segment = float(np.linalg.norm(b - a))
            lengths.append(lengths[-1] + segment)

        visited: dict[int, int] = {}
        first_i = 1 if split_chain[0] == split_chain[-1] else 0
        removed = False
        for i in range(first_i, len(split_chain)):
            vertex = split_chain[i]
            if vertex not in visited:
                visited[vertex] = i
                continue

            first = visited[vertex]
            last = i
            length = lengths[last] - lengths[first]
            outer_length = (lengths[first] - lengths[0]) + (lengths[-1] - lengths[last])

            if split_chain[0] == split_chain[-1] and outer_length < length:
                remove_from_triangle_split(0, first)
                remove_from_triangle_split(last, len(split_chain) - 1)
                del split_chain[last + 1 :]
                del split_chain[:first]
            else:
                remove_from_triangle_split(first, last)
                del split_chain[first + 1 : last + 1]
            removed = True
            break

        if not removed:
            break
