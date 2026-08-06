"""AutoKnit's row/column-graph pre-scheduling trace.

This is a dependency-light port of ``ak-trace_graph.cpp``.  The public record
types intentionally contain only the fields needed to audit graph legality;
machine scheduling and file export remain outside the peeling core.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from knitting.graph import RowColGraph
from geometry import Mesh
from knitting._peeling.surface import UINT32_MAX

NO_STITCH = UINT32_MAX
FORWARD = "a"
BACKWARD = "c"
KNIT_TYPES = frozenset({"s", "d", "k", "e", "i"})


@dataclass
class _VertexInfo:
    row: int = NO_STITCH
    knits: int = 0
    last_stitch: int = NO_STITCH


@dataclass
class _TraceRecord:
    yarn: int
    stitch_type: str
    direction: str
    in_indices: list[int]
    out_indices: list[int]
    vertex: int
    at: np.ndarray


@dataclass(frozen=True)
class TracedGraphStitch:
    yarn: int
    stitch_type: str
    direction: str
    in_indices: tuple[int, int]
    out_indices: tuple[int, int]
    vertex: int
    at: np.ndarray


def trace_graph_records(
    graph: RowColGraph,
    mesh: Mesh | None = None,
) -> list[TracedGraphStitch]:
    """Trace a graph while retaining the graph vertex for every record."""

    return [_public_record(record) for record in _trace_graph_records(graph, mesh)]


def _trace_graph_records(graph: RowColGraph, mesh: Mesh | None) -> list[_TraceRecord]:
    vertices = graph.vertices
    try:
        graph.validate()
    except ValueError as exc:
        raise RuntimeError(f"invalid row/column graph: {exc}") from exc

    info, row_pending = _initial_trace_state(graph)
    traced: list[_TraceRecord] = []
    fresh_yarn_id = 1

    def trace_yarn(yarn: int) -> bool:
        at = NO_STITCH
        direction = FORWARD

        def get_next(vertex: int) -> int:
            return vertices[vertex].row_out if direction == FORWARD else vertices[vertex].row_in

        def get_prev_child(vertex: int) -> int:
            return (
                vertices[vertex].col_out[0] if direction == FORWARD else vertices[vertex].col_out[1]
            )

        def get_next_child(vertex: int) -> int:
            return (
                vertices[vertex].col_out[1] if direction == FORWARD else vertices[vertex].col_out[0]
            )

        def get_prev_parent(vertex: int) -> int:
            return (
                vertices[vertex].col_in[0] if direction == FORWARD else vertices[vertex].col_in[1]
            )

        def get_next_parent(vertex: int) -> int:
            return (
                vertices[vertex].col_in[1] if direction == FORWARD else vertices[vertex].col_in[0]
            )

        def is_covered(vertex: int) -> bool:
            return any(
                child != NO_STITCH and info[child].knits != 0 for child in vertices[vertex].col_out
            )

        def make_stitch(next_vertex: int, base_type: str) -> None:
            nonlocal at
            at = next_vertex
            if base_type == "k":
                if row_pending[info[at].row] != 0:
                    raise RuntimeError("cannot knit a row before its parents are ready")
                if info[at].knits >= 2:
                    raise RuntimeError("row/column graph vertex was knit more than twice")

            stitch_type = _fancy_stitch_type(graph, info, at, base_type)
            in_indices = [NO_STITCH, NO_STITCH]
            if info[at].last_stitch != NO_STITCH:
                in_indices[0] = info[at].last_stitch
            else:
                if vertices[at].col_in[0] != NO_STITCH:
                    in_indices[0] = info[vertices[at].col_in[0]].last_stitch
                if vertices[at].col_in[1] != NO_STITCH:
                    in_indices[1] = info[vertices[at].col_in[1]].last_stitch
                if (
                    direction == BACKWARD
                    and in_indices[0] != NO_STITCH
                    and in_indices[1] != NO_STITCH
                ):
                    in_indices[0], in_indices[1] = in_indices[1], in_indices[0]

            if base_type == "k":
                info[at].knits += 1
                if info[at].knits == 2:
                    for child in vertices[at].col_out:
                        if child == NO_STITCH:
                            continue
                        if row_pending[info[child].row] <= 0:
                            raise RuntimeError("row pending count underflowed while tracing graph")
                        row_pending[info[child].row] -= 1

            info[at].last_stitch = len(traced)
            traced.append(
                _TraceRecord(
                    yarn=yarn,
                    stitch_type=stitch_type,
                    direction=direction,
                    in_indices=in_indices,
                    out_indices=[NO_STITCH, NO_STITCH],
                    vertex=at,
                    at=np.full(3, np.nan, dtype=np.float64),
                )
            )

        def knit(next_vertex: int) -> None:
            make_stitch(next_vertex, "k")

        def tuck(next_vertex: int) -> None:
            make_stitch(next_vertex, "t")

        def miss(next_vertex: int) -> None:
            make_stitch(next_vertex, "m")

        found = NO_STITCH
        for vertex_index, vertex_info in enumerate(info):
            if row_pending[vertex_info.row] == 0 and vertex_info.knits < 2:
                found = vertex_index
                break
        if found == NO_STITCH:
            return False

        def advance_to_row_start(vertex_index: int) -> tuple[bool, int]:
            previous = vertices[vertex_index].row_in
            while previous != NO_STITCH and info[previous].knits >= 2:
                candidate = NO_STITCH
                for slot in (1, 0):
                    child = vertices[previous].col_out[slot]
                    if child == NO_STITCH:
                        continue
                    if row_pending[info[child].row] != 0:
                        continue
                    candidate = child
                    break
                previous = candidate
            if previous == NO_STITCH:
                return False, vertex_index
            if row_pending[info[previous].row] != 0 or info[previous].knits >= 2:
                raise RuntimeError("failed to advance to a ready row start")
            return True, previous

        found2 = found
        while True:
            ok, found = advance_to_row_start(found)
            if not ok or found == found2:
                break
            ok, found = advance_to_row_start(found)
            if not ok or found == found2:
                break
            ok, found2 = advance_to_row_start(found2)
            if not ok:
                raise RuntimeError("row-start advancement became inconsistent")
            if found == found2:
                break

        direction = (
            BACKWARD
            if vertices[found].row_out == NO_STITCH or info[vertices[found].row_out].knits == 2
            else FORWARD
        )
        knit(found)
        if at == NO_STITCH:
            return False

        def rule2() -> bool:
            nonlocal direction
            up = NO_STITCH
            for child in (get_next_child(at), get_prev_child(at)):
                if child != NO_STITCH and row_pending[info[child].row] == 0:
                    up = child
                    break
            if up == NO_STITCH:
                return False

            up_next = get_next(up)
            if up_next != NO_STITCH:
                if (
                    vertices[up].col_in[0] != NO_STITCH
                    and vertices[up].col_in[1] != NO_STITCH
                    and get_prev_parent(up) == at
                    and get_next_parent(up) != at
                ):
                    miss(get_next_parent(up))
                knit(up_next)
                return True

            next_vertex = get_next(at)
            if next_vertex != NO_STITCH and is_covered(next_vertex):
                next_vertex = NO_STITCH
            if (
                next_vertex != NO_STITCH
                and info[next_vertex].knits == 2
                and vertices[next_vertex].col_out[0] != NO_STITCH
                and vertices[next_vertex].col_out[1] != NO_STITCH
            ):
                next_vertex = get_prev_child(next_vertex)
            if (
                next_vertex != NO_STITCH
                and info[next_vertex].last_stitch != NO_STITCH
                and traced[info[next_vertex].last_stitch].stitch_type == "e"
            ):
                next_vertex = NO_STITCH

            if next_vertex != NO_STITCH and info[next_vertex].knits:
                tuck(next_vertex)
                direction = BACKWARD if direction == FORWARD else FORWARD
                miss(next_vertex)
                knit(up)
            else:
                direction = BACKWARD if direction == FORWARD else FORWARD
                knit(up)
            return True

        def rule3() -> bool:
            next_vertex = get_next(at)
            if next_vertex == NO_STITCH or info[next_vertex].knits >= 2:
                return False
            knit(next_vertex)
            return True

        def rule4() -> bool:
            nonlocal direction
            if info[at].knits == 2:
                return False
            if info[at].knits != 1:
                raise RuntimeError("short-row turn expected a once-knit vertex")
            if get_next(at) != NO_STITCH:
                return False

            down = NO_STITCH
            for parent in (get_next_parent(at), get_prev_parent(at)):
                if parent != NO_STITCH:
                    down = parent
                    break
            if down == NO_STITCH:
                return False
            if info[down].knits != 2:
                raise RuntimeError("short-row parent was not fully knit")

            down_next = get_next(down)
            if down_next != NO_STITCH and is_covered(down_next):
                down_next = NO_STITCH
            if (
                down_next != NO_STITCH
                and info[down_next].knits == 2
                and vertices[down_next].col_out[0] != NO_STITCH
                and vertices[down_next].col_out[1] != NO_STITCH
            ):
                down_next = get_prev_child(down_next)
            if (
                down_next != NO_STITCH
                and info[down_next].last_stitch != NO_STITCH
                and traced[info[down_next].last_stitch].stitch_type == "e"
            ):
                down_next = NO_STITCH

            here = at
            if down_next != NO_STITCH:
                tuck(down_next)
            direction = BACKWARD if direction == FORWARD else FORWARD
            if down_next != NO_STITCH:
                miss(down_next)
            knit(here)
            return True

        def rule5() -> bool:
            if info[at].knits != 2:
                return False

            parent = at
            parent_next = NO_STITCH
            while parent_next == NO_STITCH:
                found_parent = False
                for candidate in (get_next_parent(parent), get_prev_parent(parent)):
                    if candidate != NO_STITCH:
                        parent = candidate
                        found_parent = True
                        break
                if not found_parent:
                    return False
                parent_next = get_next(parent)

            if info[parent_next].knits == 2:
                return False
            knit(parent_next)
            return True

        guard = max(1000, len(vertices) * 20)
        operations = 0
        while rule2() or rule3() or rule4() or rule5():
            operations += 1
            if operations > guard:
                raise RuntimeError("trace_graph rule loop exceeded its operation guard")
        return True

    while trace_yarn(fresh_yarn_id):
        fresh_yarn_id += 1

    unfinished = [index for index, vertex_info in enumerate(info) if vertex_info.knits < 2]
    if unfinished:
        preview = ", ".join(str(index) for index in unfinished[:8])
        suffix = "" if len(unfinished) <= 8 else ", ..."
        raise RuntimeError(
            f"trace_graph left {len(unfinished)} graph vertices unfinished: {preview}{suffix}"
        )

    _fill_out_links(traced)
    _sort_double_outs(graph, traced)
    if mesh is not None:
        _assign_positions(graph, mesh, traced)
    return traced


def _public_record(record: _TraceRecord) -> TracedGraphStitch:
    return TracedGraphStitch(
        yarn=record.yarn,
        stitch_type=record.stitch_type,
        direction=record.direction,
        in_indices=(record.in_indices[0], record.in_indices[1]),
        out_indices=(record.out_indices[0], record.out_indices[1]),
        vertex=record.vertex,
        at=record.at,
    )


def _fancy_stitch_type(
    graph: RowColGraph,
    info: list[_VertexInfo],
    at: int,
    base_type: str,
) -> str:
    vertex = graph.vertices[at]
    if base_type != "k":
        if base_type not in {"t", "m"}:
            raise RuntimeError(f"unknown stitch type {base_type!r}")
        return base_type

    if info[at].last_stitch == NO_STITCH:
        if vertex.col_in[0] == NO_STITCH and vertex.col_in[1] == NO_STITCH:
            return "s"
        if vertex.col_in[0] != NO_STITCH and vertex.col_in[1] != NO_STITCH:
            return "d"
        return "k"

    if vertex.col_out[0] == NO_STITCH and vertex.col_out[1] == NO_STITCH:
        return "e"
    if vertex.col_out[0] != NO_STITCH and vertex.col_out[1] != NO_STITCH:
        return "i"
    return "k"


def _initial_trace_state(graph: RowColGraph) -> tuple[list[_VertexInfo], list[int]]:
    vertices = graph.vertices
    info = [_VertexInfo() for _ in vertices]
    row_pending: list[int] = []

    for seed in range(len(vertices)):
        if info[seed].row != NO_STITCH:
            continue
        row = len(row_pending)
        row_pending.append(0)
        info[seed].row = row
        todo = [seed]
        while todo:
            at = todo.pop()
            for neighbor in (vertices[at].row_in, vertices[at].row_out):
                if neighbor == NO_STITCH:
                    continue
                if info[neighbor].row == row:
                    continue
                if info[neighbor].row != NO_STITCH:
                    raise RuntimeError(
                        "row traversal reached a vertex already assigned to another row"
                    )
                info[neighbor].row = row
                todo.append(neighbor)

    for vertex in vertices:
        for child in vertex.col_out:
            if child != NO_STITCH:
                row_pending[info[child].row] += 1
    return info, row_pending


def _fill_out_links(traced: list[_TraceRecord]) -> None:
    def add_out(source: int, target: int) -> None:
        if source >= len(traced):
            raise RuntimeError("stitch input references an unknown source")
        if traced[source].out_indices[0] == NO_STITCH:
            traced[source].out_indices[0] = target
        elif traced[source].out_indices[1] == NO_STITCH:
            traced[source].out_indices[1] = target
        else:
            raise RuntimeError("stitch has too many outgoing links")

    for target, stitch in enumerate(traced):
        if stitch.in_indices[0] != NO_STITCH:
            add_out(stitch.in_indices[0], target)
        if stitch.in_indices[1] != NO_STITCH:
            add_out(stitch.in_indices[1], target)


def _sort_double_outs(graph: RowColGraph, traced: list[_TraceRecord]) -> None:
    for stitch in traced:
        if stitch.out_indices[0] == NO_STITCH or stitch.out_indices[1] == NO_STITCH:
            continue
        vertex = graph.vertices[stitch.vertex]
        first_out_vertex = traced[stitch.out_indices[0]].vertex
        second_out_vertex = traced[stitch.out_indices[1]].vertex
        if vertex.col_out[0] == second_out_vertex and vertex.col_out[1] == first_out_vertex:
            stitch.out_indices[0], stitch.out_indices[1] = (
                stitch.out_indices[1],
                stitch.out_indices[0],
            )
        if stitch.direction == BACKWARD:
            stitch.out_indices[0], stitch.out_indices[1] = (
                stitch.out_indices[1],
                stitch.out_indices[0],
            )


def _assign_positions(graph: RowColGraph, mesh: Mesh, traced: list[_TraceRecord]) -> None:
    vertices = graph.vertices
    graph_positions = np.asarray(
        [vertex.at.interpolate(mesh.vertices) for vertex in vertices],
        dtype=np.float64,
    )
    up_vectors: list[np.ndarray] = []
    for vertex_index, vertex in enumerate(vertices):
        vertex_position = graph_positions[vertex_index]
        acc = np.zeros(3, dtype=np.float64)
        count = 0
        for parent in vertex.col_in:
            if parent != NO_STITCH:
                acc += vertex_position - graph_positions[parent]
                count += 1
        for child in vertex.col_out:
            if child != NO_STITCH:
                acc += graph_positions[child] - vertex_position
                count += 1
        acc = np.asarray([0.0, 0.0, 1.0], dtype=np.float64) if count == 0 else acc / float(count)
        up_vectors.append(acc)

    counts = [0 for _ in vertices]
    local_indices: list[int] = []
    for stitch in traced:
        local_indices.append(counts[stitch.vertex])
        counts[stitch.vertex] += 1

    for stitch, local_index in zip(traced, local_indices, strict=True):
        count = counts[stitch.vertex]
        amount = float(local_index + 0.5) / float(count) - 0.5
        stitch.at = graph_positions[stitch.vertex] + amount * up_vectors[stitch.vertex]


__all__ = [
    "KNIT_TYPES",
    "NO_STITCH",
    "TracedGraphStitch",
    "trace_graph_records",
]
