from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from geometry import Mesh
from knitting._peeling.surface import UINT32_MAX, SurfacePoint
from knitting._peeling.surface.path import find_surface_path
from knitting._peeling.config import PeelingConfig
from knitting._peeling.planning.chain_geometry import chain_lengths, chain_location
from knitting._peeling.planning.stitch_state import FrontStitchSample, LinkPolicy
from knitting._peeling.planning.types import (
    ActiveFront,
    FrontGraphVertices,
    FrontStitches,
    SliceChains,
    SliceToModelMap,
    StitchRef,
)

RowAdjacencyPlan = list[list[tuple[bool, bool]]]
StitchLinkIndex = dict[StitchRef, list[StitchRef]]


class PathNodeSide(IntEnum):
    FRONT = 1
    NEXT = 2


class PathNodeKind(IntEnum):
    BEGIN = 0
    STITCH = 1
    END = 2


@dataclass(frozen=True)
class FrontPathNode:
    side: PathNodeSide
    chain: int
    stitch: int
    kind: PathNodeKind = PathNodeKind.STITCH


def reconstruct_next_front(
    parameters: PeelingConfig,
    slice_mesh: Mesh,
    slice_to_model: SliceToModelMap,
    front_slice_chains: SliceChains,
    front_stitches: FrontStitches,
    front_graph_vertices: FrontGraphVertices,
    next_slice_chains: SliceChains,
    next_stitches: FrontStitches,
    kept_row_adjacency: RowAdjacencyPlan,
    front_to_next: StitchLinkIndex,
    next_to_front: StitchLinkIndex,
    discard_front: list[bool],
    next_graph_vertices: FrontGraphVertices,
) -> ActiveFront:
    """Trace the next active front after its graph vertices have been added."""

    front_lengths = [chain_lengths(slice_mesh, chain) for chain in front_slice_chains]
    next_lengths = [chain_lengths(slice_mesh, chain) for chain in next_slice_chains]
    path_edges = _build_front_path_edges(
        front_slice_chains,
        front_stitches,
        next_slice_chains,
        next_stitches,
        kept_row_adjacency,
        front_to_next,
        next_to_front,
        discard_front,
    )
    ordered_paths = _trace_front_paths(path_edges)

    built_front_chains = []
    built_front_stitches: FrontStitches = []
    built_front_graph_vertices: FrontGraphVertices = []
    for path in ordered_paths:
        chain_on_slice, output_stitches, output_graph_vertices = _output_front_path(
            parameters,
            slice_mesh,
            slice_to_model,
            front_slice_chains,
            front_stitches,
            front_graph_vertices,
            front_lengths,
            next_slice_chains,
            next_stitches,
            next_lengths,
            next_graph_vertices,
            path,
        )
        if chain_on_slice:
            built_front_chains.append([vertex.compose(slice_to_model) for vertex in chain_on_slice])
            built_front_stitches.append(output_stitches)
            built_front_graph_vertices.append(output_graph_vertices)

    for chain in built_front_chains:
        index = 1
        while index < len(chain):
            if chain[index - 1] == chain[index]:
                del chain[index]
            else:
                index += 1

    return ActiveFront(
        built_front_chains,
        built_front_stitches,
        built_front_graph_vertices,
    )


def _build_front_path_edges(
    front_slice_chains: SliceChains,
    front_stitches: FrontStitches,
    next_slice_chains: SliceChains,
    next_stitches: FrontStitches,
    kept_row_adjacency: RowAdjacencyPlan,
    front_to_next: StitchLinkIndex,
    next_to_front: StitchLinkIndex,
    discard_front: list[bool],
) -> dict[tuple[FrontPathNode, FrontPathNode], FrontPathNode]:
    next_node: dict[tuple[FrontPathNode, FrontPathNode], FrontPathNode] = {}

    def insert_edge(a: FrontPathNode, b: FrontPathNode, c: FrontPathNode) -> None:
        key = (a, b)
        if key in next_node:
            raise ValueError("duplicate next-front path edge")
        next_node[key] = c

    for next_chain_index, (chain, stitches) in enumerate(
        zip(next_slice_chains, next_stitches, strict=True)
    ):
        is_loop = chain[0] == chain[-1]
        for stitch_index, stitch in enumerate(stitches):
            if stitch.link_policy == LinkPolicy.DISCARD:
                continue
            current = FrontPathNode(PathNodeSide.NEXT, next_chain_index, stitch_index)
            if kept_row_adjacency[next_chain_index][stitch_index][0]:
                if stitch_index > 0 or is_loop:
                    previous = FrontPathNode(
                        PathNodeSide.NEXT,
                        next_chain_index,
                        stitch_index - 1 if stitch_index > 0 else len(stitches) - 1,
                    )
                else:
                    previous = FrontPathNode(
                        PathNodeSide.NEXT, next_chain_index, UINT32_MAX, PathNodeKind.BEGIN
                    )
            else:
                front = next_to_front[StitchRef.of(next_chain_index, stitch_index)][0]
                previous = FrontPathNode(
                    PathNodeSide.FRONT,
                    front.chain_index,
                    front.stitch_index,
                )

            if kept_row_adjacency[next_chain_index][stitch_index][1]:
                if stitch_index + 1 < len(stitches) or is_loop:
                    following = FrontPathNode(
                        PathNodeSide.NEXT,
                        next_chain_index,
                        stitch_index + 1 if stitch_index + 1 < len(stitches) else 0,
                    )
                else:
                    following = FrontPathNode(
                        PathNodeSide.NEXT, next_chain_index, UINT32_MAX, PathNodeKind.END
                    )
            else:
                front = next_to_front[StitchRef.of(next_chain_index, stitch_index)][-1]
                following = FrontPathNode(
                    PathNodeSide.FRONT,
                    front.chain_index,
                    front.stitch_index,
                )

            insert_edge(previous, current, following)

    for front_chain_index, (chain, stitches) in enumerate(
        zip(front_slice_chains, front_stitches, strict=True)
    ):
        if discard_front[front_chain_index]:
            continue
        is_loop = chain[0] == chain[-1]
        for stitch_index, _stitch in enumerate(stitches):
            current = FrontPathNode(PathNodeSide.FRONT, front_chain_index, stitch_index)
            previous = FrontPathNode(
                PathNodeSide.FRONT,
                front_chain_index,
                stitch_index - 1
                if stitch_index > 0
                else (len(stitches) - 1 if is_loop else UINT32_MAX),
                PathNodeKind.STITCH if stitch_index > 0 or is_loop else PathNodeKind.BEGIN,
            )
            following = FrontPathNode(
                PathNodeSide.FRONT,
                front_chain_index,
                stitch_index + 1
                if stitch_index + 1 < len(stitches)
                else (0 if is_loop else UINT32_MAX),
                PathNodeKind.STITCH
                if stitch_index + 1 < len(stitches) or is_loop
                else PathNodeKind.END,
            )

            linked = front_to_next.get(StitchRef.of(front_chain_index, stitch_index))
            if linked is None:
                insert_edge(previous, current, following)
                continue

            first_next = linked[0]
            if not kept_row_adjacency[first_next.chain_index][first_next.stitch_index][0]:
                incoming = next_to_front[first_next]
                if incoming[0] == StitchRef.of(front_chain_index, stitch_index):
                    insert_edge(
                        previous,
                        current,
                        FrontPathNode(
                            PathNodeSide.NEXT,
                            first_next.chain_index,
                            first_next.stitch_index,
                        ),
                    )

            last_next = linked[-1]
            if not kept_row_adjacency[last_next.chain_index][last_next.stitch_index][1]:
                incoming = next_to_front[last_next]
                if incoming[-1] == StitchRef.of(front_chain_index, stitch_index):
                    insert_edge(
                        FrontPathNode(
                            PathNodeSide.NEXT,
                            last_next.chain_index,
                            last_next.stitch_index,
                        ),
                        current,
                        following,
                    )

    return next_node


def _trace_front_paths(
    next_node: dict[tuple[FrontPathNode, FrontPathNode], FrontPathNode],
) -> list[list[FrontPathNode]]:
    loops: list[list[FrontPathNode]] = []
    partials: dict[tuple[FrontPathNode, FrontPathNode], list[FrontPathNode]] = {}
    while next_node:
        key = min(next_node, key=_edge_sort_key)
        chain = [key[0], key[1], next_node.pop(key)]
        while (chain[-2], chain[-1]) in next_node:
            chain.append(next_node.pop((chain[-2], chain[-1])))

        partial_key = (chain[-2], chain[-1])
        if partial_key in partials:
            tail = partials.pop(partial_key)
            chain.pop()
            chain.pop()
            chain.extend(tail)

        if chain[0] == chain[-2] and chain[1] == chain[-1]:
            chain.pop()
            loops.append(chain)
        else:
            partials[(chain[0], chain[1])] = chain

    ordered_partials = [
        partials[key]
        for key in sorted(
            partials, key=lambda item: (_node_sort_key(item[0]), _node_sort_key(item[1]))
        )
    ]
    return loops + ordered_partials


def _node_sort_key(stitch: FrontPathNode) -> tuple[int, int, int]:
    return (stitch.side, stitch.chain, stitch.stitch)


def _edge_sort_key(
    edge: tuple[FrontPathNode, FrontPathNode],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return (_node_sort_key(edge[0]), _node_sort_key(edge[1]))


def _output_front_path(
    parameters: PeelingConfig,
    slice_mesh: Mesh,
    slice_to_model: SliceToModelMap,
    front_slice_chains: SliceChains,
    front_stitches: FrontStitches,
    front_graph_vertices: FrontGraphVertices,
    front_lengths: list[list[float]],
    next_slice_chains: SliceChains,
    next_stitches: FrontStitches,
    next_lengths: list[list[float]],
    next_graph_vertices: FrontGraphVertices,
    path: list[FrontPathNode],
) -> tuple[list[SurfacePoint], list[FrontStitchSample], list[int]]:
    slice_vertices = np.asarray(slice_mesh.vertices, dtype=float)
    path_vertices: list[SurfacePoint] = []
    path_lefts: list[int] = []
    for stitch in path:
        if stitch.side == PathNodeSide.FRONT:
            source_chain = front_slice_chains[stitch.chain]
            source_stitches = front_stitches[stitch.chain]
            source_lengths = front_lengths[stitch.chain]
        else:
            source_chain = next_slice_chains[stitch.chain]
            source_stitches = next_stitches[stitch.chain]
            source_lengths = next_lengths[stitch.chain]

        if stitch.kind == PathNodeKind.BEGIN:
            vertex = SurfacePoint.on_vertex(source_chain[0])
            path_vertices.append(vertex)
            path_lefts.append(0)
        elif stitch.kind == PathNodeKind.END:
            vertex = SurfacePoint.on_vertex(source_chain[-1])
            left = len(source_chain) - 2
            path_vertices.append(vertex)
            path_lefts.append(left)
        else:
            right, mix = chain_location(
                source_lengths,
                source_stitches[stitch.stitch].chain_t,
            )
            vertex = SurfacePoint.on_edge(source_chain[right - 1], source_chain[right], float(mix))
            path_vertices.append(vertex)
            path_lefts.append(right - 1)

    chain: list[SurfacePoint] = []
    output_stitches: list[FrontStitchSample] = []
    output_graph_vertices: list[int] = []
    remove_stitches: list[int] = []
    path_length = 0.0

    def append_vertex(vertex: SurfacePoint) -> None:
        nonlocal path_length
        if chain:
            if chain[-1] == vertex:
                return
            previous_position = chain[-1].interpolate(slice_vertices)
            current_position = vertex.interpolate(slice_vertices)
            segment = float(np.linalg.norm(current_position - previous_position))
            path_length = float(path_length + segment)
        chain.append(vertex)

    def stitch_payload(stitch: FrontPathNode, at_length: float) -> tuple[FrontStitchSample, int]:
        if stitch.side == PathNodeSide.FRONT:
            source = front_stitches[stitch.chain][stitch.stitch]
            vertex = front_graph_vertices[stitch.chain][stitch.stitch]
        else:
            source = next_stitches[stitch.chain][stitch.stitch]
            vertex = next_graph_vertices[stitch.chain][stitch.stitch]
        return FrontStitchSample(at_length, source.link_policy), vertex

    for path_index in range(len(path) - 1):
        a = path[path_index]
        b = path[path_index + 1]
        a_vertex = path_vertices[path_index]
        b_vertex = path_vertices[path_index + 1]
        a_left = path_lefts[path_index]
        b_left = path_lefts[path_index + 1]
        a_chain = (front_slice_chains if a.side == PathNodeSide.FRONT else next_slice_chains)[
            a.chain
        ]

        if path_index == 0:
            append_vertex(a_vertex)
            if a.kind == PathNodeKind.STITCH:
                sample, graph_vertex = stitch_payload(a, path_length)
                output_stitches.append(sample)
                output_graph_vertices.append(graph_vertex)
        elif chain[-1] != a_vertex:
            raise ValueError("build path did not continue")

        if a.kind == PathNodeKind.STITCH and a.side != b.side and a.side == PathNodeSide.FRONT:
            remove_stitches.append(len(output_stitches) - 1)

        if a.side == b.side:
            if a.chain != b.chain:
                raise ValueError("same-source build path changed chains")
            is_loop = a_chain[0] == a_chain[-1]
            if a_left != b_left:
                source_index = a_left
                while True:
                    source_index += 1
                    if source_index + 1 == len(a_chain):
                        if not is_loop:
                            raise ValueError("open chain path wrapped unexpectedly")
                        source_index = 0
                    append_vertex(SurfacePoint.on_vertex(a_chain[source_index]))
                    if source_index == b_left:
                        break
        else:
            bridge = find_surface_path(
                parameters.max_path_sample_spacing,
                slice_mesh,
                a_vertex,
                b_vertex,
            )
            if bridge[0] != a_vertex or bridge[-1] != b_vertex:
                raise ValueError("surface path did not preserve endpoints")
            for vertex in bridge[1:-1]:
                append_vertex(vertex)

        append_vertex(b_vertex)
        if b.kind == PathNodeKind.STITCH:
            sample, graph_vertex = stitch_payload(b, path_length)
            output_stitches.append(sample)
            output_graph_vertices.append(graph_vertex)
            if a.side != b.side and b.side == PathNodeSide.FRONT:
                remove_stitches.append(len(output_stitches) - 1)

    if path[0] == path[-1]:
        if not output_stitches:
            raise ValueError("closed output path has no stitches")
        if output_graph_vertices[-1] == output_graph_vertices[0]:
            if remove_stitches and remove_stitches[-1] == len(output_stitches) - 1:
                remove_stitches[-1] = 0
            output_stitches.pop()
            output_graph_vertices.pop()

    remove_set = set(remove_stitches)
    filtered_pairs = [
        (stitch, graph_vertex)
        for index, (stitch, graph_vertex) in enumerate(
            zip(output_stitches, output_graph_vertices, strict=True)
        )
        if index not in remove_set and stitch.link_policy != LinkPolicy.DISCARD
    ]
    filtered_stitches = [stitch for stitch, _graph_vertex in filtered_pairs]
    filtered_graph_vertices = [graph_vertex for _stitch, graph_vertex in filtered_pairs]
    if path_length == 0.0:
        raise ValueError("next front chain has zero length")
    for stitch in filtered_stitches:
        stitch.chain_t = float(stitch.chain_t / path_length)
    return chain, filtered_stitches, filtered_graph_vertices
