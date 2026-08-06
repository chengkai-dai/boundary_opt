from __future__ import annotations

from dataclasses import dataclass

from knitting.graph import RowColGraph, RowColVertex
from geometry import Mesh
from knitting._peeling.surface import UINT32_MAX, SurfacePoint
from knitting._peeling.config import PeelingConfig
from knitting._peeling.planning.chain_geometry import chain_lengths, chain_location
from knitting._peeling.planning.front_path_reconstruction import (
    RowAdjacencyPlan,
    StitchLinkIndex,
    reconstruct_next_front,
)
from knitting._peeling.planning.stitch_state import LinkPolicy
from knitting._peeling.planning.types import (
    BuildNextFrontResult,
    FrontGraphVertices,
    FrontStitches,
    SliceChains,
    SliceToModelMap,
    StitchLink,
    StitchRef,
)


@dataclass(frozen=True)
class FrontTransitionPlan:
    kept_row_adjacency: RowAdjacencyPlan
    front_to_next: StitchLinkIndex
    next_to_front: StitchLinkIndex
    discard_front: list[bool]
    column_links: list[StitchLink]


def build_next_front(
    parameters: PeelingConfig,
    slice_mesh: Mesh,
    slice_to_model: SliceToModelMap,
    front_slice_chains: SliceChains,
    front_stitches: FrontStitches,
    front_graph_vertices: FrontGraphVertices,
    next_slice_chains: SliceChains,
    next_stitches: FrontStitches,
    links: list[StitchLink],
    graph: RowColGraph,
) -> BuildNextFrontResult:
    if len(front_slice_chains) != len(front_stitches):
        raise ValueError("front_slice_chains and front_stitches must align")
    if len(front_graph_vertices) != len(front_stitches):
        raise ValueError("front_graph_vertices and front_stitches must align")
    for stitches, vertices in zip(front_stitches, front_graph_vertices, strict=True):
        if len(stitches) != len(vertices):
            raise ValueError("front stitch samples and graph vertices must align")
    if len(next_slice_chains) != len(next_stitches):
        raise ValueError("next_slice_chains and next_stitches must align")

    transition = plan_front_transition(
        front_slice_chains,
        front_stitches,
        next_slice_chains,
        next_stitches,
        links,
    )

    next_graph_vertices = add_next_front_to_graph(
        slice_mesh,
        slice_to_model,
        next_slice_chains,
        next_stitches,
        transition.kept_row_adjacency,
        graph,
    )
    apply_column_links(graph, front_graph_vertices, next_graph_vertices, transition.column_links)

    built_front = reconstruct_next_front(
        parameters,
        slice_mesh,
        slice_to_model,
        front_slice_chains,
        front_stitches,
        front_graph_vertices,
        next_slice_chains,
        next_stitches,
        transition.kept_row_adjacency,
        transition.front_to_next,
        transition.next_to_front,
        transition.discard_front,
        next_graph_vertices,
    )

    return BuildNextFrontResult(front=built_front)


def plan_front_transition(
    front_slice_chains: SliceChains,
    front_stitches: FrontStitches,
    next_slice_chains: SliceChains,
    next_stitches: FrontStitches,
    links: list[StitchLink],
) -> FrontTransitionPlan:
    """Interpret stitch links as row-adjacency and front-path topology."""

    column_links = [
        link
        for link in links
        if next_stitches[link.next_ref.chain_index][link.next_ref.stitch_index].link_policy
        != LinkPolicy.DISCARD
    ]
    front_to_next: StitchLinkIndex = {}
    next_to_front: StitchLinkIndex = {}
    discard_front = [True] * len(front_slice_chains)
    for link in links:
        discard_front[link.front_ref.chain_index] = False
    for link in column_links:
        front_to_next.setdefault(link.front_ref, []).append(link.next_ref)
        next_to_front.setdefault(link.next_ref, []).append(link.front_ref)

    kept_row_adjacency: RowAdjacencyPlan = []
    for next_chain_index, (chain, stitches) in enumerate(
        zip(next_slice_chains, next_stitches, strict=True)
    ):
        is_loop = chain[0] == chain[-1]
        chain_keep = [(True, True) for _ in stitches]
        for stitch_index, stitch in enumerate(stitches):
            discard_before = stitch.link_policy == LinkPolicy.DISCARD
            discard_after = stitch.link_policy == LinkPolicy.DISCARD

            incoming = next_to_front.get(StitchRef.of(next_chain_index, stitch_index))
            if incoming:
                front = incoming[0]
                linked = front_to_next[front]
                if not (
                    len(linked) == 2 and linked[-1] == StitchRef.of(next_chain_index, stitch_index)
                ):
                    front_chain = front.chain_index
                    front_stitch = front.stitch_index
                    if (
                        front_stitch > 0
                        or front_slice_chains[front_chain][0] == front_slice_chains[front_chain][-1]
                    ):
                        previous_front = StitchRef.of(
                            front_chain,
                            front_stitch - 1
                            if front_stitch > 0
                            else len(front_stitches[front_chain]) - 1,
                        )
                        if previous_front not in front_to_next:
                            discard_before = True

                front = incoming[-1]
                linked = front_to_next[front]
                if not (
                    len(linked) == 2 and linked[0] == StitchRef.of(next_chain_index, stitch_index)
                ):
                    front_chain = front.chain_index
                    front_stitch = front.stitch_index
                    if (
                        front_stitch + 1 < len(front_stitches[front_chain])
                        or front_slice_chains[front_chain][0] == front_slice_chains[front_chain][-1]
                    ):
                        next_front_stitch = StitchRef.of(
                            front_chain,
                            front_stitch + 1
                            if front_stitch + 1 < len(front_stitches[front_chain])
                            else 0,
                        )
                        if next_front_stitch not in front_to_next:
                            discard_after = True

            if discard_before:
                if stitch_index > 0:
                    before, _ = chain_keep[stitch_index - 1]
                    chain_keep[stitch_index - 1] = (before, False)
                elif is_loop and chain_keep:
                    before, _ = chain_keep[-1]
                    chain_keep[-1] = (before, False)
                _, after = chain_keep[stitch_index]
                chain_keep[stitch_index] = (False, after)

            if discard_after:
                before, _ = chain_keep[stitch_index]
                chain_keep[stitch_index] = (before, False)
                if stitch_index + 1 < len(stitches):
                    _, after = chain_keep[stitch_index + 1]
                    chain_keep[stitch_index + 1] = (False, after)
                elif is_loop and chain_keep:
                    _, after = chain_keep[0]
                    chain_keep[0] = (False, after)

        kept_row_adjacency.append(chain_keep)

    return FrontTransitionPlan(
        kept_row_adjacency=kept_row_adjacency,
        front_to_next=front_to_next,
        next_to_front=next_to_front,
        discard_front=discard_front,
        column_links=column_links,
    )


def apply_column_links(
    graph: RowColGraph,
    front_graph_vertices: FrontGraphVertices,
    next_graph_vertices: FrontGraphVertices,
    links: list[StitchLink],
) -> None:
    for link in links:
        source = front_graph_vertices[link.front_ref.chain_index][link.front_ref.stitch_index]
        target = next_graph_vertices[link.next_ref.chain_index][link.next_ref.stitch_index]
        if source == UINT32_MAX or target == UINT32_MAX:
            raise ValueError("cannot add a graph link for an unplaced stitch")
        graph.link_column(source, target)


def add_next_front_to_graph(
    slice_mesh: Mesh,
    slice_to_model: SliceToModelMap,
    next_slice_chains: SliceChains,
    next_stitches: FrontStitches,
    keep_adj: list[list[tuple[bool, bool]]],
    graph: RowColGraph,
) -> FrontGraphVertices:
    next_lengths = [chain_lengths(slice_mesh, chain) for chain in next_slice_chains]
    next_vertices: list[list[int]] = []
    for next_chain_index, (chain, stitches) in enumerate(
        zip(next_slice_chains, next_stitches, strict=True)
    ):
        lengths = next_lengths[next_chain_index]
        vertices: list[int] = []
        for stitch in stitches:
            if stitch.link_policy == LinkPolicy.DISCARD:
                vertices.append(UINT32_MAX)
                continue
            right, mix = chain_location(lengths, stitch.chain_t)
            graph_vertex = graph.add_vertex(
                RowColVertex(
                    SurfacePoint.mix(
                        slice_to_model[chain[right - 1]],
                        slice_to_model[chain[right]],
                        mix,
                    )
                )
            )
            vertices.append(graph_vertex)

        previous = vertices[-1] if chain[0] == chain[-1] and vertices else UINT32_MAX
        for stitch_index, current in enumerate(vertices):
            if keep_adj[next_chain_index][stitch_index][0] and previous != UINT32_MAX:
                if current == UINT32_MAX:
                    raise ValueError("kept next row segment touches a discarded stitch")
                graph.link_row(previous, current)
            previous = current
        next_vertices.append(vertices)
    return next_vertices
