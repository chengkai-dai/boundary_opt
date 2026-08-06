from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from knitting.graph import RowColGraph, RowColVertex
from geometry import Mesh
from knitting._peeling.contours import extract_level_chains
from knitting._peeling.surface import (
    UINT32_MAX,
    SurfacePoint,
    trim_surface,
)
from knitting._peeling.config import PeelingConfig
from knitting._peeling.planning.chain_geometry import chain_location
from knitting._peeling.planning.stitch_state import FrontStitchSample, LinkPolicy
from knitting._peeling.planning.types import (
    ActiveFront,
    FrontGraphVertices,
    FrontStitches,
    ModelChain,
    ModelChains,
    SliceChains,
    SliceToModelMap,
)


@dataclass(frozen=True)
class InitialActiveFront:
    front: ActiveFront
    graph: RowColGraph


@dataclass(frozen=True)
class PeelSlice:
    slice_mesh: Mesh
    slice_to_model: SliceToModelMap
    front_chain_indices: list[int]
    front_slice_chains: SliceChains
    next_slice_chains: SliceChains


def sample_chain(spacing: float, mesh: Mesh, chain: ModelChain) -> ModelChain:
    if not chain:
        return []

    sampled: list[SurfacePoint] = []
    for current, next_vertex in pairwise(chain):
        sampled.append(current)
        a = current.interpolate(mesh.vertices)
        b = next_vertex.interpolate(mesh.vertices)
        length = float(np.linalg.norm(b - a))
        insert = math.floor(length / spacing)
        for i in range(insert):
            mix = (i + 1) / float(insert + 1)
            sampled.append(SurfacePoint.mix(current, next_vertex, mix))
    sampled.append(chain[-1])
    return sampled


def initialize_active_front(
    parameters: PeelingConfig,
    mesh: Mesh,
    times: np.ndarray,
    initial_chains: Sequence[ModelChain],
) -> InitialActiveFront:
    if len(times) != len(mesh.vertices):
        raise ValueError("times must have one entry per mesh vertex")

    front_chains: ModelChains = []
    front_stitches: FrontStitches = []
    if not initial_chains:
        raise ValueError("at least one explicit initial chain is required")
    for chain in initial_chains:
        if len(chain) < 2:
            raise ValueError("initial chains need at least two surface points")
        _append_initial_front_chain(parameters, mesh, list(chain), front_chains, front_stitches)
    if not front_chains:
        raise ValueError("initial chains have zero length")

    graph = RowColGraph()
    front_graph_vertices: FrontGraphVertices = []
    for chain, stitches in zip(front_chains, front_stitches, strict=True):
        lengths = cumulative_lengths(mesh, chain)
        chain_graph_vertices: list[int] = []
        for stitch in stitches:
            right, mix = chain_location(lengths, stitch.chain_t)
            graph_vertex = graph.add_vertex(
                RowColVertex(SurfacePoint.mix(chain[right - 1], chain[right], mix))
            )
            chain_graph_vertices.append(graph_vertex)

        previous = (
            chain_graph_vertices[-1]
            if chain[0] == chain[-1] and chain_graph_vertices
            else UINT32_MAX
        )
        for graph_vertex in chain_graph_vertices:
            if previous != UINT32_MAX:
                graph.link_row(previous, graph_vertex)
            previous = graph_vertex
        front_graph_vertices.append(chain_graph_vertices)

    return InitialActiveFront(
        ActiveFront(front_chains, front_stitches, front_graph_vertices),
        graph,
    )


def _append_initial_front_chain(
    parameters: PeelingConfig,
    mesh: Mesh,
    chain: ModelChain,
    front_chains: ModelChains,
    front_stitches: FrontStitches,
) -> None:
    divided_chain = sample_chain(parameters.chain_sample_spacing, mesh, chain)
    total_length = chain_length(mesh, divided_chain)
    if total_length <= 0.0:
        return

    stitch_count = max(3, math.ceil(total_length / parameters.stitch_spacing))
    front_chains.append(divided_chain)
    front_stitches.append(
        [
            FrontStitchSample(float((index + 0.5) / float(stitch_count)), LinkPolicy.LINK_ANY)
            for index in range(stitch_count)
        ]
    )


def peel_slice(
    parameters: PeelingConfig,
    mesh: Mesh,
    front_chains: ModelChains,
) -> PeelSlice:
    clipped = trim_surface(mesh, front_chains, [])
    values = _distance_to_chains(clipped.mesh.vertices, mesh, front_chains)
    level = parameters.course_spacing
    level_chains = extract_level_chains(clipped.mesh.faces, values, level)

    next_front_chains: ModelChains = []
    for clipped_chain in level_chains:
        on_model = [vertex.compose(clipped.vertex_sources) for vertex in clipped_chain]
        next_front_chains.append(sample_chain(parameters.chain_sample_spacing, mesh, on_model))

    sliced = trim_surface(mesh, front_chains, next_front_chains)
    front_chain_indices: list[int] = []
    front_slice_chains: SliceChains = []
    for index, raw_chain in enumerate(sliced.left_chains):
        chain = _clean_clipped_chain(raw_chain)
        if not chain:
            continue
        if UINT32_MAX in raw_chain:
            raise ValueError("an active front was only partially retained by surface trimming")
        if not _slice_chain_has_positive_length(sliced.mesh, chain):
            raise ValueError("a retained active front has zero geometric length")
        front_chain_indices.append(index)
        front_slice_chains.append(chain)
    next_slice_chains = [
        chain
        for raw_chain in sliced.right_chains
        if (chain := _clean_clipped_chain(raw_chain))
        and _slice_chain_has_positive_length(sliced.mesh, chain)
    ]

    return PeelSlice(
        slice_mesh=sliced.mesh,
        slice_to_model=sliced.vertex_sources,
        front_chain_indices=front_chain_indices,
        front_slice_chains=front_slice_chains,
        next_slice_chains=next_slice_chains,
    )


def cumulative_lengths(mesh: Mesh, chain: list[SurfacePoint]) -> list[float]:
    lengths = [0.0]
    for previous, current in pairwise(chain):
        a = previous.interpolate(mesh.vertices)
        b = current.interpolate(mesh.vertices)
        segment = float(np.linalg.norm(b - a))
        lengths.append(float(lengths[-1] + segment))
    return lengths


def chain_length(mesh: Mesh, chain: list[SurfacePoint]) -> float:
    return cumulative_lengths(mesh, chain)[-1]


def _distance_to_chains(
    points: np.ndarray, mesh: Mesh, chains: list[list[SurfacePoint]]
) -> np.ndarray:
    point_array = np.asarray(points, dtype=float)
    vertices = np.asarray(mesh.vertices, dtype=float)
    values = np.full(len(point_array), np.inf, dtype=float)
    segments = [
        (current.interpolate(vertices), next_vertex.interpolate(vertices))
        for chain in chains
        for current, next_vertex in pairwise(chain)
    ]
    if not segments:
        return np.sqrt(values)
    starts, ends = (np.asarray(part) for part in zip(*segments, strict=True))
    directions = ends - starts
    limits = np.einsum("ij,ij->i", directions, directions)
    valid = limits > 0.0
    starts, directions, limits = starts[valid], directions[valid], limits[valid]

    # Bound the temporary (points x segments x xyz) array on large fronts.
    for begin in range(0, len(starts), 64):
        batch_starts = starts[begin : begin + 64]
        batch_directions = directions[begin : begin + 64]
        batch_limits = limits[begin : begin + 64]
        offsets = point_array[:, None, :] - batch_starts[None, :, :]
        amounts = np.einsum("nsi,si->ns", offsets, batch_directions)
        amounts = np.clip(amounts / batch_limits, 0.0, 1.0)
        residuals = offsets - amounts[:, :, None] * batch_directions
        distances = np.einsum("nsi,nsi->ns", residuals, residuals)
        values = np.minimum(values, np.min(distances, axis=1))
    return np.sqrt(values)


def _clean_clipped_chain(chain: list[int]) -> list[int]:
    valid = [index for index, vertex in enumerate(chain) if vertex != UINT32_MAX]
    if not valid:
        return []
    first, last = valid[0], valid[-1]
    if UINT32_MAX in chain[first : last + 1]:
        raise ValueError("surface trimming split one chain into multiple components")
    out = [chain[first]]
    for vertex in chain[first + 1 : last + 1]:
        if vertex != out[-1]:
            out.append(vertex)
    return out if len(out) >= 2 else []


def _slice_chain_has_positive_length(mesh: Mesh, chain: list[int]) -> bool:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return any(
        np.linalg.norm(vertices[current] - vertices[previous]) > 0.0
        for previous, current in pairwise(chain)
    )
