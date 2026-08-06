from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from knitting.graph import RowColGraph
from geometry import Mesh
from knitting._peeling.config import PeelingConfig
from knitting._peeling.planning.chain_geometry import chain_lengths
from knitting._peeling.planning.front_transition import build_next_front
from knitting._peeling.planning.linking import link_chains
from knitting._peeling.planning.matching import Matches, build_chain_matches
from knitting._peeling.planning.peeling import initialize_active_front, peel_slice
from knitting._peeling.planning.types import (
    ActiveFront,
    FrontGraphVertices,
    FrontStitches,
    ModelChains,
    SliceChains,
)


@dataclass(frozen=True)
class PeelingStepSummary:
    step: int
    front_chain_point_counts: list[int]
    next_front_chain_point_counts: list[int]
    next_front_stitch_counts: list[int]
    link_count: int
    graph_vertex_count: int


@dataclass
class PeelingRun:
    current_front: ActiveFront
    graph: RowColGraph
    initial_graph_vertex_count: int = 0
    steps: list[PeelingStepSummary] = field(default_factory=list)
    finished: bool = False
    finish_reason: str = "not-started"

    @property
    def front_chains(self) -> ModelChains:
        return self.current_front.model_chains

    @property
    def front_stitches(self) -> FrontStitches:
        return self.current_front.stitches

    @property
    def front_graph_vertices(self) -> FrontGraphVertices:
        return self.current_front.graph_vertices


def run_open_peeling(
    parameters: PeelingConfig,
    mesh: Mesh,
    times: np.ndarray,
    initial_chains: ModelChains,
    *,
    progress: Callable[[PeelingStepSummary], None] | None = None,
) -> PeelingRun:
    """Peel, link, and build a graph from explicit initial-front chains."""

    times = np.asarray(times, dtype=np.float64)
    if times.shape != (mesh.vertex_count,):
        raise ValueError("times must have one entry per mesh vertex")
    if not np.isfinite(times).all():
        raise ValueError("times must contain only finite values")

    initial = initialize_active_front(
        parameters,
        mesh,
        times,
        initial_chains,
    )
    run = PeelingRun(
        current_front=initial.front,
        graph=initial.graph,
        initial_graph_vertex_count=len(initial.graph.vertices),
        finished=False,
        finish_reason="not-finished",
    )

    step = 0
    while True:
        front_chain_point_counts = [len(chain) for chain in run.front_chains]
        sliced = peel_slice(parameters, mesh, run.front_chains)
        if not sliced.next_slice_chains:
            run.finished = True
            run.finish_reason = "no-next-chains"
            break
        if not sliced.front_slice_chains:
            run.finished = True
            run.finish_reason = "no-active-front-chains"
            break

        active_front_stitches = [run.front_stitches[index] for index in sliced.front_chain_indices]
        active_front_graph_vertices = [
            run.front_graph_vertices[index] for index in sliced.front_chain_indices
        ]

        slice_times = np.asarray(
            [float(vertex.interpolate(times)) for vertex in sliced.slice_to_model],
            dtype=np.float64,
        )
        try:
            advances, matches = _matched_fronts_advance_in_time(
                sliced.slice_mesh,
                slice_times,
                sliced.front_slice_chains,
                active_front_stitches,
                sliced.next_slice_chains,
            )
        except RuntimeError:
            run.finished = False
            run.finish_reason = "matching-failed"
            break
        if not advances:
            run.finished = True
            run.finish_reason = "non-increasing-time"
            break
        try:
            linked = link_chains(
                parameters,
                sliced.slice_mesh,
                slice_times,
                sliced.front_slice_chains,
                active_front_stitches,
                sliced.next_slice_chains,
                matches,
            )
        except RuntimeError:
            run.finished = False
            run.finish_reason = "linking-failed"
            break
        built = build_next_front(
            parameters,
            sliced.slice_mesh,
            sliced.slice_to_model,
            sliced.front_slice_chains,
            active_front_stitches,
            active_front_graph_vertices,
            sliced.next_slice_chains,
            linked.next_stitches,
            linked.links,
            run.graph,
        )

        summary = PeelingStepSummary(
            step=step,
            front_chain_point_counts=front_chain_point_counts,
            next_front_chain_point_counts=[len(chain) for chain in built.front_chains],
            next_front_stitch_counts=[len(stitches) for stitches in built.front_stitches],
            link_count=len(linked.links),
            graph_vertex_count=len(run.graph.vertices),
        )
        run.steps.append(summary)
        if progress is not None:
            progress(summary)

        run.current_front = built.front
        if not run.front_chains:
            run.finished = True
            run.finish_reason = "empty-front-chains"
            break
        step += 1

    return run


def _matched_fronts_advance_in_time(
    mesh: Mesh,
    times: np.ndarray,
    front_chains: SliceChains,
    front_stitches: FrontStitches,
    next_chains: SliceChains,
) -> tuple[bool, Matches]:
    """Return whether at least one matched next chain advances in harmonic time."""

    front_lengths = [chain_lengths(mesh, chain) for chain in front_chains]
    next_lengths = [chain_lengths(mesh, chain) for chain in next_chains]
    front_times = [
        _length_weighted_chain_time(times, chain, lengths)
        for chain, lengths in zip(front_chains, front_lengths, strict=True)
    ]
    next_times = [
        _length_weighted_chain_time(times, chain, lengths)
        for chain, lengths in zip(next_chains, next_lengths, strict=True)
    ]
    matches = build_chain_matches(
        mesh,
        front_chains,
        front_lengths,
        front_stitches,
        next_chains,
        next_lengths,
    )

    # Harmonic time has an arbitrary additive gauge.  Only its range can
    # define a progress tolerance; using absolute values made the decision
    # change when the same field was shifted by a constant.
    tolerance = 1.0e-7 * float(np.ptp(times))
    matched_deltas = [
        next_times[next_index] - front_times[front_index]
        for (front_index, next_index), match in matches.items()
        if front_index < len(front_times)
        and next_index < len(next_times)
        and match.front
        and match.next
    ]
    if matched_deltas:
        return max(matched_deltas) > tolerance, matches

    front_time = _length_weighted_front_time(front_times, front_lengths)
    next_time = _length_weighted_front_time(next_times, next_lengths)
    return next_time > front_time + tolerance, matches


def _length_weighted_chain_time(
    times: np.ndarray,
    chain: list[int],
    lengths: list[float],
) -> float:
    segment_lengths = np.diff(np.asarray(lengths, dtype=np.float64))
    total_length = float(np.sum(segment_lengths))
    if total_length <= 0.0:
        return float(np.mean(times[np.asarray(chain, dtype=np.int64)]))
    indices = np.asarray(chain, dtype=np.int64)
    segment_times = 0.5 * (times[indices[:-1]] + times[indices[1:]])
    return float(np.dot(segment_lengths, segment_times) / total_length)


def _length_weighted_front_time(
    representatives: list[float],
    lengths: list[list[float]],
) -> float:
    weights = np.asarray([chain[-1] for chain in lengths], dtype=np.float64)
    if float(np.sum(weights)) <= 0.0:
        return float(np.mean(representatives))
    return float(np.average(np.asarray(representatives), weights=weights))
