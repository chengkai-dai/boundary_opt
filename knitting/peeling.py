"""Public front-peeling operation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from geometry import Mesh, boundary_loop
from knitting._peeling.config import PeelingConfig
from knitting._peeling.planning.pipeline import run_open_peeling
from knitting._peeling.surface import SurfacePoint
from knitting.graph import KnittingGraph
from knitting.tracing import trace_graph_records


@dataclass(frozen=True, slots=True)
class PeelingResult:
    graph: KnittingGraph
    peel_vertex_counts: np.ndarray
    finished: bool
    finish_reason: str


PeelingProgress = Callable[[int], None]


def peel(
    mesh: Mesh,
    scalar_field: np.ndarray,
    initial_front_vertices: np.ndarray,
    config: PeelingConfig = PeelingConfig(),
    *,
    progress: PeelingProgress | None = None,
) -> PeelingResult:
    """Peel a mesh from one consecutive open boundary chain."""
    times = np.asarray(scalar_field, dtype=np.float64).reshape(-1)
    front = np.asarray(initial_front_vertices)
    if front.dtype.kind not in "iu":
        raise ValueError("initial front must contain integer vertex indices")
    front = front.astype(np.int64, copy=False).reshape(-1)
    if times.shape != (len(mesh.vertices),) or not np.isfinite(times).all():
        raise ValueError("scalar field must contain one finite value per mesh vertex")
    if len(front) < 2 or len(np.unique(front)) != len(front):
        raise ValueError("initial front must contain at least two distinct vertices")

    loop = boundary_loop(mesh.faces)
    boundary_edges = {
        tuple(sorted((int(first), int(second))))
        for first, second in zip(loop, np.roll(loop, -1), strict=True)
    }
    if any(
        tuple(sorted((int(first), int(second)))) not in boundary_edges
        for first, second in pairwise(front)
    ):
        raise ValueError("initial front must be one consecutive boundary chain")

    run = run_open_peeling(
        config,
        mesh,
        times,
        [[SurfacePoint.on_vertex(int(vertex)) for vertex in front]],
        progress=(
            None
            if progress is None
            else lambda summary: progress(summary.step + 1)
        ),
    )
    run.graph.validate()
    if run.finished:
        trace_graph_records(run.graph)

    return PeelingResult(
        graph=KnittingGraph(
            points=run.graph.surface_positions(mesh.vertices),
            course_edges=run.graph.row_edges(),
            wale_edges=run.graph.column_edges(),
        ),
        peel_vertex_counts=np.asarray(
            [run.initial_graph_vertex_count, *(step.graph_vertex_count for step in run.steps)],
            dtype=np.int64,
        ),
        finished=run.finished,
        finish_reason=run.finish_reason,
    )


__all__ = ["PeelingConfig", "PeelingResult", "peel"]
