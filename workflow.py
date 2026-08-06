"""Application workflow connecting boundary optimization to front peeling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from boundary_opt import BackendName, BoundaryOptimizer, OptimizationResult, random_knots
from geometry import Mesh
from knitting import PeelingConfig, PeelingResult, peel, sample_boundary_course


@dataclass(frozen=True, slots=True)
class PipelineResult:
    mesh: Mesh
    boundary: OptimizationResult
    initial_course: np.ndarray
    peeling: PeelingResult


def run_pipeline(
    mesh: Mesh,
    *,
    seed: int = 0,
    backend: BackendName = "slsqp",
    max_iterations: int = 500,
    config: PeelingConfig = PeelingConfig(),
    progress: Callable[[int], None] | None = None,
) -> PipelineResult:
    """Optimize a uniform harmonic field and peel its knitting graph."""
    optimizer = BoundaryOptimizer(mesh)
    boundary = optimizer.optimize_multistart(
        random_knots(seed, optimizer.minimum_gap),
        backend=backend,
        max_iterations=max_iterations,
        seed=seed,
    )
    initial_course = sample_boundary_course(
        mesh,
        optimizer.harmonic.boundary_vertices,
        optimizer.boundary_positions,
        boundary.knots[0],
        boundary.knots[1],
        config.stitch_spacing,
    )
    if progress is not None:
        progress(0)
    peeling = peel(
        mesh,
        boundary.field,
        initial_course,
        config,
        progress=progress,
    )
    return PipelineResult(mesh, boundary, initial_course, peeling)
