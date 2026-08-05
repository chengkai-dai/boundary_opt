"""High-level harmonic problem and backend-independent optimization API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .boundary import (
    canonical_knots,
    cyclic_boundary_profile,
    gaps_from_knots,
    knot_gap_jacobian,
    knots_from_parameters,
    parameters_from_knots,
)
from .harmonic import HarmonicField
from .loss import (
    FieldStatistics,
    uniformity_loss_and_gradient,
    width_loss_and_gradient,
)
from .mesh import (
    Mesh,
    boundary_arclength,
    face_gradient_basis,
)
from .simplex import (
    constraint_violation,
    project_parameters,
    projected_gradient_residual,
)
from .slsqp_backend import minimize_slsqp
from .spg_backend import minimize_spg

FloatArray = NDArray[np.float64]
BackendName = Literal["slsqp", "spg"]


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    backend: BackendName
    seed: int | None
    initial_loss: float
    final_loss: float
    uniformity_loss: float
    width_loss: float
    history: FloatArray
    parameter_history: FloatArray
    parameters: FloatArray
    knots: FloatArray
    gaps: FloatArray
    field: FloatArray
    statistics: FieldStatistics
    iterations: int
    evaluations: int
    gradient_norm: float
    kkt_residual: float
    constraint_violation: float


@dataclass(frozen=True, slots=True)
class _PhysicalEvaluation:
    loss: float
    uniformity_loss: float
    width_loss: float
    knot_gradient: FloatArray
    field: FloatArray
    statistics: FieldStatistics


class _Objective:
    """Cache repeated evaluations and count physical solves."""

    def __init__(self, optimizer: BoundaryOptimizer) -> None:
        self.optimizer = optimizer
        self.cache: dict[bytes, tuple[float, FloatArray]] = {}
        self.evaluations = 0

    def __call__(self, parameters: FloatArray) -> tuple[float, FloatArray]:
        parameters = np.asarray(parameters, dtype=np.float64).reshape(-1)
        key = parameters.tobytes()
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        self.evaluations += 1
        loss, gradient = self.optimizer._state_loss_and_gradient(
            parameters, enforce_minimum_gap=False
        )
        record = float(loss), np.asarray(gradient, dtype=np.float64)
        self.cache[key] = record
        return record


class BoundaryOptimizer:
    """Optimize a four-knot Dirichlet trace on one-boundary triangle meshes."""

    def __init__(
        self,
        mesh: Mesh,
        *,
        minimum_gap: float = 0.03,
        target_arc_width: float | None = None,
        width_weight: float = 0.0,
    ) -> None:
        if not np.isfinite(minimum_gap) or not 0.0 < minimum_gap < 0.25:
            raise ValueError("minimum_gap must lie in (0, 0.25)")
        if target_arc_width is not None and (
            not np.isfinite(target_arc_width)
            or not minimum_gap < target_arc_width < 0.5 - minimum_gap
        ):
            raise ValueError(
                "target_arc_width must lie between minimum_gap and 0.5 - minimum_gap"
            )
        if not np.isfinite(width_weight) or width_weight < 0.0:
            raise ValueError("width_weight must be finite and non-negative")
        if width_weight > 0.0 and target_arc_width is None:
            raise ValueError("positive width_weight requires target_arc_width")

        self.mesh = mesh
        self.minimum_gap = float(minimum_gap)
        self.target_arc_width = (
            None if target_arc_width is None else float(target_arc_width)
        )
        self.width_weight = float(width_weight)
        self.harmonic = HarmonicField(mesh)
        self.boundary_positions = boundary_arclength(
            mesh.vertices, self.harmonic.boundary_vertices
        )

        self.face_areas, self._gradient_basis = face_gradient_basis(mesh)
        self._face_weights = self.face_areas / self.face_areas.sum()

    def _evaluate_knots(self, knots: FloatArray) -> _PhysicalEvaluation:
        boundary_values, profile_jacobian = cyclic_boundary_profile(
            self.boundary_positions, knots
        )
        field = self.harmonic.solve(boundary_values)
        uniformity_loss, field_sensitivity, statistics = uniformity_loss_and_gradient(
            field,
            self.mesh.faces,
            self._gradient_basis,
            self._face_weights,
        )
        boundary_sensitivity = self.harmonic.solve_adjoint(field_sensitivity)
        knot_gradient = profile_jacobian.T @ boundary_sensitivity

        gaps = gaps_from_knots(knots)
        width_loss, gap_gradient = width_loss_and_gradient(
            gaps, self.target_arc_width, self.width_weight
        )
        knot_gradient = knot_gradient + knot_gap_jacobian().T @ gap_gradient
        return _PhysicalEvaluation(
            loss=uniformity_loss + width_loss,
            uniformity_loss=uniformity_loss,
            width_loss=width_loss,
            knot_gradient=knot_gradient,
            field=field,
            statistics=statistics,
        )

    def loss_and_knot_gradient(self, knots: FloatArray) -> tuple[float, FloatArray]:
        """Return total loss and exact gradient with respect to four knots."""
        evaluation = self._evaluate_knots(np.asarray(knots, dtype=np.float64))
        return evaluation.loss, evaluation.knot_gradient

    def _state_loss_and_gradient(
        self, parameters: FloatArray, *, enforce_minimum_gap: bool
    ) -> tuple[float, FloatArray]:
        knots, knot_jacobian, _ = knots_from_parameters(
            parameters,
            self.minimum_gap,
            enforce_minimum_gap=enforce_minimum_gap,
        )
        evaluation = self._evaluate_knots(knots)
        return evaluation.loss, knot_jacobian.T @ evaluation.knot_gradient

    def loss_and_gradient(self, parameters: FloatArray) -> tuple[float, FloatArray]:
        """Return loss and gradient in centered-phase full-gap coordinates."""
        return self._state_loss_and_gradient(parameters, enforce_minimum_gap=True)

    def optimize(
        self,
        initial_knots: FloatArray,
        *,
        backend: BackendName = "slsqp",
        max_iterations: int = 100,
        seed: int | None = None,
    ) -> OptimizationResult:
        """Optimize from four knots with the selected interchangeable backend."""
        if backend not in ("slsqp", "spg"):
            raise ValueError("backend must be 'slsqp' or 'spg'")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")

        initial_parameters = parameters_from_knots(initial_knots, self.minimum_gap)
        initial_parameters = project_parameters(initial_parameters, self.minimum_gap)
        objective = _Objective(self)

        if backend == "slsqp":
            solver_result = minimize_slsqp(
                objective, initial_parameters, self.minimum_gap, max_iterations
            )
        else:
            solver_result = minimize_spg(
                objective, initial_parameters, self.minimum_gap, max_iterations
            )

        if not solver_result.success:
            raise RuntimeError(f"{backend} failed: {solver_result.message}")
        final_parameters = np.asarray(solver_result.x, dtype=np.float64).copy()

        parameter_history = np.asarray(
            solver_result.iterate_history, dtype=np.float64
        ).copy()
        history = np.asarray(solver_result.loss_history, dtype=np.float64).copy()
        initial_loss = float(history[0])

        _, final_gradient = objective(final_parameters)
        final_knots, _, final_gaps = knots_from_parameters(
            final_parameters, self.minimum_gap
        )
        physical = self._evaluate_knots(final_knots)
        if np.array_equal(final_parameters, parameter_history[-1]):
            history[-1] = physical.loss
        else:
            parameter_history = np.vstack((parameter_history, final_parameters))
            history = np.append(history, physical.loss)

        final_parameters[0] %= 1.0
        parameter_history[:, 0] %= 1.0
        tangent_gradient = final_gradient.copy()
        tangent_gradient[1:] -= tangent_gradient[1:].mean()

        return OptimizationResult(
            backend=backend,
            seed=seed,
            initial_loss=float(initial_loss),
            final_loss=float(physical.loss),
            uniformity_loss=float(physical.uniformity_loss),
            width_loss=float(physical.width_loss),
            history=history,
            parameter_history=parameter_history,
            parameters=final_parameters,
            knots=canonical_knots(final_knots),
            gaps=final_gaps,
            field=physical.field,
            statistics=physical.statistics,
            iterations=int(solver_result.nit),
            evaluations=objective.evaluations,
            gradient_norm=float(np.linalg.norm(tangent_gradient)),
            kkt_residual=projected_gradient_residual(
                final_parameters, final_gradient, self.minimum_gap
            ),
            constraint_violation=constraint_violation(
                final_parameters, self.minimum_gap
            ),
        )

    def optimize_backends(
        self,
        initial_knots: FloatArray,
        *,
        max_iterations: int = 100,
        seed: int | None = None,
    ) -> dict[BackendName, OptimizationResult]:
        """Run both backends from exactly the same physical initialization."""
        return {
            backend: self.optimize(
                initial_knots,
                backend=backend,
                max_iterations=max_iterations,
                seed=seed,
            )
            for backend in ("slsqp", "spg")
        }
