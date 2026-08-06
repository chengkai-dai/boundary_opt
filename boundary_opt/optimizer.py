"""High-level harmonic problem and backend-independent optimization API."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from geometry import Mesh

from .boundary import (
    boundary_arclength,
    canonical_knots,
    cyclic_boundary_profile,
    knots_from_parameters,
    parameters_from_knots,
)
from .defaults import (
    DEFAULT_LENGTH_SMOOTHNESS_WEIGHT,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MINIMUM_GAP,
    DEFAULT_UNIFORMITY_WEIGHT,
)
from .fem import face_gradient_basis
from .harmonic import HarmonicField
from .loss import (
    FieldStatistics,
    length_smoothness_loss_and_gradient,
    uniformity_loss_and_gradient,
)
from .simplex import (
    constraint_violation,
    project_parameters,
    projected_gradient_residual,
)
from .slsqp_backend import minimize_slsqp
from .spg_backend import minimize_spg

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BackendName = Literal["slsqp", "spg"]


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    backend: BackendName
    seed: int | None
    initial_loss: float
    final_loss: float
    uniformity_loss: float
    length_smoothness_loss: float
    history: FloatArray
    parameter_history: FloatArray
    parameters: FloatArray
    knots: FloatArray
    gaps: FloatArray
    field: FloatArray
    minimum_vertices: IntArray
    maximum_vertices: IntArray
    statistics: FieldStatistics
    iterations: int
    evaluations: int
    gradient_norm: float
    kkt_residual: float
    constraint_violation: float


@dataclass(frozen=True, slots=True)
class _Evaluation:
    loss: float
    uniformity_loss: float
    length_smoothness_loss: float
    knot_gradient: FloatArray
    boundary_values: FloatArray
    field: FloatArray
    statistics: FieldStatistics


def _ordered_cyclic_vertices(vertices: IntArray, selected: np.ndarray) -> IntArray:
    """Preserve the boundary order of one possibly wrapped vertex run."""
    if not np.any(selected):
        return np.empty(0, dtype=np.int64)
    if np.all(selected):
        return np.asarray(vertices, dtype=np.int64).copy()
    starts = np.flatnonzero(selected & ~np.roll(selected, 1))
    if len(starts) != 1:
        raise RuntimeError("boundary plateau is not one nonempty cyclic run")
    order = (int(starts[0]) + np.arange(np.count_nonzero(selected))) % len(selected)
    return np.asarray(vertices[order], dtype=np.int64)


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
            parameters,
            enforce_minimum_gap=False,
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
        minimum_gap: float = DEFAULT_MINIMUM_GAP,
        uniformity_weight: float = DEFAULT_UNIFORMITY_WEIGHT,
        length_smoothness_weight: float = DEFAULT_LENGTH_SMOOTHNESS_WEIGHT,
    ) -> None:
        if not np.isfinite(minimum_gap) or not 0.0 < minimum_gap < 0.25:
            raise ValueError("minimum_gap must lie in (0, 0.25)")
        if not np.isfinite(uniformity_weight) or uniformity_weight < 0.0:
            raise ValueError("uniformity_weight must be finite and non-negative")
        if (
            not np.isfinite(length_smoothness_weight)
            or length_smoothness_weight < 0.0
        ):
            raise ValueError(
                "length_smoothness_weight must be finite and non-negative"
            )

        self.mesh = mesh
        self.minimum_gap = float(minimum_gap)
        self.uniformity_weight = float(uniformity_weight)
        self.length_smoothness_weight = float(length_smoothness_weight)
        self._weight_scale = max(
            self.uniformity_weight, self.length_smoothness_weight
        )
        if self._weight_scale == 0.0:
            raise ValueError("at least one loss weight must be positive")
        self._relative_weights = np.asarray(
            [self.uniformity_weight, self.length_smoothness_weight]
        ) / self._weight_scale
        self.harmonic = HarmonicField(mesh)
        self.boundary_positions = boundary_arclength(
            mesh.vertices, self.harmonic.boundary_vertices
        )

        self.face_areas, self._gradient_basis = face_gradient_basis(mesh)
        self._face_weights = self.face_areas / self.face_areas.sum()

    def _evaluate_knots(self, knots: FloatArray) -> _Evaluation:
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
        length_smoothness_loss, length_smoothness_sensitivity = (
            length_smoothness_loss_and_gradient(
                field,
                self.mesh.faces,
                self._gradient_basis,
                self._face_weights,
            )
        )
        uniformity_weight, length_smoothness_weight = self._relative_weights
        field_sensitivity = (
            uniformity_weight * field_sensitivity
            + length_smoothness_weight * length_smoothness_sensitivity
        )
        boundary_sensitivity = self.harmonic.solve_adjoint(field_sensitivity)
        knot_gradient = profile_jacobian.T @ boundary_sensitivity

        return _Evaluation(
            loss=(
                uniformity_weight * uniformity_loss
                + length_smoothness_weight * length_smoothness_loss
            ),
            uniformity_loss=uniformity_loss,
            length_smoothness_loss=length_smoothness_loss,
            knot_gradient=knot_gradient,
            boundary_values=boundary_values,
            field=field,
            statistics=statistics,
        )

    def loss_and_knot_gradient(self, knots: FloatArray) -> tuple[float, FloatArray]:
        """Return total loss and exact gradient with respect to four knots."""
        evaluation = self._evaluate_knots(np.asarray(knots, dtype=np.float64))
        return (
            self._weight_scale * evaluation.loss,
            self._weight_scale * evaluation.knot_gradient,
        )

    def _state_loss_and_gradient(
        self,
        parameters: FloatArray,
        *,
        enforce_minimum_gap: bool,
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
        loss, gradient = self._state_loss_and_gradient(
            parameters, enforce_minimum_gap=True
        )
        return self._weight_scale * loss, self._weight_scale * gradient

    def optimize(
        self,
        initial_knots: FloatArray,
        *,
        backend: BackendName = "slsqp",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
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
        history = self._weight_scale * np.asarray(
            solver_result.loss_history, dtype=np.float64
        )
        initial_loss = float(history[0])

        _, final_gradient = objective(final_parameters)
        final_knots, _, final_gaps = knots_from_parameters(
            final_parameters, self.minimum_gap
        )
        evaluation = self._evaluate_knots(final_knots)
        if np.array_equal(final_parameters, parameter_history[-1]):
            history[-1] = self._weight_scale * evaluation.loss
        else:
            parameter_history = np.vstack((parameter_history, final_parameters))
            history = np.append(history, self._weight_scale * evaluation.loss)

        final_parameters[0] %= 1.0
        parameter_history[:, 0] %= 1.0
        tangent_gradient = final_gradient.copy()
        tangent_gradient[1:] -= tangent_gradient[1:].mean()

        return OptimizationResult(
            backend=backend,
            seed=seed,
            initial_loss=float(initial_loss),
            final_loss=float(self._weight_scale * evaluation.loss),
            uniformity_loss=float(evaluation.uniformity_loss),
            length_smoothness_loss=float(evaluation.length_smoothness_loss),
            history=history,
            parameter_history=parameter_history,
            parameters=final_parameters,
            knots=canonical_knots(final_knots),
            gaps=final_gaps,
            field=evaluation.field,
            minimum_vertices=_ordered_cyclic_vertices(
                self.harmonic.boundary_vertices,
                evaluation.boundary_values == 0.0,
            ),
            maximum_vertices=_ordered_cyclic_vertices(
                self.harmonic.boundary_vertices,
                evaluation.boundary_values == 1.0,
            ),
            statistics=evaluation.statistics,
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
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
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

    def optimize_multistart(
        self,
        initial_knots: FloatArray,
        *,
        backend: BackendName = "slsqp",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        seed: int | None = None,
    ) -> OptimizationResult:
        """Return the best of the input and two balanced physical starts."""
        initial_parameters = parameters_from_knots(initial_knots, self.minimum_gap)
        balanced = np.concatenate(([initial_parameters[0]], np.full(4, 0.25)))
        shifted = balanced.copy()
        shifted[0] += 0.25
        starts = [
            np.asarray(initial_knots, dtype=np.float64).copy(),
            knots_from_parameters(balanced, self.minimum_gap)[0],
            knots_from_parameters(shifted, self.minimum_gap)[0],
        ]
        runs = [
            self.optimize(
                knots,
                backend=backend,
                max_iterations=max_iterations,
                seed=seed,
            )
            for knots in starts
        ]
        best = min(runs, key=lambda result: result.final_loss)
        return replace(
            best,
            iterations=sum(run.iterations for run in runs),
            evaluations=sum(run.evaluations for run in runs),
        )
