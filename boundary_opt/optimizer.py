"""High-level harmonic problem and backend-independent optimization API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .boundary import (
    canonical_knots,
    cyclic_boundary_profile,
    knots_from_parameters,
    parameters_from_knots,
)
from .defaults import (
    DEFAULT_AREA_WEIGHT,
    DEFAULT_ISOLINE_WEIGHT,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MINIMUM_GAP,
    DEFAULT_UNIFORMITY_WEIGHT,
)
from .harmonic import HarmonicField
from .loss import (
    FieldStatistics,
    area_balance_loss_and_gradient,
    isoline_length_loss_and_gradient,
    uniformity_loss_and_gradient,
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
    area_loss: float
    isoline_loss: float
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
class _Evaluation:
    loss: float
    uniformity_loss: float
    area_loss: float
    isoline_loss: float
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
        area_weight: float = DEFAULT_AREA_WEIGHT,
        isoline_weight: float = DEFAULT_ISOLINE_WEIGHT,
    ) -> None:
        if not np.isfinite(minimum_gap) or not 0.0 < minimum_gap < 0.25:
            raise ValueError("minimum_gap must lie in (0, 0.25)")
        if not np.isfinite(area_weight) or area_weight < 0.0:
            raise ValueError("area_weight must be finite and non-negative")
        if not np.isfinite(uniformity_weight) or uniformity_weight < 0.0:
            raise ValueError("uniformity_weight must be finite and non-negative")
        if not np.isfinite(isoline_weight) or isoline_weight < 0.0:
            raise ValueError("isoline_weight must be finite and non-negative")

        self.mesh = mesh
        self.minimum_gap = float(minimum_gap)
        self.uniformity_weight = float(uniformity_weight)
        self.area_weight = float(area_weight)
        self.isoline_weight = float(isoline_weight)
        self._weight_scale = max(
            self.uniformity_weight, self.area_weight, self.isoline_weight
        )
        if self._weight_scale == 0.0:
            raise ValueError("at least one loss weight must be positive")
        self._relative_weights = np.asarray(
            [self.uniformity_weight, self.area_weight, self.isoline_weight]
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
        area_loss, area_sensitivity = area_balance_loss_and_gradient(
            field, self.mesh.faces, self._face_weights
        )
        isoline_loss, isoline_sensitivity = isoline_length_loss_and_gradient(
            field,
            self.mesh.faces,
            self._gradient_basis,
            self._face_weights,
        )
        uniformity_weight, area_weight, isoline_weight = self._relative_weights
        field_sensitivity = (
            uniformity_weight * field_sensitivity
            + area_weight * area_sensitivity
            + isoline_weight * isoline_sensitivity
        )
        boundary_sensitivity = self.harmonic.solve_adjoint(field_sensitivity)
        knot_gradient = profile_jacobian.T @ boundary_sensitivity

        return _Evaluation(
            loss=(
                uniformity_weight * uniformity_loss
                + area_weight * area_loss
                + isoline_weight * isoline_loss
            ),
            uniformity_loss=uniformity_loss,
            area_loss=area_loss,
            isoline_loss=isoline_loss,
            knot_gradient=knot_gradient,
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
            area_loss=float(evaluation.area_loss),
            isoline_loss=float(evaluation.isoline_loss),
            history=history,
            parameter_history=parameter_history,
            parameters=final_parameters,
            knots=canonical_knots(final_knots),
            gaps=final_gaps,
            field=evaluation.field,
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
