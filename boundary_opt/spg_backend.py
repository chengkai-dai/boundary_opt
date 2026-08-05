"""Spectral projected-gradient optimization on the centered full-gap simplex."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.optimize
from numpy.typing import NDArray

from .simplex import (
    parameters_are_feasible,
    project_parameters,
    projected_gradient_residual,
)

FloatArray = NDArray[np.float64]
ValueAndGradient = Callable[[FloatArray], tuple[float, FloatArray]]

_ALPHA_MIN = 1.0e-10
_ARMIJO = 1.0e-4
_BACKTRACK = 0.5
_MAX_BACKTRACKS = 40
_NONMONOTONE_WINDOW = 10


def minimize_spg(
    value_and_grad: ValueAndGradient,
    initial: FloatArray,
    minimum_gap: float,
    max_iterations: int,
    *,
    tolerance: float = 1.0e-6,
) -> scipy.optimize.OptimizeResult:
    """Minimize an objective on the centered full-gap simplex with SPG."""
    parameters = np.asarray(initial, dtype=np.float64).reshape(-1).copy()
    if not parameters_are_feasible(parameters, minimum_gap):
        raise ValueError("initial parameters must be feasible")

    evaluations = 0

    def evaluate(values: FloatArray) -> tuple[float, FloatArray]:
        nonlocal evaluations
        evaluations += 1
        return value_and_grad(values)

    loss, gradient = evaluate(parameters)

    iterates = [parameters.copy()]
    losses = [float(loss)]
    alpha = max(_ALPHA_MIN, 1.0 / max(float(np.linalg.norm(gradient)), 1.0e-12))
    alpha_max = max(1.0, alpha)
    status = 1
    message = "Maximum iterations reached"

    for _ in range(max_iterations):
        residual = projected_gradient_residual(parameters, gradient, minimum_gap)
        if residual <= tolerance:
            break

        projected = project_parameters(parameters - alpha * gradient, minimum_gap)
        direction = projected - parameters
        slope = float(gradient @ direction)
        if slope >= 0.0:
            status = 2
            message = "Projected direction is not a descent direction"
            break

        reference = max(losses[-_NONMONOTONE_WINDOW:])
        step = 1.0
        for _ in range(_MAX_BACKTRACKS):
            trial = project_parameters(parameters + step * direction, minimum_gap)
            trial_loss, trial_gradient = evaluate(trial)
            if trial_loss <= reference + _ARMIJO * step * slope:
                break
            step *= _BACKTRACK
        else:
            status = 3
            message = "Nonmonotone Armijo line search failed"
            break

        displacement = trial - parameters
        gradient_change = trial_gradient - gradient
        squared_displacement = float(displacement @ displacement)
        curvature = float(displacement @ gradient_change)
        if curvature > 0.0:
            alpha = float(
                np.clip(
                    squared_displacement / curvature,
                    _ALPHA_MIN,
                    alpha_max,
                )
            )
        else:
            alpha = alpha_max

        parameters, loss, gradient = trial, trial_loss, trial_gradient
        iterates.append(parameters.copy())
        losses.append(float(loss))

    residual = float(projected_gradient_residual(parameters, gradient, minimum_gap))
    success = residual <= tolerance
    if success:
        status = 0
        message = "Projected-gradient tolerance reached"

    return scipy.optimize.OptimizeResult(
        x=parameters,
        fun=loss,
        jac=gradient,
        nit=len(iterates) - 1,
        nfev=evaluations,
        success=success,
        status=status,
        message=message,
        projected_gradient_residual=residual,
        iterate_history=np.vstack(iterates),
        loss_history=np.asarray(losses, dtype=np.float64),
    )
