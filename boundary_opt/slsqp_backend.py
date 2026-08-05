"""SciPy SLSQP backend for the centered full-gap simplex."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.optimize
from numpy.typing import NDArray

from .simplex import parameters_are_feasible, projected_gradient_residual

FloatArray = NDArray[np.float64]
ValueAndGradient = Callable[[FloatArray], tuple[float, FloatArray]]


def minimize_slsqp(
    value_and_grad: ValueAndGradient,
    initial: FloatArray,
    minimum_gap: float,
    max_iterations: int,
    *,
    tolerance: float = 1.0e-12,
) -> scipy.optimize.OptimizeResult:
    """Run one exact-gradient SLSQP solve on the full-gap simplex."""
    initial = np.asarray(initial, dtype=np.float64).reshape(-1).copy()
    if not parameters_are_feasible(initial, minimum_gap):
        raise ValueError("initial parameters must be feasible")

    initial_loss, _ = value_and_grad(initial)
    objective_scale = max(abs(float(initial_loss)), np.finfo(np.float64).eps)
    iterates = [initial.copy()]
    losses = [float(initial_loss)]

    def scipy_objective(values: FloatArray) -> tuple[float, FloatArray]:
        loss, gradient = value_and_grad(values)
        return float(loss) / objective_scale, gradient / objective_scale

    def record(values: FloatArray, loss: float) -> None:
        values = np.asarray(values, dtype=np.float64)
        if not parameters_are_feasible(values, minimum_gap):
            return
        if np.array_equal(values, iterates[-1]):
            losses[-1] = float(loss)
        else:
            iterates.append(values.copy())
            losses.append(float(loss))

    def callback(values: FloatArray) -> None:
        loss, _ = value_and_grad(values)
        record(values, loss)

    constraint = scipy.optimize.LinearConstraint(
        np.asarray([0.0, 1.0, 1.0, 1.0, 1.0]), 1.0, 1.0
    )
    bounds = [(None, None), *[(minimum_gap, None)] * 4]
    result = scipy.optimize.minimize(
        scipy_objective,
        initial,
        jac=True,
        method="SLSQP",
        bounds=bounds,
        constraints=constraint,
        callback=callback,
        options={"maxiter": max_iterations, "ftol": tolerance, "disp": False},
    )
    terminal = np.asarray(result.x, dtype=np.float64)
    terminal_loss, terminal_gradient = value_and_grad(terminal)
    residual = projected_gradient_residual(
        terminal, terminal_gradient / objective_scale, minimum_gap
    )
    result.fun = float(terminal_loss)
    result.jac = terminal_gradient
    result.projected_gradient_residual = residual
    result.objective_scale = objective_scale
    if parameters_are_feasible(terminal, minimum_gap):
        record(terminal, terminal_loss)
    else:
        result.success = False
        result.status = -1
        result.message = f"{result.message}; terminal parameters are infeasible"
    result.iterate_history = np.vstack(iterates)
    result.loss_history = np.asarray(losses, dtype=np.float64)
    return result
