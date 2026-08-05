"""Closed-simplex constraints used by both optimization backends."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

FEASIBILITY_TOLERANCE = 1.0e-9


def validate_minimum_gap(minimum_gap: float) -> float:
    if not np.isfinite(minimum_gap) or not 0.0 < minimum_gap < 0.25:
        raise ValueError("minimum_gap must lie in (0, 0.25)")
    return float(minimum_gap)


def project_gaps(gaps: FloatArray, minimum_gap: float) -> FloatArray:
    """Project four gaps onto {g >= minimum_gap, sum(g) = 1}."""
    minimum_gap = validate_minimum_gap(minimum_gap)
    gaps = np.asarray(gaps, dtype=np.float64).reshape(-1)
    if gaps.shape != (4,) or not np.isfinite(gaps).all():
        raise ValueError("gaps must contain four finite values")
    if gaps.min() >= minimum_gap and float(gaps.sum()) == 1.0:
        return gaps.copy()

    capacity = 1.0 - 4.0 * minimum_gap
    shifted = gaps - minimum_gap
    ordered = np.sort(shifted)[::-1]
    thresholds = (np.cumsum(ordered) - capacity) / np.arange(1.0, 5.0)
    threshold = thresholds[np.flatnonzero(ordered > thresholds)[-1]]
    projected = minimum_gap + np.maximum(shifted - threshold, 0.0)
    projected[np.argmax(projected)] += 1.0 - projected.sum()
    return projected


def project_parameters(parameters: FloatArray, minimum_gap: float) -> FloatArray:
    """Project the four gaps while leaving the center phase unchanged."""
    parameters = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if parameters.shape != (5,) or not np.isfinite(parameters).all():
        raise ValueError("parameters must contain center phase and four gaps")
    projected = parameters.copy()
    projected[1:] = project_gaps(projected[1:], minimum_gap)
    return projected


def parameters_are_feasible(parameters: FloatArray, minimum_gap: float) -> bool:
    parameters = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if parameters.shape != (5,) or not np.isfinite(parameters).all():
        return False
    gaps = parameters[1:]
    return bool(
        gaps.min() >= minimum_gap - FEASIBILITY_TOLERANCE
        and abs(float(gaps.sum()) - 1.0) <= FEASIBILITY_TOLERANCE
    )


def constraint_violation(parameters: FloatArray, minimum_gap: float) -> float:
    gaps = np.asarray(parameters, dtype=np.float64).reshape(-1)[1:]
    return max(minimum_gap - float(gaps.min()), abs(float(gaps.sum()) - 1.0), 0.0)


def projected_gradient_residual(
    parameters: FloatArray, gradient: FloatArray, minimum_gap: float
) -> float:
    """Infinity norm of the unit-step projected-gradient mapping."""
    parameters = np.asarray(parameters, dtype=np.float64)
    gradient = np.asarray(gradient, dtype=np.float64)
    mapped = parameters - project_parameters(parameters - gradient, minimum_gap)
    return float(np.linalg.norm(mapped, ord=np.inf))
