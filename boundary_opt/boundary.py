"""The cyclic four-knot boundary profile and its parameterization."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .defaults import DEFAULT_MINIMUM_GAP
from .simplex import (
    FEASIBILITY_TOLERANCE,
    parameters_are_feasible,
    validate_minimum_gap,
)

FloatArray = NDArray[np.float64]

_CENTER_GAP_TO_KNOTS = np.asarray(
    [
        [1.0, -0.75, -0.50, -0.25, 0.0],
        [1.0, 0.25, -0.50, -0.25, 0.0],
        [1.0, 0.25, 0.50, -0.25, 0.0],
        [1.0, 0.25, 0.50, 0.75, 0.0],
    ]
)


def boundary_arclength(vertices: FloatArray, loop: NDArray[np.int64]) -> FloatArray:
    """Normalized cumulative arc length at ordered boundary vertices."""
    following = np.roll(loop, -1)
    lengths = np.linalg.norm(vertices[following] - vertices[loop], axis=1)
    perimeter = float(lengths.sum())
    if not np.isfinite(perimeter) or perimeter <= 0.0 or np.any(lengths <= 0.0):
        raise ValueError("boundary loop contains a zero or invalid edge")
    return np.concatenate(([0.0], np.cumsum(lengths[:-1]))) / perimeter


def cyclic_boundary_profile(
    positions: FloatArray, knots: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Evaluate the cyclic 0/rise/1/fall profile and its knot Jacobian."""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1)
    knots = np.asarray(knots, dtype=np.float64).reshape(-1)
    if knots.shape != (4,) or not np.isfinite(knots).all():
        raise ValueError("knots must contain four finite values")
    if not np.isfinite(positions).all():
        raise ValueError("positions contain NaN or infinite values")
    if np.any(np.diff(knots) <= 0.0) or knots[3] >= knots[0] + 1.0:
        raise ValueError("knots must satisfy k0 < k1 < k2 < k3 < k0 + 1")

    local = np.mod(positions - knots[0], 1.0)
    first, second, third = knots[1:] - knots[0]
    values = np.zeros_like(local)
    jacobian = np.zeros((len(local), 4), dtype=np.float64)

    rising = (local > first) & (local < second)
    rise_width = knots[2] - knots[1]
    fraction = (local[rising] - first) / rise_width
    values[rising] = fraction
    jacobian[rising, 1] = -(1.0 - fraction) / rise_width
    jacobian[rising, 2] = -fraction / rise_width

    values[(local >= second) & (local <= third)] = 1.0

    falling = local > third
    fall_width = knots[0] + 1.0 - knots[3]
    fraction = (local[falling] - third) / fall_width
    values[falling] = 1.0 - fraction
    jacobian[falling, 0] = fraction / fall_width
    jacobian[falling, 3] = (1.0 - fraction) / fall_width
    return values, jacobian


def gaps_from_knots(knots: FloatArray) -> FloatArray:
    """Return four positive cyclic gaps from ordered unwrapped knots."""
    knots = np.asarray(knots, dtype=np.float64).reshape(-1)
    if knots.shape != (4,) or not np.isfinite(knots).all():
        raise ValueError("knots must contain four finite values")
    gaps = np.append(np.diff(knots), knots[0] + 1.0 - knots[3])
    if np.any(gaps <= 0.0):
        raise ValueError("knots must satisfy k0 < k1 < k2 < k3 < k0 + 1")
    return gaps


def parameters_from_knots(knots: FloatArray, minimum_gap: float) -> FloatArray:
    """Map four knots to centered phase plus four full-simplex gaps."""
    minimum_gap = validate_minimum_gap(minimum_gap)
    knots = np.asarray(knots, dtype=np.float64).reshape(-1)
    gaps = gaps_from_knots(knots)
    if np.any(gaps < minimum_gap - FEASIBILITY_TOLERANCE):
        raise ValueError("each cyclic knot gap must be at least minimum_gap")
    return np.concatenate(([float(knots.mean() % 1.0)], gaps))


def knots_from_parameters(
    parameters: FloatArray,
    minimum_gap: float,
    *,
    enforce_minimum_gap: bool = True,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Map centered phase and four gaps to knots and the exact Jacobian."""
    minimum_gap = validate_minimum_gap(minimum_gap)
    parameters = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if parameters.shape != (5,) or not np.isfinite(parameters).all():
        raise ValueError("parameters must contain center phase and four gaps")
    if enforce_minimum_gap and not parameters_are_feasible(parameters, minimum_gap):
        raise ValueError("parameters lie outside the minimum-gap simplex")

    shifted = parameters[1:] - minimum_gap
    shifted_sum = float(shifted.sum())
    if shifted_sum <= 0.0:
        raise ValueError("gap parameters cannot all equal minimum_gap")
    capacity = 1.0 - 4.0 * minimum_gap
    gaps = minimum_gap + capacity * shifted / shifted_sum
    gap_jacobian = (
        capacity
        / shifted_sum
        * (np.eye(4) - np.outer(shifted, np.ones(4)) / shifted_sum)
    )

    local = np.concatenate(([parameters[0]], gaps))
    state_jacobian = np.zeros((5, 5), dtype=np.float64)
    state_jacobian[0, 0] = 1.0
    state_jacobian[1:, 1:] = gap_jacobian
    return (
        _CENTER_GAP_TO_KNOTS @ local,
        _CENTER_GAP_TO_KNOTS @ state_jacobian,
        gaps,
    )


def canonical_knots(knots: FloatArray) -> FloatArray:
    """Shift all unwrapped knots so the first lies in [0, 1)."""
    knots = np.asarray(knots, dtype=np.float64).reshape(-1)
    return knots + (knots[0] % 1.0 - knots[0])


def random_knots(seed: int, minimum_gap: float = DEFAULT_MINIMUM_GAP) -> FloatArray:
    """Draw an ordered cyclic four-knot initialization for one seed."""
    minimum_gap = validate_minimum_gap(minimum_gap)
    rng = np.random.default_rng(seed)
    gaps = minimum_gap + (1.0 - 4.0 * minimum_gap) * rng.dirichlet(np.ones(4))
    return float(rng.uniform()) + np.concatenate(([0.0], np.cumsum(gaps[:3])))
