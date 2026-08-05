"""Loss functions and field diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class DegenerateFieldError(ValueError):
    """Raised when the scale-free field loss is undefined."""


@dataclass(frozen=True, slots=True)
class FieldStatistics:
    spacing_cv: float
    minimum_gradient: float
    maximum_gradient: float


def uniformity_loss_and_gradient(
    field: FloatArray,
    faces: IntArray,
    gradient_basis: FloatArray,
    face_weights: FloatArray,
) -> tuple[float, FloatArray, FieldStatistics]:
    """Return CV^2(|grad u|^2), its field gradient, and diagnostics."""
    face_values = np.asarray(field, dtype=np.float64)[faces]
    gradients = np.einsum("fij,fi->fj", gradient_basis, face_values)
    squared_norms = np.einsum("ij,ij->i", gradients, gradients)
    mean_squared = float(face_weights @ squared_norms)
    if mean_squared <= 0.0 or not np.isfinite(mean_squared):
        raise DegenerateFieldError("harmonic field has zero or invalid gradient energy")
    second_moment = float(face_weights @ squared_norms**2)
    loss = second_moment / mean_squared**2 - 1.0

    coefficients = (
        4.0
        * face_weights
        * (squared_norms - second_moment / mean_squared)
        / mean_squared**2
    )
    face_sensitivity = coefficients[:, None] * gradients
    corner_sensitivity = np.einsum("fij,fj->fi", gradient_basis, face_sensitivity)
    field_sensitivity = np.bincount(
        faces.reshape(-1),
        weights=corner_sensitivity.reshape(-1),
        minlength=len(field),
    ).astype(np.float64)

    norms = np.sqrt(squared_norms)
    mean_norm = float(face_weights @ norms)
    variance = float(face_weights @ (norms - mean_norm) ** 2)
    statistics = FieldStatistics(
        spacing_cv=float(np.sqrt(max(variance, 0.0)) / max(mean_norm, 1.0e-15)),
        minimum_gradient=float(norms.min()),
        maximum_gradient=float(norms.max()),
    )
    return float(loss), field_sensitivity, statistics


def width_loss_and_gradient(
    gaps: FloatArray, target: float | None, weight: float
) -> tuple[float, FloatArray]:
    """Optional quadratic prior on the zero and one plateau widths."""
    if weight == 0.0 or target is None:
        return 0.0, np.zeros(4, dtype=np.float64)
    residual = (np.asarray(gaps)[[0, 2]] - target) / target
    gradient = np.zeros(4, dtype=np.float64)
    gradient[[0, 2]] = 2.0 * weight * residual / target
    return weight * float(residual @ residual), gradient
