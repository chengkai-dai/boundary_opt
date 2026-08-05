"""Loss functions and field diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, ndtr

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_FIELD_LEVELS = np.linspace(0.05, 0.95, 19)
_FIELD_LEVEL_STEPS = np.diff(_FIELD_LEVELS)
_AREA_SMOOTHING = 0.01
_AREA_TARGET = _AREA_SMOOTHING * (
    np.logaddexp(0.0, _FIELD_LEVELS / _AREA_SMOOTHING)
    - np.logaddexp(0.0, (_FIELD_LEVELS - 1.0) / _AREA_SMOOTHING)
)
_ISOLINE_SMOOTHING = 0.03
_ISOLINE_KERNEL_MASS = ndtr((1.0 - _FIELD_LEVELS) / _ISOLINE_SMOOTHING) - ndtr(
    -_FIELD_LEVELS / _ISOLINE_SMOOTHING
)
_TRIANGLE_QUADRATURE = np.asarray(
    [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ]
)


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


def area_balance_loss_and_gradient(
    field: FloatArray,
    faces: IntArray,
    face_weights: FloatArray,
) -> tuple[float, FloatArray]:
    """Match the field's area-weighted CDF to a uniform distribution."""
    face_values = np.asarray(field, dtype=np.float64)[faces].mean(axis=1)
    sigmoid = expit((_FIELD_LEVELS[None, :] - face_values[:, None]) / _AREA_SMOOTHING)
    cdf = face_weights @ sigmoid
    residual = cdf - _AREA_TARGET
    loss = float(np.mean(residual**2))

    cdf_gradient = 2.0 * residual / len(_FIELD_LEVELS)
    face_gradient = (
        -face_weights / _AREA_SMOOTHING * ((sigmoid * (1.0 - sigmoid)) @ cdf_gradient)
    )
    field_gradient = np.bincount(
        faces.reshape(-1),
        weights=np.repeat(face_gradient / 3.0, 3),
        minlength=len(field),
    ).astype(np.float64)
    return loss, field_gradient


def length_smoothness_loss_and_gradient(
    field: FloatArray,
    faces: IntArray,
    gradient_basis: FloatArray,
    face_weights: FloatArray,
) -> tuple[float, FloatArray]:
    """Penalize rapid changes between neighboring soft isoline lengths."""
    face_values = np.asarray(field, dtype=np.float64)[faces]
    gradients = np.einsum("fij,fi->fj", gradient_basis, face_values)
    gradient_norms = np.sqrt(np.einsum("ij,ij->i", gradients, gradients) + 1.0e-24)

    samples = face_values @ _TRIANGLE_QUADRATURE.T
    offsets = samples[:, :, None] - _FIELD_LEVELS
    kernels = (
        np.exp(-0.5 * (offsets / _ISOLINE_SMOOTHING) ** 2)
        / (np.sqrt(2.0 * np.pi) * _ISOLINE_SMOOTHING)
        / _ISOLINE_KERNEL_MASS
    )
    mean_kernels = kernels.mean(axis=1)
    lengths = (face_weights * gradient_norms) @ mean_kernels

    length_rates = np.diff(lengths) / _FIELD_LEVEL_STEPS
    numerator = float(np.mean(length_rates**2))
    mean_length = float(lengths.mean())
    denominator = mean_length**2 + 1.0e-15
    loss = numerator / denominator

    length_gradient = np.zeros_like(lengths)
    difference_gradient = (
        2.0 * length_rates / _FIELD_LEVEL_STEPS / len(length_rates)
    )
    length_gradient[:-1] -= difference_gradient
    length_gradient[1:] += difference_gradient
    length_gradient = (
        length_gradient / denominator
        - numerator * (2.0 * mean_length / len(lengths)) / denominator**2
    )

    kernel_derivatives = -offsets / _ISOLINE_SMOOTHING**2 * kernels
    sample_sensitivity = (
        face_weights[:, None]
        * gradient_norms[:, None]
        * np.einsum("fql,l->fq", kernel_derivatives, length_gradient)
        / len(_TRIANGLE_QUADRATURE)
    )
    corner_sensitivity = sample_sensitivity @ _TRIANGLE_QUADRATURE

    norm_sensitivity = face_weights * (mean_kernels @ length_gradient)
    gradient_sensitivity = (
        norm_sensitivity[:, None] * gradients / gradient_norms[:, None]
    )
    corner_sensitivity += np.einsum("fij,fj->fi", gradient_basis, gradient_sensitivity)
    field_gradient = np.bincount(
        faces.reshape(-1),
        weights=corner_sensitivity.reshape(-1),
        minlength=len(field),
    ).astype(np.float64)
    return float(loss), field_gradient
