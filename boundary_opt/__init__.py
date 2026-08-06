"""High-level API for harmonic boundary optimization."""

from .boundary import random_knots
from .defaults import (
    DEFAULT_LENGTH_SMOOTHNESS_WEIGHT,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MINIMUM_GAP,
    DEFAULT_UNIFORMITY_WEIGHT,
)
from .harmonic import HarmonicField
from .optimizer import (
    BackendName,
    BoundaryOptimizer,
    OptimizationResult,
)

__all__ = [
    "DEFAULT_LENGTH_SMOOTHNESS_WEIGHT",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MINIMUM_GAP",
    "DEFAULT_UNIFORMITY_WEIGHT",
    "BackendName",
    "BoundaryOptimizer",
    "HarmonicField",
    "OptimizationResult",
    "random_knots",
]
