from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PeelingConfig:
    """Distances used by the peeling core, expressed in mesh model units."""

    course_spacing: float = 0.15
    stitch_spacing: float = 0.15

    def __post_init__(self) -> None:
        if not math.isfinite(self.stitch_spacing) or self.stitch_spacing <= 0.0:
            raise ValueError("stitch_spacing must be positive and finite")
        if not math.isfinite(self.course_spacing) or self.course_spacing <= 0.0:
            raise ValueError("course_spacing must be positive and finite")

    @property
    def chain_sample_spacing(self) -> float:
        return 0.25 * self.stitch_spacing

    @property
    def max_path_sample_spacing(self) -> float:
        return 0.02 * min(self.stitch_spacing, self.course_spacing)
