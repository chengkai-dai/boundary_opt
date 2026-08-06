"""Initial knitting-course selection on a mesh boundary."""

from __future__ import annotations

import numpy as np

from geometry import Mesh


def sample_boundary_course(
    mesh: Mesh,
    loop: np.ndarray,
    positions: np.ndarray,
    start: float,
    end: float,
    stitch_spacing: float,
) -> np.ndarray:
    """Sample one cyclic boundary interval in the loop's orientation."""
    loop = np.asarray(loop, dtype=np.int64).reshape(-1)
    positions = np.asarray(positions, dtype=np.float64).reshape(-1)
    local = np.mod(positions - start, 1.0)
    selected = local <= end - start
    starts = np.flatnonzero(selected & ~np.roll(selected, 1))
    if len(starts) != 1:
        raise ValueError("initial course is not one nonempty boundary chain")

    first = int(starts[0])
    count = int(np.count_nonzero(selected))
    last = (first + count - 1) % len(loop)
    perimeter = float(
        np.linalg.norm(
            mesh.vertices[np.roll(loop, -1)] - mesh.vertices[loop], axis=1
        ).sum()
    )
    include_before = (
        (positions[first] - start) % 1.0 > 0.0
        and ((start - positions[(first - 1) % len(loop)]) % 1.0) * perimeter
        <= 0.5 * stitch_spacing
    )
    include_after = (
        (end - positions[last]) % 1.0 > 0.0
        and ((positions[(last + 1) % len(loop)] - end) % 1.0) * perimeter
        <= 0.5 * stitch_spacing
    )
    first -= int(include_before)
    count += int(include_before) + int(include_after)
    course = loop[(first + np.arange(count)) % len(loop)]
    if len(course) < 2:
        raise ValueError("initial course needs at least two boundary vertices")
    return course
