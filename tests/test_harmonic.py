from pathlib import Path

import numpy as np
import pytest

from boundary_opt import HarmonicField
from geometry import load_obj

ROOT = Path(__file__).resolve().parent.parent


def test_solve_adjoint_is_the_transpose_of_solve() -> None:
    harmonic = HarmonicField(load_obj(ROOT / "data" / "disk.obj"))
    rng = np.random.default_rng(4)
    boundary = rng.normal(size=len(harmonic.boundary_vertices))
    sensitivity = rng.normal(size=len(harmonic.mesh.vertices))
    forward = float(sensitivity @ harmonic.solve(boundary))
    backward = float(harmonic.solve_adjoint(sensitivity) @ boundary)
    assert forward == pytest.approx(backward, rel=2.0e-11, abs=2.0e-11)
