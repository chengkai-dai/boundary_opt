import numpy as np
import pytest

from boundary_opt.boundary import parameters_from_knots, random_knots
from boundary_opt.simplex import project_gaps, project_parameters


def test_lower_bounded_simplex_projection() -> None:
    minimum_gap = 0.03
    feasible = parameters_from_knots(random_knots(0), minimum_gap)[1:]
    np.testing.assert_array_equal(project_gaps(feasible, minimum_gap), feasible)
    projected = project_gaps(np.asarray([-2.0, 0.2, 0.4, 3.0]), minimum_gap)
    assert projected.sum() == pytest.approx(1.0, abs=2.0e-16)
    assert projected.min() >= minimum_gap
    state = project_parameters(np.r_[0.7, -2.0, 0.2, 0.4, 3.0], minimum_gap)
    assert state[0] == 0.7
    np.testing.assert_allclose(state[1:], projected)
