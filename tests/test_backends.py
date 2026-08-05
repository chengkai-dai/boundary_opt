import numpy as np
import pytest

from boundary_opt.simplex import projected_gradient_residual
from boundary_opt.slsqp_backend import minimize_slsqp
from boundary_opt.spg_backend import minimize_spg


@pytest.mark.parametrize("backend", [minimize_slsqp, minimize_spg])
def test_backend_solves_active_simplex_quadratic(backend) -> None:
    minimum_gap = 0.03
    target = np.asarray([0.2, minimum_gap, 0.27, minimum_gap, 0.67])
    initial = np.asarray([0.8, 0.25, 0.25, 0.25, 0.25])

    def objective(parameters):
        difference = parameters - target
        return 0.5 * float(difference @ difference), difference

    result = backend(objective, initial, minimum_gap, 200)
    assert result.success
    assert result.fun < 1.0e-10
    assert result.x[1] == pytest.approx(minimum_gap, abs=1.0e-6)
    assert result.x[3] == pytest.approx(minimum_gap, abs=1.0e-6)
    assert projected_gradient_residual(result.x, result.jac, minimum_gap) < 2.0e-6
    assert result.iterate_history.shape[1] == 5
    assert len(result.iterate_history) == len(result.loss_history)
    np.testing.assert_allclose(result.iterate_history[:, 1:].sum(axis=1), 1.0)
    assert result.iterate_history[:, 1:].min() >= minimum_gap - 1.0e-12


@pytest.mark.parametrize("backend", [minimize_slsqp, minimize_spg])
@pytest.mark.parametrize("scale", [1.0e-4, 1.0, 1.0e4])
def test_backend_is_robust_to_objective_scaling(backend, scale: float) -> None:
    minimum_gap = 0.03
    target = np.asarray([0.2, minimum_gap, 0.27, minimum_gap, 0.67])
    initial = np.asarray([0.8, 0.25, 0.25, 0.25, 0.25])

    def objective(parameters):
        difference = parameters - target
        return (
            0.5 * scale * float(difference @ difference),
            scale * difference,
        )

    result = backend(objective, initial, minimum_gap, 200)
    assert result.success
    np.testing.assert_allclose(result.x, target, atol=2.0e-5)
