from pathlib import Path

import numpy as np
import pytest

from boundary_opt import BoundaryOptimizer, load_obj, random_knots
from boundary_opt.boundary import (
    knots_from_parameters,
    parameters_from_knots,
)
from boundary_opt.loss import DegenerateFieldError

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def disk_optimizer() -> BoundaryOptimizer:
    return BoundaryOptimizer(load_obj(ROOT / "data" / "disk.obj"))


def plane_corner_knots(optimizer: BoundaryOptimizer) -> np.ndarray:
    points = optimizer.mesh.vertices[optimizer.harmonic.boundary_vertices]
    incoming = points - np.roll(points, 1, axis=0)
    outgoing = np.roll(points, -1, axis=0) - points
    cosines = np.einsum("ij,ij->i", incoming, outgoing) / (
        np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
    )
    indices = np.sort(np.argsort(np.arccos(np.clip(cosines, -1.0, 1.0)))[-4:])
    return optimizer.boundary_positions[indices]


def test_exact_knot_gradient_matches_finite_difference(
    disk_optimizer: BoundaryOptimizer,
) -> None:
    knots = random_knots(7)
    _, gradient = disk_optimizer.loss_and_knot_gradient(knots)
    step = 1.0e-6
    finite_difference = np.asarray(
        [
            (
                disk_optimizer.loss_and_knot_gradient(knots + step * direction)[0]
                - disk_optimizer.loss_and_knot_gradient(knots - step * direction)[0]
            )
            / (2.0 * step)
            for direction in np.eye(4)
        ]
    )
    np.testing.assert_allclose(gradient, finite_difference, rtol=2.0e-7)


def test_centered_state_gradient_on_tangent_with_width_prior() -> None:
    optimizer = BoundaryOptimizer(
        load_obj(ROOT / "data" / "triple_peak.obj"),
        target_arc_width=0.1,
        width_weight=0.1,
    )
    parameters = parameters_from_knots(random_knots(7), optimizer.minimum_gap)
    _, gradient = optimizer.loss_and_gradient(parameters)
    directions = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, -1.0],
        ]
    )
    step = 1.0e-6
    for direction in directions:
        forward = optimizer.loss_and_gradient(parameters + step * direction)[0]
        backward = optimizer.loss_and_gradient(parameters - step * direction)[0]
        assert (forward - backward) / (2.0 * step) == pytest.approx(
            float(gradient @ direction), rel=2.0e-7, abs=2.0e-7
        )


def test_plane_corners_are_global_zero_loss_for_both_backends() -> None:
    optimizer = BoundaryOptimizer(load_obj(ROOT / "data" / "plane.obj"))
    corners = plane_corner_knots(optimizer)
    assert optimizer.loss_and_knot_gradient(corners)[0] < 1.0e-10
    for backend in ("slsqp", "spg"):
        result = optimizer.optimize(corners, backend=backend, max_iterations=10)
        assert result.final_loss < 1.0e-10
        assert result.constraint_violation <= 1.0e-9


@pytest.mark.parametrize("backend", ["slsqp", "spg"])
def test_high_level_history_and_improvement(
    disk_optimizer: BoundaryOptimizer, backend: str
) -> None:
    initial = random_knots(3)
    result = disk_optimizer.optimize(initial, backend=backend, max_iterations=100)
    assert result.backend == backend
    assert result.final_loss < result.initial_loss
    assert result.history[0] == pytest.approx(result.initial_loss)
    assert result.history[-1] == pytest.approx(result.final_loss)
    assert result.parameter_history.shape == (len(result.history), 5)
    np.testing.assert_allclose(result.parameter_history[-1], result.parameters)
    assert result.constraint_violation <= 1.0e-9
    assert np.isfinite(result.field).all()
    for parameters, loss in zip(result.parameter_history, result.history):
        replayed, _ = disk_optimizer.loss_and_gradient(parameters)
        assert replayed == pytest.approx(loss, abs=1.0e-10)
        _, _, gaps = knots_from_parameters(parameters, disk_optimizer.minimum_gap)
        assert gaps.min() >= disk_optimizer.minimum_gap - 1.0e-9


def test_optimize_backends_uses_same_initial_condition(
    disk_optimizer: BoundaryOptimizer,
) -> None:
    results = disk_optimizer.optimize_backends(random_knots(0), max_iterations=120)
    assert set(results) == {"slsqp", "spg"}
    assert results["slsqp"].initial_loss == pytest.approx(
        results["spg"].initial_loss, abs=1.0e-12
    )
    np.testing.assert_array_equal(
        results["slsqp"].parameter_history[0],
        results["spg"].parameter_history[0],
    )


def test_objective_is_periodic_and_complement_symmetric(
    disk_optimizer: BoundaryOptimizer,
) -> None:
    knots = random_knots(13)
    parameters = parameters_from_knots(knots, disk_optimizer.minimum_gap)
    loss, gradient = disk_optimizer.loss_and_gradient(parameters)
    shifted = parameters.copy()
    shifted[0] += 1.0
    shifted_loss, shifted_gradient = disk_optimizer.loss_and_gradient(shifted)
    assert shifted_loss == pytest.approx(loss, abs=1.0e-12)
    np.testing.assert_allclose(shifted_gradient, gradient, atol=1.0e-10)

    complement = np.concatenate((knots[2:], knots[:2] + 1.0))
    complement_loss, _ = disk_optimizer.loss_and_knot_gradient(complement)
    assert complement_loss == pytest.approx(loss, abs=1.0e-12)


@pytest.mark.parametrize("backend", ["slsqp", "spg"])
def test_iteration_limit_raises(backend: str) -> None:
    optimizer = BoundaryOptimizer(load_obj(ROOT / "data" / "plane.obj"))
    with pytest.raises(RuntimeError, match=f"{backend} failed"):
        optimizer.optimize(random_knots(11), backend=backend, max_iterations=1, seed=11)


def test_unknown_backend_is_rejected(
    disk_optimizer: BoundaryOptimizer,
) -> None:
    with pytest.raises(ValueError, match="backend must be"):
        disk_optimizer.optimize(random_knots(0), backend="unknown")


@pytest.mark.parametrize("backend", ["slsqp", "spg"])
def test_degenerate_initial_field_raises(backend: str) -> None:
    minimum_gap = 0.001
    optimizer = BoundaryOptimizer(
        load_obj(ROOT / "data" / "plane.obj"), minimum_gap=minimum_gap
    )
    knots = np.asarray([0.0, 0.997, 0.998, 0.999])
    parameters = parameters_from_knots(knots, minimum_gap)
    with pytest.raises(DegenerateFieldError):
        optimizer.loss_and_gradient(parameters)
    with pytest.raises(DegenerateFieldError):
        optimizer.optimize(knots, backend=backend, max_iterations=40)


@pytest.mark.parametrize(
    ("mesh_name", "maximum_loss"),
    [("disk.obj", 0.05), ("plane.obj", 0.01), ("triple_peak.obj", 0.5)],
)
@pytest.mark.parametrize("backend", ["slsqp", "spg"])
def test_backend_mesh_quality(
    mesh_name: str, maximum_loss: float, backend: str
) -> None:
    optimizer = BoundaryOptimizer(load_obj(ROOT / "data" / mesh_name))
    result = optimizer.optimize(random_knots(0), backend=backend, max_iterations=100)
    assert result.final_loss <= result.initial_loss
    assert result.final_loss < maximum_loss
    assert result.constraint_violation <= 1.0e-9
    assert np.isfinite(result.field).all()
