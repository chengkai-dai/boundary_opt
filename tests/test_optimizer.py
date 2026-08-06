from pathlib import Path

import numpy as np
import pytest

from boundary_opt import (
    DEFAULT_LENGTH_SMOOTHNESS_WEIGHT,
    DEFAULT_MINIMUM_GAP,
    DEFAULT_UNIFORMITY_WEIGHT,
    BoundaryOptimizer,
    random_knots,
)
from boundary_opt.boundary import (
    knots_from_parameters,
    parameters_from_knots,
)
from boundary_opt.loss import DegenerateFieldError
from geometry import load_obj, normalize_mesh

ROOT = Path(__file__).resolve().parent.parent
TEST_LENGTH_SMOOTHNESS_WEIGHT = 1.0
TEST_UNIFORMITY_WEIGHT = 1.0


@pytest.fixture(scope="module")
def disk_optimizer() -> BoundaryOptimizer:
    return BoundaryOptimizer(
        load_obj(ROOT / "data" / "disk.obj"),
        uniformity_weight=TEST_UNIFORMITY_WEIGHT,
        length_smoothness_weight=TEST_LENGTH_SMOOTHNESS_WEIGHT,
    )


def test_default_weights() -> None:
    optimizer = BoundaryOptimizer(load_obj(ROOT / "data" / "disk.obj"))
    assert optimizer.minimum_gap == DEFAULT_MINIMUM_GAP
    assert (
        optimizer.length_smoothness_weight == DEFAULT_LENGTH_SMOOTHNESS_WEIGHT
    )
    assert optimizer.uniformity_weight == DEFAULT_UNIFORMITY_WEIGHT


def test_loss_weights_scale_value_and_gradient() -> None:
    mesh = load_obj(ROOT / "data" / "disk.obj")
    knots = random_knots(7)
    uniformity = BoundaryOptimizer(
        mesh, uniformity_weight=1.0, length_smoothness_weight=0.0
    )
    length_smoothness = BoundaryOptimizer(
        mesh, uniformity_weight=0.0, length_smoothness_weight=1.0
    )
    weighted = BoundaryOptimizer(
        mesh,
        uniformity_weight=3.0,
        length_smoothness_weight=11.0,
    )

    uniformity_loss, uniformity_gradient = uniformity.loss_and_knot_gradient(knots)
    length_loss, length_gradient = length_smoothness.loss_and_knot_gradient(knots)
    weighted_loss, weighted_gradient = weighted.loss_and_knot_gradient(knots)

    assert weighted_loss == pytest.approx(3.0 * uniformity_loss + 11.0 * length_loss)
    np.testing.assert_allclose(
        weighted_gradient,
        3.0 * uniformity_gradient + 11.0 * length_gradient,
    )


def test_mesh_normalization_preserves_objective() -> None:
    mesh = load_obj(ROOT / "data" / "peak.obj")
    original = BoundaryOptimizer(mesh)
    normalized = BoundaryOptimizer(normalize_mesh(mesh, 10.0))
    knots = random_knots(7)
    original_loss, original_gradient = original.loss_and_knot_gradient(knots)
    normalized_loss, normalized_gradient = normalized.loss_and_knot_gradient(knots)
    assert normalized_loss == pytest.approx(original_loss, abs=1.0e-11)
    np.testing.assert_allclose(normalized_gradient, original_gradient, atol=1.0e-9)


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


def test_centered_state_gradient_on_tangent() -> None:
    optimizer = BoundaryOptimizer(
        load_obj(ROOT / "data" / "triple_peak.obj"),
        uniformity_weight=TEST_UNIFORMITY_WEIGHT,
        length_smoothness_weight=TEST_LENGTH_SMOOTHNESS_WEIGHT,
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


def test_plane_corners_have_zero_loss() -> None:
    optimizer = BoundaryOptimizer(
        load_obj(ROOT / "data" / "plane.obj"),
        uniformity_weight=TEST_UNIFORMITY_WEIGHT,
        length_smoothness_weight=TEST_LENGTH_SMOOTHNESS_WEIGHT,
    )
    corners = plane_corner_knots(optimizer)
    assert optimizer.loss_and_knot_gradient(corners)[0] < 1.0e-8


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("backend", ["slsqp", "spg"])
def test_multistart_uniformity_finds_plane_corners(
    backend: str, normalized: bool
) -> None:
    mesh = load_obj(ROOT / "data" / "plane.obj")
    optimizer = BoundaryOptimizer(
        normalize_mesh(mesh, 10.0) if normalized else mesh,
        uniformity_weight=1.0,
        length_smoothness_weight=0.0,
    )
    initial = random_knots(0)
    result = optimizer.optimize_multistart(
        initial, backend=backend, max_iterations=500
    )
    corners = plane_corner_knots(optimizer)
    distances = np.abs((result.knots % 1.0)[:, None] - corners)
    circular_distances = np.minimum(distances, 1.0 - distances)

    assert result.final_loss < 1.0e-10
    assert np.max(np.min(circular_distances, axis=0)) < 1.0e-5
    assert np.max(np.min(circular_distances, axis=1)) < 1.0e-5
    assert result.history[0] == pytest.approx(result.initial_loss)
    assert result.history[-1] == pytest.approx(result.final_loss)
    assert len(result.history) == len(result.parameter_history)
    np.testing.assert_allclose(result.parameter_history[-1], result.parameters)
    for parameters, loss in zip(result.parameter_history, result.history):
        assert optimizer.loss_and_gradient(parameters)[0] == pytest.approx(
            loss, abs=1.0e-10
        )


def test_global_weight_scale_does_not_change_slsqp_path() -> None:
    mesh = load_obj(ROOT / "data" / "plane.obj")
    initial = random_knots(0)
    unit = BoundaryOptimizer(
        mesh, uniformity_weight=0.0, length_smoothness_weight=1.0
    ).optimize(initial, backend="slsqp", max_iterations=500)
    scaled = BoundaryOptimizer(
        mesh,
        uniformity_weight=0.0,
        length_smoothness_weight=100.0,
    ).optimize(initial, backend="slsqp", max_iterations=500)

    np.testing.assert_array_equal(scaled.parameters, unit.parameters)
    assert scaled.length_smoothness_loss == unit.length_smoothness_loss
    assert scaled.final_loss == pytest.approx(100.0 * unit.final_loss)


@pytest.mark.parametrize("backend", ["slsqp", "spg"])
def test_high_level_history_and_improvement(
    disk_optimizer: BoundaryOptimizer, backend: str
) -> None:
    initial = random_knots(3)
    result = disk_optimizer.optimize(initial, backend=backend, max_iterations=200)
    assert result.backend == backend
    assert result.final_loss < result.initial_loss
    assert result.history[0] == pytest.approx(result.initial_loss)
    assert result.history[-1] == pytest.approx(result.final_loss)
    assert result.parameter_history.shape == (len(result.history), 5)
    np.testing.assert_allclose(result.parameter_history[-1], result.parameters)
    assert result.constraint_violation <= 1.0e-9
    assert np.isfinite(result.field).all()
    np.testing.assert_array_equal(result.field[result.minimum_vertices], 0.0)
    np.testing.assert_array_equal(result.field[result.maximum_vertices], 1.0)
    boundary_edges = {
        tuple(sorted(edge))
        for edge in zip(
            disk_optimizer.harmonic.boundary_vertices,
            np.roll(disk_optimizer.harmonic.boundary_vertices, -1),
            strict=True,
        )
    }
    for curve in (result.minimum_vertices, result.maximum_vertices):
        assert len(curve) >= 2
        assert all(
            tuple(sorted(edge)) in boundary_edges
            for edge in zip(curve[:-1], curve[1:], strict=True)
        )
    for parameters, loss in zip(result.parameter_history, result.history):
        replayed, _ = disk_optimizer.loss_and_gradient(parameters)
        assert replayed == pytest.approx(loss, abs=1.0e-10)
        _, _, gaps = knots_from_parameters(parameters, disk_optimizer.minimum_gap)
        assert gaps.min() >= disk_optimizer.minimum_gap - 1.0e-9


def test_optimize_backends_uses_same_initial_condition(
    disk_optimizer: BoundaryOptimizer,
) -> None:
    results = disk_optimizer.optimize_backends(random_knots(0), max_iterations=240)
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
    ("mesh_name", "maximum_uniformity"),
    [
        ("disk.obj", 0.03),
        ("peak.obj", 0.03),
        ("triple_peak.obj", 0.36),
    ],
)
@pytest.mark.parametrize("backend", ["slsqp", "spg"])
def test_multistart_mesh_quality(
    mesh_name: str,
    maximum_uniformity: float,
    backend: str,
) -> None:
    optimizer = BoundaryOptimizer(
        normalize_mesh(load_obj(ROOT / "data" / mesh_name), 10.0),
        uniformity_weight=1.0,
        length_smoothness_weight=0.0,
    )
    initial = random_knots(0)
    local = optimizer.optimize(initial, backend=backend, max_iterations=500)
    result = optimizer.optimize_multistart(
        initial, backend=backend, max_iterations=500
    )
    assert result.final_loss <= local.final_loss
    assert result.uniformity_loss < maximum_uniformity
    assert result.constraint_violation <= 1.0e-9
    assert np.isfinite(result.field).all()
