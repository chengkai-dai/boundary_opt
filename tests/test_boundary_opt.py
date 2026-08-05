from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boundary_opt import (
    HarmonicBoundaryOptimizer,
    cotangent_stiffness,
    cyclic_boundary_profile,
    face_gradient_basis,
    knots_from_parameters,
    load_obj,
    parameters_from_knots,
    random_knots,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def optimizer() -> HarmonicBoundaryOptimizer:
    return HarmonicBoundaryOptimizer(load_obj(ROOT / "data" / "disk.obj"))


def test_linear_boundary_profile_jacobian_matches_finite_difference() -> None:
    positions = np.linspace(0.0, 1.0, 101, endpoint=False)
    knots = np.asarray([0.07, 0.24, 0.58, 0.81])
    values, jacobian = cyclic_boundary_profile(positions, knots)
    assert values.min() == 0.0
    assert values.max() == 1.0
    step = 1.0e-6
    for index in range(4):
        plus, minus = knots.copy(), knots.copy()
        plus[index] += step
        minus[index] -= step
        forward, _ = cyclic_boundary_profile(positions, plus)
        backward, _ = cyclic_boundary_profile(positions, minus)
        finite_difference = (forward - backward) / (2.0 * step)
        np.testing.assert_allclose(jacobian[:, index], finite_difference, atol=2.0e-7)


def test_direct_parameters_include_closed_simplex_boundary() -> None:
    minimum_gap = 0.03
    parameters = np.asarray([0.11, minimum_gap, 0.47, minimum_gap])
    knots, jacobian, gaps = knots_from_parameters(parameters, minimum_gap)
    np.testing.assert_allclose(gaps, [0.03, 0.47, 0.03, 0.47], atol=1.0e-15)
    np.testing.assert_allclose(jacobian, np.tril(np.ones((4, 4))))
    np.testing.assert_allclose(
        parameters_from_knots(knots, minimum_gap), parameters, atol=1.0e-15
    )
    shifted_knots = knots + 3.0
    np.testing.assert_allclose(
        parameters_from_knots(shifted_knots, minimum_gap), parameters, atol=1.0e-15
    )

    with pytest.raises(ValueError, match="outside the minimum-gap simplex"):
        knots_from_parameters(parameters + [0.0, -1.0e-4, 0.0, 0.0], minimum_gap)


def test_harmonic_extension_adjoint_identity(
    optimizer: HarmonicBoundaryOptimizer,
) -> None:
    rng = np.random.default_rng(4)
    boundary = rng.normal(size=len(optimizer.boundary_vertices))
    field_sensitivity = rng.normal(size=len(optimizer.mesh.vertices))
    forward = float(field_sensitivity @ optimizer.extend(boundary))
    backward = float(optimizer.extend_adjoint(field_sensitivity) @ boundary)
    assert forward == pytest.approx(backward, rel=2.0e-11, abs=2.0e-11)


def test_cotangent_energy_matches_face_gradient_energy(
    optimizer: HarmonicBoundaryOptimizer,
) -> None:
    rng = np.random.default_rng(9)
    field = rng.normal(size=len(optimizer.mesh.vertices))
    stiffness = cotangent_stiffness(optimizer.mesh)
    areas, basis = face_gradient_basis(optimizer.mesh)
    gradients = np.einsum("fij,fi->fj", basis, field[optimizer.mesh.faces])
    stiffness_energy = float(field @ (stiffness @ field))
    gradient_energy = float(areas @ np.einsum("ij,ij->i", gradients, gradients))
    assert stiffness_energy == pytest.approx(gradient_energy, rel=2.0e-12)


def test_full_four_parameter_gradient(optimizer: HarmonicBoundaryOptimizer) -> None:
    parameters = parameters_from_knots(random_knots(7), optimizer.minimum_gap)
    _, gradient = optimizer.loss_and_gradient(parameters)
    step = 1.0e-6
    finite_difference = np.empty(4)
    for index in range(4):
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += step
        minus[index] -= step
        forward, _ = optimizer.loss_and_gradient(plus)
        backward, _ = optimizer.loss_and_gradient(minus)
        finite_difference[index] = (forward - backward) / (2.0 * step)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2.0e-5, atol=2.0e-7)


def test_direct_objective_is_periodic_in_offset(
    optimizer: HarmonicBoundaryOptimizer,
) -> None:
    parameters = np.asarray([0.07, 0.17, 0.34, 0.23])
    loss, gradient = optimizer.loss_and_gradient(parameters)
    for shift in (-1.0, 1.0):
        shifted = parameters.copy()
        shifted[0] += shift
        shifted_loss, shifted_gradient = optimizer.loss_and_gradient(shifted)
        assert shifted_loss == pytest.approx(loss, abs=1.0e-12)
        np.testing.assert_allclose(
            shifted_gradient, gradient, rtol=1.0e-10, atol=1.0e-10
        )


def test_plane_corners_are_an_affine_zero_loss_solution() -> None:
    optimizer = HarmonicBoundaryOptimizer(load_obj(ROOT / "data" / "plane.obj"))
    boundary_points = optimizer.mesh.vertices[optimizer.boundary_vertices]
    incoming = boundary_points - np.roll(boundary_points, 1, axis=0)
    outgoing = np.roll(boundary_points, -1, axis=0) - boundary_points
    turning_cosines = np.einsum("ij,ij->i", incoming, outgoing) / (
        np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
    )
    turning_angles = np.arccos(np.clip(turning_cosines, -1.0, 1.0))
    corner_indices = np.sort(np.argsort(turning_angles)[-4:])
    corner_knots = optimizer.boundary_positions[corner_indices]

    boundary_values, _ = cyclic_boundary_profile(
        optimizer.boundary_positions, corner_knots
    )
    field = optimizer.extend(boundary_values)
    corner_loss, _, _ = optimizer._loss_and_field_gradient(field)
    assert corner_loss < 1.0e-10

    result = optimizer.optimize(random_knots(0), max_iterations=100, seed=0)
    endpoint_distances = np.abs(
        (result.knots[:, None] - corner_knots[None, :] + 0.5) % 1.0 - 0.5
    )
    assert result.final_loss < 1.0e-10
    assert np.all(endpoint_distances.min(axis=1) < 1.0e-5)


def test_optimization_decreases_loss(optimizer: HarmonicBoundaryOptimizer) -> None:
    initial_knots = random_knots(3)
    result = optimizer.optimize(initial_knots, max_iterations=8, seed=3)
    assert np.isfinite(result.field).all()
    assert result.final_loss < result.initial_loss
    assert result.history[0] == pytest.approx(result.initial_loss)
    assert result.history[-1] == pytest.approx(result.final_loss)
    assert result.parameter_history.shape == (len(result.history), 4)
    assert result.total_evaluations >= result.evaluations
    np.testing.assert_allclose(
        result.parameter_history[0],
        parameters_from_knots(initial_knots, optimizer.minimum_gap),
        rtol=0.0,
        atol=5.0e-14,
    )
    np.testing.assert_allclose(
        result.parameter_history[-1], result.parameters, rtol=0.0, atol=0.0
    )
    for parameters, loss in zip(result.parameter_history, result.history):
        assert optimizer.loss_and_gradient(parameters)[0] == pytest.approx(loss)
        _, _, gaps = knots_from_parameters(parameters, optimizer.minimum_gap)
        assert gaps.min() >= optimizer.minimum_gap - 1.0e-9
    assert result.constraint_violation <= 1.0e-9
    assert result.statistics.spacing_cv > 0.0


def test_complementary_initialization_is_numerically_equivariant() -> None:
    optimizer = HarmonicBoundaryOptimizer(load_obj(ROOT / "data" / "plane.obj"))
    initial = random_knots(13)
    complement = np.concatenate((initial[2:], initial[:2] + 1.0))
    first = optimizer.optimize(initial, max_iterations=100, seed=13)
    second = optimizer.optimize(complement, max_iterations=100, seed=13)
    assert second.final_loss == pytest.approx(first.final_loss, abs=1.0e-11)


def test_slsqp_recovers_from_out_of_domain_trials() -> None:
    optimizer = HarmonicBoundaryOptimizer(load_obj(ROOT / "data" / "triple_peak.obj"))
    result = optimizer.optimize(random_knots(60), max_iterations=100, seed=60)
    assert np.isfinite(result.final_loss)
    assert result.constraint_violation <= 1.0e-9
    for parameters in result.parameter_history:
        _, _, gaps = knots_from_parameters(parameters, optimizer.minimum_gap)
        assert gaps.min() >= optimizer.minimum_gap - 1.0e-9


def test_iteration_limit_returns_best_feasible_evaluation() -> None:
    optimizer = HarmonicBoundaryOptimizer(load_obj(ROOT / "data" / "plane.obj"))
    result = optimizer.optimize(random_knots(11), max_iterations=1, seed=11)
    assert not result.success
    assert result.final_loss < 0.4
    assert result.final_loss <= result.initial_loss
    assert result.history[-1] == pytest.approx(result.final_loss)
    assert "returned best feasible evaluation" in result.message


def test_solver_recovers_from_discretely_constant_boundary_trial() -> None:
    minimum_gap = 0.001
    optimizer = HarmonicBoundaryOptimizer(
        load_obj(ROOT / "data" / "plane.obj"), minimum_gap=minimum_gap
    )
    result = optimizer.optimize(random_knots(0, minimum_gap), max_iterations=40, seed=0)
    assert np.isfinite(result.final_loss)
    assert result.final_loss < result.initial_loss
    assert np.ptp(result.field) > 0.9
    assert result.constraint_violation <= 1.0e-9


def test_solver_recovers_from_discretely_constant_initial_field() -> None:
    minimum_gap = 0.001
    optimizer = HarmonicBoundaryOptimizer(
        load_obj(ROOT / "data" / "plane.obj"), minimum_gap=minimum_gap
    )
    parameters = np.asarray([0.0, 0.997, minimum_gap, minimum_gap])
    knots, _, _ = knots_from_parameters(parameters, minimum_gap)
    with pytest.raises(ValueError, match="zero or invalid gradient energy"):
        optimizer.loss_and_gradient(parameters)

    result = optimizer.optimize(knots, max_iterations=40, seed=0)
    assert result.initial_loss >= 1.0e6
    assert np.isfinite(result.final_loss)
    assert result.final_loss < result.initial_loss
    assert np.ptp(result.field) > 0.9
    assert result.constraint_violation <= 1.0e-9


def test_slsqp_can_reach_and_diagnose_active_simplex_face() -> None:
    optimizer = HarmonicBoundaryOptimizer(load_obj(ROOT / "data" / "plane.obj"))
    result = optimizer.optimize(random_knots(2), max_iterations=100, seed=2)
    assert result.success
    assert result.final_loss < 0.003
    assert result.gaps[0] == pytest.approx(optimizer.minimum_gap, abs=1.0e-8)
    assert result.gaps[2] == pytest.approx(optimizer.minimum_gap, abs=1.0e-8)
    assert result.gradient_norm > 0.1
    assert result.kkt_residual < 5.0e-6


def test_optimization_handles_curved_mesh() -> None:
    optimizer = HarmonicBoundaryOptimizer(
        load_obj(ROOT / "data" / "triple_peak.obj"),
        target_arc_width=0.1,
        width_weight=0.1,
    )
    parameters = parameters_from_knots(random_knots(7), optimizer.minimum_gap)
    _, gradient = optimizer.loss_and_gradient(parameters)
    step = 1.0e-6
    finite_difference = np.asarray(
        [
            (
                optimizer.loss_and_gradient(parameters + np.eye(4)[index] * step)[0]
                - optimizer.loss_and_gradient(parameters - np.eye(4)[index] * step)[0]
            )
            / (2.0 * step)
            for index in range(4)
        ]
    )
    np.testing.assert_allclose(gradient, finite_difference, rtol=2.0e-8)

    result = optimizer.optimize(random_knots(0), max_iterations=10, seed=0)
    assert result.final_loss < result.initial_loss
    assert result.field.min() >= -1.0e-12
    assert result.field.max() <= 1.0 + 1.0e-12
