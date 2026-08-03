from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boundary_opt import (
    HarmonicBoundaryOptimizer,
    Mesh,
    cotangent_stiffness,
    cyclic_arc_edge_weights,
    face_gradient_basis,
    knots_from_parameters,
    load_obj,
    parameters_from_knots,
    random_knots,
)
from plot_loss_curves import chart_svg
from scan_mesh_seeds import relative_reduction

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def optimizer() -> HarmonicBoundaryOptimizer:
    return HarmonicBoundaryOptimizer(
        load_obj(ROOT / "data" / "disk.obj"),
        target_arc_width=0.1,
        width_weight=0.1,
    )


def _finite_difference_gradient(
    optimizer: HarmonicBoundaryOptimizer, parameters: np.ndarray, step: float = 1.0e-6
) -> np.ndarray:
    directions = np.eye(4) * step
    return np.asarray(
        [
            (
                optimizer.loss_and_gradient(parameters + direction)[0]
                - optimizer.loss_and_gradient(parameters - direction)[0]
            )
            / (2.0 * step)
            for direction in directions
        ]
    )


def test_cyclic_arc_weights_are_exact_even_when_an_arc_wraps() -> None:
    positions = np.asarray([0.0, 0.2, 0.5, 0.7])
    edge_lengths = np.diff(np.append(positions, 1.0))
    knots = np.asarray([0.85, 1.05, 1.25, 1.45])
    zero, one = cyclic_arc_edge_weights(positions, knots)
    assert np.all((0.0 <= zero) & (zero <= 1.0))
    assert np.all((0.0 <= one) & (one <= 1.0))
    assert np.all(zero + one <= 1.0)
    assert np.any((zero == 0.0) & (one == 0.0))
    assert edge_lengths @ zero == pytest.approx(knots[1] - knots[0])
    assert edge_lengths @ one == pytest.approx(knots[3] - knots[2])


def test_robin_system_residual(optimizer: HarmonicBoundaryOptimizer) -> None:
    knots = random_knots(4, optimizer.minimum_gap)
    field = optimizer.field_from_knots(knots)
    zero_mass = optimizer._arc_mass(knots[0], knots[1])
    one_mass = optimizer._arc_mass(knots[2], knots[3])
    residual = np.asarray(cotangent_stiffness(optimizer.mesh) @ field)
    boundary_field = field[optimizer.boundary_vertices]
    residual[optimizer.boundary_vertices] += optimizer.boundary_penalty * (
        (zero_mass + one_mass) @ boundary_field - one_mass.sum(axis=1)
    )
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=2.0e-12)


def test_moving_arc_mass_derivative_is_continuous_at_a_vertex(
    optimizer: HarmonicBoundaryOptimizer,
) -> None:
    boundary_index = len(optimizer.boundary_vertices) // 3
    start = float(optimizer.boundary_positions[boundary_index])
    end = start + 0.2
    step = 1.0e-8
    derivative = (
        optimizer._arc_mass(start + step, end) - optimizer._arc_mass(start - step, end)
    ) / (2.0 * step)
    rng = np.random.default_rng(12)
    left = rng.normal(size=len(optimizer.boundary_vertices))
    right = rng.normal(size=len(optimizer.boundary_vertices))
    actual = float(left @ (derivative @ right))
    expected = -float(left[boundary_index] * right[boundary_index])
    assert actual == pytest.approx(expected, rel=3.0e-6, abs=2.0e-9)


def test_mesh_owns_read_only_canonical_arrays() -> None:
    source_vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    source_faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    mesh = Mesh(source_vertices, source_faces)
    source_vertices[0, 0] = 9.0
    source_faces[0, 0] = 2
    assert mesh.vertices.dtype == np.float64
    assert mesh.faces.dtype == np.int64
    assert mesh.vertices[0, 0] == 0.0
    assert mesh.faces[0, 0] == 0
    with pytest.raises(ValueError):
        mesh.vertices[0, 0] = 1.0
    with pytest.raises(ValueError):
        mesh.faces[0, 0] = 1


def test_optimizer_cached_geometry_is_read_only(
    optimizer: HarmonicBoundaryOptimizer,
) -> None:
    for array in (
        optimizer.boundary_vertices,
        optimizer.boundary_positions,
        optimizer.interior_vertices,
        optimizer.face_areas,
        optimizer._harmonic_lift,
    ):
        with pytest.raises(ValueError):
            array.flat[0] = array.flat[0]


def test_harmonic_lift_preserves_boundary_and_solves_the_interior(
    optimizer: HarmonicBoundaryOptimizer,
) -> None:
    boundary_field = np.random.default_rng(21).normal(
        size=len(optimizer.boundary_vertices)
    )
    field = optimizer._harmonic_lift @ boundary_field
    np.testing.assert_array_equal(field[optimizer.boundary_vertices], boundary_field)
    stiffness = cotangent_stiffness(optimizer.mesh)
    residual = stiffness @ field
    np.testing.assert_allclose(
        residual[optimizer.interior_vertices], 0.0, rtol=0.0, atol=2.0e-14
    )
    np.testing.assert_allclose(
        optimizer._harmonic_lift.T @ (stiffness @ optimizer._harmonic_lift),
        optimizer._boundary_schur,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_mesh_rejects_noninteger_faces() -> None:
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    with pytest.raises(TypeError, match="integer"):
        Mesh(vertices, np.asarray([[0.0, 1.0, 2.0]]))
    with pytest.raises(TypeError, match="integer"):
        Mesh(vertices, np.asarray([[0.0, np.nan, 2.0]]))


def test_obj_rejects_zero_index_even_before_later_vertices(tmp_path: Path) -> None:
    source = tmp_path / "zero-index.obj"
    source.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\nv 0 0 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one-based"):
        load_obj(source)


def test_disconnected_unused_vertex_is_rejected() -> None:
    mesh = load_obj(ROOT / "data" / "plane.obj")
    vertices = np.vstack((mesh.vertices, [[0.0, 1.0, 0.0]]))
    with pytest.raises(ValueError, match="connected"):
        HarmonicBoundaryOptimizer(Mesh(vertices, mesh.faces))


def test_mesh_without_interior_vertices_uses_boundary_system() -> None:
    mesh = Mesh(
        np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        np.asarray([[0, 1, 2], [0, 2, 3]]),
    )
    optimizer = HarmonicBoundaryOptimizer(mesh)
    parameters = parameters_from_knots(random_knots(2), optimizer.minimum_gap)
    loss, gradient = optimizer.loss_and_gradient(parameters)

    assert len(optimizer.interior_vertices) == 0
    assert np.isfinite(loss)
    assert np.isfinite(gradient).all()
    finite_difference = _finite_difference_gradient(optimizer, parameters)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2.0e-5, atol=2.0e-7)


def test_loss_plot_handles_zero_and_constant_histories() -> None:
    zero_svg = chart_svg([("zero", {0: [0.0, 0.0]})], title="zero", themed=False)
    constant_svg = chart_svg(
        [("constant", {0: [1.0, 1.0]})], title="constant", themed=False
    )
    assert zero_svg.startswith("<svg")
    assert constant_svg.startswith("<svg")
    with pytest.raises(ValueError, match="non-empty"):
        chart_svg([], title="empty", themed=False)


def test_zero_minimum_gap_is_rejected_consistently() -> None:
    with pytest.raises(ValueError, match="minimum_gap"):
        random_knots(0, minimum_gap=0.0)
    with pytest.raises(ValueError, match="minimum_gap"):
        knots_from_parameters(np.zeros(4), minimum_gap=0.0)
    with pytest.raises(ValueError, match="minimum_gap"):
        parameters_from_knots(np.asarray([0.0, 0.2, 0.5, 0.8]), minimum_gap=0.0)


def test_zero_initial_loss_has_finite_relative_reduction() -> None:
    assert relative_reduction(0.0, 0.0) == 0.0


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
    finite_difference = _finite_difference_gradient(optimizer, parameters)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2.0e-5, atol=2.0e-7)


def test_plane_corner_arcs_produce_affine_field() -> None:
    mesh = load_obj(ROOT / "data" / "plane.obj")
    optimizer = HarmonicBoundaryOptimizer(mesh, width_weight=0.0)
    boundary_points = mesh.vertices[optimizer.boundary_vertices]
    incoming = boundary_points - np.roll(boundary_points, 1, axis=0)
    outgoing = np.roll(boundary_points, -1, axis=0) - boundary_points
    incoming /= np.linalg.norm(incoming, axis=1)[:, None]
    outgoing /= np.linalg.norm(outgoing, axis=1)[:, None]
    corner_indices = np.sort(
        np.argsort(np.linalg.norm(outgoing - incoming, axis=1))[-4:]
    )
    knots = optimizer.boundary_positions[corner_indices]

    field, zero_weights, one_weights = optimizer.field_and_arc_weights(knots)
    loss, _ = optimizer._uniformity_loss_and_gradient(field)
    gradients = np.einsum("fij,fi->fj", optimizer._gradient_basis, field[mesh.faces])

    assert loss < 1.0e-10
    assert loss >= 0.0
    mean_gradient = np.average(gradients, axis=0, weights=optimizer.face_areas)
    np.testing.assert_allclose(
        gradients,
        np.broadcast_to(mean_gradient, gradients.shape),
        rtol=0.0,
        atol=2.0e-7,
    )

    # Only the two opposite sides are constrained.  Vertices strictly inside
    # the other sides obey the natural Neumann row (K u)_i = 0.
    constrained_edges = zero_weights + one_weights
    free_boundary = (constrained_edges == 0.0) & (np.roll(constrained_edges, 1) == 0.0)
    residual = cotangent_stiffness(mesh) @ field
    np.testing.assert_allclose(
        residual[optimizer.boundary_vertices[free_boundary]],
        0.0,
        rtol=0.0,
        atol=2.0e-12,
    )

    result = optimizer.optimize(random_knots(0), max_iterations=100)
    endpoint_distances = np.abs(
        (result.knots[:, None] - knots[None, :] + 0.5) % 1.0 - 0.5
    )
    assert result.final_loss < 1.0e-10
    assert np.all(endpoint_distances.min(axis=1) < 1.0e-5)


def test_optimization_decreases_loss(optimizer: HarmonicBoundaryOptimizer) -> None:
    initial_knots = random_knots(3)
    result = optimizer.optimize(initial_knots, max_iterations=8)
    assert np.isfinite(result.field).all()
    assert result.final_loss < result.initial_loss
    assert result.history[0] == pytest.approx(result.initial_loss)
    assert result.history[-1] == pytest.approx(result.final_loss)
    assert result.parameter_history.shape == (len(result.history), 4)
    assert len(result.history) == result.iterations + 1
    np.testing.assert_allclose(
        result.parameter_history[0],
        parameters_from_knots(initial_knots, optimizer.minimum_gap),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.parameter_history[-1], result.parameters, rtol=0.0, atol=0.0
    )
    for parameters, loss in zip(result.parameter_history, result.history):
        assert optimizer.loss_and_gradient(parameters)[0] == pytest.approx(loss)
    assert np.all(np.diff(result.history) <= 1.0e-10)
    assert result.statistics.spacing_cv > 0.0


def test_optimization_handles_curved_mesh() -> None:
    optimizer = HarmonicBoundaryOptimizer(
        load_obj(ROOT / "data" / "triple_peak.obj"),
        target_arc_width=0.1,
        width_weight=0.1,
    )
    parameters = parameters_from_knots(random_knots(7), optimizer.minimum_gap)
    _, gradient = optimizer.loss_and_gradient(parameters)
    finite_difference = _finite_difference_gradient(optimizer, parameters)
    np.testing.assert_allclose(gradient, finite_difference, rtol=5.0e-7, atol=2.0e-7)

    result = optimizer.optimize(random_knots(0), max_iterations=10)
    assert result.final_loss < result.initial_loss
    assert result.field.min() >= -2.0e-3
    assert result.field.max() <= 1.0 + 2.0e-3
