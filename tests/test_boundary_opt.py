from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boundary_opt import (
    HarmonicBoundaryOptimizer,
    Mesh,
    _gap_kkt_residual,
    cotangent_stiffness,
    cyclic_arc_edge_weights,
    face_gradient_basis,
    knots_from_parameters,
    load_obj,
    parameters_from_knots,
    random_knots,
)
from plot_loss_curves import chart_svg, read_histories
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
    directions = np.eye(len(parameters)) * step
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


def _arc_mean(
    optimizer: HarmonicBoundaryOptimizer,
    field: np.ndarray,
    start: float,
    end: float,
) -> float:
    mass = optimizer._arc_mass(start, end)
    boundary_field = field[optimizer.boundary_vertices]
    return float(mass.sum(axis=0) @ boundary_field / mass.sum())


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
    field = optimizer.robin_field_from_knots(knots)
    zero_mass = optimizer._arc_mass(knots[0], knots[1])
    one_mass = optimizer._arc_mass(knots[2], knots[3])
    residual = np.asarray(cotangent_stiffness(optimizer.mesh) @ field)
    boundary_field = field[optimizer.boundary_vertices]
    residual[optimizer.boundary_vertices] += optimizer.boundary_penalty * (
        (zero_mass + one_mass) @ boundary_field - one_mass.sum(axis=1)
    )
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=2.0e-12)


def test_canonical_field_has_exact_target_arc_means_when_an_arc_wraps(
    optimizer: HarmonicBoundaryOptimizer,
) -> None:
    knots = np.asarray([0.85, 1.05, 1.25, 1.45])
    field = optimizer.field_from_knots(knots)
    assert _arc_mean(optimizer, field, knots[0], knots[1]) == pytest.approx(
        0.0, abs=5.0e-14
    )
    assert _arc_mean(optimizer, field, knots[2], knots[3]) == pytest.approx(
        1.0, abs=5.0e-14
    )


def test_canonicalization_preserves_harmonicity_and_uniformity(
    optimizer: HarmonicBoundaryOptimizer,
) -> None:
    knots = random_knots(5, optimizer.minimum_gap)
    robin_field = optimizer.robin_field_from_knots(knots)
    field, boundary_statistics = optimizer.field_and_boundary_statistics_from_knots(
        knots
    )
    np.testing.assert_allclose(
        field,
        (robin_field - boundary_statistics.raw_zero_mean)
        / boundary_statistics.raw_span,
        rtol=0.0,
        atol=2.0e-15,
    )
    residual = cotangent_stiffness(optimizer.mesh) @ field
    np.testing.assert_allclose(
        residual[optimizer.interior_vertices], 0.0, rtol=0.0, atol=5.0e-13
    )
    robin_loss, _ = optimizer._uniformity_loss_and_gradient(robin_field)
    canonical_loss, _ = optimizer._uniformity_loss_and_gradient(field)
    assert canonical_loss == pytest.approx(robin_loss, rel=2.0e-14)

    robin_boundary = robin_field[optimizer.boundary_vertices]
    canonical_boundary = field[optimizer.boundary_vertices]
    zero_mass = optimizer._arc_mass(knots[0], knots[1])
    one_mass = optimizer._arc_mass(knots[2], knots[3])
    zero_target_rms = np.sqrt(
        robin_boundary @ (zero_mass @ robin_boundary) / zero_mass.sum()
    )
    one_residual = robin_boundary - 1.0
    one_target_rms = np.sqrt(one_residual @ (one_mass @ one_residual) / one_mass.sum())
    zero_shape_rms = np.sqrt(
        canonical_boundary @ (zero_mass @ canonical_boundary) / zero_mass.sum()
    )
    canonical_one_residual = canonical_boundary - 1.0
    one_shape_rms = np.sqrt(
        canonical_one_residual @ (one_mass @ canonical_one_residual) / one_mass.sum()
    )
    assert boundary_statistics.raw_zero_target_rms == pytest.approx(zero_target_rms)
    assert boundary_statistics.raw_one_target_rms == pytest.approx(one_target_rms)
    assert boundary_statistics.canonical_zero_target_rms == pytest.approx(
        zero_shape_rms
    )
    assert boundary_statistics.canonical_one_target_rms == pytest.approx(one_shape_rms)


@pytest.mark.parametrize("case", ["constant", "tiny", "negative"])
def test_canonicalization_rejects_a_field_without_stable_positive_span(
    optimizer: HarmonicBoundaryOptimizer,
    case: str,
) -> None:
    knots = random_knots(6, optimizer.minimum_gap)
    baseline = optimizer.robin_field_from_knots(knots)
    fields = {
        "constant": np.ones(len(optimizer.mesh.vertices)),
        "tiny": 0.5 + 1.0e-12 * (baseline - 0.5),
        "negative": -baseline,
    }
    with pytest.raises(ValueError, match="mean span is too small"):
        optimizer._canonicalize_field(fields[case], knots)


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
    parameters = parameters_from_knots(random_knots(2))
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


def test_loss_plot_reads_recorded_state_history(tmp_path: Path) -> None:
    history = tmp_path / "history.csv"
    history.write_text("seed,recorded_state,loss\n0,1,0.5\n0,0,1.0\n", encoding="utf-8")
    assert read_histories(history) == {0: [1.0, 0.5]}


def test_zero_minimum_gap_is_rejected_consistently() -> None:
    with pytest.raises(ValueError, match="minimum_gap"):
        random_knots(0, minimum_gap=0.0)
    with pytest.raises(ValueError, match="minimum_gap"):
        HarmonicBoundaryOptimizer(
            load_obj(ROOT / "data" / "plane.obj"), minimum_gap=0.0
        )


def test_target_arc_width_accepts_exactly_the_feasible_interval(
    optimizer: HarmonicBoundaryOptimizer,
) -> None:
    for target in (0.1, 0.4):
        configured = HarmonicBoundaryOptimizer(
            optimizer.mesh,
            minimum_gap=0.1,
            target_arc_width=target,
            width_weight=1.0,
        )
        assert configured.target_arc_width == target
    for target in (0.099, 0.401):
        with pytest.raises(ValueError, match="target_arc_width"):
            HarmonicBoundaryOptimizer(
                optimizer.mesh,
                minimum_gap=0.1,
                target_arc_width=target,
                width_weight=1.0,
            )


def test_direct_gap_parameters_and_jacobian() -> None:
    knots = random_knots(3)
    parameters = parameters_from_knots(knots)
    recovered, jacobian, gaps = knots_from_parameters(parameters)
    np.testing.assert_allclose(recovered, knots, rtol=0.0, atol=2.0e-16)
    np.testing.assert_allclose(gaps.sum(), 1.0, rtol=0.0, atol=2.0e-16)
    step = 1.0e-7
    finite_difference = np.column_stack(
        [
            (
                knots_from_parameters(parameters + step * direction)[0]
                - knots_from_parameters(parameters - step * direction)[0]
            )
            / (2.0 * step)
            for direction in np.eye(5)
        ]
    )
    np.testing.assert_allclose(jacobian, finite_difference, rtol=2.0e-9, atol=2.0e-9)


def test_gap_kkt_residual_understands_an_active_lower_bound() -> None:
    minimum_gap = 0.03
    gaps = np.asarray([minimum_gap, 0.3, 0.3, 0.37])
    assert (
        _gap_kkt_residual(np.asarray([0.0, 1.0, 0.0, 0.0, 0.0]), gaps, minimum_gap)
        == 0.0
    )
    assert _gap_kkt_residual(
        np.asarray([0.0, -1.0, 0.0, 0.0, 0.0]), gaps, minimum_gap
    ) == pytest.approx(1.0)


def test_gap_kkt_residual_does_not_hide_near_bound_interior_gradients() -> None:
    gaps = np.asarray([0.030000005, 0.3, 0.3, 0.369999995])
    residual = _gap_kkt_residual(np.asarray([0.0, 100.0, 0.0, 0.0, 0.0]), gaps, 0.03)
    assert residual == pytest.approx(75.0)

    near_quarter = np.full(4, 0.25)
    residual = _gap_kkt_residual(
        np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
        near_quarter,
        0.249999999,
    )
    assert residual == pytest.approx(1.5)


def test_small_minimum_gap_keeps_slsqp_trial_fields_valid() -> None:
    optimizer = HarmonicBoundaryOptimizer(
        load_obj(ROOT / "data" / "disk.obj"), minimum_gap=1.0e-6
    )
    result = optimizer.optimize(
        random_knots(0, minimum_gap=optimizer.minimum_gap), max_iterations=5
    )
    assert np.isfinite(result.field).all()


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


def test_full_five_coordinate_gradient(optimizer: HarmonicBoundaryOptimizer) -> None:
    parameters = parameters_from_knots(random_knots(7))
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
    assert result.success
    assert result.kkt_residual <= 1.0e-5
    assert np.all(endpoint_distances.min(axis=1) < 1.0e-5)


def test_optimization_decreases_loss(optimizer: HarmonicBoundaryOptimizer) -> None:
    initial_knots = random_knots(3)
    result = optimizer.optimize(initial_knots, max_iterations=8)
    assert np.isfinite(result.field).all()
    assert result.final_loss < result.initial_loss
    assert result.history[0] == pytest.approx(result.initial_loss)
    assert result.history[-1] == pytest.approx(result.final_loss)
    assert result.parameter_history.shape == (len(result.history), 5)
    np.testing.assert_allclose(
        result.parameter_history[0],
        parameters_from_knots(initial_knots),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.parameter_history[-1], result.parameters, rtol=0.0, atol=0.0
    )
    for parameters, loss in zip(result.parameter_history, result.history):
        assert optimizer.loss_and_gradient(parameters)[0] == pytest.approx(loss)
    assert np.all(result.gaps >= optimizer.minimum_gap - 1.0e-10)
    assert result.constraint_violation <= 1.0e-10
    assert result.statistics.gradient_cv > 0.0
    assert result.statistics.spacing_cv > 0.0
    np.testing.assert_allclose(
        result.field, optimizer.field_from_knots(result.knots), rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        result.raw_field,
        optimizer.robin_field_from_knots(result.knots),
        rtol=0.0,
        atol=0.0,
    )
    assert result.boundary_statistics.raw_span > 0.0

    norms = np.linalg.norm(optimizer._face_gradients(result.field), axis=1)
    gradient_mean = float(optimizer._face_weights @ norms)
    gradient_cv = (
        np.sqrt(optimizer._face_weights @ (norms - gradient_mean) ** 2) / gradient_mean
    )
    spacings = 1.0 / norms
    spacing_mean = float(optimizer._face_weights @ spacings)
    spacing_cv = (
        np.sqrt(optimizer._face_weights @ (spacings - spacing_mean) ** 2) / spacing_mean
    )
    assert result.statistics.gradient_cv == pytest.approx(gradient_cv)
    assert result.statistics.spacing_cv == pytest.approx(spacing_cv)


def test_zero_gradient_field_reports_infinite_spacing_cv(
    optimizer: HarmonicBoundaryOptimizer,
) -> None:
    statistics = optimizer._field_statistics(
        np.zeros(len(optimizer.mesh.vertices), dtype=np.float64)
    )
    assert statistics.gradient_cv == 0.0
    assert statistics.spacing_cv == np.inf


def test_optimization_handles_curved_mesh() -> None:
    optimizer = HarmonicBoundaryOptimizer(
        load_obj(ROOT / "data" / "triple_peak.obj"),
        target_arc_width=0.1,
        width_weight=0.1,
    )
    parameters = parameters_from_knots(random_knots(7))
    _, gradient = optimizer.loss_and_gradient(parameters)
    finite_difference = _finite_difference_gradient(optimizer, parameters)
    np.testing.assert_allclose(gradient, finite_difference, rtol=5.0e-7, atol=2.0e-7)

    result = optimizer.optimize(random_knots(0), max_iterations=10)
    assert result.final_loss < result.initial_loss
    assert np.isfinite(result.field).all()
    assert _arc_mean(
        optimizer, result.field, result.knots[0], result.knots[1]
    ) == pytest.approx(0.0, abs=5.0e-13)
    assert _arc_mean(
        optimizer, result.field, result.knots[2], result.knots[3]
    ) == pytest.approx(1.0, abs=5.0e-13)
