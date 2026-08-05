from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boundary_opt import (
    Mesh,
    cotangent_stiffness,
    face_gradient_basis,
    load_obj,
    random_knots,
)
from continuous_partial_opt import (
    ContinuousPartialBoundaryOptimizer,
    _canonical_knots,
)
from visualize_continuous_partial import display_mesh

ROOT = Path(__file__).resolve().parent.parent


def plane_corners(optimizer: ContinuousPartialBoundaryOptimizer) -> np.ndarray:
    points = optimizer.mesh.vertices[optimizer.boundary_vertices]
    incoming = points - np.roll(points, 1, axis=0)
    outgoing = np.roll(points, -1, axis=0) - points
    cosines = np.einsum("ij,ij->i", incoming, outgoing) / (
        np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
    )
    indices = np.sort(np.argsort(np.arccos(np.clip(cosines, -1.0, 1.0)))[-4:])
    return optimizer.boundary_positions[indices]


@pytest.mark.parametrize("boundary_smoothing", [0.0, 0.1, 1.0, 100.0])
def test_plane_corners_reproduce_affine_partial_dirichlet_solution(
    boundary_smoothing: float,
) -> None:
    optimizer = ContinuousPartialBoundaryOptimizer(
        load_obj(ROOT / "data" / "plane.obj"),
        boundary_smoothing=boundary_smoothing,
    )
    evaluation = optimizer.evaluate(plane_corners(optimizer))

    assert evaluation.loss < 1.0e-10
    assert evaluation.system_residual < 1.0e-11
    assert len(evaluation.cut_points) == 0
    assert len(evaluation.field) == len(optimizer.mesh.vertices)


def test_display_mesh_handles_solution_without_cut_patches() -> None:
    optimizer = ContinuousPartialBoundaryOptimizer(
        load_obj(ROOT / "data" / "plane.obj")
    )
    evaluation = optimizer.evaluate(plane_corners(optimizer))

    vertices, faces, values = display_mesh(
        optimizer, evaluation.knots, evaluation.field
    )

    assert vertices.shape == (len(optimizer.mesh.vertices) + 4, 3)
    np.testing.assert_array_equal(faces, optimizer.mesh.faces)
    assert values.shape == (len(vertices),)


def test_interior_endpoints_are_eliminated_from_global_field() -> None:
    mesh = load_obj(ROOT / "data" / "plane.obj")
    original_vertices = mesh.vertices.copy()
    original_faces = mesh.faces.copy()
    optimizer = ContinuousPartialBoundaryOptimizer(mesh)
    corners = plane_corners(optimizer)
    knots = corners + np.asarray([0.01, -0.01, 0.01, -0.01])
    evaluation = optimizer.evaluate(knots)

    assert evaluation.cut_points.shape == (4, 3)
    assert evaluation.cut_values.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert len(evaluation.field) == len(optimizer.mesh.vertices)
    assert evaluation.integration_faces == len(optimizer.mesh.faces) + 12
    assert evaluation.system_residual < 1.0e-10
    np.testing.assert_allclose(evaluation.field.min(), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(evaluation.field.max(), 1.0, atol=1.0e-12)
    np.testing.assert_array_equal(mesh.vertices, original_vertices)
    np.testing.assert_array_equal(mesh.faces, original_faces)


def test_cycle_shift_does_not_change_solution() -> None:
    optimizer = ContinuousPartialBoundaryOptimizer(load_obj(ROOT / "data" / "disk.obj"))
    knots = random_knots(5)
    first = optimizer.evaluate(knots)
    shifted = optimizer.evaluate(knots + 1.0)

    assert first.loss == pytest.approx(shifted.loss, rel=1.0e-13)
    np.testing.assert_allclose(first.field, shifted.field, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(first.endpoint_points, shifted.endpoint_points)


def test_wrapped_saved_knots_round_trip() -> None:
    optimizer = ContinuousPartialBoundaryOptimizer(load_obj(ROOT / "data" / "disk.obj"))
    knots = np.asarray([0.82, 0.93, 1.17, 1.51])
    unwrapped = optimizer.evaluate(knots)
    wrapped = optimizer.evaluate(np.mod(knots, 1.0))

    assert wrapped.loss == pytest.approx(unwrapped.loss, rel=1.0e-13)
    np.testing.assert_allclose(wrapped.field, unwrapped.field, atol=1.0e-13)
    np.testing.assert_allclose(wrapped.endpoint_points, unwrapped.endpoint_points)


def test_zero_boundary_smoothing_is_an_exact_regression() -> None:
    mesh = load_obj(ROOT / "data" / "disk.obj")
    knots = random_knots(6)
    default = ContinuousPartialBoundaryOptimizer(mesh).evaluate(knots)
    explicit_zero = ContinuousPartialBoundaryOptimizer(
        mesh, boundary_smoothing=0.0
    ).evaluate(knots)

    assert explicit_zero.loss == default.loss
    np.testing.assert_array_equal(explicit_zero.field, default.field)


@pytest.mark.parametrize("boundary_smoothing", [0.0, 0.3])
def test_swapping_zero_and_one_arcs_complements_the_field(
    boundary_smoothing: float,
) -> None:
    optimizer = ContinuousPartialBoundaryOptimizer(
        load_obj(ROOT / "data" / "disk.obj"),
        boundary_smoothing=boundary_smoothing,
    )
    knots = random_knots(8)
    rotated = np.concatenate((knots[2:], knots[:2] + 1.0))
    first = optimizer.evaluate(knots)
    complemented = optimizer.evaluate(rotated)

    assert complemented.loss == pytest.approx(first.loss, rel=1.0e-12)
    np.testing.assert_allclose(complemented.field, 1.0 - first.field, atol=1.0e-12)


def test_local_elimination_matches_an_explicit_augmented_mesh() -> None:
    optimizer = ContinuousPartialBoundaryOptimizer(
        load_obj(ROOT / "data" / "plane.obj")
    )
    knots, _ = _canonical_knots(
        plane_corners(optimizer) + np.asarray([0.01, -0.01, 0.01, -0.01])
    )
    evaluation = optimizer.evaluate(knots)
    assembly = optimizer._assemble(knots)
    original_count = len(optimizer.mesh.vertices)

    unaffected = np.ones(len(optimizer.mesh.faces), dtype=bool)
    unaffected[assembly.affected_faces] = False
    center_offset = original_count + 4
    local_faces = []
    for patch_index, patch in enumerate(assembly.local_patches):
        center_reference = center_offset + patch_index
        for index, reference in enumerate(patch.boundary_references):
            following = patch.boundary_references[
                (index + 1) % len(patch.boundary_references)
            ]
            local_faces.append((reference, following, center_reference))
    augmented = Mesh(
        np.vstack(
            (
                optimizer.mesh.vertices,
                assembly.endpoint_points,
                np.asarray([patch.center for patch in assembly.local_patches]),
            )
        ),
        np.asarray(
            [
                *optimizer.mesh.faces[unaffected].tolist(),
                *local_faces,
            ],
            dtype=np.int64,
        ),
    )
    augmented_stiffness = cotangent_stiffness(augmented)
    boundary_count = original_count + 4
    boundary_block = augmented_stiffness[:boundary_count, :boundary_count]
    center_coupling = augmented_stiffness[:boundary_count, boundary_count:]
    center_block = augmented_stiffness[boundary_count:, boundary_count:].toarray()
    condensed = boundary_block.toarray() - center_coupling @ np.linalg.solve(
        center_block, center_coupling.T.toarray()
    )
    matrix_difference = (
        assembly.stiffness.toarray() - condensed[:original_count, :original_count]
    )
    assert np.max(np.abs(matrix_difference), initial=0.0) < 1.0e-11

    endpoint_values = np.asarray([0.0, 0.0, 1.0, 1.0])
    expected_rhs = -(condensed[:original_count, original_count:] @ endpoint_values)
    np.testing.assert_allclose(assembly.cut_rhs, expected_rhs, atol=1.0e-11)

    boundary_field = np.concatenate((evaluation.field, endpoint_values))
    center_field = -np.linalg.solve(
        center_block,
        augmented_stiffness[boundary_count:, :boundary_count] @ boundary_field,
    )
    augmented_field = np.concatenate((boundary_field, center_field))
    areas, basis = face_gradient_basis(augmented)
    gradients = np.einsum("fij,fi->fj", basis, augmented_field[augmented.faces])
    squared_norms = np.einsum("ij,ij->i", gradients, gradients)
    weights = areas / areas.sum()
    mean = float(weights @ squared_norms)
    expected_loss = float(weights @ squared_norms**2) / mean**2 - 1.0
    assert evaluation.loss == pytest.approx(expected_loss, abs=1.0e-12)


def test_boundary_smoothing_matches_explicit_one_dimensional_fem() -> None:
    vertices = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.35, 1.2, 0.0]])
    mesh = Mesh(vertices, np.asarray([[0, 1, 2]], dtype=np.int64))
    knots = np.asarray([0.08, 0.38, 0.67, 0.91])
    baseline = ContinuousPartialBoundaryOptimizer(
        mesh, minimum_gap=0.01, snap_tolerance=1.0e-12
    )
    smoothed = ContinuousPartialBoundaryOptimizer(
        mesh,
        minimum_gap=0.01,
        snap_tolerance=1.0e-12,
        boundary_smoothing=0.3,
    )
    base_assembly = baseline._assemble(knots)
    smooth_assembly = smoothed._assemble(knots)
    original_count = len(vertices)

    references = smooth_assembly.local_patches[0].boundary_references
    points = smoothed._reference_points(references, smooth_assembly.endpoint_points)
    perimeter = sum(
        np.linalg.norm(
            vertices[mesh.faces[0, (index + 1) % 3]] - vertices[mesh.faces[0, index]]
        )
        for index in range(3)
    )
    boundary_matrix = np.zeros((original_count + 4, original_count + 4))
    for index, start in enumerate(references):
        following = (index + 1) % len(references)
        end = references[following]
        normalized_length = (
            np.linalg.norm(points[following] - points[index]) / perimeter
        )
        local = 0.3 / normalized_length * np.asarray([[1.0, -1.0], [-1.0, 1.0]])
        boundary_matrix[np.ix_([start, end], [start, end])] += local

    np.testing.assert_allclose(
        (smooth_assembly.stiffness - base_assembly.stiffness).toarray(),
        boundary_matrix[:original_count, :original_count],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        smooth_assembly.cut_rhs - base_assembly.cut_rhs,
        -(
            boundary_matrix[:original_count, original_count:]
            @ np.asarray([0.0, 0.0, 1.0, 1.0])
        ),
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    "knots",
    [
        np.asarray([0.02, 0.05, 0.08, 0.11]),
        np.asarray([0.10, 0.20, 0.40, 0.50]),
    ],
)
def test_multiple_cuts_in_one_face_cover_it_once(knots: np.ndarray) -> None:
    height = np.sqrt(3.0) / 2.0
    mesh = Mesh(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, height, 0.0]]),
        np.asarray([[0, 1, 2]], dtype=np.int64),
    )
    optimizer = ContinuousPartialBoundaryOptimizer(
        mesh, minimum_gap=0.01, snap_tolerance=1.0e-10
    )
    evaluation = optimizer.evaluate(knots)
    assembly = optimizer._assemble(evaluation.knots)

    cell_area = 0.0
    patch = assembly.local_patches[0]
    boundary_points = optimizer._reference_points(
        patch.boundary_references, assembly.endpoint_points
    )
    for index, point in enumerate(boundary_points):
        following = boundary_points[(index + 1) % len(boundary_points)]
        cell_area += 0.5 * np.linalg.norm(
            np.cross(following - point, patch.center - point)
        )
    assert cell_area == pytest.approx(np.sqrt(3.0) / 4.0, rel=1.0e-13)
    assert len(patch.boundary_references) == 7
    assert evaluation.integration_faces == 7
    assert evaluation.system_residual < 1.0e-10
    assert np.isfinite(evaluation.field).all()


def test_snap_tolerance_cannot_merge_distinct_arcs() -> None:
    with pytest.raises(ValueError, match="snap_tolerance"):
        ContinuousPartialBoundaryOptimizer(
            load_obj(ROOT / "data" / "plane.obj"),
            minimum_gap=0.03,
            snap_tolerance=0.015,
        )
    with pytest.raises(ValueError, match="boundary_smoothing"):
        ContinuousPartialBoundaryOptimizer(
            load_obj(ROOT / "data" / "plane.obj"), boundary_smoothing=-1.0
        )


def test_internal_finite_difference_can_cross_an_active_gap_constraint() -> None:
    optimizer = ContinuousPartialBoundaryOptimizer(
        load_obj(ROOT / "data" / "plane.obj")
    )
    just_outside = np.asarray([0.0, 0.03 - 1.0e-6, 0.50, 0.75])

    with pytest.raises(ValueError, match="minimum_gap"):
        optimizer.evaluate(just_outside)
    internal = optimizer._evaluate(just_outside, enforce_minimum_gap=False)
    assert np.isfinite(internal.loss)


@pytest.mark.parametrize("boundary_smoothing", [0.0, 0.3])
def test_local_assembly_is_invariant_to_cyclic_vertex_relabeling(
    boundary_smoothing: float,
) -> None:
    vertices = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.35, 1.2, 0.0]])
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    knots = np.asarray([0.08, 0.38, 0.67, 0.91])
    base = ContinuousPartialBoundaryOptimizer(
        Mesh(vertices, faces),
        minimum_gap=0.01,
        snap_tolerance=1.0e-12,
        boundary_smoothing=boundary_smoothing,
    )
    expected = base.evaluate(knots)

    permutation = np.asarray([1, 2, 0])
    relabeled = ContinuousPartialBoundaryOptimizer(
        Mesh(vertices[permutation], faces.copy()),
        minimum_gap=0.01,
        snap_tolerance=1.0e-12,
        boundary_smoothing=boundary_smoothing,
    )
    old_start_slot = int(np.flatnonzero(base.boundary_vertices == permutation[0])[0])
    shift = float(base.boundary_positions[old_start_slot])
    actual = relabeled.evaluate(np.mod(knots - shift, 1.0))

    field_in_old_order = np.empty_like(actual.field)
    field_in_old_order[permutation] = actual.field
    np.testing.assert_allclose(field_in_old_order, expected.field, atol=1.0e-11)
    np.testing.assert_allclose(
        actual.endpoint_points, expected.endpoint_points, atol=1.0e-12
    )
    assert actual.loss == pytest.approx(expected.loss, abs=1.0e-11)
    assert actual.integration_faces == expected.integration_faces


def test_boundary_smoothing_is_scale_invariant() -> None:
    mesh = load_obj(ROOT / "data" / "plane.obj")
    knots = random_knots(4)
    original = ContinuousPartialBoundaryOptimizer(mesh, boundary_smoothing=0.3)
    scaled = ContinuousPartialBoundaryOptimizer(
        Mesh(7.0 * mesh.vertices, mesh.faces.copy()), boundary_smoothing=0.3
    )

    first = original.evaluate(knots)
    second = scaled.evaluate(knots)
    np.testing.assert_allclose(first.field, second.field, atol=1.0e-12)
    assert first.loss == pytest.approx(second.loss, abs=1.0e-12)


def test_plane_feature_start_finds_the_four_corner_solution() -> None:
    optimizer = ContinuousPartialBoundaryOptimizer(
        load_obj(ROOT / "data" / "plane.obj")
    )
    result = optimizer.optimize(random_knots(0), max_iterations=3, starts=1)

    assert result.final_loss < 1.0e-10
    assert np.all(np.diff(result.history) <= 0.0)
    assert len(result.field) == len(optimizer.mesh.vertices)
    np.testing.assert_allclose(
        np.sort(np.mod(result.knots, 1.0)), plane_corners(optimizer), atol=1.0e-14
    )
