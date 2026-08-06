from pathlib import Path

import numpy as np

from boundary_opt import BoundaryOptimizer
from geometry import Mesh, load_obj, normalize_mesh
from knitting import sample_boundary_course
from visualize_pipeline import _boundary_curve_points

ROOT = Path(__file__).resolve().parent.parent


def test_boundary_curve_includes_interpolated_endpoints() -> None:
    mesh = Mesh(
        vertices=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        ),
        faces=np.asarray(((0, 1, 2), (0, 2, 3))),
    )

    points = _boundary_curve_points(mesh, 0.125, 0.625)

    np.testing.assert_allclose(
        points,
        ((0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.5, 1.0, 0.0)),
    )


def test_plane_initial_front_keeps_corners_within_half_stitch() -> None:
    mesh = normalize_mesh(load_obj(ROOT / "data" / "plane.obj"), 1.0)
    optimizer = BoundaryOptimizer(
        mesh,
        uniformity_weight=1.0,
        length_smoothness_weight=0.0,
    )
    points = mesh.vertices[optimizer.harmonic.boundary_vertices]
    incoming = points - np.roll(points, 1, axis=0)
    outgoing = np.roll(points, -1, axis=0) - points
    cosines = np.einsum("ij,ij->i", incoming, outgoing) / (
        np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
    )
    corners = np.sort(
        optimizer.boundary_positions[
            np.argsort(np.arccos(np.clip(cosines, -1.0, 1.0)))[-4:]
        ]
    )
    knots = corners.copy()
    knots[0] += 1.0e-8
    knots[1] -= 1.0e-8

    front = sample_boundary_course(
        mesh,
        optimizer.harmonic.boundary_vertices,
        optimizer.boundary_positions,
        knots[0],
        knots[1],
        stitch_spacing=0.15,
    )

    assert len(front) == 33
    np.testing.assert_allclose(
        mesh.vertices[front, 0], mesh.vertices[:, 0].min(), atol=3.0e-7
    )
