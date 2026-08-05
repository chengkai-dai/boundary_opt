from pathlib import Path

import numpy as np
import pytest

from boundary_opt import load_obj
from boundary_opt.loss import (
    area_balance_loss_and_gradient,
    isoline_length_loss_and_gradient,
)
from boundary_opt.mesh import face_gradient_basis

ROOT = Path(__file__).resolve().parent.parent


def test_area_balance_gradient_matches_finite_difference() -> None:
    mesh = load_obj(ROOT / "data" / "plane.obj")
    areas, _ = face_gradient_basis(mesh)
    weights = areas / areas.sum()
    rng = np.random.default_rng(3)
    field = rng.uniform(size=len(mesh.vertices))
    direction = rng.normal(size=len(mesh.vertices))
    loss, gradient = area_balance_loss_and_gradient(field, mesh.faces, weights)
    step = 1.0e-6
    forward = area_balance_loss_and_gradient(
        field + step * direction, mesh.faces, weights
    )[0]
    backward = area_balance_loss_and_gradient(
        field - step * direction, mesh.faces, weights
    )[0]
    assert loss >= 0.0
    assert (forward - backward) / (2.0 * step) == pytest.approx(
        float(gradient @ direction), rel=2.0e-7, abs=2.0e-9
    )


def test_area_balance_distinguishes_axis_and_diagonal_plane_fields() -> None:
    mesh = load_obj(ROOT / "data" / "plane.obj")
    areas, _ = face_gradient_basis(mesh)
    weights = areas / areas.sum()
    x = mesh.vertices[:, 0]
    z = mesh.vertices[:, 2]
    x = (x - x.min()) / np.ptp(x)
    z = (z - z.min()) / np.ptp(z)
    axis_loss = area_balance_loss_and_gradient(x, mesh.faces, weights)[0]
    diagonal_loss = area_balance_loss_and_gradient(0.5 * (x + z), mesh.faces, weights)[
        0
    ]
    assert axis_loss < 1.0e-6
    assert diagonal_loss > 1.0e-3


def test_isoline_length_gradient_matches_finite_difference() -> None:
    mesh = load_obj(ROOT / "data" / "plane.obj")
    areas, gradient_basis = face_gradient_basis(mesh)
    weights = areas / areas.sum()
    rng = np.random.default_rng(5)
    field = rng.uniform(size=len(mesh.vertices))
    direction = rng.normal(size=len(mesh.vertices))
    loss, gradient = isoline_length_loss_and_gradient(
        field, mesh.faces, gradient_basis, weights
    )
    step = 1.0e-6
    forward = isoline_length_loss_and_gradient(
        field + step * direction, mesh.faces, gradient_basis, weights
    )[0]
    backward = isoline_length_loss_and_gradient(
        field - step * direction, mesh.faces, gradient_basis, weights
    )[0]
    assert loss >= 0.0
    assert (forward - backward) / (2.0 * step) == pytest.approx(
        float(gradient @ direction), rel=2.0e-7, abs=2.0e-9
    )


def test_isoline_length_distinguishes_axis_and_diagonal_plane_fields() -> None:
    mesh = load_obj(ROOT / "data" / "plane.obj")
    areas, gradient_basis = face_gradient_basis(mesh)
    weights = areas / areas.sum()
    x = mesh.vertices[:, 0]
    z = mesh.vertices[:, 2]
    x = (x - x.min()) / np.ptp(x)
    z = (z - z.min()) / np.ptp(z)
    axis_loss = isoline_length_loss_and_gradient(
        x, mesh.faces, gradient_basis, weights
    )[0]
    diagonal_loss = isoline_length_loss_and_gradient(
        0.5 * (x + z), mesh.faces, gradient_basis, weights
    )[0]
    assert axis_loss < 1.0e-8
    assert diagonal_loss > 1.0e-3
