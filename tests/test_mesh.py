from pathlib import Path

import numpy as np
import pytest

from boundary_opt.mesh import (
    boundary_arclength,
    boundary_loop,
    cotangent_stiffness,
    face_gradient_basis,
    load_obj,
)

ROOT = Path(__file__).resolve().parent.parent


def test_obj_boundary_and_arclength() -> None:
    mesh = load_obj(ROOT / "data" / "plane.obj")
    loop = boundary_loop(mesh.faces)
    positions = boundary_arclength(mesh.vertices, loop)
    assert len(mesh.vertices) == 1089
    assert len(mesh.faces) == 2048
    assert len(loop) == 128
    assert positions[0] == 0.0
    assert np.all(np.diff(positions) > 0.0)
    assert positions[-1] < 1.0


def test_cotangent_energy_matches_face_gradient_energy() -> None:
    mesh = load_obj(ROOT / "data" / "disk.obj")
    rng = np.random.default_rng(9)
    field = rng.normal(size=len(mesh.vertices))
    stiffness = cotangent_stiffness(mesh)
    areas, basis = face_gradient_basis(mesh)
    gradients = np.einsum("fij,fi->fj", basis, field[mesh.faces])
    stiffness_energy = float(field @ (stiffness @ field))
    gradient_energy = float(areas @ np.einsum("ij,ij->i", gradients, gradients))
    assert stiffness_energy == pytest.approx(gradient_energy, rel=2.0e-12)
