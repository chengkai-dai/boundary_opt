from pathlib import Path

import numpy as np
import pytest

from boundary_opt.boundary import boundary_arclength
from boundary_opt.fem import cotangent_stiffness, face_gradient_basis
from geometry import boundary_loop, load_obj, normalize_mesh

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


@pytest.mark.parametrize("name", ["disk", "plane", "peak", "triple_peak"])
def test_normalize_mesh_uses_one_centered_unit_box(name: str) -> None:
    mesh = normalize_mesh(load_obj(ROOT / "data" / f"{name}.obj"), 1.0)
    lower = mesh.vertices.min(axis=0)
    upper = mesh.vertices.max(axis=0)
    np.testing.assert_allclose(0.5 * (lower + upper), 0.0, atol=1.0e-15)
    assert float((upper - lower).max()) == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [0.5, 2.5])
def test_normalize_mesh_sets_target_scale(scale: float) -> None:
    mesh = normalize_mesh(load_obj(ROOT / "data" / "plane.obj"), scale)
    assert float(np.ptp(mesh.vertices, axis=0).max()) == pytest.approx(scale)


@pytest.mark.parametrize("scale", [0.0, -1.0, np.inf])
def test_normalize_mesh_rejects_invalid_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="normalization scale"):
        normalize_mesh(load_obj(ROOT / "data" / "plane.obj"), scale)


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
