"""Four-parameter harmonic boundary optimization on triangle meshes.

The four design variables produce four ordered cyclic knots.  They define a
P1-compatible boundary profile with a zero plateau, a linear rise, a one
plateau, and a linear fall.  The interior is the cotangent-harmonic extension
of that fixed boundary data.  Within each boundary-edge cell, gradients use
one exact adjoint backsolve; no knitting code, autodiff framework, moving
vertex set, or contour extraction is involved.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Mesh:
    vertices: FloatArray
    faces: IntArray

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (V, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if not np.isfinite(self.vertices).all():
            raise ValueError("vertices contain NaN or infinite values")
        if len(self.vertices) == 0 or len(self.faces) == 0:
            raise ValueError("mesh must contain vertices and faces")
        if self.faces.min() < 0 or self.faces.max() >= len(self.vertices):
            raise ValueError("faces reference a vertex outside the mesh")


@dataclass(frozen=True, slots=True)
class FieldStatistics:
    spacing_cv: float
    minimum_gradient: float
    maximum_gradient: float


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    seed: int | None
    initial_loss: float
    final_loss: float
    uniformity_loss: float
    width_loss: float
    history: FloatArray
    parameter_history: FloatArray
    parameters: FloatArray
    knots: FloatArray
    gaps: FloatArray
    field: FloatArray
    statistics: FieldStatistics
    iterations: int
    evaluations: int
    gradient_norm: float
    success: bool
    message: str


def load_obj(path: str | Path) -> Mesh:
    """Load OBJ positions and polygon faces, triangulating polygons by a fan."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    source = Path(path)
    for line in source.read_text(encoding="utf-8", errors="strict").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if fields[0] == "v":
            if len(fields) < 4:
                raise ValueError(f"invalid vertex in {source}")
            vertices.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "f":
            if len(fields) < 4:
                raise ValueError(f"face has fewer than three vertices in {source}")
            polygon: list[int] = []
            for token in fields[1:]:
                raw = int(token.split("/", 1)[0])
                index = raw - 1 if raw > 0 else len(vertices) + raw
                polygon.append(index)
            faces.extend(
                (polygon[0], polygon[index], polygon[index + 1])
                for index in range(1, len(polygon) - 1)
            )
    return Mesh(
        np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    )


def boundary_loop(faces: IntArray) -> IntArray:
    """Return the sole manifold boundary loop, in cyclic vertex order."""
    counts: Counter[tuple[int, int]] = Counter()
    for triangle in np.asarray(faces, dtype=np.int64):
        for start, end in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            counts[tuple(sorted((start, end)))] += 1
    if any(count > 2 for count in counts.values()):
        raise ValueError("mesh is not edge-manifold")
    boundary_edges = [edge for edge, count in counts.items() if count == 1]
    if not boundary_edges:
        raise ValueError("mesh has no boundary")

    adjacency: defaultdict[int, list[int]] = defaultdict(list)
    for start, end in boundary_edges:
        adjacency[start].append(end)
        adjacency[end].append(start)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("mesh boundary is not a collection of manifold loops")

    start = min(adjacency)
    loop = [start]
    previous = -1
    current = start
    while True:
        first, second = adjacency[current]
        following = first if first != previous else second
        if following == start:
            break
        if following in loop:
            raise ValueError("boundary contains a repeated vertex before closing")
        loop.append(following)
        previous, current = current, following
    if len(loop) != len(adjacency):
        raise ValueError("optimizer requires exactly one boundary loop")
    return np.asarray(loop, dtype=np.int64)


def boundary_arclength(vertices: FloatArray, loop: IntArray) -> FloatArray:
    """Normalized cumulative arc length at each ordered boundary vertex."""
    following = np.roll(loop, -1)
    lengths = np.linalg.norm(vertices[following] - vertices[loop], axis=1)
    perimeter = float(lengths.sum())
    if not np.isfinite(perimeter) or perimeter <= 0.0 or np.any(lengths <= 0.0):
        raise ValueError("boundary loop contains a zero or invalid edge")
    return np.concatenate(([0.0], np.cumsum(lengths[:-1]))) / perimeter


def cyclic_boundary_profile(
    positions: FloatArray, knots: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Evaluate the cyclic 0/linear-rise/1/linear-fall profile and Jacobian.

    ``knots`` are unwrapped and must satisfy ``k0 < k1 < k2 < k3 < k0 + 1``.
    The returned Jacobian has shape ``(len(positions), 4)``.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1)
    knots = np.asarray(knots, dtype=np.float64).reshape(-1)
    if knots.shape != (4,) or not np.isfinite(knots).all():
        raise ValueError("knots must contain four finite values")
    if not np.isfinite(positions).all():
        raise ValueError("positions contain NaN or infinite values")
    if np.any(np.diff(knots) <= 0.0) or knots[3] >= knots[0] + 1.0:
        raise ValueError("knots must satisfy k0 < k1 < k2 < k3 < k0 + 1")

    local = np.mod(positions - knots[0], 1.0)
    first, second, third = knots[1:] - knots[0]
    values = np.zeros_like(local)
    jacobian = np.zeros((len(local), 4), dtype=np.float64)

    rising = (local > first) & (local < second)
    rise_width = knots[2] - knots[1]
    z = (local[rising] - first) / rise_width
    values[rising] = z
    jacobian[rising, 1] = -(1.0 - z) / rise_width
    jacobian[rising, 2] = -z / rise_width

    values[(local >= second) & (local <= third)] = 1.0

    falling = local > third
    fall_width = knots[0] + 1.0 - knots[3]
    z = (local[falling] - third) / fall_width
    values[falling] = 1.0 - z
    jacobian[falling, 0] = z / fall_width
    jacobian[falling, 3] = (1.0 - z) / fall_width
    return values, jacobian


def knots_from_parameters(
    parameters: FloatArray, minimum_gap: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Map one origin and three gauge-fixed logits to four ordered knots."""
    parameters = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if parameters.shape != (4,) or not np.isfinite(parameters).all():
        raise ValueError("parameters must contain four finite values")
    if not np.isfinite(minimum_gap) or not 0.0 <= minimum_gap < 0.25:
        raise ValueError("minimum_gap must lie in [0, 0.25)")

    logits = np.concatenate((parameters[1:], [0.0]))
    exponentials = np.exp(logits - logits.max())
    probabilities = exponentials / exponentials.sum()
    scale = 1.0 - 4.0 * minimum_gap
    gaps = minimum_gap + scale * probabilities
    knots = parameters[0] + np.concatenate(([0.0], np.cumsum(gaps[:3])))

    softmax_jacobian = np.diag(probabilities) - np.outer(probabilities, probabilities)
    gap_jacobian = scale * softmax_jacobian[:, :3]
    jacobian = np.zeros((4, 4), dtype=np.float64)
    jacobian[:, 0] = 1.0
    for index in range(1, 4):
        jacobian[index, 1:] = gap_jacobian[:index].sum(axis=0)
    return knots, jacobian, gaps


def parameters_from_knots(knots: FloatArray, minimum_gap: float) -> FloatArray:
    """Invert :func:`knots_from_parameters` after fixing the softmax gauge."""
    knots = np.asarray(knots, dtype=np.float64).reshape(-1)
    if knots.shape != (4,) or not np.isfinite(knots).all():
        raise ValueError("knots must contain four finite values")
    if not np.isfinite(minimum_gap) or not 0.0 <= minimum_gap < 0.25:
        raise ValueError("minimum_gap must lie in [0, 0.25)")
    gaps = np.append(np.diff(knots), knots[0] + 1.0 - knots[3])
    probabilities = (gaps - minimum_gap) / (1.0 - 4.0 * minimum_gap)
    if np.any(probabilities <= 0.0):
        raise ValueError("each cyclic knot gap must exceed minimum_gap")
    return np.concatenate(([knots[0]], np.log(probabilities[:3] / probabilities[3])))


def random_knots(seed: int, minimum_gap: float = 0.03) -> FloatArray:
    """Draw an ordered cyclic four-knot initialization for one random seed."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not 0.0 <= minimum_gap < 0.25:
        raise ValueError("minimum_gap must lie in [0, 0.25)")
    rng = np.random.default_rng(seed)
    gaps = minimum_gap + (1.0 - 4.0 * minimum_gap) * rng.dirichlet(np.ones(4))
    origin = float(rng.uniform())
    return origin + np.concatenate(([0.0], np.cumsum(gaps[:3])))


def cotangent_stiffness(mesh: Mesh) -> scipy.sparse.csr_matrix:
    """Positive-semidefinite cotangent stiffness matrix."""
    triangles = mesh.vertices[mesh.faces]
    edge01 = triangles[:, 1] - triangles[:, 0]
    edge02 = triangles[:, 2] - triangles[:, 0]
    double_area = np.linalg.norm(np.cross(edge01, edge02), axis=1)
    scale = np.linalg.norm(edge01, axis=1) * np.linalg.norm(edge02, axis=1)
    if np.any(double_area <= np.finfo(np.float64).eps * scale * 16.0):
        raise ValueError("mesh contains a degenerate triangle")

    cot0 = np.einsum("ij,ij->i", edge01, edge02) / double_area
    edge10 = triangles[:, 0] - triangles[:, 1]
    edge12 = triangles[:, 2] - triangles[:, 1]
    cot1 = np.einsum("ij,ij->i", edge10, edge12) / double_area
    edge20 = triangles[:, 0] - triangles[:, 2]
    edge21 = triangles[:, 1] - triangles[:, 2]
    cot2 = np.einsum("ij,ij->i", edge20, edge21) / double_area

    starts = np.concatenate((mesh.faces[:, 1], mesh.faces[:, 2], mesh.faces[:, 0]))
    ends = np.concatenate((mesh.faces[:, 2], mesh.faces[:, 0], mesh.faces[:, 1]))
    weights = 0.5 * np.concatenate((cot0, cot1, cot2))
    rows = np.concatenate((starts, starts, ends, ends))
    columns = np.concatenate((starts, ends, starts, ends))
    data = np.concatenate((weights, -weights, -weights, weights))
    matrix = scipy.sparse.coo_matrix(
        (data, (rows, columns)), shape=(len(mesh.vertices), len(mesh.vertices))
    )
    return matrix.tocsr()


def face_gradient_basis(mesh: Mesh) -> tuple[FloatArray, FloatArray]:
    """Return face areas and gradients of each triangle's barycentric basis."""
    triangles = mesh.vertices[mesh.faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    double_area = np.linalg.norm(cross, axis=1)
    scale = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1) * np.linalg.norm(
        triangles[:, 2] - triangles[:, 0], axis=1
    )
    if np.any(double_area <= np.finfo(np.float64).eps * scale * 16.0):
        raise ValueError("mesh contains a degenerate triangle")
    normals = cross / double_area[:, None]
    basis = (
        np.stack(
            (
                np.cross(normals, triangles[:, 2] - triangles[:, 1]),
                np.cross(normals, triangles[:, 0] - triangles[:, 2]),
                np.cross(normals, triangles[:, 1] - triangles[:, 0]),
            ),
            axis=1,
        )
        / double_area[:, None, None]
    )
    return 0.5 * double_area, basis


class HarmonicBoundaryOptimizer:
    """Optimize two constant arcs on a mesh with one manifold boundary loop."""

    def __init__(
        self,
        mesh: Mesh,
        *,
        minimum_gap: float = 0.03,
        target_arc_width: float | None = None,
        width_weight: float = 0.0,
    ) -> None:
        if not np.isfinite(minimum_gap) or not 0.0 < minimum_gap < 0.25:
            raise ValueError("minimum_gap must lie in (0, 0.25)")
        self.mesh = mesh
        self.minimum_gap = float(minimum_gap)
        if target_arc_width is not None and (
            not np.isfinite(target_arc_width)
            or not minimum_gap < target_arc_width < 0.5 - minimum_gap
        ):
            raise ValueError(
                "target_arc_width must lie between minimum_gap and 0.5 - minimum_gap"
            )
        if not np.isfinite(width_weight) or width_weight < 0.0:
            raise ValueError("width_weight must be finite and non-negative")
        if width_weight > 0.0 and target_arc_width is None:
            raise ValueError("positive width_weight requires target_arc_width")
        self.target_arc_width = (
            None if target_arc_width is None else float(target_arc_width)
        )
        self.width_weight = float(width_weight)
        self.boundary_vertices = boundary_loop(mesh.faces)
        self.boundary_positions = boundary_arclength(
            mesh.vertices, self.boundary_vertices
        )
        interior_mask = np.ones(len(mesh.vertices), dtype=bool)
        interior_mask[self.boundary_vertices] = False
        self.interior_vertices = np.flatnonzero(interior_mask).astype(np.int64)
        if len(self.interior_vertices) == 0:
            raise ValueError("mesh has no interior vertices")

        stiffness = cotangent_stiffness(mesh)
        self._interior_block = stiffness[self.interior_vertices][
            :, self.interior_vertices
        ].tocsc()
        self._coupling_block = stiffness[self.interior_vertices][
            :, self.boundary_vertices
        ].tocsr()
        try:
            self._factorization = scipy.sparse.linalg.splu(self._interior_block)
        except RuntimeError as exc:
            raise ValueError(
                "mesh interior is not connected to the selected boundary loop"
            ) from exc

        self.face_areas, self._gradient_basis = face_gradient_basis(mesh)
        self._face_weights = self.face_areas / self.face_areas.sum()

    def extend(self, boundary_values: FloatArray) -> FloatArray:
        """Harmonically extend one value per ordered boundary vertex."""
        boundary_values = np.asarray(boundary_values, dtype=np.float64).reshape(-1)
        if boundary_values.shape != self.boundary_vertices.shape:
            raise ValueError(
                "boundary_values must contain one value per boundary vertex"
            )
        field = np.empty(len(self.mesh.vertices), dtype=np.float64)
        field[self.boundary_vertices] = boundary_values
        right_hand_side = -(self._coupling_block @ boundary_values)
        field[self.interior_vertices] = self._factorization.solve(right_hand_side)
        return field

    def extend_adjoint(self, field_sensitivity: FloatArray) -> FloatArray:
        """Apply the exact transpose of :meth:`extend`."""
        field_sensitivity = np.asarray(field_sensitivity, dtype=np.float64).reshape(-1)
        if field_sensitivity.shape != (len(self.mesh.vertices),):
            raise ValueError("field_sensitivity must contain one value per vertex")
        interior_adjoint = self._factorization.solve(
            field_sensitivity[self.interior_vertices], trans="T"
        )
        return np.asarray(
            field_sensitivity[self.boundary_vertices]
            - self._coupling_block.T @ interior_adjoint,
            dtype=np.float64,
        )

    def _loss_and_field_gradient(
        self, field: FloatArray
    ) -> tuple[float, FloatArray, FieldStatistics]:
        face_values = field[self.mesh.faces]
        gradients = np.einsum("fij,fi->fj", self._gradient_basis, face_values)
        squared_norms = np.einsum("ij,ij->i", gradients, gradients)
        mean_squared = float(self._face_weights @ squared_norms)
        if not np.isfinite(mean_squared) or mean_squared <= 0.0:
            raise ValueError("harmonic field has zero or invalid gradient energy")
        second_moment = float(self._face_weights @ squared_norms**2)
        loss = second_moment / mean_squared**2 - 1.0

        coefficients = (
            4.0
            * self._face_weights
            * (squared_norms - second_moment / mean_squared)
            / mean_squared**2
        )
        face_sensitivity = coefficients[:, None] * gradients
        corner_sensitivity = np.einsum(
            "fij,fj->fi", self._gradient_basis, face_sensitivity
        )
        field_sensitivity = np.bincount(
            self.mesh.faces.reshape(-1),
            weights=corner_sensitivity.reshape(-1),
            minlength=len(self.mesh.vertices),
        ).astype(np.float64)

        norms = np.sqrt(squared_norms)
        mean_norm = float(self._face_weights @ norms)
        variance = float(self._face_weights @ (norms - mean_norm) ** 2)
        statistics = FieldStatistics(
            spacing_cv=float(np.sqrt(max(variance, 0.0)) / max(mean_norm, 1.0e-15)),
            minimum_gradient=float(norms.min()),
            maximum_gradient=float(norms.max()),
        )
        return loss, field_sensitivity, statistics

    def _width_loss_and_gradient(
        self, gaps: FloatArray, knot_jacobian: FloatArray
    ) -> tuple[float, FloatArray]:
        if self.width_weight == 0.0 or self.target_arc_width is None:
            return 0.0, np.zeros(4, dtype=np.float64)
        target = self.target_arc_width
        plateau_indices = np.asarray([0, 2])
        residual = (gaps[plateau_indices] - target) / target
        loss = self.width_weight * float(residual @ residual)
        gap_gradient = np.zeros(4, dtype=np.float64)
        gap_gradient[plateau_indices] = 2.0 * self.width_weight * residual / target
        gap_jacobian = np.vstack(
            (
                knot_jacobian[1] - knot_jacobian[0],
                knot_jacobian[2] - knot_jacobian[1],
                knot_jacobian[3] - knot_jacobian[2],
                knot_jacobian[0] - knot_jacobian[3],
            )
        )
        return loss, gap_jacobian.T @ gap_gradient

    def loss_and_gradient(self, parameters: FloatArray) -> tuple[float, FloatArray]:
        """Return the scale-invariant field loss and exact four-vector gradient."""
        knots, knot_jacobian, gaps = knots_from_parameters(parameters, self.minimum_gap)
        boundary_values, profile_jacobian = cyclic_boundary_profile(
            self.boundary_positions, knots
        )
        field = self.extend(boundary_values)
        uniformity_loss, field_sensitivity, _ = self._loss_and_field_gradient(field)
        boundary_sensitivity = self.extend_adjoint(field_sensitivity)
        knot_gradient = profile_jacobian.T @ boundary_sensitivity
        width_loss, width_gradient = self._width_loss_and_gradient(gaps, knot_jacobian)
        return (
            uniformity_loss + width_loss,
            knot_jacobian.T @ knot_gradient + width_gradient,
        )

    def optimize(
        self,
        initial_knots: FloatArray,
        *,
        max_iterations: int = 60,
        seed: int | None = None,
    ) -> OptimizationResult:
        """Run L-BFGS from one ordered cyclic four-knot initialization."""
        if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
            raise TypeError("max_iterations must be an integer")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        initial_parameters = parameters_from_knots(initial_knots, self.minimum_gap)
        initial_loss, _ = self.loss_and_gradient(initial_parameters)
        history = [float(initial_loss)]
        parameter_history = [initial_parameters.copy()]

        def callback(intermediate_result: scipy.optimize.OptimizeResult) -> None:
            parameter_history.append(
                np.asarray(intermediate_result.x, dtype=np.float64).copy()
            )
            history.append(float(intermediate_result.fun))

        result = scipy.optimize.minimize(
            self.loss_and_gradient,
            initial_parameters,
            jac=True,
            method="L-BFGS-B",
            callback=callback,
            options={
                "maxiter": max_iterations,
                "ftol": 1.0e-12,
                "gtol": 1.0e-8,
                "maxls": 30,
            },
        )

        parameters = np.asarray(result.x, dtype=np.float64)
        knots, knot_jacobian, gaps = knots_from_parameters(parameters, self.minimum_gap)
        boundary_values, _ = cyclic_boundary_profile(self.boundary_positions, knots)
        field = self.extend(boundary_values)
        uniformity_loss, _, statistics = self._loss_and_field_gradient(field)
        width_loss, _ = self._width_loss_and_gradient(gaps, knot_jacobian)
        final_loss = uniformity_loss + width_loss
        if np.array_equal(parameter_history[-1], parameters):
            history[-1] = float(final_loss)
        else:
            parameter_history.append(parameters.copy())
            history.append(float(final_loss))
        return OptimizationResult(
            seed=seed,
            initial_loss=float(initial_loss),
            final_loss=float(final_loss),
            uniformity_loss=float(uniformity_loss),
            width_loss=float(width_loss),
            history=np.asarray(history, dtype=np.float64),
            parameter_history=np.vstack(parameter_history),
            parameters=parameters,
            knots=knots,
            gaps=gaps,
            field=field,
            statistics=statistics,
            iterations=int(result.nit),
            evaluations=int(result.nfev),
            gradient_norm=float(np.linalg.norm(result.jac)),
            success=bool(result.success),
            message=str(result.message),
        )
