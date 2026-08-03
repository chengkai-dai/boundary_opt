"""Differentiable four-endpoint harmonic boundary optimization.

Four design variables define two ordered arcs on one boundary loop.  The first
arc targets field value zero and the second targets one; the rest of the
boundary has the natural Neumann condition.  Moving arcs are represented by
exact P1 boundary-mass integrals, whose endpoint derivatives remain continuous
as an endpoint crosses a mesh vertex.  A precomputed interior Schur reduction,
one boundary solve, one harmonic lift, and one adjoint backsolve provide the
exact gradient.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Mesh:
    vertices: FloatArray
    faces: IntArray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices)
        faces = np.asarray(self.faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (V, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if not np.issubdtype(faces.dtype, np.integer):
            raise TypeError("faces must contain integer vertex indices")

        vertices = np.array(vertices, dtype=np.float64, order="C", copy=True)
        faces = np.array(faces, dtype=np.int64, order="C", copy=True)
        if not np.isfinite(vertices).all():
            raise ValueError("vertices contain NaN or infinite values")
        if len(vertices) == 0 or len(faces) == 0:
            raise ValueError("mesh must contain vertices and faces")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("faces reference a vertex outside the mesh")
        vertices.setflags(write=False)
        faces.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)


@dataclass(frozen=True, slots=True)
class FieldStatistics:
    spacing_cv: float
    minimum_gradient: float
    maximum_gradient: float


@dataclass(frozen=True, slots=True)
class OptimizationResult:
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
                if raw == 0:
                    raise ValueError(f"OBJ vertex indices are one-based in {source}")
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
    visited = {start}
    previous = -1
    current = start
    while True:
        first, second = adjacency[current]
        following = first if first != previous else second
        if following == start:
            break
        if following in visited:
            raise ValueError("boundary contains a repeated vertex before closing")
        loop.append(following)
        visited.add(following)
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


def _validated_knots(knots: FloatArray) -> FloatArray:
    knots = np.asarray(knots, dtype=np.float64).reshape(-1)
    if knots.shape != (4,) or not np.isfinite(knots).all():
        raise ValueError("knots must contain four finite values")
    if np.any(np.diff(knots) <= 0.0) or knots[3] >= knots[0] + 1.0:
        raise ValueError("knots must satisfy k0 < k1 < k2 < k3 < k0 + 1")
    return knots


def _validated_minimum_gap(minimum_gap: float) -> float:
    if not np.isfinite(minimum_gap) or not 0.0 < minimum_gap < 0.25:
        raise ValueError("minimum_gap must lie in (0, 0.25)")
    return float(minimum_gap)


def _validated_boundary_positions(positions: FloatArray) -> FloatArray:
    positions = np.asarray(positions, dtype=np.float64).reshape(-1)
    if len(positions) < 3 or not np.isfinite(positions).all():
        raise ValueError("positions must contain at least three finite values")
    if abs(float(positions[0])) > 1.0e-12:
        raise ValueError("the first boundary position must be zero")
    if np.any(np.diff(positions) <= 0.0) or positions[-1] >= 1.0:
        raise ValueError("boundary positions must increase within [0, 1)")
    return positions


def _arc_edge_coordinates(
    positions: FloatArray,
    edge_lengths: FloatArray,
    start: float,
    end: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return local coordinates of one cyclic arc on every boundary edge."""
    origin = start % 1.0
    stop = origin + end - start
    local_start = np.clip((origin - positions) / edge_lengths, 0.0, 1.0)
    local_end = np.clip((min(stop, 1.0) - positions) / edge_lengths, 0.0, 1.0)
    wrapped_end = np.clip((max(stop - 1.0, 0.0) - positions) / edge_lengths, 0.0, 1.0)
    return local_start, local_end, wrapped_end


def cyclic_arc_edge_weights(
    positions: FloatArray, knots: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return exact fractional coverage of each boundary edge by both arcs.

    The first array describes the zero-target arc ``[k0, k1]`` and the second
    the one-target arc ``[k2, k3]``.  These weights are primarily intended for
    visualization; the PDE uses exact consistent edge-mass integration.
    """
    positions = _validated_boundary_positions(positions)
    knots = _validated_knots(knots)
    edge_lengths = np.diff(np.append(positions, 1.0))
    weights = []
    for start, end in ((knots[0], knots[1]), (knots[2], knots[3])):
        local_start, local_end, wrapped_end = _arc_edge_coordinates(
            positions, edge_lengths, float(start), float(end)
        )
        weights.append(np.clip(local_end - local_start + wrapped_end, 0.0, 1.0))
    return weights[0], weights[1]


def knots_from_parameters(
    parameters: FloatArray, minimum_gap: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Map one origin and three gauge-fixed logits to four ordered knots."""
    parameters = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if parameters.shape != (4,) or not np.isfinite(parameters).all():
        raise ValueError("parameters must contain four finite values")
    minimum_gap = _validated_minimum_gap(minimum_gap)

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
    knots = _validated_knots(knots)
    minimum_gap = _validated_minimum_gap(minimum_gap)
    gaps = np.append(np.diff(knots), knots[0] + 1.0 - knots[3])
    probabilities = (gaps - minimum_gap) / (1.0 - 4.0 * minimum_gap)
    if np.any(probabilities <= 0.0):
        raise ValueError("each cyclic knot gap must exceed minimum_gap")
    return np.concatenate(([knots[0]], np.log(probabilities[:3] / probabilities[3])))


def random_knots(seed: int, minimum_gap: float = 0.03) -> FloatArray:
    """Draw an ordered cyclic four-knot initialization for one random seed."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    minimum_gap = _validated_minimum_gap(minimum_gap)
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
    """Optimize zero- and one-target arcs on one manifold boundary loop."""

    def __init__(
        self,
        mesh: Mesh,
        *,
        minimum_gap: float = 0.03,
        target_arc_width: float | None = None,
        width_weight: float = 0.0,
        boundary_penalty: float = 100.0,
    ) -> None:
        minimum_gap = _validated_minimum_gap(minimum_gap)
        self.mesh = mesh
        self.minimum_gap = minimum_gap
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
        if not np.isfinite(boundary_penalty) or boundary_penalty <= 0.0:
            raise ValueError("boundary_penalty must be finite and positive")
        self.target_arc_width = (
            None if target_arc_width is None else float(target_arc_width)
        )
        self.width_weight = float(width_weight)
        self.boundary_penalty = float(boundary_penalty)
        self.boundary_vertices = boundary_loop(mesh.faces)
        self.boundary_positions = boundary_arclength(
            mesh.vertices, self.boundary_vertices
        )
        self._boundary_edge_lengths = np.diff(np.append(self.boundary_positions, 1.0))
        stiffness = cotangent_stiffness(mesh).tocsc()
        component_count = scipy.sparse.csgraph.connected_components(
            stiffness, directed=False, return_labels=False
        )
        if component_count != 1:
            raise ValueError("mesh must be connected")

        interior_mask = np.ones(len(mesh.vertices), dtype=bool)
        interior_mask[self.boundary_vertices] = False
        self.interior_vertices = np.flatnonzero(interior_mask).astype(np.int64)

        # E maps boundary values to the complete harmonic field: u = E @ u_B.
        harmonic_lift = np.zeros(
            (len(mesh.vertices), len(self.boundary_vertices)),
            dtype=np.float64,
        )
        harmonic_lift[self.boundary_vertices] = np.eye(len(self.boundary_vertices))
        boundary_block = stiffness[self.boundary_vertices][
            :, self.boundary_vertices
        ].toarray()
        if len(self.interior_vertices):
            interior_block = stiffness[self.interior_vertices][
                :, self.interior_vertices
            ].tocsc()
            coupling = stiffness[self.interior_vertices][
                :, self.boundary_vertices
            ].toarray()
            try:
                interior_factorization = scipy.sparse.linalg.splu(interior_block)
            except RuntimeError as exc:
                raise ValueError("mesh interior harmonic block is singular") from exc
            harmonic_extension = -interior_factorization.solve(coupling)
            harmonic_lift[self.interior_vertices] = harmonic_extension
            boundary_block += coupling.T @ harmonic_extension
        self._harmonic_lift = harmonic_lift

        # ponytail: dense Schur assumes B << V; add a sparse fallback only if
        # thin meshes with boundary-scale B become a measured workload.
        self._boundary_schur = np.asarray(
            0.5 * (boundary_block + boundary_block.T), dtype=np.float64
        )

        self.face_areas, self._gradient_basis = face_gradient_basis(mesh)
        self._face_weights = self.face_areas / self.face_areas.sum()

        for array in (
            self.boundary_vertices,
            self.boundary_positions,
            self.interior_vertices,
            self.face_areas,
            self._boundary_edge_lengths,
            self._harmonic_lift,
            self._boundary_schur,
            self._gradient_basis,
            self._face_weights,
        ):
            array.setflags(write=False)

    @staticmethod
    def _mass_antiderivative(value: FloatArray) -> FloatArray:
        """Antiderivatives of ``(1-t)^2``, ``(1-t)t``, and ``t^2``."""
        return np.stack(
            (
                value - value**2 + value**3 / 3.0,
                value**2 / 2.0 - value**3 / 3.0,
                value**3 / 3.0,
            ),
            axis=-1,
        )

    def _arc_mass(self, start: float, end: float) -> FloatArray:
        """Exactly integrate the boundary P1 mass over one arc."""
        matrix = np.zeros(
            (len(self.boundary_vertices), len(self.boundary_vertices)),
            dtype=np.float64,
        )
        local_start, local_end, wrapped_end = _arc_edge_coordinates(
            self.boundary_positions, self._boundary_edge_lengths, start, end
        )
        integral = self._boundary_edge_lengths[:, None] * (
            self._mass_antiderivative(local_end)
            - self._mass_antiderivative(local_start)
            + self._mass_antiderivative(wrapped_end)
        )
        edge = np.arange(len(self.boundary_vertices))
        following = (edge + 1) % len(self.boundary_vertices)
        matrix[edge, edge] += integral[:, 0]
        matrix[edge, following] += integral[:, 1]
        matrix[following, edge] += integral[:, 1]
        matrix[following, following] += integral[:, 2]
        return matrix

    def _system_from_knots(self, knots: FloatArray) -> tuple[FloatArray, FloatArray]:
        knots = _validated_knots(knots)
        zero_mass = self._arc_mass(float(knots[0]), float(knots[1]))
        one_mass = self._arc_mass(float(knots[2]), float(knots[3]))
        matrix = self._boundary_schur + self.boundary_penalty * (zero_mass + one_mass)
        right_hand_side = self.boundary_penalty * one_mass.sum(axis=1)
        return matrix, right_hand_side

    def _solve(self, knots: FloatArray) -> tuple[FloatArray, tuple[FloatArray, bool]]:
        matrix, right_hand_side = self._system_from_knots(knots)
        try:
            factorization = scipy.linalg.cho_factor(
                matrix, lower=True, check_finite=False
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "reduced harmonic system is not positive definite"
            ) from exc
        boundary_field = scipy.linalg.cho_solve(
            factorization, right_hand_side, check_finite=False
        )
        field = self._harmonic_lift @ boundary_field
        return field, factorization

    def field_from_knots(self, knots: FloatArray) -> FloatArray:
        """Solve the partial-boundary harmonic field for four ordered knots."""
        field, _ = self._solve(knots)
        return field

    def field_and_arc_weights(
        self, knots: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return the field and per-edge zero/one arc coverage for display."""
        field = self.field_from_knots(knots)
        zero_weights, one_weights = cyclic_arc_edge_weights(
            self.boundary_positions, knots
        )
        return field, zero_weights, one_weights

    def _face_gradients(self, field: FloatArray) -> FloatArray:
        face_values = field[self.mesh.faces]
        return np.einsum("fij,fi->fj", self._gradient_basis, face_values)

    def _uniformity_loss_and_gradient(
        self, field: FloatArray
    ) -> tuple[float, FloatArray]:
        gradients = self._face_gradients(field)
        squared_norms = np.einsum("ij,ij->i", gradients, gradients)
        mean_squared = float(self._face_weights @ squared_norms)
        if not np.isfinite(mean_squared) or mean_squared <= 0.0:
            raise ValueError("harmonic field has zero or invalid gradient energy")
        normalized_squared_norms = squared_norms / mean_squared
        loss = float(self._face_weights @ (normalized_squared_norms - 1.0) ** 2)
        normalized_second_moment = float(
            self._face_weights @ normalized_squared_norms**2
        )

        coefficients = (
            4.0
            * self._face_weights
            * (normalized_squared_norms - normalized_second_moment)
            / mean_squared
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
        return loss, field_sensitivity

    def _field_statistics(self, field: FloatArray) -> FieldStatistics:
        norms = np.linalg.norm(self._face_gradients(field), axis=1)
        mean_norm = float(self._face_weights @ norms)
        variance = float(self._face_weights @ (norms - mean_norm) ** 2)
        return FieldStatistics(
            spacing_cv=float(np.sqrt(max(variance, 0.0)) / max(mean_norm, 1.0e-15)),
            minimum_gradient=float(norms.min()),
            maximum_gradient=float(norms.max()),
        )

    def _width_loss_and_knot_gradient(
        self, gaps: FloatArray
    ) -> tuple[float, FloatArray]:
        if self.width_weight == 0.0 or self.target_arc_width is None:
            return 0.0, np.zeros(4, dtype=np.float64)
        target = self.target_arc_width
        residual = (gaps[[0, 2]] - target) / target
        loss = self.width_weight * float(residual @ residual)
        gradient = (2.0 * self.width_weight / target) * np.asarray(
            (-residual[0], residual[0], -residual[1], residual[1])
        )
        return loss, gradient

    def loss_and_gradient(self, parameters: FloatArray) -> tuple[float, FloatArray]:
        """Return the scale-invariant field loss and exact four-vector gradient."""
        knots, knot_jacobian, gaps = knots_from_parameters(parameters, self.minimum_gap)
        field, factorization = self._solve(knots)
        uniformity_loss, d_loss_d_field = self._uniformity_loss_and_gradient(field)
        boundary_field = field[self.boundary_vertices]
        d_loss_d_boundary_field = self._harmonic_lift.T @ d_loss_d_field
        adjoint = scipy.linalg.cho_solve(
            factorization, d_loss_d_boundary_field, check_finite=False
        )
        field_at_knots = np.interp(
            knots % 1.0, self.boundary_positions, boundary_field, period=1.0
        )
        adjoint_at_knots = np.interp(
            knots % 1.0, self.boundary_positions, adjoint, period=1.0
        )
        d_loss_d_knots = self.boundary_penalty * np.asarray(
            (
                adjoint_at_knots[0] * field_at_knots[0],
                -adjoint_at_knots[1] * field_at_knots[1],
                -adjoint_at_knots[2] * (1.0 - field_at_knots[2]),
                adjoint_at_knots[3] * (1.0 - field_at_knots[3]),
            )
        )
        width_loss, width_knot_gradient = self._width_loss_and_knot_gradient(gaps)
        return (
            uniformity_loss + width_loss,
            knot_jacobian.T @ (d_loss_d_knots + width_knot_gradient),
        )

    def optimize(
        self,
        initial_knots: FloatArray,
        *,
        max_iterations: int = 100,
    ) -> OptimizationResult:
        """Run L-BFGS from one ordered cyclic four-knot initialization."""
        if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
            raise TypeError("max_iterations must be an integer")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        initial_parameters = parameters_from_knots(initial_knots, self.minimum_gap)
        history: list[float] = []
        parameter_history: list[FloatArray] = []

        def objective(parameters: FloatArray) -> tuple[float, FloatArray]:
            loss, gradient = self.loss_and_gradient(parameters)
            if not history:
                history.append(float(loss))
                parameter_history.append(np.asarray(parameters).copy())
            return loss, gradient

        def callback(intermediate_result: scipy.optimize.OptimizeResult) -> None:
            parameter_history.append(
                np.asarray(intermediate_result.x, dtype=np.float64).copy()
            )
            history.append(float(intermediate_result.fun))

        result = scipy.optimize.minimize(
            objective,
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
        knots, _, gaps = knots_from_parameters(parameters, self.minimum_gap)
        field = self.field_from_knots(knots)
        uniformity_loss, _ = self._uniformity_loss_and_gradient(field)
        width_loss, _ = self._width_loss_and_knot_gradient(gaps)
        statistics = self._field_statistics(field)
        final_loss = uniformity_loss + width_loss
        if np.array_equal(parameter_history[-1], parameters):
            history[-1] = float(final_loss)
        else:
            parameter_history.append(parameters.copy())
            history.append(float(final_loss))
        return OptimizationResult(
            initial_loss=history[0],
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
            iterations=len(history) - 1,
            evaluations=int(result.nfev),
            gradient_norm=float(np.linalg.norm(result.jac)),
            success=bool(result.success),
            message=str(result.message),
        )
