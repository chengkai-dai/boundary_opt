"""Differentiable four-endpoint harmonic boundary optimization.

Four ordered endpoints define two arcs on one boundary loop.  The first
arc targets field value zero and the second targets one; the rest of the
boundary has the natural Neumann condition.  Moving arcs are represented by
exact P1 boundary-mass integrals, whose endpoint derivatives remain continuous
as an endpoint crosses a mesh vertex.  A precomputed interior Schur reduction,
one boundary solve, one harmonic lift, and one adjoint backsolve provide the
exact gradient.  The latent Robin solution is affinely calibrated so the two
target-arc means of the public field are exactly zero and one.  Direct gap
variables and a linear simplex constraint keep the four endpoints ordered.
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
    gradient_cv: float
    spacing_cv: float
    minimum_gradient: float
    maximum_gradient: float


@dataclass(frozen=True, slots=True)
class BoundaryStatistics:
    raw_zero_mean: float
    raw_one_mean: float
    raw_span: float
    raw_zero_target_rms: float
    raw_one_target_rms: float
    canonical_zero_target_rms: float
    canonical_one_target_rms: float


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
    raw_field: FloatArray
    field: FloatArray
    statistics: FieldStatistics
    boundary_statistics: BoundaryStatistics
    iterations: int
    evaluations: int
    constraint_violation: float
    kkt_residual: float
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
    parameters: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Map one origin and four positive gap coordinates to four knots.

    SLSQP may evaluate away from the sum-to-one constraint, so the gaps are
    normalized here.  On the feasible simplex this is exactly the identity.
    """
    parameters = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if parameters.shape != (5,) or not np.isfinite(parameters).all():
        raise ValueError("parameters must contain one origin and four finite gaps")
    raw_gaps = parameters[1:]
    if np.any(raw_gaps <= 0.0):
        raise ValueError("direct cyclic gaps must be positive")
    total = float(raw_gaps.sum())
    gaps = raw_gaps / total
    knots = parameters[0] + np.concatenate(([0.0], np.cumsum(gaps[:3])))
    normalization_jacobian = (
        np.eye(4, dtype=np.float64) * total - np.outer(raw_gaps, np.ones(4))
    ) / total**2
    cumulative_jacobian = np.tril(np.ones((4, 4), dtype=np.float64), k=-1)
    jacobian = np.empty((4, 5), dtype=np.float64)
    jacobian[:, 0] = 1.0
    jacobian[:, 1:] = cumulative_jacobian @ normalization_jacobian
    return knots, jacobian, gaps


def parameters_from_knots(knots: FloatArray) -> FloatArray:
    """Return one origin and all four direct cyclic gaps."""
    knots = _validated_knots(knots)
    gaps = np.append(np.diff(knots), knots[0] + 1.0 - knots[3])
    return np.concatenate(([knots[0]], gaps))


def _gap_kkt_residual(
    gradient: FloatArray, gaps: FloatArray, minimum_gap: float
) -> float:
    """Return the infinity-norm KKT residual on the cyclic gap simplex."""
    gradient = np.asarray(gradient, dtype=np.float64)
    gaps = np.asarray(gaps, dtype=np.float64)
    gap_gradient = gradient[1:]
    tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * max(1.0, abs(minimum_gap), float(np.abs(gaps).max()))
    )
    active = gaps <= minimum_gap + tolerance
    free = ~active
    if not np.any(free):
        return np.inf
    multiplier = float(gap_gradient[free].mean())
    residuals = [abs(float(gradient[0]))]
    residuals.extend(np.abs(gap_gradient[free] - multiplier))
    residuals.extend(np.maximum(multiplier - gap_gradient[active], 0.0))
    return float(max(residuals))


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
            or not minimum_gap <= target_arc_width <= 0.5 - minimum_gap
        ):
            raise ValueError(
                "target_arc_width must lie in [minimum_gap, 0.5 - minimum_gap]"
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

    def robin_field_from_knots(self, knots: FloatArray) -> FloatArray:
        """Solve the latent Robin field for four ordered knots."""
        field, _ = self._solve(knots)
        return field

    def _canonicalize_field(
        self, robin_field: FloatArray, knots: FloatArray
    ) -> tuple[FloatArray, BoundaryStatistics]:
        """Fix the affine gauge using exact target-arc means."""
        knots = _validated_knots(knots)
        boundary_field = np.asarray(robin_field, dtype=np.float64)[
            self.boundary_vertices
        ]
        zero_mass = self._arc_mass(float(knots[0]), float(knots[1]))
        one_mass = self._arc_mass(float(knots[2]), float(knots[3]))
        zero_length = float(zero_mass.sum())
        one_length = float(one_mass.sum())
        zero_mean = float(zero_mass.sum(axis=0) @ boundary_field / zero_length)
        one_mean = float(one_mass.sum(axis=0) @ boundary_field / one_length)
        span = one_mean - zero_mean
        threshold = np.sqrt(np.finfo(np.float64).eps) * max(
            1.0, abs(zero_mean), abs(one_mean)
        )
        if not np.isfinite(span) or span <= threshold:
            raise ValueError(
                "target-arc mean span is too small for canonicalization: "
                f"zero_mean={zero_mean:.16g}, one_mean={one_mean:.16g}, "
                f"span={span:.16g}"
            )

        zero_second_moment = float(
            boundary_field @ (zero_mass @ boundary_field) / zero_length
        )
        one_residual = boundary_field - 1.0
        one_target_second_moment = float(
            one_residual @ (one_mass @ one_residual) / one_length
        )
        field = (np.asarray(robin_field, dtype=np.float64) - zero_mean) / span
        canonical_boundary_field = field[self.boundary_vertices]
        canonical_one_residual = canonical_boundary_field - 1.0
        canonical_zero_second_moment = float(
            canonical_boundary_field
            @ (zero_mass @ canonical_boundary_field)
            / zero_length
        )
        canonical_one_second_moment = float(
            canonical_one_residual @ (one_mass @ canonical_one_residual) / one_length
        )
        statistics = BoundaryStatistics(
            raw_zero_mean=zero_mean,
            raw_one_mean=one_mean,
            raw_span=span,
            raw_zero_target_rms=float(np.sqrt(max(zero_second_moment, 0.0))),
            raw_one_target_rms=float(np.sqrt(max(one_target_second_moment, 0.0))),
            canonical_zero_target_rms=float(
                np.sqrt(max(canonical_zero_second_moment, 0.0))
            ),
            canonical_one_target_rms=float(
                np.sqrt(max(canonical_one_second_moment, 0.0))
            ),
        )
        return field, statistics

    def field_and_boundary_statistics_from_knots(
        self, knots: FloatArray
    ) -> tuple[FloatArray, BoundaryStatistics]:
        """Return the canonical field and gauge-explicit boundary diagnostics."""
        robin_field = self.robin_field_from_knots(knots)
        return self._canonicalize_field(robin_field, knots)

    def field_from_knots(self, knots: FloatArray) -> FloatArray:
        """Return the canonical harmonic field with target-arc means zero and one."""
        field, _ = self.field_and_boundary_statistics_from_knots(knots)
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
        gradient_variance = float(self._face_weights @ (norms - mean_norm) ** 2)
        if np.any(norms <= 0.0):
            spacing_cv = np.inf
        else:
            spacings = float(norms.min()) / norms
            mean_spacing = float(self._face_weights @ spacings)
            spacing_variance = float(
                self._face_weights @ (spacings - mean_spacing) ** 2
            )
            spacing_cv = float(np.sqrt(max(spacing_variance, 0.0)) / mean_spacing)
        return FieldStatistics(
            gradient_cv=float(
                np.sqrt(max(gradient_variance, 0.0)) / max(mean_norm, 1.0e-15)
            ),
            spacing_cv=spacing_cv,
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
        """Return loss and its exact gradient in five simplex coordinates."""
        knots, knot_jacobian, gaps = knots_from_parameters(parameters)
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
        """Run direct-simplex SLSQP from one ordered four-knot initialization."""
        if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
            raise TypeError("max_iterations must be an integer")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        initial_parameters = parameters_from_knots(initial_knots)
        _, _, initial_gaps = knots_from_parameters(initial_parameters)
        if np.any(initial_gaps < self.minimum_gap - 1.0e-12):
            raise ValueError("each initial cyclic gap must be at least minimum_gap")

        cached_parameters: FloatArray | None = None
        cached_value: tuple[float, FloatArray] | None = None
        evaluation_count = 0

        def objective(parameters: FloatArray) -> tuple[float, FloatArray]:
            nonlocal cached_parameters, cached_value, evaluation_count
            parameters = np.asarray(parameters, dtype=np.float64)
            if cached_parameters is not None and np.array_equal(
                parameters, cached_parameters
            ):
                assert cached_value is not None
                return cached_value
            cached_parameters = parameters.copy()
            cached_value = self.loss_and_gradient(parameters)
            evaluation_count += 1
            return cached_value

        initial_loss, _ = objective(initial_parameters)
        history = [float(initial_loss)]
        parameter_history = [initial_parameters.copy()]

        def record(parameters: FloatArray) -> None:
            parameters = np.asarray(parameters, dtype=np.float64)
            loss, _ = objective(parameters)
            if np.array_equal(parameter_history[-1], parameters):
                history[-1] = float(loss)
            else:
                parameter_history.append(parameters.copy())
                history.append(float(loss))

        result = scipy.optimize.minimize(
            objective,
            initial_parameters,
            jac=True,
            method="SLSQP",
            bounds=scipy.optimize.Bounds(
                [-np.inf] + [self.minimum_gap] * 4,
                [np.inf] * 5,
            ),
            constraints=scipy.optimize.LinearConstraint(
                np.asarray([0.0, 1.0, 1.0, 1.0, 1.0]),
                1.0,
                1.0,
            ),
            callback=record,
            options={
                "maxiter": max_iterations,
                "ftol": 1.0e-14,
            },
        )

        parameters = np.asarray(result.x, dtype=np.float64)
        knots, _, gaps = knots_from_parameters(parameters)
        robin_field = self.robin_field_from_knots(knots)
        field, boundary_statistics = self._canonicalize_field(robin_field, knots)
        uniformity_loss, _ = self._uniformity_loss_and_gradient(robin_field)
        width_loss, _ = self._width_loss_and_knot_gradient(gaps)
        statistics = self._field_statistics(field)
        final_loss = uniformity_loss + width_loss
        record(parameters)
        history[-1] = float(final_loss)
        _, final_gradient = objective(parameters)
        raw_gaps = parameters[1:]
        constraint_violation = float(
            max(
                0.0,
                self.minimum_gap - raw_gaps.min(),
                self.minimum_gap - gaps.min(),
                abs(float(raw_gaps.sum()) - 1.0),
            )
        )
        kkt_residual = _gap_kkt_residual(final_gradient, raw_gaps, self.minimum_gap)
        success = bool(
            result.success and constraint_violation <= 1.0e-8 and kkt_residual <= 1.0e-5
        )
        message = (
            f"{result.message}; constraint_violation={constraint_violation:.3e}; "
            f"kkt_residual={kkt_residual:.3e}"
        )
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
            raw_field=robin_field,
            field=field,
            statistics=statistics,
            boundary_statistics=boundary_statistics,
            iterations=int(result.nit),
            evaluations=evaluation_count,
            constraint_violation=constraint_violation,
            kkt_residual=kkt_residual,
            success=success,
            message=message,
        )
