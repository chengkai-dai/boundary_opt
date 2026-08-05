"""Continuous partial-Dirichlet harmonic boundary optimization.

Four normalized cyclic coordinates define a zero arc and a one arc.  Every
other original mesh vertex is a variational unknown.  Interior free vertices
satisfy the harmonic equation; unconstrained boundary vertices satisfy either
the natural Neumann condition or its Wentzell extension below.  An endpoint
inside an edge is used only by local cut-cell integration.  Each affected face
has a deterministic centroid fan; the known cut values are eliminated and the
free centroid is Schur-condensed locally.  Neither the original ``V/F`` arrays
nor the global unknown vector gain a vertex.

An optional dimensionless ``boundary_smoothing = eta`` adds
``eta/2 * integral_0^1 |du/dxi|^2 dxi`` to the inner state problem, where
``xi`` is normalized boundary arclength.  On the hard constant arcs this energy
is identically zero; on free arcs it yields a Wentzell boundary condition and
lets the transition trace be solved rather than prescribed.  The outer loss is
still the bulk gradient-uniformity measure.  ``eta = 0`` is the exact
natural-Neumann model.

Away from mesh vertices and the numerical snap threshold, the hard-boundary
objective is smooth while endpoints stay in the same boundary edges.  Crossing
those active-set boundaries is generally nonsmooth.  The reference optimizer
therefore uses constrained SLSQP finite differences and several inexpensive
starts; it does not claim a global optimum.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg

from boundary_opt import (
    FieldStatistics,
    FloatArray,
    IntArray,
    Mesh,
    boundary_arclength,
    boundary_loop,
    cotangent_stiffness,
    face_gradient_basis,
    load_obj,
    random_knots,
)

_ENDPOINT_VALUES = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
_GAP_JACOBIAN = np.asarray(
    [
        [-1.0, 1.0, 0.0, 0.0],
        [0.0, -1.0, 1.0, 0.0],
        [0.0, 0.0, -1.0, 1.0],
        [1.0, 0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class PartialEvaluation:
    loss: float
    knots: FloatArray
    gaps: FloatArray
    field: FloatArray
    endpoint_points: FloatArray
    cut_points: FloatArray
    cut_values: FloatArray
    statistics: FieldStatistics
    system_residual: float
    integration_faces: int


@dataclass(frozen=True, slots=True)
class PartialOptimizationResult:
    initial_loss: float
    final_loss: float
    history: FloatArray
    knot_history: FloatArray
    knots: FloatArray
    gaps: FloatArray
    field: FloatArray
    endpoint_points: FloatArray
    statistics: FieldStatistics
    iterations: int
    evaluations: int
    success: bool
    message: str


@dataclass(frozen=True, slots=True)
class _CutPoint:
    reference: int
    fraction: float


@dataclass(frozen=True, slots=True)
class _LocalPatch:
    boundary_references: tuple[int, ...]
    center: FloatArray
    center_weights: FloatArray
    condensed_stiffness: FloatArray


@dataclass(frozen=True, slots=True)
class _CutAssembly:
    stiffness: scipy.sparse.csr_matrix
    cut_rhs: FloatArray
    known_indices: IntArray
    known_values: FloatArray
    endpoint_points: FloatArray
    endpoint_is_cut: np.ndarray
    affected_faces: IntArray
    local_patches: tuple[_LocalPatch, ...]


def _canonical_knots(knots: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return one unwrapped cycle, accepting wrapped or unwrapped input."""
    raw = np.asarray(knots, dtype=np.float64).reshape(-1)
    if raw.shape != (4,) or not np.isfinite(raw).all():
        raise ValueError("knots must contain four finite values")

    wrapped = np.mod(raw, 1.0)
    canonical = np.empty(4, dtype=np.float64)
    canonical[0] = wrapped[0]
    for index in range(1, 4):
        canonical[index] = wrapped[index]
        if canonical[index] <= canonical[index - 1]:
            canonical[index] += 1.0

    gaps = np.append(np.diff(canonical), canonical[0] + 1.0 - canonical[3])
    if np.any(gaps <= 0.0):
        raise ValueError("knots must be strictly ordered around one cycle")
    return canonical, gaps


def _triangle_geometry(points: FloatArray) -> tuple[float, FloatArray]:
    """Return area and barycentric-gradient rows for one 3D triangle."""
    edge01 = points[1] - points[0]
    edge02 = points[2] - points[0]
    cross = np.cross(edge01, edge02)
    double_area = float(np.linalg.norm(cross))
    scale = float(np.linalg.norm(edge01) * np.linalg.norm(edge02))
    if double_area <= np.finfo(np.float64).eps * scale * 16.0:
        raise ValueError("a boundary cut produced a degenerate triangle")
    normal = cross / double_area
    basis = (
        np.stack(
            (
                np.cross(normal, points[2] - points[1]),
                np.cross(normal, points[0] - points[2]),
                np.cross(normal, points[1] - points[0]),
            )
        )
        / double_area
    )
    return 0.5 * double_area, basis


def _statistics_and_loss(
    areas: FloatArray, squared_norms: FloatArray
) -> tuple[float, FieldStatistics]:
    weights = areas / areas.sum()
    mean_squared = float(weights @ squared_norms)
    if not np.isfinite(mean_squared) or mean_squared <= 0.0:
        raise ValueError("harmonic field has zero or invalid gradient energy")
    second_moment = float(weights @ squared_norms**2)
    loss = second_moment / mean_squared**2 - 1.0

    norms = np.sqrt(squared_norms)
    mean_norm = float(weights @ norms)
    variance = float(weights @ (norms - mean_norm) ** 2)
    statistics = FieldStatistics(
        spacing_cv=float(np.sqrt(max(variance, 0.0)) / max(mean_norm, 1.0e-15)),
        minimum_gradient=float(norms.min()),
        maximum_gradient=float(norms.max()),
    )
    return float(loss), statistics


class ContinuousPartialBoundaryOptimizer:
    """Optimize two hard arcs with optional Wentzell boundary smoothing."""

    def __init__(
        self,
        mesh: Mesh,
        *,
        minimum_gap: float = 0.03,
        snap_tolerance: float = 1.0e-9,
        boundary_smoothing: float = 0.0,
    ) -> None:
        if not np.isfinite(minimum_gap) or not 0.0 < minimum_gap < 0.25:
            raise ValueError("minimum_gap must lie in (0, 0.25)")
        if not np.isfinite(snap_tolerance) or snap_tolerance <= 0.0:
            raise ValueError("snap_tolerance must be finite and positive")
        if 2.0 * snap_tolerance >= minimum_gap:
            raise ValueError("2 * snap_tolerance must be smaller than minimum_gap")
        if not np.isfinite(boundary_smoothing) or boundary_smoothing < 0.0:
            raise ValueError("boundary_smoothing must be finite and non-negative")

        self.mesh = mesh
        self.minimum_gap = float(minimum_gap)
        self.snap_tolerance = float(snap_tolerance)
        self.boundary_smoothing = float(boundary_smoothing)
        self.boundary_vertices = boundary_loop(mesh.faces)
        self.boundary_positions = boundary_arclength(
            mesh.vertices, self.boundary_vertices
        )
        self._stiffness = cotangent_stiffness(mesh)
        self._face_areas, self._gradient_basis = face_gradient_basis(mesh)
        self._face_stiffness = self._face_areas[:, None, None] * np.einsum(
            "fik,fjk->fij", self._gradient_basis, self._gradient_basis
        )

        edge_faces: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
        for face_index, face in enumerate(mesh.faces):
            for start, end in (
                (int(face[0]), int(face[1])),
                (int(face[1]), int(face[2])),
                (int(face[2]), int(face[0])),
            ):
                edge_faces[tuple(sorted((start, end)))].append(face_index)

        following = np.roll(self.boundary_vertices, -1)
        boundary_face_indices = []
        for start, end in zip(self.boundary_vertices, following, strict=True):
            incident = edge_faces[tuple(sorted((int(start), int(end))))]
            if len(incident) != 1:
                raise ValueError("each boundary edge must have one incident face")
            boundary_face_indices.append(incident[0])
        self._boundary_face_indices = np.asarray(boundary_face_indices, dtype=np.int64)

    def _reference_points(
        self, references: tuple[int, ...], endpoint_points: FloatArray
    ) -> FloatArray:
        original_count = len(self.mesh.vertices)
        return np.asarray(
            [
                self.mesh.vertices[reference]
                if reference < original_count
                else endpoint_points[reference - original_count]
                for reference in references
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _add_known(known: dict[int, float], index: int, value: float) -> None:
        previous = known.get(index)
        if previous is not None and previous != value:
            raise ValueError("zero and one Dirichlet arcs overlap")
        known[index] = value

    def _locate_endpoints(
        self, knots: FloatArray
    ) -> tuple[
        FloatArray,
        np.ndarray,
        IntArray,
        defaultdict[int, list[_CutPoint]],
    ]:
        original_count = len(self.mesh.vertices)
        endpoint_points = np.empty((4, 3), dtype=np.float64)
        endpoint_is_cut = np.zeros(4, dtype=bool)
        snapped_vertices = np.full(4, -1, dtype=np.int64)
        cuts_by_edge: defaultdict[int, list[_CutPoint]] = defaultdict(list)

        for endpoint, knot in enumerate(knots):
            position = float(knot % 1.0)
            cyclic_distance = np.abs(
                (self.boundary_positions - position + 0.5) % 1.0 - 0.5
            )
            nearest = int(np.argmin(cyclic_distance))
            if cyclic_distance[nearest] <= self.snap_tolerance:
                vertex = int(self.boundary_vertices[nearest])
                snapped_vertices[endpoint] = vertex
                endpoint_points[endpoint] = self.mesh.vertices[vertex]
                continue

            right = int(
                np.searchsorted(self.boundary_positions, position, side="right")
            )
            left = right - 1
            if left < 0:
                raise ValueError("failed to locate a boundary endpoint")
            next_index = (left + 1) % len(self.boundary_vertices)
            left_position = float(self.boundary_positions[left])
            right_position = (
                1.0 if next_index == 0 else float(self.boundary_positions[next_index])
            )
            fraction = (position - left_position) / (right_position - left_position)
            start = int(self.boundary_vertices[left])
            end = int(self.boundary_vertices[next_index])
            endpoint_points[endpoint] = (1.0 - fraction) * self.mesh.vertices[
                start
            ] + fraction * self.mesh.vertices[end]
            endpoint_is_cut[endpoint] = True
            cuts_by_edge[left].append(
                _CutPoint(
                    reference=original_count + endpoint,
                    fraction=float(fraction),
                )
            )
        return endpoint_points, endpoint_is_cut, snapped_vertices, cuts_by_edge

    def _build_local_patches(
        self,
        cuts_by_edge: defaultdict[int, list[_CutPoint]],
        endpoint_points: FloatArray,
    ) -> tuple[IntArray, tuple[_LocalPatch, ...]]:
        affected_faces = np.unique(
            [self._boundary_face_indices[edge] for edge in cuts_by_edge]
        ).astype(np.int64)
        edge_cuts: dict[tuple[int, int], tuple[int, int, list[_CutPoint]]] = {}
        for edge, cuts in cuts_by_edge.items():
            start = int(self.boundary_vertices[edge])
            end = int(self.boundary_vertices[(edge + 1) % len(self.boundary_vertices)])
            edge_cuts[tuple(sorted((start, end)))] = (start, end, cuts)

        patches: list[_LocalPatch] = []
        for face_index_value in affected_faces:
            face_index = int(face_index_value)
            face = tuple(map(int, self.mesh.faces[face_index]))
            boundary: list[int] = []
            for index, start in enumerate(face):
                end = face[(index + 1) % 3]
                boundary.append(start)
                entry = edge_cuts.get(tuple(sorted((start, end))))
                if entry is None:
                    continue
                loop_start, loop_end, cuts = entry
                if (start, end) == (loop_start, loop_end):
                    ordered = sorted(cuts, key=lambda item: item.fraction)
                elif (start, end) == (loop_end, loop_start):
                    ordered = sorted(cuts, key=lambda item: item.fraction, reverse=True)
                else:  # pragma: no cover - guarded by the undirected key
                    raise ValueError("boundary edge orientation is inconsistent")
                boundary.extend(cut.reference for cut in ordered)

            boundary_references = tuple(boundary)
            boundary_points = self._reference_points(
                boundary_references, endpoint_points
            )
            center = self.mesh.vertices[self.mesh.faces[face_index]].mean(axis=0)
            boundary_count = len(boundary_references)
            local_stiffness = np.zeros(
                (boundary_count + 1, boundary_count + 1), dtype=np.float64
            )
            fan_area = 0.0
            for index in range(boundary_count):
                following = (index + 1) % boundary_count
                points = np.asarray(
                    [boundary_points[index], boundary_points[following], center]
                )
                area, basis = _triangle_geometry(points)
                fan_area += area
                triangle_stiffness = area * (basis @ basis.T)
                local_indices = (index, following, boundary_count)
                for row, local_row in enumerate(local_indices):
                    for column, local_column in enumerate(local_indices):
                        local_stiffness[local_row, local_column] += triangle_stiffness[
                            row, column
                        ]

            original_area = float(self._face_areas[face_index])
            if not np.isclose(
                fan_area,
                original_area,
                rtol=1.0e-11,
                atol=1.0e-13 * max(1.0, original_area),
            ):
                raise ValueError("local cut-cell fan does not cover its source face")
            center_stiffness = float(local_stiffness[-1, -1])
            if not np.isfinite(center_stiffness) or center_stiffness <= 0.0:
                raise ValueError("local cut-cell center cannot be eliminated")
            coupling = local_stiffness[:-1, -1]
            # Eliminate the one free centroid from [Kbb Kbc; Kcb Kcc].
            condensed = (
                local_stiffness[:-1, :-1]
                - np.outer(coupling, coupling) / center_stiffness
            )
            condensed = 0.5 * (condensed + condensed.T)
            patches.append(
                _LocalPatch(
                    boundary_references=boundary_references,
                    center=center,
                    center_weights=-local_stiffness[-1, :-1] / center_stiffness,
                    condensed_stiffness=condensed,
                )
            )
        return affected_faces, tuple(patches)

    def _add_boundary_smoothing(
        self,
        stiffness: scipy.sparse.lil_matrix,
        cut_rhs: FloatArray,
        cuts_by_edge: defaultdict[int, list[_CutPoint]],
    ) -> None:
        """Add eta * integral |du/dxi|^2 on normalized boundary arclength."""
        if self.boundary_smoothing == 0.0:
            return
        original_count = len(self.mesh.vertices)
        boundary_count = len(self.boundary_vertices)
        for edge in range(boundary_count):
            cuts = sorted(cuts_by_edge.get(edge, ()), key=lambda item: item.fraction)
            references = [
                int(self.boundary_vertices[edge]),
                *(cut.reference for cut in cuts),
                int(self.boundary_vertices[(edge + 1) % boundary_count]),
            ]
            fractions = [0.0, *(cut.fraction for cut in cuts), 1.0]
            edge_span = (
                1.0 - float(self.boundary_positions[edge])
                if edge + 1 == boundary_count
                else float(
                    self.boundary_positions[edge + 1] - self.boundary_positions[edge]
                )
            )
            for segment in range(len(references) - 1):
                span = edge_span * (fractions[segment + 1] - fractions[segment])
                coefficient = self.boundary_smoothing / span
                local_matrix = coefficient * np.asarray([[1.0, -1.0], [-1.0, 1.0]])
                pair = references[segment : segment + 2]
                for row, row_reference in enumerate(pair):
                    if row_reference >= original_count:
                        continue
                    for column, column_reference in enumerate(pair):
                        value = local_matrix[row, column]
                        if column_reference < original_count:
                            stiffness[row_reference, column_reference] += value
                        else:
                            endpoint = column_reference - original_count
                            cut_rhs[row_reference] -= value * _ENDPOINT_VALUES[endpoint]

    def _assemble(self, knots: FloatArray) -> _CutAssembly:
        (
            endpoint_points,
            endpoint_is_cut,
            snapped_vertices,
            cuts_by_edge,
        ) = self._locate_endpoints(knots)
        affected_faces, local_patches = self._build_local_patches(
            cuts_by_edge, endpoint_points
        )

        local = np.mod(self.boundary_positions - knots[0], 1.0)
        first, second, third = knots[1:] - knots[0]
        zero_arc = local <= first + self.snap_tolerance
        one_arc = (local >= second - self.snap_tolerance) & (
            local <= third + self.snap_tolerance
        )
        known: dict[int, float] = {}
        for vertex in self.boundary_vertices[zero_arc]:
            self._add_known(known, int(vertex), 0.0)
        for vertex in self.boundary_vertices[one_arc]:
            self._add_known(known, int(vertex), 1.0)
        for endpoint, vertex in enumerate(snapped_vertices):
            if vertex >= 0:
                self._add_known(known, int(vertex), float(_ENDPOINT_VALUES[endpoint]))

        stiffness = self._stiffness.tolil(copy=True)
        cut_rhs = np.zeros(len(self.mesh.vertices), dtype=np.float64)
        for face_index in affected_faces:
            references = self.mesh.faces[face_index]
            original_matrix = self._face_stiffness[face_index]
            for row, row_reference in enumerate(references):
                for column, column_reference in enumerate(references):
                    stiffness[row_reference, column_reference] -= original_matrix[
                        row, column
                    ]

        original_count = len(self.mesh.vertices)
        for patch in local_patches:
            for row, row_reference in enumerate(patch.boundary_references):
                if row_reference >= original_count:
                    continue
                for column, column_reference in enumerate(patch.boundary_references):
                    coefficient = patch.condensed_stiffness[row, column]
                    if column_reference < original_count:
                        stiffness[row_reference, column_reference] += coefficient
                    else:
                        endpoint = column_reference - original_count
                        cut_rhs[row_reference] -= (
                            coefficient * _ENDPOINT_VALUES[endpoint]
                        )

        self._add_boundary_smoothing(stiffness, cut_rhs, cuts_by_edge)

        known_indices = np.asarray(sorted(known), dtype=np.int64)
        known_values = np.asarray(
            [known[int(index)] for index in known_indices], dtype=np.float64
        )
        return _CutAssembly(
            stiffness=stiffness.tocsr(),
            cut_rhs=cut_rhs,
            known_indices=known_indices,
            known_values=known_values,
            endpoint_points=endpoint_points,
            endpoint_is_cut=endpoint_is_cut,
            affected_faces=affected_faces,
            local_patches=local_patches,
        )

    def _loss(
        self, field: FloatArray, assembly: _CutAssembly
    ) -> tuple[float, FieldStatistics]:
        unaffected = np.ones(len(self.mesh.faces), dtype=bool)
        unaffected[assembly.affected_faces] = False
        original_values = field[self.mesh.faces[unaffected]]
        original_gradients = np.einsum(
            "fij,fi->fj", self._gradient_basis[unaffected], original_values
        )
        areas = list(self._face_areas[unaffected])
        squared_norms = list(
            np.einsum("ij,ij->i", original_gradients, original_gradients)
        )

        original_count = len(self.mesh.vertices)
        for patch in assembly.local_patches:
            boundary_points = self._reference_points(
                patch.boundary_references, assembly.endpoint_points
            )
            boundary_values = np.asarray(
                [
                    field[reference]
                    if reference < original_count
                    else _ENDPOINT_VALUES[reference - original_count]
                    for reference in patch.boundary_references
                ],
                dtype=np.float64,
            )
            center_value = float(patch.center_weights @ boundary_values)
            for index in range(len(patch.boundary_references)):
                following = (index + 1) % len(patch.boundary_references)
                points = np.asarray(
                    [
                        boundary_points[index],
                        boundary_points[following],
                        patch.center,
                    ]
                )
                area, basis = _triangle_geometry(points)
                values = np.asarray(
                    [
                        boundary_values[index],
                        boundary_values[following],
                        center_value,
                    ]
                )
                gradient = values @ basis
                areas.append(area)
                squared_norms.append(float(gradient @ gradient))
        return _statistics_and_loss(
            np.asarray(areas, dtype=np.float64),
            np.asarray(squared_norms, dtype=np.float64),
        )

    def _evaluate(
        self, knots: FloatArray, *, enforce_minimum_gap: bool
    ) -> PartialEvaluation:
        knots, gaps = _canonical_knots(knots)
        if enforce_minimum_gap and np.any(gaps < self.minimum_gap):
            raise ValueError("knot gaps are smaller than minimum_gap")
        assembly = self._assemble(knots)

        free_mask = np.ones(len(self.mesh.vertices), dtype=bool)
        free_mask[assembly.known_indices] = False
        free = np.flatnonzero(free_mask).astype(np.int64)
        field = np.empty(len(self.mesh.vertices), dtype=np.float64)
        field[assembly.known_indices] = assembly.known_values

        if len(free):
            system = assembly.stiffness[free][:, free].tocsc()
            right_hand_side = assembly.cut_rhs[free].copy()
            if len(assembly.known_indices):
                right_hand_side -= (
                    assembly.stiffness[free][:, assembly.known_indices]
                    @ assembly.known_values
                )
            try:
                factorization = scipy.sparse.linalg.splu(system)
            except RuntimeError as exc:
                raise ValueError(
                    "partial Dirichlet constraints do not anchor the mesh"
                ) from exc
            field[free] = factorization.solve(right_hand_side)
            residual = float(np.max(np.abs(system @ field[free] - right_hand_side)))
        else:
            residual = 0.0

        loss, statistics = self._loss(field, assembly)
        return PartialEvaluation(
            loss=loss,
            knots=knots,
            gaps=gaps,
            field=field,
            endpoint_points=assembly.endpoint_points.copy(),
            cut_points=assembly.endpoint_points[assembly.endpoint_is_cut].copy(),
            cut_values=_ENDPOINT_VALUES[assembly.endpoint_is_cut].copy(),
            statistics=statistics,
            system_residual=residual,
            integration_faces=(
                len(self.mesh.faces)
                - len(assembly.affected_faces)
                + sum(
                    len(patch.boundary_references) for patch in assembly.local_patches
                )
            ),
        )

    def evaluate(self, knots: FloatArray) -> PartialEvaluation:
        """Solve the exact partial-Dirichlet field for four feasible knots."""
        return self._evaluate(knots, enforce_minimum_gap=True)

    def _feature_starts(self) -> list[FloatArray]:
        if len(self.boundary_vertices) < 4:
            return []
        points = self.mesh.vertices[self.boundary_vertices]
        incoming = points - np.roll(points, 1, axis=0)
        outgoing = np.roll(points, -1, axis=0) - points
        denominators = np.linalg.norm(incoming, axis=1) * np.linalg.norm(
            outgoing, axis=1
        )
        cosines = np.einsum("ij,ij->i", incoming, outgoing) / denominators
        turning = np.arccos(np.clip(cosines, -1.0, 1.0))
        indices = np.sort(np.argsort(turning)[-4:])
        candidate, gaps = _canonical_knots(self.boundary_positions[indices])
        if np.any(gaps < self.minimum_gap):
            return []
        rotated, _ = _canonical_knots(np.roll(candidate, -1))
        return [candidate, rotated]

    def _local_optimize(
        self,
        initial_knots: FloatArray,
        *,
        max_iterations: int,
    ) -> PartialOptimizationResult:
        cache: dict[bytes, PartialEvaluation] = {}
        evaluations = 0
        best = self.evaluate(initial_knots)
        evaluations += 1
        history = [best.loss]
        knot_history = [best.knots.copy()]

        def raw_gaps(values: FloatArray) -> FloatArray:
            values = np.asarray(values, dtype=np.float64)
            return np.append(np.diff(values), values[0] + 1.0 - values[3])

        def feasible(values: FloatArray) -> bool:
            gaps = raw_gaps(values)
            return bool(
                np.isfinite(values).all()
                and 0.0 <= values[0] <= 1.0
                and np.all(gaps >= self.minimum_gap - 1.0e-10)
            )

        def cached_evaluation(values: FloatArray) -> PartialEvaluation:
            nonlocal evaluations
            values = np.asarray(values, dtype=np.float64)
            key = values.tobytes()
            if key not in cache:
                evaluations += 1
                cache[key] = self._evaluate(values, enforce_minimum_gap=False)
            return cache[key]

        def objective(values: FloatArray) -> float:
            nonlocal best
            try:
                evaluation = cached_evaluation(values)
            except (ValueError, RuntimeError):
                return 1.0e12
            if feasible(values) and evaluation.loss < best.loss:
                best = evaluation
            return evaluation.loss

        def callback(values: FloatArray) -> None:
            objective(values)
            history.append(best.loss)
            knot_history.append(best.knots.copy())

        result = scipy.optimize.minimize(
            objective,
            initial_knots,
            method="SLSQP",
            bounds=((0.0, 1.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0)),
            constraints={
                "type": "ineq",
                "fun": lambda values: raw_gaps(values) - self.minimum_gap,
                "jac": lambda _values: _GAP_JACOBIAN,
            },
            callback=callback,
            options={
                "maxiter": max_iterations,
                "ftol": 1.0e-10,
                "eps": 1.0e-6,
                "disp": False,
            },
        )
        if feasible(result.x):
            objective(np.asarray(result.x, dtype=np.float64))
        if history[-1] != best.loss:
            history.append(best.loss)
            knot_history.append(best.knots.copy())

        return PartialOptimizationResult(
            initial_loss=float(history[0]),
            final_loss=float(best.loss),
            history=np.asarray(history, dtype=np.float64),
            knot_history=np.vstack(knot_history),
            knots=best.knots,
            gaps=best.gaps,
            field=best.field,
            endpoint_points=best.endpoint_points,
            statistics=best.statistics,
            iterations=int(result.nit),
            evaluations=evaluations,
            success=bool(result.success),
            message=str(result.message),
        )

    def optimize(
        self,
        initial_knots: FloatArray,
        *,
        max_iterations: int = 60,
        starts: int = 2,
    ) -> PartialOptimizationResult:
        """Refine at most ``starts`` of the best deterministic candidates."""
        if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
            raise TypeError("max_iterations must be an integer")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not isinstance(starts, int) or isinstance(starts, bool):
            raise TypeError("starts must be an integer")
        if starts < 1:
            raise ValueError("starts must be positive")

        initial_knots, initial_gaps = _canonical_knots(initial_knots)
        if np.any(initial_gaps < self.minimum_gap):
            raise ValueError("initial knot gaps are smaller than minimum_gap")

        candidates = [initial_knots, np.arange(4, dtype=np.float64) / 4.0]
        candidates.extend(self._feature_starts())
        unique_candidates: list[FloatArray] = []
        for candidate in candidates:
            candidate, gaps = _canonical_knots(candidate)
            if np.any(gaps < self.minimum_gap):
                continue
            if not any(
                np.allclose(candidate, previous, rtol=0.0, atol=1.0e-14)
                for previous in unique_candidates
            ):
                unique_candidates.append(candidate)

        candidate_evaluations = [
            self.evaluate(candidate) for candidate in unique_candidates
        ]
        initial_evaluation = candidate_evaluations[0]
        best = min(candidate_evaluations, key=lambda evaluation: evaluation.loss)
        global_history = [initial_evaluation.loss]
        global_knots = [initial_evaluation.knots.copy()]
        if best.loss < global_history[-1]:
            global_history.append(best.loss)
            global_knots.append(best.knots.copy())

        runs: list[PartialOptimizationResult] = []
        extra_evaluations = 0
        if best.loss > 1.0e-10:
            ranked = sorted(
                candidate_evaluations, key=lambda evaluation: evaluation.loss
            )
            for candidate in ranked[: min(starts, len(ranked))]:
                run = self._local_optimize(
                    candidate.knots, max_iterations=max_iterations
                )
                runs.append(run)
                if run.final_loss < best.loss:
                    best = self.evaluate(run.knots)
                    extra_evaluations += 1
                for loss, knots in zip(run.history, run.knot_history, strict=True):
                    if loss < global_history[-1]:
                        global_history.append(float(loss))
                        global_knots.append(knots.copy())

        if global_history[-1] != best.loss:
            global_history.append(best.loss)
            global_knots.append(best.knots.copy())
        success = best.loss <= 1.0e-10 or any(run.success for run in runs)
        if best.loss <= 1.0e-10 and not runs:
            message = "near-zero candidate; local optimization skipped"
        else:
            message = f"best of {len(candidate_evaluations)} candidates and {len(runs)} local runs"
        return PartialOptimizationResult(
            initial_loss=float(initial_evaluation.loss),
            final_loss=float(best.loss),
            history=np.asarray(global_history, dtype=np.float64),
            knot_history=np.vstack(global_knots),
            knots=best.knots,
            gaps=best.gaps,
            field=best.field,
            endpoint_points=best.endpoint_points,
            statistics=best.statistics,
            iterations=sum(run.iterations for run in runs),
            evaluations=(
                len(candidate_evaluations)
                + sum(run.evaluations for run in runs)
                + extra_evaluations
            ),
            success=success,
            message=message,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=Path("data/disk.obj"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument(
        "--starts",
        type=int,
        default=2,
        help="number of best candidate starts refined by SLSQP",
    )
    parser.add_argument("--minimum-gap", type=float, default=0.03)
    parser.add_argument(
        "--boundary-smoothing",
        type=float,
        default=0.0,
        help="dimensionless Wentzell boundary-energy weight",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimizer = ContinuousPartialBoundaryOptimizer(
        load_obj(args.mesh),
        minimum_gap=args.minimum_gap,
        boundary_smoothing=args.boundary_smoothing,
    )
    initial = random_knots(args.seed, args.minimum_gap)
    result = optimizer.optimize(
        initial, max_iterations=args.iterations, starts=args.starts
    )
    print(
        f"mesh={args.mesh} loss {result.initial_loss:.9g} -> "
        f"{result.final_loss:.9g}; iterations={result.iterations}; "
        f"evaluations={result.evaluations}; eta={args.boundary_smoothing:g}; "
        f"success={result.success}"
    )
    print(f"knots[0,1)={np.mod(result.knots, 1.0)}")
    print(f"angles[0,2pi)={2.0 * np.pi * np.mod(result.knots, 1.0)}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output,
            field=result.field,
            knots=np.mod(result.knots, 1.0),
            angles=2.0 * np.pi * np.mod(result.knots, 1.0),
            endpoint_points=result.endpoint_points,
            history=result.history,
            boundary_smoothing=np.asarray(args.boundary_smoothing),
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
