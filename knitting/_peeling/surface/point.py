from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UINT32_MAX = 2**32 - 1


@dataclass(frozen=True)
class SurfacePoint:
    simplex: tuple[int, int, int]
    weights: tuple[float, float, float]

    @staticmethod
    def on_vertex(vertex: int) -> SurfacePoint:
        return SurfacePoint((vertex, UINT32_MAX, UINT32_MAX), (1.0, 0.0, 0.0))

    @staticmethod
    def on_edge(a: int, b: int, mix: float) -> SurfacePoint:
        mix = float(mix)
        if a > b:
            a, b = b, a
            mix = 1.0 - mix
        return SurfacePoint.canonicalize(
            (a, b, UINT32_MAX),
            (1.0 - mix, mix, 0.0),
        )

    @staticmethod
    def canonicalize(
        simplex: tuple[int, int, int], weights: tuple[float, float, float]
    ) -> SurfacePoint:
        combined: dict[int, float] = {}
        for index, weight in zip(simplex, weights, strict=True):
            if index == UINT32_MAX or weight == 0.0:
                continue
            combined[index] = combined.get(index, 0.0) + float(weight)
        pairs = sorted((index, weight) for index, weight in combined.items() if weight != 0.0)
        if not pairs:
            raise ValueError("surface point must have at least one nonzero weight")
        if len(pairs) > 3:
            raise ValueError("surface point cannot reference more than three vertices")
        pairs.extend([(UINT32_MAX, 0.0)] * (3 - len(pairs)))
        return SurfacePoint(
            tuple(index for index, _ in pairs),
            tuple(float(weight) for _, weight in pairs),
        )

    @staticmethod
    def common_simplex(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
        common = sorted((set(a) | set(b)) - {UINT32_MAX})
        if len(common) > 3:
            raise ValueError(f"surface points do not share a simplex: {a} vs {b}")
        common.extend([UINT32_MAX] * (3 - len(common)))
        return tuple(common)

    @staticmethod
    def mix(a: SurfacePoint, b: SurfacePoint, amount: float) -> SurfacePoint:
        a = SurfacePoint.canonicalize(a.simplex, a.weights)
        b = SurfacePoint.canonicalize(b.simplex, b.weights)
        common = SurfacePoint.common_simplex(a.simplex, b.simplex)
        a_weights = a.weights_on(common)
        b_weights = b.weights_on(common)
        amount = float(amount)
        one_minus = 1.0 - amount
        weights = tuple(
            one_minus * aw + amount * bw for aw, bw in zip(a_weights, b_weights, strict=True)
        )
        return SurfacePoint.canonicalize(common, weights)

    def weights_on(self, simplex: tuple[int, int, int]) -> tuple[float, float, float]:
        out = [0.0, 0.0, 0.0]
        for index, weight in zip(self.simplex, self.weights, strict=True):
            if index == UINT32_MAX or weight == 0.0:
                continue
            out[simplex.index(index)] = weight
        return tuple(out)

    def interpolate(self, vertices: np.ndarray) -> np.ndarray:
        vertices = np.asarray(vertices, dtype=np.float64)
        return sum(
            (
                weight * vertices[index]
                for index, weight in zip(self.simplex, self.weights, strict=True)
                if index != UINT32_MAX
            ),
            start=np.zeros(vertices.shape[1:], dtype=np.float64),
        )

    def compose(self, source_points: list[SurfacePoint]) -> SurfacePoint:
        simplex = source_points[self.simplex[0]].simplex
        for local_index in self.simplex[1:]:
            if local_index != UINT32_MAX:
                simplex = SurfacePoint.common_simplex(
                    simplex,
                    source_points[local_index].simplex,
                )

        weights = [0.0, 0.0, 0.0]
        for local_index, weight in zip(self.simplex, self.weights, strict=True):
            if local_index == UINT32_MAX:
                continue
            for index, local_weight in enumerate(source_points[local_index].weights_on(simplex)):
                weights[index] += weight * local_weight
        return SurfacePoint.canonicalize(simplex, tuple(weights))
