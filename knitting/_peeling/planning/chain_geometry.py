from __future__ import annotations

from bisect import bisect_right
from itertools import pairwise

import numpy as np

from geometry import Mesh
from knitting._peeling.planning.stitch_state import FrontStitchSample, LinkPolicy


def chain_lengths(mesh: Mesh, chain: list[int]) -> list[float]:
    lengths = [0.0]
    for previous, current in pairwise(chain):
        segment = float(np.linalg.norm(mesh.vertices[current] - mesh.vertices[previous]))
        lengths.append(float(lengths[-1] + segment))
    return lengths


def chain_location(lengths: list[float], chain_t: float) -> tuple[int, float]:
    distance = lengths[-1] * chain_t
    right = bisect_right(lengths, distance)
    if right == 0 or right == len(lengths):
        raise ValueError("stitch position fell outside chain")
    mix = (distance - lengths[right - 1]) / (lengths[right] - lengths[right - 1])
    return right, float(mix)


def segment_weights(lengths: list[float]) -> list[float]:
    return [float(lengths[index] - lengths[index - 1]) for index in range(1, len(lengths))]


def stitch_info(
    mesh: Mesh,
    chain: list[int],
    lengths: list[float],
    stitches: list[FrontStitchSample],
) -> tuple[np.ndarray, list[bool]]:
    locations: list[np.ndarray] = []
    linkones: list[bool] = []
    for stitch in stitches:
        right, mix = chain_location(lengths, stitch.chain_t)
        locations.append(
            (1.0 - mix) * mesh.vertices[chain[right - 1]] + mix * mesh.vertices[chain[right]]
        )
        linkones.append(stitch.link_policy == LinkPolicy.LINK_ONE)
    return np.asarray(locations, dtype=np.float64), linkones
