from __future__ import annotations

from heapq import heappop, heappush

import numpy as np

from geometry import Mesh
from knitting._peeling.surface import UINT32_MAX
from knitting._peeling.config import PeelingConfig
from knitting._peeling.planning.allocation import (
    AllocationPlan,
    allocate_next_stitches_for_matches,
    discard_ranges,
    is_discarded,
)
from knitting._peeling.planning.chain_geometry import chain_lengths, stitch_info
from knitting._peeling.planning.matching import Matches, build_chain_matches, match_counts
from knitting._peeling.planning.stitch_state import LinkPolicy
from knitting._peeling.planning.types import (
    FrontStitches,
    LinkChainsResult,
    SliceChains,
    StitchLink,
    StitchRef,
)

__all__ = [
    "LinkChainsResult",
    "StitchLink",
    "link_chains",
    "optimal_link",
]


def optimal_link(
    target_distance: float,
    do_roll: bool,
    source: np.ndarray,
    source_linkone: list[bool],
    target: np.ndarray,
    target_linkone: list[bool],
) -> list[tuple[int, int]]:
    if len(source) != len(source_linkone):
        raise ValueError("source and source_linkone sizes differ")
    if len(target) != len(target_linkone):
        raise ValueError("target and target_linkone sizes differ")
    if len(source) == 0 or len(target) == 0:
        return []

    source_points = np.asarray(source, dtype=float)
    target_points = np.asarray(target, dtype=float)
    target_distance = float(target_distance)
    squared_distances = (
        np.einsum("ij,ij->i", source_points, source_points)[:, None]
        + np.einsum("ij,ij->i", target_points, target_points)[None, :]
        - 2.0 * source_points @ target_points.T
    )
    np.maximum(squared_distances, 0.0, out=squared_distances)
    penalty_table = (np.sqrt(squared_distances) - target_distance) ** 2

    state_distance: dict[tuple[int, int, int, int], tuple[float, tuple[int, int]]] = {}
    heap: list[tuple[float, int, tuple[int, int, int, int]]] = []

    def state_pack(state: tuple[int, int, int, int]) -> int:
        source_index, target_index, source_remain, target_remain = state
        return (
            int(source_index)
            | (int(target_index) << 16)
            | (int(source_remain) << 32)
            | (int(target_remain) << 48)
        )

    def visit(state: tuple[int, int, int, int], distance: float, action: tuple[int, int]) -> None:
        distance = float(distance)
        current = state_distance.get(state)
        if current is None or distance < current[0]:
            state_distance[state] = (distance, action)
            heappush(heap, (distance, -state_pack(state), state))

    def possible(state: tuple[int, int, int, int]) -> bool:
        _, _, source_remain, target_remain = state
        if source_remain * 2 < target_remain:
            return False
        return target_remain * 2 >= source_remain

    def penalty(source_index: int, target_index: int) -> float:
        return float(penalty_table[source_index, target_index])

    if do_roll:
        for target_index in range(len(target)):
            visit((0, target_index, len(source), len(target)), 0.0, (0, 0))
            visit((1 % len(source), target_index, len(source), len(target)), 0.0, (0, 0))
    else:
        visit((0, 0, len(source), len(target)), 0.0, (0, 0))

    best: tuple[int, int, int, int] | None = None
    while heap:
        distance, _, state = heappop(heap)
        known = state_distance[state][0]
        if known < distance:
            continue
        source_index, target_index, source_remain, target_remain = state
        if source_remain == 0 and target_remain == 0:
            best = state
            break
        if not possible(state):
            continue

        next_state = (
            (source_index + 1) % len(source),
            (target_index + 1) % len(target),
            source_remain - 1,
            target_remain - 1,
        )
        if possible(next_state):
            visit(next_state, float(distance + penalty(source_index, target_index)), (1, 1))

        if (
            target_remain >= 2
            and not source_linkone[source_index]
            and not target_linkone[target_index]
            and not target_linkone[(target_index + 1) % len(target)]
        ):
            next_state = (
                (source_index + 1) % len(source),
                (target_index + 2) % len(target),
                source_remain - 1,
                target_remain - 2,
            )
            if possible(next_state):
                visit(
                    next_state,
                    float(
                        float(distance + penalty(source_index, target_index))
                        + penalty(source_index, (target_index + 1) % len(target))
                    ),
                    (1, 2),
                )

        if (
            source_remain >= 2
            and not source_linkone[source_index]
            and not source_linkone[(source_index + 1) % len(source)]
            and not target_linkone[target_index]
        ):
            next_state = (
                (source_index + 2) % len(source),
                (target_index + 1) % len(target),
                source_remain - 2,
                target_remain - 1,
            )
            if possible(next_state):
                visit(
                    next_state,
                    float(
                        float(distance + penalty(source_index, target_index))
                        + penalty((source_index + 1) % len(source), target_index)
                    ),
                    (2, 1),
                )

    if best is None:
        raise RuntimeError("failed to link stitch sequences")

    source_size = len(source)
    target_size = len(target)
    at = best
    links: list[tuple[int, int]] = []
    while True:
        action = state_distance[at][1]
        take_source, take_target = action
        if take_source == 0 and take_target == 0:
            break
        source_index, target_index, source_remain, target_remain = at
        if take_source == 1:
            source_index = (source_index + source_size - 1) % source_size
            source_remain += 1
            while take_target > 0:
                take_target -= 1
                target_index = (target_index + target_size - 1) % target_size
                target_remain += 1
                links.append((source_index, target_index))
        elif take_target == 1:
            target_index = (target_index + target_size - 1) % target_size
            target_remain += 1
            while take_source > 0:
                take_source -= 1
                source_index = (source_index + source_size - 1) % source_size
                source_remain += 1
                links.append((source_index, target_index))
        else:
            raise AssertionError("optimal link action should consume source or target")
        at = (source_index, target_index, source_remain, target_remain)

    links.reverse()
    return links


def link_chains(
    parameters: PeelingConfig,
    slice_mesh: Mesh,
    slice_times: np.ndarray,
    front_slice_chains: SliceChains,
    front_stitches: FrontStitches,
    next_slice_chains: SliceChains,
    matches: Matches | None = None,
) -> LinkChainsResult:
    if len(slice_times) != len(slice_mesh.vertices):
        raise ValueError("slice_times must have one entry per slice vertex")
    if len(front_stitches) != len(front_slice_chains):
        raise ValueError("front_stitches must align with front_slice_chains")

    front_lengths = [chain_lengths(slice_mesh, chain) for chain in front_slice_chains]
    next_lengths = [chain_lengths(slice_mesh, chain) for chain in next_slice_chains]
    front_max_time = max(
        float(slice_times[vertex]) for chain in front_slice_chains for vertex in chain
    )

    next_discard_after = [
        discard_ranges(parameters, slice_times, chain, lengths, front_max_time)
        for chain, lengths in zip(next_slice_chains, next_lengths, strict=True)
    ]
    if all(discard for discard_after in next_discard_after for _, discard in discard_after):
        next_discard_after = [[(0.0, False)] for _ in next_slice_chains]

    if matches is None:
        matches = build_chain_matches(
            slice_mesh,
            front_slice_chains,
            front_lengths,
            front_stitches,
            next_slice_chains,
            next_lengths,
        )
    front_matches, next_matches = match_counts(
        matches, len(front_slice_chains), len(next_slice_chains)
    )

    to_accept: set[int] = set()
    for front_index, next_index in matches:
        if front_index == UINT32_MAX or next_index == UINT32_MAX:
            continue
        if front_matches[front_index] > 1 or next_matches[next_index] > 1:
            to_accept.add(next_index)
    for next_index in to_accept:
        next_discard_after[next_index] = [(0.0, False)]

    allocation = allocate_next_stitches_for_matches(
        parameters,
        matches,
        front_matches,
        next_matches,
        front_stitches,
        next_slice_chains,
        next_lengths,
        next_discard_after,
    )
    next_stitches = allocation.next_stitches

    all_front_locations: list[np.ndarray] = []
    all_front_linkones: list[list[bool]] = []
    for chain, lengths, stitches in zip(
        front_slice_chains,
        front_lengths,
        front_stitches,
        strict=True,
    ):
        locations, linkones = stitch_info(slice_mesh, chain, lengths, stitches)
        all_front_locations.append(locations)
        all_front_linkones.append(linkones)

    all_next_locations: list[np.ndarray] = []
    all_next_linkones: list[list[bool]] = []
    for chain, lengths, stitches in zip(
        next_slice_chains,
        next_lengths,
        next_stitches,
        strict=True,
    ):
        locations, linkones = stitch_info(slice_mesh, chain, lengths, stitches)
        all_next_locations.append(locations)
        all_next_linkones.append(linkones)

    links = _link_matched_stitches(
        parameters,
        matches,
        allocation.plan,
        front_slice_chains,
        next_slice_chains,
        next_stitches,
        all_front_locations,
        all_front_linkones,
        all_next_locations,
        all_next_linkones,
    )

    for next_index, stitches in enumerate(next_stitches):
        discard_after = next_discard_after[next_index]
        for stitch in stitches:
            if is_discarded(stitch.chain_t, discard_after):
                stitch.link_policy = LinkPolicy.DISCARD

    return LinkChainsResult(
        next_stitches=next_stitches,
        links=links,
    )


def _link_matched_stitches(
    parameters: PeelingConfig,
    matches,
    allocation_plan: AllocationPlan,
    front_slice_chains: SliceChains,
    next_slice_chains: SliceChains,
    next_stitches: FrontStitches,
    all_front_locations: list[np.ndarray],
    all_front_linkones: list[list[bool]],
    all_next_locations: list[np.ndarray],
    all_next_linkones: list[list[bool]],
) -> list[StitchLink]:
    links: list[StitchLink] = []
    target_distance = parameters.course_spacing

    for key in sorted(matches):
        front_index, next_index = key
        match = matches[key]
        if not match.front or not match.next:
            continue

        next_stitch_indices: list[int] = []
        next_locations: list[np.ndarray] = []
        next_linkones: list[bool] = []
        for segment_index, segment in enumerate(match.next):
            count = 0

            def collect_range(
                begin: float,
                end: float,
                next_index: int = next_index,
                next_stitch_indices: list[int] = next_stitch_indices,
                next_locations: list[np.ndarray] = next_locations,
                next_linkones: list[bool] = next_linkones,
            ) -> int:
                range_count = 0
                for stitch_index, stitch in enumerate(next_stitches[next_index]):
                    if begin <= stitch.chain_t < end:
                        next_stitch_indices.append(stitch_index)
                        next_locations.append(all_next_locations[next_index][stitch_index])
                        next_linkones.append(all_next_linkones[next_index][stitch_index])
                        range_count += 1
                return range_count

            if segment.begin <= segment.end:
                count = collect_range(segment.begin, segment.end)
            else:
                count = collect_range(segment.begin, 1.0)
                count += collect_range(0.0, segment.end)
            if count != allocation_plan.count_for((front_index, next_index, segment_index)):
                raise RuntimeError("next stitch range assignment count mismatch")

        front_stitch_indices: list[int] = []
        front_locations: list[np.ndarray] = []
        front_linkones: list[bool] = []
        for segment in match.front:
            for stitch_index in segment.stitches:
                front_stitch_indices.append(stitch_index)
                front_locations.append(all_front_locations[front_index][stitch_index])
                front_linkones.append(all_front_linkones[front_index][stitch_index])

        if not front_stitch_indices or not next_stitch_indices:
            continue

        do_roll = (
            front_slice_chains[front_index][0] == front_slice_chains[front_index][-1]
            and next_slice_chains[next_index][0] == next_slice_chains[next_index][-1]
        )
        local_links = optimal_link(
            target_distance,
            do_roll,
            np.asarray(front_locations, dtype=float),
            front_linkones,
            np.asarray(next_locations, dtype=float),
            next_linkones,
        )
        for source_index, target_index in local_links:
            links.append(
                StitchLink(
                    front_ref=StitchRef.of(front_index, front_stitch_indices[source_index]),
                    next_ref=StitchRef.of(next_index, next_stitch_indices[target_index]),
                )
            )

    return links
