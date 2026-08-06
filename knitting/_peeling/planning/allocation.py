from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from knitting._peeling.surface import UINT32_MAX
from knitting._peeling.config import PeelingConfig
from knitting._peeling.planning.matching import Matches, NextMatchSegment
from knitting._peeling.planning.stitch_state import FrontStitchSample, LinkPolicy

NextSegmentKey = tuple[int, int, int]


@dataclass
class AllocationPlan:
    next_segment_stitch_counts: dict[NextSegmentKey, int] = field(default_factory=dict)

    def count_for(self, key: NextSegmentKey) -> int:
        return self.next_segment_stitch_counts.get(key, 0)

    def set_count(self, key: NextSegmentKey, count: int) -> None:
        self.next_segment_stitch_counts[key] = count


@dataclass
class AllocationResult:
    next_stitches: list[list[FrontStitchSample]]
    plan: AllocationPlan


@dataclass
class _AllocRange:
    begin: float
    end: float
    first_one: bool
    last_one: bool
    stitches: int = 0
    length: float = 0.0


def allocate_next_stitches_for_matches(
    parameters: PeelingConfig,
    matches: Matches,
    front_matches: list[int],
    next_matches: list[int],
    front_stitches: list[list[FrontStitchSample]],
    next_chains: list[list[int]],
    next_lengths: list[list[float]],
    next_discard_after: list[list[tuple[float, bool]]],
) -> AllocationResult:
    next_stitches: list[list[FrontStitchSample]] = [[] for _ in next_chains]
    allocation_plan = AllocationPlan()
    for key in sorted(matches):
        front_index, next_index = key
        match = matches[key]
        if not match.front or not match.next:
            continue

        is_split_or_merge = front_matches[front_index] > 1 or next_matches[next_index] > 1
        front_ones = 0
        front_anys = 0
        for segment in match.front:
            for stitch_index in segment.stitches:
                link_mode = front_stitches[front_index][stitch_index].link_policy
                if link_mode == LinkPolicy.LINK_ONE:
                    front_ones += 1
                elif link_mode == LinkPolicy.LINK_ANY:
                    front_anys += 1

        next_is_loop = next_chains[next_index][0] == next_chains[next_index][-1]
        discard_after = next_discard_after[next_index]
        next_ones = _count_next_ones_for_match(match.next, discard_after, next_is_loop)

        total_length = 0.0
        for segment in match.next:
            if segment.begin <= segment.end:
                total_length = float(total_length + float(segment.end - segment.begin))
            else:
                total_length = float(total_length + float(segment.end + 1.0 - segment.begin))
        total_length = float(total_length * next_lengths[next_index][-1])

        target_count = max(1, _round_positive_to_int(total_length / parameters.stitch_spacing))
        lower = _minimum_next_stitches(front_ones, front_anys, next_ones)
        upper = front_ones + 2 * front_anys
        if is_split_or_merge:
            target_count = front_ones + front_anys
        target_count = max(lower, min(upper, target_count))

        segment_keys = [
            (front_index, next_index, segment_index)
            for segment_index, _segment in enumerate(match.next)
        ]
        new_stitches, segment_counts = _allocate_next_stitches_for_match(
            target_count,
            match.next,
            discard_after,
            next_lengths[next_index][-1],
            next_is_loop=next_is_loop,
        )
        for segment_key, count in zip(segment_keys, segment_counts, strict=True):
            allocation_plan.set_count(segment_key, count)
        next_stitches[next_index].extend(new_stitches)

    _balance_merge_next_stitches(matches, next_chains, next_stitches, allocation_plan)

    for stitches in next_stitches:
        stitches.sort(key=lambda stitch: stitch.chain_t)
    return AllocationResult(next_stitches=next_stitches, plan=allocation_plan)


def discard_ranges(
    parameters: PeelingConfig,
    slice_times: np.ndarray,
    chain: list[int],
    lengths: list[float],
    front_max_time: float,
) -> list[tuple[float, bool]]:
    discard_after: list[tuple[float, bool]] = [(0.0, bool(slice_times[chain[0]] > front_max_time))]
    for index in range(len(chain) - 1):
        ta = float(slice_times[chain[index]])
        tb = float(slice_times[chain[index + 1]])
        if (ta <= front_max_time < tb) or (tb <= front_max_time < ta):
            mix = (front_max_time - ta) / (tb - ta)
            length = mix * (lengths[index + 1] - lengths[index]) + lengths[index]
            discard_after.append((length / lengths[-1], bool(tb > front_max_time)))
        elif (ta > front_max_time) != (tb > front_max_time):
            raise ValueError("inconsistent discard range crossing")

    is_loop = chain[0] == chain[-1]
    discard_after = _remove_short_ranges(
        discard_after, lengths[-1], is_loop, False, 1.5 * parameters.stitch_spacing
    )
    discard_after = _remove_short_ranges(
        discard_after, lengths[-1], is_loop, True, 0.5 * parameters.stitch_spacing
    )
    return discard_after


def is_discarded(t: float, discard_after: list[tuple[float, bool]]) -> bool:
    current = discard_after[0][1]
    for begin, discard in discard_after[1:]:
        if begin <= t:
            current = discard
        else:
            break
    return current


def _round_positive_to_int(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _minimum_next_stitches(front_ones: int, front_anys: int, next_ones: int) -> int:
    paired_ones = min(front_ones, next_ones)
    front_ones_without_next_one = front_ones - paired_ones
    next_ones_without_front_one = next_ones - paired_ones
    front_anys_after_next_ones = max(0, front_anys - next_ones_without_front_one)
    return next_ones + front_ones_without_next_one + (front_anys_after_next_ones + 1) // 2


def _count_next_ones_for_match(
    next_segments: list[NextMatchSegment],
    discard_after: list[tuple[float, bool]],
    next_is_loop: bool,
) -> int:
    count = 0
    for segment in next_segments:
        for index, (transition, _discard) in enumerate(discard_after):
            if next_is_loop and index == 0:
                continue
            if segment.begin <= transition < segment.end:
                count += 1
            if segment.begin < transition <= segment.end:
                count += 1
    return count


def _allocate_next_stitches_for_match(
    stitch_count: int,
    next_segments: list[NextMatchSegment],
    discard_after: list[tuple[float, bool]],
    total_length: float,
    *,
    next_is_loop: bool,
) -> tuple[list[FrontStitchSample], list[int]]:
    allocations: list[_AllocRange] = []
    allocation_segment_indices: list[int] = []

    def split_back() -> None:
        for transition_index, (transition, _discard) in enumerate(discard_after):
            if next_is_loop and transition_index == 0:
                continue
            current = allocations[-1]
            if transition < current.begin:
                continue
            if transition == current.begin:
                current.first_one = True
            elif transition < current.end:
                old_end = current.end
                current.last_one = True
                current.end = transition
                allocations.append(_AllocRange(transition, old_end, True, False))
                allocation_segment_indices.append(allocation_segment_indices[-1])
            elif transition == current.end:
                current.last_one = True

    for segment_index, segment in enumerate(next_segments):
        if segment.begin <= segment.end:
            allocations.append(_AllocRange(segment.begin, segment.end, False, False))
            allocation_segment_indices.append(segment_index)
            split_back()
        else:
            allocations.append(_AllocRange(segment.begin, 1.0, False, False))
            allocation_segment_indices.append(segment_index)
            split_back()
            allocations.append(_AllocRange(0.0, segment.end, False, False))
            allocation_segment_indices.append(segment_index)
            split_back()

    total_ones = 0
    for allocation in allocations:
        allocation.stitches = int(allocation.first_one) + int(allocation.last_one)
        total_ones += allocation.stitches
        allocation.length = float((allocation.end - allocation.begin) * total_length)

    for _ in range(total_ones, stitch_count):
        best_index = 0
        best_density = 0.0
        for index, allocation in enumerate(allocations):
            density = allocation.length / float(allocation.stitches + 1)
            if density > best_density:
                best_index = index
                best_density = density
        allocations[best_index].stitches += 1

    stitches: list[FrontStitchSample] = []
    segment_counts = [0 for _segment in next_segments]
    for allocation, segment_index in zip(
        allocations,
        allocation_segment_indices,
        strict=True,
    ):
        for stitch_index in range(allocation.stitches):
            t = float(
                (stitch_index + 0.5)
                / float(allocation.stitches)
                * (allocation.end - allocation.begin)
                + allocation.begin
            )
            link_mode = LinkPolicy.LINK_ANY
            if stitch_index == 0 and allocation.first_one:
                link_mode = LinkPolicy.LINK_ONE
            if stitch_index + 1 == allocation.stitches and allocation.last_one:
                link_mode = LinkPolicy.LINK_ONE
            stitches.append(FrontStitchSample(t, link_mode))
            segment_counts[segment_index] += 1

    if len(stitches) != stitch_count:
        raise RuntimeError("next stitch allocation produced the wrong number of stitches")
    return stitches, segment_counts


def _balance_merge_next_stitches(
    matches: Matches,
    next_chains: list[list[int]],
    next_stitches: list[list[FrontStitchSample]],
    allocation_plan: AllocationPlan,
) -> None:
    next_segments: list[list[tuple[NextMatchSegment, NextSegmentKey]]] = [[] for _ in next_chains]
    for (front_index, next_index), match in matches.items():
        if next_index == UINT32_MAX:
            continue
        for segment_index, segment in enumerate(match.next):
            if segment.front != front_index:
                raise RuntimeError("next match segment points at the wrong front chain")
            next_segments[next_index].append((segment, (front_index, next_index, segment_index)))

    for next_index, segment_entries in enumerate(next_segments):
        if not segment_entries:
            continue
        segment_entries.sort(key=lambda item: item[0].begin)
        segments = [segment for segment, _key in segment_entries]
        segment_keys = [key for _segment, key in segment_entries]
        is_loop = (
            bool(next_chains[next_index])
            and next_chains[next_index][0] == next_chains[next_index][-1]
        )
        if not (
            (segments[0].begin == 0.0 and segments[-1].end == 1.0)
            or (is_loop and segments[0].begin == segments[-1].end)
        ):
            raise RuntimeError("next match segments do not partition the chain")

        front_counts: dict[int, int] = {}
        for segment in segments:
            front_counts[segment.front] = front_counts.get(segment.front, 0) + 1
        singles = sum(1 for count in front_counts.values() if count == 1)
        doubles = sum(1 for count in front_counts.values() if count == 2)
        multis = sum(1 for count in front_counts.values() if count > 2)
        if singles == 1 and doubles == 0 and multis == 0:
            continue
        if singles == 2 and doubles == 0 and multis == 0:
            continue
        if not (singles == 2 and multis == 0):
            raise RuntimeError("unhandled merge situation")

        for index, segment in enumerate(segments):
            if front_counts[segment.front] == 1:
                segments[:] = segments[index:] + segments[:index]
                segment_keys[:] = segment_keys[index:] + segment_keys[:index]
                break

        if len(segments) % 2 != 0 or len(segments) < 4:
            raise RuntimeError("unexpected merge segment layout")

        left_sum = 0
        right_sum = 0
        for index in range(1, len(segments) // 2):
            opposite = len(segments) - index
            total = allocation_plan.count_for(segment_keys[index]) + allocation_plan.count_for(
                segment_keys[opposite]
            )
            left_count = total // 2
            right_count = (total + 1) // 2
            if right_sum > left_sum:
                left_count, right_count = right_count, left_count
            allocation_plan.set_count(segment_keys[index], left_count)
            allocation_plan.set_count(segment_keys[opposite], right_count)
            left_sum += left_count
            right_sum += right_count

        old_count = len(next_stitches[next_index])
        next_stitches[next_index].clear()
        for segment, segment_key in zip(segments, segment_keys, strict=True):
            stitch_count = allocation_plan.count_for(segment_key)
            for stitch_index in range(stitch_count):
                end = segment.end if segment.begin <= segment.end else segment.end + 1.0
                t = float(
                    (stitch_index + 0.5) / float(stitch_count) * (end - segment.begin)
                    + segment.begin
                )
                if t >= 1.0:
                    t = float(t - 1.0)
                next_stitches[next_index].append(FrontStitchSample(t, LinkPolicy.LINK_ANY))
        if old_count != len(next_stitches[next_index]):
            raise RuntimeError("merge balancing changed next stitch count")


def _remove_short_ranges(
    ranges: list[tuple[float, bool]],
    total_length: float,
    is_loop: bool,
    target_state: bool,
    min_length: float,
) -> list[tuple[float, bool]]:
    if len(ranges) <= 1:
        return ranges
    ranges = list(ranges)
    first_len = ranges[1][0]
    last_len = 1.0 - ranges[-1][0]
    if is_loop:
        first_len = last_len = first_len + last_len

    for index, (begin, state) in enumerate(ranges):
        if state != target_state:
            continue
        if index == 0:
            length = first_len
        elif index + 1 == len(ranges):
            length = last_len
        else:
            length = ranges[index + 1][0] - begin
        if length * total_length < min_length:
            ranges[index] = (begin, not target_state)

    condensed: list[tuple[float, bool]] = []
    for begin, state in ranges:
        if condensed and condensed[-1][1] == state:
            continue
        condensed.append((begin, state))
    return condensed
