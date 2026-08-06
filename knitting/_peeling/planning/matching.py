from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np

from geometry import Mesh
from knitting._peeling.surface import UINT32_MAX
from knitting._peeling.planning.chain_geometry import segment_weights
from knitting._peeling.planning.stitch_state import FrontStitchSample


@dataclass
class FrontMatchSegment:
    begin: float
    end: float
    next: int
    stitches: list[int]


@dataclass
class NextMatchSegment:
    begin: float
    end: float
    front: int


@dataclass
class ChainMatch:
    front: list[FrontMatchSegment]
    next: list[NextMatchSegment]


Matches = dict[tuple[int, int], ChainMatch]


def build_chain_matches(
    mesh: Mesh,
    front_slice_chains: list[list[int]],
    front_lengths: list[list[float]],
    front_stitches: list[list[FrontStitchSample]],
    next_chains: list[list[int]],
    next_lengths: list[list[float]],
) -> Matches:
    front_closest = _closest_source_chain(mesh, next_chains, front_slice_chains)
    next_closest = _closest_source_chain(mesh, front_slice_chains, next_chains)
    for closest in front_closest:
        if closest:
            closest.pop()
    for closest in next_closest:
        if closest:
            closest.pop()

    while True:
        _discard_nonmutual(front_closest, next_closest)
        for front_index, closest in enumerate(front_closest):
            weights = segment_weights(front_lengths[front_index])
            is_loop = (
                bool(front_slice_chains[front_index])
                and front_slice_chains[front_index][0] == front_slice_chains[front_index][-1]
            )
            _fill_unassigned(closest, weights, is_loop)
            _flatten_closest(closest, weights, is_loop)
        for next_index, closest in enumerate(next_closest):
            weights = segment_weights(next_lengths[next_index])
            is_loop = (
                bool(next_chains[next_index])
                and next_chains[next_index][0] == next_chains[next_index][-1]
            )
            _fill_unassigned(closest, weights, is_loop)
            _flatten_closest(closest, weights, is_loop)
        if not _discard_nonmutual(front_closest, next_closest):
            break

    matches: Matches = {}

    def match_for(front_index: int, next_index: int) -> ChainMatch:
        return matches.setdefault((front_index, next_index), ChainMatch([], []))

    for front_index, closest in enumerate(front_closest):
        lengths = front_lengths[front_index]
        begin = 0
        while begin < len(closest):
            end = begin + 1
            while end < len(closest) and closest[end] == closest[begin]:
                end += 1
            match_for(front_index, closest[begin]).front.append(
                FrontMatchSegment(
                    float(lengths[begin] / lengths[-1]),
                    float(lengths[end] / lengths[-1]),
                    closest[begin],
                    [],
                )
            )
            begin = end

    for next_index, closest in enumerate(next_closest):
        lengths = next_lengths[next_index]
        begin = 0
        while begin < len(closest):
            end = begin + 1
            while end < len(closest) and closest[end] == closest[begin]:
                end += 1
            match_for(closest[begin], next_index).next.append(
                NextMatchSegment(
                    float(lengths[begin] / lengths[-1]),
                    float(lengths[end] / lengths[-1]),
                    closest[begin],
                )
            )
            begin = end

    _assign_front_stitches(matches, front_slice_chains, front_stitches)
    _merge_loop_wrapped_segments(matches, front_slice_chains, next_chains)
    _balance_split_front_stitches(matches, front_slice_chains)
    return matches


def match_counts(
    matches: Matches, front_count: int, next_count: int
) -> tuple[list[int], list[int]]:
    front_matches = [0 for _ in range(front_count)]
    next_matches = [0 for _ in range(next_count)]
    for front_index, next_index in matches:
        if front_index != UINT32_MAX:
            front_matches[front_index] += 1
        if next_index != UINT32_MAX:
            next_matches[next_index] += 1
    return front_matches, next_matches


def front_segments_from_matches(
    matches: Matches, front_count: int
) -> list[list[FrontMatchSegment]]:
    segments: list[list[FrontMatchSegment]] = [[] for _ in range(front_count)]
    for (front_index, next_index), match in matches.items():
        if front_index == UINT32_MAX:
            continue
        for segment in match.front:
            if segment.next != next_index:
                raise RuntimeError("front match segment points at the wrong next chain")
            segments[front_index].append(segment)
    return segments


def _assign_front_stitches(
    matches: Matches,
    front_slice_chains: list[list[int]],
    front_stitches: list[list[FrontStitchSample]],
) -> None:
    front_segments = front_segments_from_matches(matches, len(front_slice_chains))
    for front_index, segments in enumerate(front_segments):
        if not segments:
            continue
        segments.sort(key=lambda segment: segment.begin)
        stitches = front_stitches[front_index]
        stitch_index = 0
        for segment in segments:
            if stitch_index == len(stitches):
                break
            while stitch_index < len(stitches) and stitches[stitch_index].chain_t < segment.end:
                segment.stitches.append(stitch_index)
                stitch_index += 1


def _merge_loop_wrapped_segments(
    matches: Matches,
    front_slice_chains: list[list[int]],
    next_chains: list[list[int]],
) -> None:
    for (front_index, _next_index), match in sorted(matches.items()):
        if len(match.front) <= 1 or front_index == UINT32_MAX:
            continue
        is_loop = (
            bool(front_slice_chains[front_index])
            and front_slice_chains[front_index][0] == front_slice_chains[front_index][-1]
        )
        if not is_loop:
            continue
        match.front.sort(key=lambda segment: segment.begin)
        if match.front[0].begin == 0.0 and match.front[-1].end == 1.0:
            match.front[-1].end = match.front[0].end
            match.front[-1].stitches.extend(match.front[0].stitches)
            del match.front[0]

    for (_front_index, next_index), match in sorted(matches.items()):
        if len(match.next) <= 1 or next_index == UINT32_MAX:
            continue
        is_loop = (
            bool(next_chains[next_index])
            and next_chains[next_index][0] == next_chains[next_index][-1]
        )
        if not is_loop:
            continue
        match.next.sort(key=lambda segment: segment.begin)
        if match.next[0].begin == 0.0 and match.next[-1].end == 1.0:
            match.next[-1].end = match.next[0].end
            del match.next[0]


def _closest_source_chain(
    mesh: Mesh,
    sources: list[list[int]],
    targets: list[list[int]],
) -> list[list[int]]:
    adjacency = [[] for _ in range(len(mesh.vertices))]
    edges: set[tuple[int, int]] = set()
    for face in mesh.faces:
        a, b, c = map(int, face)
        for u, v in ((a, b), (b, c), (c, a)):
            if v > u:
                u, v = v, u
            edges.add((u, v))
    for a, b in sorted(edges):
        adjacency[a].append(b)
        adjacency[b].append(a)

    distances = [float("inf")] * len(mesh.vertices)
    source_from = [UINT32_MAX] * len(mesh.vertices)
    heap: list[tuple[float, int, int]] = []

    def queue(vertex: int, distance: float, source_index: int) -> None:
        distance = float(distance)
        distances[vertex] = distance
        source_from[vertex] = source_index
        heappush(heap, (distance, -vertex, vertex))

    vertices = mesh.vertices
    for source_index, chain in enumerate(sources):
        for vertex in chain:
            if vertex == UINT32_MAX:
                continue
            if distances[vertex] > 0.0:
                queue(vertex, 0.0, source_index)

    while heap:
        distance, _, vertex = heappop(heap)
        if distance > distances[vertex]:
            continue
        for neighbor in adjacency[vertex]:
            segment = float(np.linalg.norm(vertices[neighbor] - vertices[vertex]))
            next_distance = float(distance + segment)
            if next_distance < distances[neighbor]:
                queue(neighbor, next_distance, source_from[vertex])

    closest: list[list[int]] = []
    for chain in targets:
        closest.append(
            [UINT32_MAX if vertex == UINT32_MAX else source_from[vertex] for vertex in chain]
        )
    return closest


def _discard_nonmutual(front_closest: list[list[int]], next_closest: list[list[int]]) -> bool:
    front_refs = [set(closest) for closest in front_closest]
    next_refs = [set(closest) for closest in next_closest]
    discarded = False
    for front_index, closest in enumerate(front_closest):
        for index, value in enumerate(closest):
            if value != UINT32_MAX and front_index not in next_refs[value]:
                closest[index] = UINT32_MAX
                discarded = True
    for next_index, closest in enumerate(next_closest):
        for index, value in enumerate(closest):
            if value != UINT32_MAX and next_index not in front_refs[value]:
                closest[index] = UINT32_MAX
                discarded = True
    return discarded


def _fill_unassigned(closest: list[int], weights: list[float], is_loop: bool) -> bool:
    if not any(value != UINT32_MAX for value in closest):
        return False

    def do_range(first: int, last: int) -> None:
        before = closest[first - 1 if first > 0 else len(closest) - 1]
        after = closest[last + 1 if last + 1 < len(closest) else 0]
        if not is_loop:
            if first == 0:
                before = after
            if last + 1 == len(closest):
                after = before
        if before == UINT32_MAX or after == UINT32_MAX:
            raise RuntimeError("cannot fill unassigned closest range")

        total = 0.0
        index = first
        while True:
            total = float(total + weights[index])
            if index == last:
                break
            index = index + 1 if index + 1 < len(closest) else 0

        weight_sum = 0.0
        index = first
        while True:
            if float(weight_sum + float(0.5 * weights[index])) < float(0.5 * total):
                closest[index] = before
            else:
                closest[index] = after
            weight_sum = float(weight_sum + weights[index])
            if index == last:
                break
            index = index + 1 if index + 1 < len(closest) else 0

    for seed in range(len(closest)):
        if closest[seed] != UINT32_MAX:
            continue
        first = seed
        while (first > 0 or is_loop) and closest[
            first - 1 if first > 0 else len(closest) - 1
        ] == UINT32_MAX:
            first = first - 1 if first > 0 else len(closest) - 1
        last = seed
        while (last + 1 < len(closest) or is_loop) and closest[
            last + 1 if last + 1 < len(closest) else 0
        ] == UINT32_MAX:
            last = last + 1 if last + 1 < len(closest) else 0
        do_range(first, last)

    return True


def _flatten_closest(closest: list[int], weights: list[float], is_loop: bool) -> None:
    if not closest:
        return
    symbols: list[list[float | int]] = []
    for value, weight in zip(closest, weights, strict=True):
        if not symbols or int(symbols[-1][0]) != value:
            symbols.append([value, 0.0])
        symbols[-1][1] = float(symbols[-1][1]) + weight
    if len(symbols) == 1:
        return

    symbol_bits: dict[int, int] = {}
    bit_symbols: list[tuple[int, float]] = []
    for symbol, weight in symbols:
        symbol = int(symbol)
        if symbol not in symbol_bits:
            symbol_bits[symbol] = 1 << len(symbol_bits)
        bit_symbols.append((symbol_bits[symbol], float(weight)))
    if len(symbol_bits) > 16:
        raise RuntimeError("closest flattening has too many symbols")

    def pack(state: tuple[int, int, int, int]) -> int:
        used, minimum, maximum, current = state
        return int(used) | (int(minimum) << 16) | (int(maximum) << 24) | (int(current) << 32)

    def unpack(value: int) -> tuple[int, int, int, int]:
        return (value & 0xFFFF, (value >> 16) & 0xFF, (value >> 24) & 0xFF, (value >> 32) & 0xFFFF)

    finished_state: int | None = None
    finished_cost = float("inf")
    finished_from: int | None = None
    visited: dict[int, tuple[float, int]] = {}
    todo: list[tuple[float, int]] = []

    def queue_state(
        state: tuple[int, int, int, int], cost: float, from_state: tuple[int, int, int, int]
    ) -> None:
        nonlocal finished_state, finished_cost, finished_from
        cost = float(cost)
        _, minimum, maximum, _ = state
        _, from_minimum, from_maximum, _ = from_state
        if (minimum != from_minimum and minimum == from_maximum) or (
            maximum != from_maximum and maximum == from_minimum
        ):
            if cost < finished_cost:
                finished_state = pack(state)
                finished_cost = cost
                finished_from = pack(from_state)
            return

        state_key = pack(state)
        from_key = pack(from_state)
        old = visited.get(state_key)
        if old is None or old[0] > cost:
            visited[state_key] = (cost, from_key)
            heappush(todo, (cost, state_key))

    def expand_state(state: tuple[int, int, int, int], cost: float) -> None:
        used, minimum, maximum, current = state
        min_next = len(bit_symbols) - 1 if minimum == 0 else minimum - 1
        min_next_symbol = bit_symbols[min_next]
        max_next = maximum + 1 if maximum + 1 < len(bit_symbols) else 0
        # Advancing max consumes its old boundary, not the new one.
        max_next_symbol = bit_symbols[maximum]

        if min_next_symbol[0] == current or max_next_symbol[0] == current:
            next_state = (
                used,
                min_next if min_next_symbol[0] == current else minimum,
                max_next if max_next_symbol[0] == current else maximum,
                current,
            )
            queue_state(next_state, cost, state)
            return

        if not (used & min_next_symbol[0]):
            queue_state(
                (used | min_next_symbol[0], min_next, maximum, min_next_symbol[0]), cost, state
            )
        if not (used & max_next_symbol[0]):
            queue_state(
                (used | max_next_symbol[0], minimum, max_next, max_next_symbol[0]), cost, state
            )

        queue_state((used, min_next, maximum, current), float(cost + min_next_symbol[1]), state)
        queue_state((used, minimum, max_next, current), float(cost + max_next_symbol[1]), state)

    for symbol_index in range(len(bit_symbols)):
        expand_state((0, symbol_index, symbol_index, 0), 0.0)
        if not is_loop:
            break

    while todo:
        cost, state_key = heappop(todo)
        if cost >= finished_cost:
            break
        old = visited.get(state_key)
        if old is None or cost > old[0]:
            continue
        expand_state(unpack(state_key), cost)

    if finished_state is None or finished_from is None:
        raise RuntimeError("failed to flatten closest labels")

    path = [finished_state, finished_from]
    while path[-1] in visited:
        path.append(visited[path[-1]][1])
    path.reverse()

    keep = [-1] * len(bit_symbols)
    for index in range(1, len(path)):
        state = unpack(path[index - 1])
        next_state = unpack(path[index])
        _, minimum, maximum, _ = state
        _, next_minimum, next_maximum, next_current = next_state
        if minimum != next_minimum and maximum != next_maximum:
            keep[next_minimum] = 1
            keep[maximum] = 1
        elif minimum != next_minimum:
            keep[next_minimum] = 1 if bit_symbols[next_minimum][0] == next_current else 0
        else:
            keep[maximum] = 1 if bit_symbols[maximum][0] == next_current else 0

    relabel: list[bool] = []
    symbol_index = 0
    for value in closest:
        if int(symbols[symbol_index][0]) != value:
            symbol_index += 1
        relabel.append(keep[symbol_index] == 0)

    if not any(not value for value in relabel):
        raise RuntimeError("closest flattening discarded all labels")

    def relabel_range(first: int, last: int) -> None:
        before = closest[first - 1 if first > 0 else len(closest) - 1]
        after = closest[last + 1 if last + 1 < len(closest) else 0]
        total = 0.0
        index = first
        while True:
            total = float(total + weights[index])
            if index == last:
                break
            index = index + 1 if index + 1 < len(closest) else 0
        weight_sum = 0.0
        index = first
        while True:
            if float(weight_sum + float(0.5 * weights[index])) < float(0.5 * total):
                closest[index] = before
            else:
                closest[index] = after
            relabel[index] = False
            weight_sum = float(weight_sum + weights[index])
            if index == last:
                break
            index = index + 1 if index + 1 < len(closest) else 0

    for seed in range(len(closest)):
        if not relabel[seed]:
            continue
        first = seed
        while relabel[first - 1 if first > 0 else len(closest) - 1]:
            first = first - 1 if first > 0 else len(closest) - 1
        last = seed
        while relabel[last + 1 if last + 1 < len(closest) else 0]:
            last = last + 1 if last + 1 < len(closest) else 0
        relabel_range(first, last)


def _balance_split_front_stitches(matches: Matches, front_slice_chains: list[list[int]]) -> None:
    front_segments = front_segments_from_matches(matches, len(front_slice_chains))
    for front_index, segments in enumerate(front_segments):
        if not segments:
            continue
        segments.sort(key=lambda segment: segment.begin)
        is_loop = (
            bool(front_slice_chains[front_index])
            and front_slice_chains[front_index][0] == front_slice_chains[front_index][-1]
        )
        if not (
            (segments[0].begin == 0.0 and segments[-1].end == 1.0)
            or (is_loop and segments[0].begin == segments[-1].end)
        ):
            raise RuntimeError("front match segments do not partition the chain")

        next_counts: dict[int, int] = {}
        for segment in segments:
            next_counts[segment.next] = next_counts.get(segment.next, 0) + 1
        singles = sum(1 for count in next_counts.values() if count == 1)
        doubles = sum(1 for count in next_counts.values() if count == 2)
        multis = sum(1 for count in next_counts.values() if count > 2)
        if singles == 1 and doubles == 0 and multis == 0:
            continue
        if singles == 2 and doubles == 0 and multis == 0:
            continue
        if not (singles == 2 and multis == 0):
            raise RuntimeError("unhandled split situation")

        for index, segment in enumerate(segments):
            if next_counts[segment.next] == 1:
                segments[:] = segments[index:] + segments[:index]
                break

        if len(segments) % 2 != 0 or len(segments) < 4:
            raise RuntimeError("unexpected split segment layout")

        middle = (len(segments) // 2) // 2
        opposite = len(segments) - middle
        while len(segments[middle].stitches) > len(segments[opposite].stitches):
            segments[middle + 1].stitches.insert(0, segments[middle].stitches.pop())
            if len(segments[middle].stitches) > len(segments[opposite].stitches):
                segments[middle - 1].stitches.append(segments[middle].stitches.pop(0))
        while len(segments[opposite].stitches) > len(segments[middle].stitches):
            segments[opposite - 1].stitches.append(segments[opposite].stitches.pop(0))
            if len(segments[opposite].stitches) > len(segments[middle].stitches):
                opposite_next = opposite + 1 if opposite + 1 < len(segments) else 0
                segments[opposite_next].stitches.insert(0, segments[opposite].stitches.pop())

        for left in range(middle - 1, 0, -1):
            right = len(segments) - left
            right_next = right + 1 if right + 1 < len(segments) else 0
            while len(segments[left].stitches) > len(segments[right].stitches):
                segments[left - 1].stitches.append(segments[left].stitches.pop(0))
            while len(segments[right].stitches) > len(segments[left].stitches):
                segments[right_next].stitches.insert(0, segments[right].stitches.pop())

        for right_forward in range(middle + 1, len(segments) // 2):
            right = len(segments) - right_forward
            while len(segments[right_forward].stitches) > len(segments[right].stitches):
                segments[right_forward + 1].stitches.insert(
                    0, segments[right_forward].stitches.pop()
                )
            while len(segments[right].stitches) > len(segments[right_forward].stitches):
                segments[right - 1].stitches.append(segments[right].stitches.pop(0))
