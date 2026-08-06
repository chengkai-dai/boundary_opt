import numpy as np

from knitting import KnittingGraph, PeelingConfig
from knitting._peeling.contours import extract_level_chains
from knitting._peeling.planning.matching import _flatten_closest


def test_level_chain_keeps_higher_values_on_its_left() -> None:
    vertices = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    )
    faces = np.asarray(((0, 1, 2), (0, 2, 3)))
    values = vertices[:, 0]

    [chain] = extract_level_chains(faces, values, 0.5)
    points = np.asarray([point.interpolate(vertices) for point in chain])

    assert points[0, 1] > points[-1, 1]


def test_peeling_spacing_must_be_finite_and_positive() -> None:
    for spacing in (0.0, -1.0, np.nan):
        try:
            PeelingConfig(course_spacing=spacing)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid spacing {spacing}")


def test_flatten_preserves_an_already_flat_open_chain() -> None:
    closest = [0, 0, 1, 1]

    _flatten_closest(closest, [1.0] * len(closest), is_loop=False)

    assert closest == [0, 0, 1, 1]


def test_knitting_graph_derives_course_and_shaping_counts() -> None:
    graph = KnittingGraph(
        points=np.zeros((6, 3)),
        course_edges=np.asarray(((0, 1), (2, 3), (4, 5))),
        wale_edges=np.asarray(((0, 2), (1, 2), (3, 4), (3, 5))),
    )

    assert graph.course_count == 3
    assert graph.increase_count == 1
    assert graph.decrease_count == 1
