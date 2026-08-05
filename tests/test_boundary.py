import numpy as np

from boundary_opt.boundary import (
    cyclic_boundary_profile,
    knots_from_parameters,
    parameters_from_knots,
    random_knots,
)


def test_boundary_profile_jacobian_matches_finite_difference() -> None:
    positions = np.linspace(0.0, 1.0, 101, endpoint=False)
    knots = np.asarray([0.07, 0.24, 0.58, 0.81])
    values, jacobian = cyclic_boundary_profile(positions, knots)
    assert values.min() == 0.0
    assert values.max() == 1.0
    step = 1.0e-6
    for index in range(4):
        direction = np.eye(4)[index]
        forward = cyclic_boundary_profile(positions, knots + step * direction)[0]
        backward = cyclic_boundary_profile(positions, knots - step * direction)[0]
        np.testing.assert_allclose(
            jacobian[:, index], (forward - backward) / (2.0 * step), atol=2.0e-7
        )


def test_centered_full_gap_round_trip_reaches_closed_simplex_face() -> None:
    minimum_gap = 0.03
    knots = np.asarray([0.11, 0.14, 0.61, 0.64])
    parameters = parameters_from_knots(knots, minimum_gap)
    restored, jacobian, gaps = knots_from_parameters(parameters, minimum_gap)
    np.testing.assert_allclose(gaps, [0.03, 0.47, 0.03, 0.47], atol=1.0e-15)
    np.testing.assert_allclose(restored, knots, atol=1.0e-15)
    assert jacobian.shape == (4, 5)
    assert parameters.shape == (5,)
    shifted = parameters_from_knots(knots + 3.0, minimum_gap)
    np.testing.assert_allclose(shifted, parameters, atol=1.0e-15)


def test_state_to_knots_jacobian_on_simplex_tangent() -> None:
    minimum_gap = 0.03
    parameters = parameters_from_knots(random_knots(7), minimum_gap)
    _, jacobian, _ = knots_from_parameters(parameters, minimum_gap)
    directions = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, -1.0],
        ]
    )
    step = 1.0e-6
    for direction in directions:
        forward = knots_from_parameters(parameters + step * direction, minimum_gap)[0]
        backward = knots_from_parameters(parameters - step * direction, minimum_gap)[0]
        np.testing.assert_allclose(
            jacobian @ direction,
            (forward - backward) / (2.0 * step),
            rtol=0.0,
            atol=2.0e-10,
        )


def test_state_to_knots_jacobian_off_simplex() -> None:
    minimum_gap = 0.03
    parameters = np.asarray([0.4, 0.2, 0.3, 0.4, 0.5])
    _, jacobian, _ = knots_from_parameters(
        parameters, minimum_gap, enforce_minimum_gap=False
    )
    step = 1.0e-6
    for direction in np.eye(5):
        forward = knots_from_parameters(
            parameters + step * direction,
            minimum_gap,
            enforce_minimum_gap=False,
        )[0]
        backward = knots_from_parameters(
            parameters - step * direction,
            minimum_gap,
            enforce_minimum_gap=False,
        )[0]
        np.testing.assert_allclose(
            jacobian @ direction,
            (forward - backward) / (2.0 * step),
            rtol=0.0,
            atol=2.0e-10,
        )
