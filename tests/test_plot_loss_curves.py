from plot_loss_curves import chart_svg


def test_chart_handles_near_zero_plane_loss() -> None:
    svg = chart_svg(
        [("Plane", {0: [12.0, 1.0e-8, 6.0e-14, 0.0, -1.0e-16]})],
        title="Plane patch test",
        themed=False,
    )

    assert "Plane patch test" in svg
    assert "<polyline" in svg
    assert "median 0.0000" in svg
