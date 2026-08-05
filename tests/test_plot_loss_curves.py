import re

import numpy as np

from plot_loss_curves import chart_svg, read_histories


def test_read_npz_history(tmp_path) -> None:
    path = tmp_path / "optimization.npz"
    np.savez(path, history=np.asarray([2.0, 0.5, 0.25]))

    assert read_histories(path) == {0: [2.0, 0.5, 0.25]}


def test_chart_handles_near_zero_plane_loss() -> None:
    svg = chart_svg(
        [("Plane", {0: [12.0, 1.0e-8, 6.0e-14, 0.0, -1.0e-16]})],
        title="Plane patch test",
        themed=False,
    )

    assert "Plane patch test" in svg
    assert "<polyline" in svg
    assert "median 0.0000" in svg


def test_short_panel_stops_at_its_last_record() -> None:
    svg = chart_svg(
        [("Long", {0: [3.0, 2.0, 1.0]}), ("Short", {0: [3.0, 1.0]})],
        title="Different lengths",
        themed=False,
    )

    median_lines = re.findall(r'<polyline points="([^"]+)" stroke="#2563eb"', svg)
    assert [len(line.split()) for line in median_lines] == [3, 2]
