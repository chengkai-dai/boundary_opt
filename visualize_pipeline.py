"""Polyscope viewer for the optimize-and-peel workflow."""

from __future__ import annotations

import numpy as np

from boundary_opt.boundary import boundary_arclength
from geometry import Mesh, boundary_loop
from workflow import PipelineResult


def _boundary_curve_points(mesh: Mesh, start: float, end: float) -> np.ndarray:
    loop = boundary_loop(mesh.faces)
    positions = boundary_arclength(mesh.vertices, loop)

    def point_at(position: float) -> np.ndarray:
        position %= 1.0
        left = int(np.searchsorted(positions, position, side="right") - 1)
        right = (left + 1) % len(loop)
        edge_end = positions[right] if right else 1.0
        mix = (position - positions[left]) / (edge_end - positions[left])
        return (1.0 - mix) * mesh.vertices[loop[left]] + mix * mesh.vertices[loop[right]]

    width = end - start
    local = np.mod(positions - start, 1.0)
    inside = np.flatnonzero((local > 0.0) & (local < width))
    inside = loop[inside[np.argsort(local[inside])]]
    return np.vstack((point_at(start), mesh.vertices[inside], point_at(end)))


def show_pipeline(run: PipelineResult) -> None:
    import polyscope as ps
    import polyscope.imgui as psim

    mesh = run.mesh
    boundary = run.boundary
    graph = run.peeling.graph

    ps.set_program_name("Optimized field → front peeling")
    ps.set_give_focus_on_show(True)
    ps.set_up_dir("y_up")
    ps.init()
    ps.set_window_size(1400, 850)
    ps.set_ground_plane_mode("none")
    ps.set_background_color((0.96, 0.97, 0.98))
    ps.set_navigation_style("turntable")
    ps.set_build_default_gui_panels(True)

    surface = ps.register_surface_mesh(
        "optimized harmonic field",
        mesh.vertices,
        mesh.faces,
        color=(0.70, 0.73, 0.77),
        edge_width=0.0,
        smooth_shade=True,
    )
    surface.add_scalar_quantity(
        "u",
        boundary.field,
        enabled=True,
        cmap="viridis",
        vminmax=(0.0, 1.0),
        isolines_enabled=False,
        isoline_style="contour",
        isoline_period=0.05,
    )

    for name, start, end, color in (
        (
            "min boundary curve · u = 0",
            boundary.knots[0],
            boundary.knots[1],
            (0.95, 0.24, 0.08),
        ),
        (
            "max boundary curve · u = 1",
            boundary.knots[2],
            boundary.knots[3],
            (0.08, 0.28, 0.95),
        ),
    ):
        points = _boundary_curve_points(mesh, start, end)
        edges = np.column_stack(
            (np.arange(len(points) - 1), np.arange(1, len(points)))
        )
        curve = ps.register_curve_network(
            name,
            points,
            edges,
            radius=0.0045,
            color=color,
        )
        curve.set_material("flat")

    front_points = mesh.vertices[run.initial_course]
    front_edges = np.column_stack(
        (np.arange(len(front_points) - 1), np.arange(1, len(front_points)))
    )
    ps.register_curve_network(
        "initial zero front",
        front_points,
        front_edges,
        radius=0.0030,
        color=(0.10, 0.75, 0.35),
    )
    ps.register_point_cloud(
        "stitches",
        graph.points,
        radius=0.0030,
        color=(0.92, 0.93, 0.95),
    )
    ps.register_curve_network(
        "courses",
        graph.points,
        graph.course_edges,
        radius=0.0016,
        color=(0.12, 0.30, 0.95),
    )
    ps.register_curve_network(
        "wales",
        graph.points,
        graph.wale_edges,
        radius=0.0012,
        color=(0.95, 0.30, 0.10),
    )

    def panel() -> None:
        psim.TextUnformatted("Boundary optimization")
        psim.Separator()
        psim.TextUnformatted(
            f"Total loss: {boundary.initial_loss:.6f} -> {boundary.final_loss:.6f}"
        )
        psim.TextUnformatted(f"Uniformity: {boundary.uniformity_loss:.6f}")
        psim.TextUnformatted(
            f"Length smoothness: {boundary.length_smoothness_loss:.6f}"
        )
        psim.TextUnformatted(f"Initial-course vertices: {len(run.initial_course)}")
        psim.TextUnformatted(f"Courses: {graph.course_count}")
        psim.TextUnformatted(f"Stitches: {len(graph.points)}")
        psim.TextUnformatted(f"Course edges: {len(graph.course_edges)}")
        psim.TextUnformatted(f"Wale edges: {len(graph.wale_edges)}")
        psim.TextUnformatted(
            f"Increases / decreases: {graph.increase_count} / {graph.decrease_count}"
        )
        psim.TextUnformatted(f"Peeling: {run.peeling.finish_reason}")

    ps.set_user_callback(panel)
    ps.show()


__all__ = ["show_pipeline"]
