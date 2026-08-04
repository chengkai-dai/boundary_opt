"""Visualize a harmonic boundary optimization in Polyscope."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from boundary_opt import (
    HarmonicBoundaryOptimizer,
    cyclic_arc_edge_weights,
    knots_from_parameters,
    load_obj,
    random_knots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=Path("data/disk.obj"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--minimum-gap", type=float, default=0.03)
    parser.add_argument("--target-arc-width", type=float, default=None)
    parser.add_argument("--width-weight", type=float, default=0.0)
    parser.add_argument("--boundary-penalty", type=float, default=100.0)
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--show", action="store_true", help="open the interactive viewer"
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="show only the optimized result with standard Polyscope panels",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="loop over recorded optimizer states in the viewer",
    )
    parser.add_argument("--fps", type=float, default=8.0)
    return parser.parse_args()


def add_field_quantity(surface: object, field: np.ndarray) -> None:
    surface.add_scalar_quantity(
        "canonical harmonic field",
        field,
        cmap="viridis",
        vminmax=(0.0, 1.0),
        enabled=True,
        isolines_enabled=True,
        isoline_period=0.1,
        isoline_darkness=0.7,
        isoline_contour_thickness=0.25,
        onscreen_colorbar_enabled=False,
    )


def register_surface(
    ps: object,
    optimizer: HarmonicBoundaryOptimizer,
    label: str,
    vertices: np.ndarray,
    field: np.ndarray,
) -> object:
    surface = ps.register_surface_mesh(
        label,
        vertices,
        optimizer.mesh.faces,
        smooth_shade=False,
        edge_width=0.25,
        edge_color=(0.32, 0.34, 0.38),
        back_face_policy="identical",
    )
    add_field_quantity(surface, field)
    return surface


def register_state(
    ps: object,
    optimizer: HarmonicBoundaryOptimizer,
    display_vertices: np.ndarray,
    label: str,
    field: np.ndarray,
    boundary_colors: np.ndarray,
    offset: np.ndarray,
) -> tuple[object, object]:
    vertices = display_vertices + offset
    surface = register_surface(ps, optimizer, label, vertices, field)
    boundary_points = vertices[optimizer.boundary_vertices]
    boundary = ps.register_curve_network(
        f"{label} · boundary targets",
        boundary_points,
        "loop",
        radius=0.009,
    )
    boundary.add_color_quantity(
        "boundary state",
        boundary_colors,
        defined_on="edges",
        enabled=True,
    )
    boundary.set_material("flat")
    return surface, boundary


def boundary_edge_colors(
    zero_weights: np.ndarray, one_weights: np.ndarray
) -> np.ndarray:
    """Blend edge colors by exact zero-arc, one-arc, and free coverage."""
    free_weights = np.clip(1.0 - zero_weights - one_weights, 0.0, 1.0)
    return (
        free_weights[:, None] * np.asarray((0.45, 0.47, 0.50))
        + zero_weights[:, None] * np.asarray((0.08, 0.28, 0.95))
        + one_weights[:, None] * np.asarray((0.95, 0.24, 0.08))
    )


def optimization_frames(
    optimizer: HarmonicBoundaryOptimizer, parameter_history: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate harmonic fields and boundary colors at recorded optimizer states."""
    fields = []
    colors = []
    for parameters in parameter_history:
        knots, _, _ = knots_from_parameters(parameters)
        field, minimum_weights, maximum_weights = optimizer.field_and_arc_weights(knots)
        fields.append(field)
        colors.append(boundary_edge_colors(minimum_weights, maximum_weights))
    return np.stack(fields), np.stack(colors)


def principal_axis_view(vertices: np.ndarray) -> np.ndarray:
    """Rigidly align the two dominant mesh axes with screen x/z."""
    centered = vertices - vertices.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    return centered @ np.stack((axes[0], axes[2], axes[1]), axis=1)


def rgba8(red: int, green: int, blue: int, alpha: int = 255) -> int:
    """Pack an RGBA color for Dear ImGui's ImU32 representation."""
    return red | (green << 8) | (blue << 16) | (alpha << 24)


def draw_label(
    psim: object,
    center_x: float,
    title: str,
    subtitle: str,
    accent: int,
) -> None:
    draw = psim.GetForegroundDrawList()
    font = psim.GetFont()
    default_font_size = psim.GetFontSize()
    panel_min = (center_x - 170.0, 24.0)
    panel_max = (center_x + 170.0, 94.0)
    draw.AddRectFilled(panel_min, panel_max, rgba8(255, 255, 255, 235), 8.0)
    draw.AddRect(
        panel_min,
        panel_max,
        rgba8(148, 163, 184),
        8.0,
        thickness=1.0,
    )
    draw.AddRectFilled(
        panel_min,
        (panel_min[0] + 7.0, panel_max[1]),
        accent,
        8.0,
    )
    for line, y, size in ((title, 34.0, 22.0), (subtitle, 65.0, 16.0)):
        text_width = psim.CalcTextSize(line)[0] * size / default_font_size
        draw.AddText(
            font,
            size,
            (center_x - 0.5 * text_width, y),
            rgba8(31, 41, 55),
            line,
        )


def make_label_callback(
    ps: object,
    psim: object,
    initial_loss: float,
    optimized_loss: float,
):
    """Return a screen-space overlay which makes left/right unambiguous."""

    def draw_labels() -> None:
        width, _ = ps.get_window_size()
        draw_label(
            psim,
            0.25 * width,
            "BEFORE / INITIAL",
            f"loss = {initial_loss:.6f}",
            rgba8(71, 85, 105),
        )
        draw_label(
            psim,
            0.75 * width,
            "AFTER / OPTIMIZED",
            f"loss = {optimized_loss:.6f}",
            rgba8(5, 150, 105),
        )

    return draw_labels


def make_animation_callback(
    ps: object,
    psim: object,
    field_buffer: object,
    color_buffer: object,
    fields: np.ndarray,
    colors: np.ndarray,
    losses: np.ndarray,
    fps: float,
):
    """Return an auto-looping recorded-state playback callback."""
    hold = max(1, round(fps))
    sequence = np.concatenate(
        (
            np.zeros(hold, dtype=np.int64),
            np.arange(1, len(fields), dtype=np.int64),
            np.full(hold, len(fields) - 1, dtype=np.int64),
        )
    )
    state = {"elapsed": 0.0, "step": 0, "frame": 0}
    period = 1.0 / fps

    def animate() -> None:
        state["elapsed"] += min(float(psim.GetIO().DeltaTime), 0.1)
        steps = int(state["elapsed"] / period)
        if steps:
            state["elapsed"] %= period
            state["step"] = (state["step"] + steps) % len(sequence)
            frame = int(sequence[state["step"]])
            if frame != state["frame"]:
                field_buffer.update_data(fields[frame])
                color_buffer.update_data(colors[frame])
                state["frame"] = frame

        width, _ = ps.get_window_size()
        draw_label(
            psim,
            0.25 * width,
            "BEFORE / INITIAL",
            f"loss = {losses[0]:.6f}",
            rgba8(71, 85, 105),
        )
        draw_label(
            psim,
            0.75 * width,
            f"STATE {state['frame']} / {len(fields) - 1}",
            f"loss = {losses[state['frame']]:.6f}",
            rgba8(5, 150, 105),
        )

    return animate


def main() -> None:
    args = parse_args()
    if args.animate and args.final_only:
        raise SystemExit("--animate and --final-only cannot be used together")
    if args.animate and (not np.isfinite(args.fps) or not 0.0 < args.fps <= 60.0):
        raise SystemExit("--fps must lie in (0, 60]")
    try:
        import polyscope as ps
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Polyscope is optional; run with `uv run --extra visualization python "
            "visualize_mesh_optimization.py`."
        ) from exc

    mesh = load_obj(args.mesh)
    optimizer = HarmonicBoundaryOptimizer(
        mesh,
        minimum_gap=args.minimum_gap,
        target_arc_width=args.target_arc_width,
        width_weight=args.width_weight,
        boundary_penalty=args.boundary_penalty,
    )
    initial_knots = random_knots(args.seed, args.minimum_gap)
    initial_field, initial_zero, initial_one = optimizer.field_and_arc_weights(
        initial_knots
    )
    initial_colors = boundary_edge_colors(initial_zero, initial_one)
    result = optimizer.optimize(
        initial_knots,
        max_iterations=args.iterations,
    )

    display_vertices = principal_axis_view(mesh.vertices)
    extent = float(np.ptp(display_vertices[:, [0, 2]], axis=0).max())
    # With this camera, screen-right points toward world -x. Translate the
    # optimized mesh in -x so the screenshot reads initial -> optimized.
    translation = np.asarray([-1.25 * extent, 0.0, 0.0])
    center = 0.5 * (display_vertices.min(axis=0) + display_vertices.max(axis=0))
    target = center + 0.5 * translation

    ps.init()
    ps.set_program_name("Harmonic boundary optimization")
    ps.set_window_size(1600, 800)
    ps.set_ground_plane_mode("none")
    ps.set_background_color((0.96, 0.97, 0.98))
    ps.set_up_dir("z_up")
    ps.set_view_projection_mode("orthographic")
    ps.set_SSAA_factor(2)
    ps.set_build_default_gui_panels(args.final_only)
    ps.set_open_imgui_window_for_user_callback(False)

    import polyscope.imgui as psim

    if args.animate:
        fields, colors = optimization_frames(optimizer, result.parameter_history)
        register_state(
            ps,
            optimizer,
            display_vertices,
            f"initial · seed {args.seed}",
            fields[0],
            colors[0],
            np.zeros(3),
        )
        surface, boundary = register_state(
            ps,
            optimizer,
            display_vertices,
            "optimization trajectory",
            fields[0],
            colors[0],
            translation,
        )
        field_buffer = surface.get_quantity_buffer("canonical harmonic field", "values")
        color_buffer = boundary.get_quantity_buffer("boundary state", "colors")
        ps.look_at(target + np.asarray([0.0, 5.5 * extent, 2.0 * extent]), target)
        ps.set_user_callback(
            make_animation_callback(
                ps,
                psim,
                field_buffer,
                color_buffer,
                fields,
                colors,
                result.history,
                args.fps,
            )
        )
        print(
            f"seed={args.seed} loss {result.initial_loss:.6f} -> "
            f"{result.final_loss:.6f}; playing {len(fields)} recorded states"
        )
        ps.show()
        return

    optimized_zero, optimized_one = cyclic_arc_edge_weights(
        optimizer.boundary_positions, result.knots
    )
    optimized_colors = boundary_edge_colors(optimized_zero, optimized_one)
    if args.final_only:
        register_state(
            ps,
            optimizer,
            display_vertices,
            "optimized",
            result.field,
            optimized_colors,
            np.zeros(3),
        )
        ps.look_at(center + np.asarray([0.0, 5.5 * extent, 2.0 * extent]), center)
    else:
        register_state(
            ps,
            optimizer,
            display_vertices,
            f"before · seed {args.seed}",
            initial_field,
            initial_colors,
            np.zeros(3),
        )
        register_state(
            ps,
            optimizer,
            display_vertices,
            "after",
            result.field,
            optimized_colors,
            translation,
        )
        ps.look_at(target + np.asarray([0.0, 5.5 * extent, 2.0 * extent]), target)
        ps.set_user_callback(
            make_label_callback(
                ps,
                psim,
                result.initial_loss,
                result.final_loss,
            )
        )

    screenshot = args.screenshot or Path("output") / (
        f"{args.mesh.stem}_{'optimized' if args.final_only else 'before_after'}_polyscope.png"
    )
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    ps.show(forFrames=5)
    ps.screenshot(str(screenshot), transparent_bg=False, include_UI=True)
    print(
        f"seed={args.seed} loss {result.initial_loss:.6f} -> {result.final_loss:.6f}; "
        f"wrote {screenshot}"
    )
    if args.show:
        ps.show()


if __name__ == "__main__":
    main()
