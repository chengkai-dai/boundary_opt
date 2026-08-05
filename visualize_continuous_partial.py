"""Show the final continuous partial-Dirichlet solution in Polyscope."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from boundary_opt import load_obj, random_knots
from continuous_partial_opt import ContinuousPartialBoundaryOptimizer

ENDPOINT_VALUES = np.asarray([0.0, 0.0, 1.0, 1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=Path("data/disk.obj"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--starts", type=int, default=1)
    parser.add_argument("--minimum-gap", type=float, default=0.03)
    parser.add_argument("--boundary-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=Path("output/continuous_partial_disk_final_polyscope.png"),
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def display_mesh(
    optimizer: ContinuousPartialBoundaryOptimizer,
    knots: np.ndarray,
    field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the exact local fan solely for rendering the solved field."""
    assembly = optimizer._assemble(knots)
    original_count = len(optimizer.mesh.vertices)
    unaffected = np.ones(len(optimizer.mesh.faces), dtype=bool)
    unaffected[assembly.affected_faces] = False

    centers = np.asarray(
        [patch.center for patch in assembly.local_patches], dtype=np.float64
    ).reshape(-1, 3)
    vertices = np.vstack((optimizer.mesh.vertices, assembly.endpoint_points, centers))
    values = [*field, *ENDPOINT_VALUES]
    faces = optimizer.mesh.faces[unaffected].tolist()
    center_offset = original_count + 4
    for patch_index, patch in enumerate(assembly.local_patches):
        boundary_values = np.asarray(
            [
                field[reference]
                if reference < original_count
                else ENDPOINT_VALUES[reference - original_count]
                for reference in patch.boundary_references
            ]
        )
        values.append(float(patch.center_weights @ boundary_values))
        center = center_offset + patch_index
        for index, reference in enumerate(patch.boundary_references):
            following = patch.boundary_references[
                (index + 1) % len(patch.boundary_references)
            ]
            faces.append((reference, following, center))
    return vertices, np.asarray(faces, dtype=np.int64), np.asarray(values)


def principal_transform(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = vertices.mean(axis=0)
    _, _, axes = np.linalg.svd(vertices - center, full_matrices=False)
    transform = np.stack((axes[0], axes[2], axes[1]), axis=1)
    return center, transform


def arc_polyline(
    boundary_points: np.ndarray,
    boundary_positions: np.ndarray,
    start: float,
    end: float,
    start_point: np.ndarray,
    end_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    width = float((end - start) % 1.0)
    local = np.mod(boundary_positions - start, 1.0)
    selected = np.flatnonzero((local > 1.0e-12) & (local < width - 1.0e-12))
    selected = selected[np.argsort(local[selected])]
    points = np.vstack((start_point, boundary_points[selected], end_point))
    edges = np.column_stack(
        (np.arange(len(points) - 1), np.arange(1, len(points)))
    ).astype(np.int64)
    return points, edges


def register_arc(
    ps: object,
    name: str,
    points: np.ndarray,
    edges: np.ndarray,
    color: tuple[float, float, float],
    radius: float,
) -> None:
    curve = ps.register_curve_network(name, points, edges, color=color, radius=radius)
    curve.set_material("flat")


def main() -> None:
    args = parse_args()
    try:
        import polyscope as ps
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Run with `uv run --extra visualization python "
            "visualize_continuous_partial.py`."
        ) from exc

    optimizer = ContinuousPartialBoundaryOptimizer(
        load_obj(args.mesh),
        minimum_gap=args.minimum_gap,
        boundary_smoothing=args.boundary_smoothing,
    )
    result = optimizer.optimize(
        random_knots(args.seed, args.minimum_gap),
        max_iterations=args.iterations,
        starts=args.starts,
    )
    vertices, faces, values = display_mesh(optimizer, result.knots, result.field)
    center, transform = principal_transform(optimizer.mesh.vertices)
    display_vertices = (vertices - center) @ transform
    boundary_points = display_vertices[optimizer.boundary_vertices]
    endpoint_points = (result.endpoint_points - center) @ transform
    extent = float(np.ptp(display_vertices[:, [0, 2]], axis=0).max())

    ps.init()
    ps.set_program_name(
        f"Continuous partial-Dirichlet field · eta={args.boundary_smoothing:g}"
    )
    ps.set_window_size(1500, 900)
    ps.set_ground_plane_mode("none")
    ps.set_background_color((0.96, 0.97, 0.98))
    ps.set_up_dir("z_up")
    ps.set_view_projection_mode("orthographic")
    ps.set_SSAA_factor(2)
    ps.set_build_default_gui_panels(True)

    surface = ps.register_surface_mesh(
        "Harmonic field u",
        display_vertices,
        faces,
        smooth_shade=False,
        edge_width=0.0,
        back_face_policy="identical",
    )
    # Polyscope's coolwarm is blue-low/red-high, so display 1-u to honor the
    # project convention red u=0 / blue u=1. Isolines are unchanged.
    surface.add_scalar_quantity(
        "red u=0 · blue u=1",
        1.0 - values,
        cmap="coolwarm",
        vminmax=(0.0, 1.0),
        enabled=True,
        isolines_enabled=True,
        isoline_period=0.1,
        isoline_darkness=0.65,
        isoline_contour_thickness=0.2,
        onscreen_colorbar_enabled=False,
    )
    surface.add_scalar_quantity(
        "numeric u",
        values,
        cmap="viridis",
        vminmax=(0.0, 1.0),
        enabled=False,
    )

    knots = result.knots
    free_condition = (
        "Wentzell solved" if args.boundary_smoothing > 0.0 else "natural Neumann"
    )
    arcs = (
        ("Gamma0 · prescribed u=0", 0, 1, (0.90, 0.12, 0.10), 0.011),
        (
            f"Gamma free · {free_condition} (rise side)",
            1,
            2,
            (0.35, 0.37, 0.41),
            0.005,
        ),
        ("Gamma1 · prescribed u=1", 2, 3, (0.08, 0.25, 0.92), 0.011),
        (
            f"Gamma free · {free_condition} (fall side)",
            3,
            0,
            (0.35, 0.37, 0.41),
            0.005,
        ),
    )
    for name, start, end, color, radius in arcs:
        end_knot = knots[end] if end else knots[0] + 1.0
        points, edges = arc_polyline(
            boundary_points,
            optimizer.boundary_positions,
            float(knots[start]),
            float(end_knot),
            endpoint_points[start],
            endpoint_points[end],
        )
        register_arc(ps, name, points, edges, color, radius)

    ps.register_point_cloud(
        "theta0 theta1 · u=0 junctions",
        endpoint_points[:2],
        color=(0.90, 0.12, 0.10),
        radius=0.016,
    )
    ps.register_point_cloud(
        "theta2 theta3 · u=1 junctions",
        endpoint_points[2:],
        color=(0.08, 0.25, 0.92),
        radius=0.016,
    )

    scene_center = 0.5 * (display_vertices.min(axis=0) + display_vertices.max(axis=0))
    ps.look_at(
        scene_center + np.asarray([0.0, 5.5 * extent, 2.0 * extent]),
        scene_center,
    )
    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    ps.show(forFrames=5)
    ps.screenshot(str(args.screenshot), transparent_bg=False, include_UI=True)
    print(
        f"eta={args.boundary_smoothing:g}; loss={result.final_loss:.9g}; "
        f"u=[{values.min():.6g}, "
        f"{values.max():.6g}]; wrote {args.screenshot}"
    )
    if args.show:
        ps.show()


if __name__ == "__main__":
    main()
