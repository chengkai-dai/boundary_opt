"""Optimize a harmonic field, peel it, and show the knitting graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from geometry import load_obj, normalize_mesh
from workflow import run_pipeline

ROOT = Path(__file__).resolve().parent
MESH_CHOICES = ("disk", "plane", "peak", "triple_peak")
DEFAULT_MESH = "disk"
DEFAULT_NORMALIZATION_SCALE = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", choices=MESH_CHOICES, default=DEFAULT_MESH)
    parser.add_argument("--scale", type=float, default=DEFAULT_NORMALIZATION_SCALE)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main(mesh_name: str, scale: float, *, show: bool = True) -> None:
    mesh = normalize_mesh(load_obj(ROOT / "data" / f"{mesh_name}.obj"), scale)
    print(f"optimizing {mesh_name} at scale {scale:g}...", flush=True)
    try:
        run = run_pipeline(
            mesh,
            progress=lambda step: print(
                f"\rpeeling: step {step}", end="", flush=True
            ),
        )
    finally:
        print()

    boundary = run.boundary
    graph = run.peeling.graph
    print(
        f"boundary loss: {boundary.initial_loss:.6f} -> {boundary.final_loss:.6f}\n"
        f"final components: uniformity {boundary.uniformity_loss:.6f}, "
        f"length smoothness {boundary.length_smoothness_loss:.6f}\n"
        f"knitting graph: {graph.course_count} courses, "
        f"{len(graph.points)} stitches, {len(graph.wale_edges)} wale edges\n"
        f"peeling: {run.peeling.finish_reason}"
    )
    if show:
        from visualize_pipeline import show_pipeline

        show_pipeline(run)


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments.mesh, arguments.scale, show=not arguments.no_show)
