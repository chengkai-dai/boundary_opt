"""Scan harmonic boundary optimization from random initializations."""

from __future__ import annotations

import argparse
import csv
import time
from itertools import product
from pathlib import Path

import numpy as np

from boundary_opt import HarmonicBoundaryOptimizer, Mesh, load_obj, random_knots


def relative_reduction(initial: float, final: float) -> float:
    return 0.0 if initial <= 0.0 else 1.0 - final / initial


def uniformly_refined(mesh: Mesh, levels: int) -> Mesh:
    """Split every triangle into four by shared edge midpoints."""
    for _ in range(levels):
        faces = mesh.faces
        edges = np.concatenate(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])
        )
        edges.sort(axis=1)
        unique_edges, inverse = np.unique(edges, axis=0, return_inverse=True)
        midpoint_indices = len(mesh.vertices) + np.arange(len(unique_edges))
        midpoint01, midpoint12, midpoint20 = np.split(
            midpoint_indices[inverse], 3
        )
        vertex0, vertex1, vertex2 = faces.T
        mesh = Mesh(
            np.vstack(
                (
                    mesh.vertices,
                    mesh.vertices[unique_edges].mean(axis=1),
                )
            ),
            np.vstack(
                (
                    np.column_stack((vertex0, midpoint01, midpoint20)),
                    np.column_stack((midpoint01, vertex1, midpoint12)),
                    np.column_stack((midpoint20, midpoint12, vertex2)),
                    np.column_stack((midpoint01, midpoint12, midpoint20)),
                )
            ),
        )
    return mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh", type=Path, nargs="+", default=[Path("data/disk.obj")]
    )
    parser.add_argument("--seeds", type=int, default=16, help="scan seeds [0, N)")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--minimum-gap", type=float, default=0.03)
    parser.add_argument("--target-arc-width", type=float, default=None)
    parser.add_argument("--width-weight", type=float, default=0.0)
    parser.add_argument(
        "--boundary-penalty", type=float, nargs="+", default=[100.0]
    )
    parser.add_argument(
        "--refinements",
        type=int,
        nargs="+",
        default=[0],
        help="uniform 1-to-4 triangle subdivision levels",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--history-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds < 1:
        raise SystemExit("--seeds must be positive")
    if min(args.refinements) < 0:
        raise SystemExit("--refinements must be non-negative")

    configurations = list(product(args.mesh, args.boundary_penalty, args.refinements))
    rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    for mesh_path, boundary_penalty, refinement_level in configurations:
        start = time.perf_counter()
        mesh = uniformly_refined(load_obj(mesh_path), refinement_level)
        optimizer = HarmonicBoundaryOptimizer(
            mesh,
            minimum_gap=args.minimum_gap,
            target_arc_width=args.target_arc_width,
            width_weight=args.width_weight,
            boundary_penalty=boundary_penalty,
        )
        setup_seconds = time.perf_counter() - start
        print(
            f"mesh={mesh_path} refinement={refinement_level} "
            f"V={len(mesh.vertices)} F={len(mesh.faces)} "
            f"boundary={len(optimizer.boundary_vertices)} "
            f"penalty={optimizer.boundary_penalty:g} setup={setup_seconds:.4f}s"
        )

        for seed in range(args.seeds):
            initial = random_knots(seed, args.minimum_gap)
            start = time.perf_counter()
            result = optimizer.optimize(
                initial,
                max_iterations=args.iterations,
            )
            optimize_seconds = time.perf_counter() - start
            reduction = relative_reduction(result.initial_loss, result.final_loss)
            boundary = result.boundary_statistics
            field_min = float(result.field.min())
            field_max = float(result.field.max())
            run = len(rows)
            row: dict[str, object] = {
                "run": run,
                "mesh": str(mesh_path),
                "refinement_level": refinement_level,
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "boundary_vertices": len(optimizer.boundary_vertices),
                "setup_seconds": setup_seconds,
                "seed": seed,
                "minimum_gap": args.minimum_gap,
                "target_arc_width": args.target_arc_width,
                "width_weight": args.width_weight,
                "boundary_penalty": boundary_penalty,
                "initial_loss": result.initial_loss,
                "final_loss": result.final_loss,
                "uniformity_loss": result.uniformity_loss,
                "width_loss": result.width_loss,
                "reduction": reduction,
                "iterations": result.iterations,
                "evaluations": result.evaluations,
                "success": int(result.success),
                "constraint_violation": result.constraint_violation,
                "kkt_residual": result.kkt_residual,
                "minimum_projected_hessian_eigenvalue": (
                    result.minimum_projected_hessian_eigenvalue
                ),
                "optimize_seconds": optimize_seconds,
                "gradient_cv": result.statistics.gradient_cv,
                "spacing_cv": result.statistics.spacing_cv,
                "minimum_gradient": result.statistics.minimum_gradient,
                "maximum_gradient": result.statistics.maximum_gradient,
                "canonical_field_min": field_min,
                "canonical_field_max": field_max,
                "canonical_overshoot": max(0.0, -field_min, field_max - 1.0),
                "raw_zero_mean": boundary.raw_zero_mean,
                "raw_one_mean": boundary.raw_one_mean,
                "raw_span": boundary.raw_span,
                "raw_zero_target_rms": boundary.raw_zero_target_rms,
                "raw_one_target_rms": boundary.raw_one_target_rms,
                "canonical_zero_target_rms": boundary.canonical_zero_target_rms,
                "canonical_one_target_rms": boundary.canonical_one_target_rms,
                "target_arc_at_minimum": int(
                    min(result.gaps[0], result.gaps[2])
                    <= args.minimum_gap + 1.0e-4
                ),
                "any_gap_at_minimum": int(
                    result.gaps.min() <= args.minimum_gap + 1.0e-4
                ),
                "message": result.message,
            }
            row.update(
                {
                    f"knot_{index}": float(value % 1.0)
                    for index, value in enumerate(result.knots)
                }
            )
            row.update(
                {
                    f"gap_{index}": float(value)
                    for index, value in enumerate(result.gaps)
                }
            )
            rows.append(row)
            history_rows.extend(
                {
                    "run": run,
                    "mesh": str(mesh_path),
                    "refinement_level": refinement_level,
                    "boundary_penalty": boundary_penalty,
                    "seed": seed,
                    "recorded_state": state,
                    "loss": float(loss),
                }
                for state, loss in enumerate(result.history)
            )
            print(
                f"seed={seed:02d} loss {result.initial_loss:.6f} -> "
                f"{result.final_loss:.6f} ({100.0 * reduction:6.2f}%) "
                f"uniform={result.uniformity_loss:.6f} width={result.width_loss:.6f} "
                f"spacing_cv={result.statistics.spacing_cv:.4f} "
                f"span={boundary.raw_span:.4f} iter={result.iterations:02d} "
                f"time={optimize_seconds:.4f}s kkt={result.kkt_residual:.1e} "
                f"hessian_min={result.minimum_projected_hessian_eigenvalue:.3g} "
                f"success={result.success}"
            )

    output = args.output or Path("output") / (
        f"{args.mesh[0].stem}_seed_scan.csv"
        if len(configurations) == 1
        else "mesh_penalty_refinement_sweep.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    history_output = args.history_output or output.with_name(
        f"{output.stem}_history.csv"
    )
    history_output.parent.mkdir(parents=True, exist_ok=True)
    with history_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history_rows[0]))
        writer.writeheader()
        writer.writerows(history_rows)

    initial = np.asarray([float(row["initial_loss"]) for row in rows])
    final = np.asarray([float(row["final_loss"]) for row in rows])
    timings = np.asarray([float(row["optimize_seconds"]) for row in rows])
    reductions = np.asarray(
        [relative_reduction(before, after) for before, after in zip(initial, final)]
    )
    best = int(np.argmin(final))
    improved = int(np.sum(final < initial))
    succeeded = int(sum(int(row["success"]) for row in rows))
    print(
        f"saved {output} and {history_output}; success={succeeded}/{len(rows)}, "
        f"improved={improved}/{len(rows)}, "
        f"median_reduction={100.0 * float(np.median(reductions)):.2f}%, "
        f"final_p90={float(np.quantile(final, 0.9)):.6f}, "
        f"final_max={float(final.max()):.6f}, "
        f"median_time={float(np.median(timings)):.4f}s, "
        f"best_mesh={rows[best]['mesh']} "
        f"best_refinement={rows[best]['refinement_level']} "
        f"best_penalty={float(rows[best]['boundary_penalty']):g} "
        f"best_seed={rows[best]['seed']} best_loss={final[best]:.6f}"
    )


if __name__ == "__main__":
    main()
