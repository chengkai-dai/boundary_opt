"""Scan harmonic boundary optimization from random initializations."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from boundary_opt import (
    DEFAULT_AREA_WEIGHT,
    DEFAULT_LENGTH_SMOOTHNESS_WEIGHT,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MINIMUM_GAP,
    DEFAULT_UNIFORMITY_WEIGHT,
    BoundaryOptimizer,
    load_obj,
    random_knots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh",
        choices=("disk", "plane", "peak", "triple_peak"),
        default="disk",
    )
    parser.add_argument("--backend", choices=("slsqp", "spg"), default="slsqp")
    parser.add_argument("--seeds", type=int, default=16, help="scan seeds [0, N)")
    parser.add_argument("--iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--minimum-gap", type=float, default=DEFAULT_MINIMUM_GAP)
    parser.add_argument(
        "--uniformity-weight", type=float, default=DEFAULT_UNIFORMITY_WEIGHT
    )
    parser.add_argument("--area-weight", type=float, default=DEFAULT_AREA_WEIGHT)
    parser.add_argument(
        "--length-smoothness-weight",
        type=float,
        default=DEFAULT_LENGTH_SMOOTHNESS_WEIGHT,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--history-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds < 1:
        raise SystemExit("--seeds must be positive")
    start = time.perf_counter()
    mesh = load_obj(Path("data") / f"{args.mesh}.obj")
    optimizer = BoundaryOptimizer(
        mesh,
        minimum_gap=args.minimum_gap,
        uniformity_weight=args.uniformity_weight,
        area_weight=args.area_weight,
        length_smoothness_weight=args.length_smoothness_weight,
    )
    setup_seconds = time.perf_counter() - start
    print(
        f"mesh={args.mesh} V={len(mesh.vertices)} F={len(mesh.faces)} "
        f"boundary={len(optimizer.harmonic.boundary_vertices)} "
        f"setup={setup_seconds:.4f}s"
    )

    rows: list[dict[str, int | float | str]] = []
    history_rows: list[dict[str, int | float]] = []
    for seed in range(args.seeds):
        initial = random_knots(seed, args.minimum_gap)
        start = time.perf_counter()
        result = optimizer.optimize(
            initial,
            backend=args.backend,
            max_iterations=args.iterations,
            seed=seed,
        )
        optimize_seconds = time.perf_counter() - start
        reduction = 1.0 - result.final_loss / result.initial_loss
        row: dict[str, int | float | str] = {
            "mesh": args.mesh,
            "backend": args.backend,
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "boundary_vertices": len(optimizer.harmonic.boundary_vertices),
            "setup_seconds": setup_seconds,
            "seed": seed,
            "minimum_gap": args.minimum_gap,
            "uniformity_weight": args.uniformity_weight,
            "area_weight": args.area_weight,
            "length_smoothness_weight": args.length_smoothness_weight,
            "initial_loss": result.initial_loss,
            "final_loss": result.final_loss,
            "uniformity_loss": result.uniformity_loss,
            "area_loss": result.area_loss,
            "length_smoothness_loss": result.length_smoothness_loss,
            "reduction": reduction,
            "iterations": result.iterations,
            "evaluations": result.evaluations,
            "gradient_norm": result.gradient_norm,
            "kkt_residual": result.kkt_residual,
            "constraint_violation": result.constraint_violation,
            "optimize_seconds": optimize_seconds,
            "spacing_cv": result.statistics.spacing_cv,
            "minimum_gradient": result.statistics.minimum_gradient,
            "maximum_gradient": result.statistics.maximum_gradient,
            "plateau_at_minimum": int(
                min(result.gaps[0], result.gaps[2]) <= args.minimum_gap + 1.0e-4
            ),
        }
        row.update(
            {
                f"knot_{index}": float(value % 1.0)
                for index, value in enumerate(result.knots)
            }
        )
        row.update(
            {f"gap_{index}": float(value) for index, value in enumerate(result.gaps)}
        )
        rows.append(row)
        history_rows.extend(
            {
                "seed": seed,
                "record": record,
                "loss": float(loss),
            }
            for record, loss in enumerate(result.history)
        )
        print(
            f"backend={args.backend} seed={seed:02d} loss "
            f"{result.initial_loss:.6f} -> {result.final_loss:.6f} "
            f"({100.0 * reduction:6.2f}%) uniform={result.uniformity_loss:.6f} "
            f"area={result.area_loss:.6f} "
            f"length_smoothness={result.length_smoothness_loss:.6f} "
            f"cv={result.statistics.spacing_cv:.4f} "
            f"iter={result.iterations:02d} kkt={result.kkt_residual:.2e} "
            f"time={optimize_seconds:.4f}s"
        )

    output = args.output or Path("output") / (
        f"{args.mesh}_{args.backend}_seed_scan.csv"
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
    best = int(np.argmin(final))
    improved = int(np.sum(final < initial))
    print(
        f"saved {output} and {history_output}; improved={improved}/{len(rows)}, "
        f"median_reduction={100.0 * float(np.median(1.0 - final / initial)):.2f}%, "
        f"median_time={float(np.median(timings)):.4f}s, "
        f"best_seed={rows[best]['seed']} best_loss={final[best]:.6f}"
    )


if __name__ == "__main__":
    main()
