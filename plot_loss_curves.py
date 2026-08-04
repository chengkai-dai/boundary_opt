"""Plot one or more random-seed loss-history CSV files as SVG."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from xml.sax.saxutils import escape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        nargs=2,
        action="append",
        metavar=("LABEL", "CSV"),
        help="repeat for each panel",
    )
    parser.add_argument("--title", default="Harmonic boundary optimization")
    parser.add_argument("--svg", type=Path, default=Path("output/loss_curves.svg"))
    parser.add_argument("--html", type=Path)
    return parser.parse_args()


def read_histories(path: Path) -> dict[int, list[float]]:
    histories: defaultdict[int, list[tuple[int, float]]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            histories[int(row["seed"])].append(
                (int(row["iteration"]), float(row["loss"]))
            )
    return {
        seed: [value for _, value in sorted(points)]
        for seed, points in sorted(histories.items())
    }


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def chart_svg(
    panels: list[tuple[str, dict[int, list[float]]]],
    *,
    title: str,
    themed: bool,
) -> str:
    width, height = max(760, 390 * len(panels) + 90), 470
    left, right, top, bottom, gap = 66, 24, 38, 58, 58
    panel_width = (width - left - right - gap * (len(panels) - 1)) / len(panels)
    plot_height = height - top - bottom
    maximum_iteration = max(
        len(history) - 1 for _, data in panels for history in data.values()
    )
    maximum_iteration = max(10, int(math.ceil(maximum_iteration / 10.0) * 10))
    all_values = [
        value for _, data in panels for history in data.values() for value in history
    ]
    if not all_values or not all(math.isfinite(value) for value in all_values):
        raise ValueError("loss histories must contain finite values")
    positive_values = [value for value in all_values if value > 0.0]
    display_floor = min(positive_values, default=1.0e-15) / 10.0
    display_ceiling = max(max(all_values), display_floor * 10.0)
    minimum_exponent = math.floor(math.log10(display_floor)) - 1
    maximum_exponent = math.ceil(math.log10(display_ceiling)) + 1
    candidate_ticks = [
        value
        for exponent in range(minimum_exponent, maximum_exponent + 1)
        for multiplier in (1, 3)
        if (value := multiplier * 10**exponent) > 0.0
    ]
    lower = max(value for value in candidate_ticks if value <= display_floor)
    upper = min(value for value in candidate_ticks if value >= display_ceiling)
    y_ticks = [value for value in candidate_ticks if lower <= value <= upper]
    x_ticks = list(range(0, maximum_iteration + 1, 10))

    colors = (
        {
            "text": "var(--foreground)",
            "muted": "var(--muted-foreground)",
            "grid": "var(--border)",
            "peer": "var(--muted-foreground)",
            "median": "var(--viz-series-1)",
        }
        if themed
        else {
            "text": "#202124",
            "muted": "#667085",
            "grid": "#d9dee7",
            "peer": "#98a2b3",
            "median": "#2563eb",
        }
    )

    def x_coordinate(iteration: int, panel: int) -> float:
        return (
            left
            + panel * (panel_width + gap)
            + panel_width * iteration / maximum_iteration
        )

    def y_coordinate(value: float) -> float:
        value = max(value, display_floor)
        fraction = (math.log10(value) - math.log10(lower)) / (
            math.log10(upper) - math.log10(lower)
        )
        return top + plot_height * (1.0 - fraction)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="width:100%;height:auto;display:block" role="img" aria-labelledby="loss-title loss-desc">',
        f'<title id="loss-title">{escape(title)} loss curves</title>',
        '<desc id="loss-desc">Random-seed optimization histories; thin lines are runs and the thick line is the median. The vertical axis is logarithmic.</desc>',
    ]
    if not themed:
        parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append('<g fill="none" stroke-linecap="round" stroke-linejoin="round">')
    for panel, (label, histories) in enumerate(panels):
        x0 = left + panel * (panel_width + gap)
        for tick in y_ticks:
            y = y_coordinate(tick)
            parts.append(
                f'<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x0 + panel_width:.2f}" y2="{y:.2f}" stroke="{colors["grid"]}" stroke-width="1"/>'
            )
            if panel == 0:
                parts.append(
                    f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" fill="{colors["muted"]}" font-size="12">{tick:g}</text>'
                )
        for tick in x_ticks:
            x = x_coordinate(tick, panel)
            parts.append(
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="{colors["grid"]}" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{x:.2f}" y="{top + plot_height + 22}" text-anchor="middle" fill="{colors["muted"]}" font-size="12">{tick}</text>'
            )

        for history in histories.values():
            points = [
                (x_coordinate(iteration, panel), y_coordinate(value))
                for iteration, value in enumerate(history)
            ]
            parts.append(
                f'<polyline points="{polyline(points)}" stroke="{colors["peer"]}" stroke-opacity="0.32" stroke-width="1.2"/>'
            )

        median_history = [
            median(
                history[min(iteration, len(history) - 1)]
                for history in histories.values()
            )
            for iteration in range(maximum_iteration + 1)
        ]
        median_points = [
            (x_coordinate(iteration, panel), y_coordinate(value))
            for iteration, value in enumerate(median_history)
        ]
        parts.append(
            f'<polyline points="{polyline(median_points)}" stroke="{colors["median"]}" stroke-width="3"/>'
        )
        parts.extend(
            (
                f'<text x="{x0:.2f}" y="22" fill="{colors["text"]}" font-size="15" font-weight="500">{escape(label)}</text>',
                f'<text x="{x0 + panel_width - 4:.2f}" y="{y_coordinate(median_history[-1]) - 9:.2f}" text-anchor="end" fill="{colors["text"]}" font-size="12">median {max(median_history[-1], 0.0):.4f}</text>',
            )
        )

    parts.extend(
        (
            f'<text x="{width / 2:.2f}" y="{height - 8}" text-anchor="middle" fill="{colors["text"]}" font-size="13">Accepted L-BFGS iteration</text>',
            f'<text x="16" y="{top + plot_height / 2:.2f}" text-anchor="middle" transform="rotate(-90 16 {top + plot_height / 2:.2f})" fill="{colors["text"]}" font-size="13">Total loss · log scale</text>',
            "</g></svg>",
        )
    )
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    requested = args.history or [
        ("Disk · width prior", "output/disk_seed_scan_history.csv"),
        ("Disk · no width prior", "output/disk_seed_scan_unregularized_history.csv"),
    ]
    panels = [(label, read_histories(Path(path))) for label, path in requested]
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.svg.write_text(
        chart_svg(panels, title=args.title, themed=False), encoding="utf-8"
    )
    if args.html is not None:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        fragment = (
            '<div id="mesh-loss-benchmark" style="width:100%;color:var(--foreground)">\n'
            + chart_svg(panels, title=args.title, themed=True)
            + "\n</div>\n"
        )
        args.html.write_text(fragment, encoding="utf-8")
    print(f"wrote {args.svg}" + (f" and {args.html}" if args.html else ""))


if __name__ == "__main__":
    main()
