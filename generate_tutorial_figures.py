"""Generate the exact SVG figures used by the beginner algorithm guide."""

from __future__ import annotations

import math
from pathlib import Path

from boundary_opt import HarmonicBoundaryOptimizer, load_obj, random_knots
from plot_loss_curves import chart_svg

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "docs" / "figures"


def polar(cx: float, cy: float, radius: float, angle: float) -> tuple[float, float]:
    radians = math.radians(angle)
    return cx + radius * math.cos(radians), cy + radius * math.sin(radians)


def arc_path(cx: float, cy: float, radius: float, start: float, end: float) -> str:
    x0, y0 = polar(cx, cy, radius, start)
    x1, y1 = polar(cx, cy, radius, end)
    large = int(end - start > 180.0)
    return f"M {x0:.2f} {y0:.2f} A {radius} {radius} 0 {large} 1 {x1:.2f} {y1:.2f}"


def svg_page(width: int, height: int, body: str, title: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
<title>{title}</title>
<desc>{title}</desc>
<rect width="{width}" height="{height}" fill="#ffffff"/>
<style>
text {{ font-family: Inter, ui-sans-serif, system-ui, sans-serif; fill: #1f2937; }}
.label {{ font-size: 22px; }}
.small {{ font-size: 17px; fill: #64748b; }}
.title {{ font-size: 28px; font-weight: 600; }}
.tiny {{ font-size: 15px; fill: #64748b; }}
.math {{ font-family: Georgia, serif; font-style: italic; }}
.formula {{ font-family: "STIX Two Math", "Cambria Math", Georgia, serif; font-style: normal; }}
</style>
{body}
</svg>
"""


def boundary_parameterization_svg() -> str:
    cx, cy, radius = 310.0, 285.0, 205.0
    angles = (205.0, 315.0, 380.0, 490.0, 565.0)
    colors = ("#2563eb", "#7c8797", "#ef3e23", "#7c8797")
    labels = ("target 0", "free", "target 1", "free")
    arcs = []
    knots = []
    for index, (start, end, color, label) in enumerate(
        zip(angles, angles[1:], colors, labels)
    ):
        arcs.append(
            f'<path d="{arc_path(cx, cy, radius, start, end)}" fill="none" '
            f'stroke="{color}" stroke-width="18" stroke-linecap="round"/>'
        )
        mid_x, mid_y = polar(cx, cy, radius + 38.0, 0.5 * (start + end))
        arcs.append(
            f'<text x="{mid_x:.1f}" y="{mid_y:.1f}" text-anchor="middle" '
            f'class="small">{label}</text>'
        )
        x, y = polar(cx, cy, radius, start)
        label_x, label_y = polar(cx, cy, radius + 28.0, start)
        knots.extend(
            (
                (
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="9" fill="#ffffff" '
                    'stroke="#111827" stroke-width="3"/>'
                ),
                (
                    f'<text x="{label_x:.2f}" y="{label_y + 7:.2f}" '
                    f'text-anchor="middle" class="label math">k{index}</text>'
                ),
            )
        )

    mesh_lines = []
    for angle in range(0, 360, 30):
        x, y = polar(cx, cy, radius - 12.0, angle)
        mesh_lines.append(
            f'<line x1="{cx}" y1="{cy}" x2="{x:.2f}" y2="{y:.2f}" '
            'stroke="#cbd5e1" stroke-width="1"/>'
        )
    for inner_radius in (68.0, 136.0):
        mesh_lines.append(
            f'<circle cx="{cx}" cy="{cy}" r="{inner_radius}" fill="none" '
            'stroke="#cbd5e1" stroke-width="1"/>'
        )

    strip_x, strip_y, strip_w = 660.0, 212.0, 365.0
    fractions = (0.0, 0.24, 0.42, 0.72, 1.0)
    strip = [
        f'<text x="{strip_x}" y="145" class="label">Unwrap the boundary once</text>',
        f'<line x1="{strip_x}" y1="{strip_y}" x2="{strip_x + strip_w}" y2="{strip_y}" stroke="#d1d5db" stroke-width="14" stroke-linecap="round"/>',
    ]
    for left, right, color in zip(fractions, fractions[1:], colors):
        strip.append(
            f'<line x1="{strip_x + strip_w * left:.1f}" y1="{strip_y}" '
            f'x2="{strip_x + strip_w * right:.1f}" y2="{strip_y}" '
            f'stroke="{color}" stroke-width="14"/>'
        )
    for index, fraction in enumerate(fractions):
        x = strip_x + strip_w * fraction
        label = "k0 + 1" if index == 4 else f"k{index}"
        strip.extend(
            (
                f'<circle cx="{x:.1f}" cy="{strip_y}" r="7" fill="#ffffff" stroke="#111827" stroke-width="2"/>',
                f'<text x="{x:.1f}" y="{strip_y + 38}" text-anchor="middle" class="label math">{label}</text>',
            )
        )
    strip.extend(
        (
            f'<path d="M {strip_x} 328 H {strip_x + strip_w - 18}" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>',
            f'<text x="{strip_x + strip_w / 2}" y="360" text-anchor="middle" class="label math">normalized arc length s ∈ [0, 1)</text>',
            '<text x="660" y="430" class="small">Only the blue and red arcs receive target values.</text>',
            '<text x="660" y="462" class="small">Gray arcs remain natural Neumann boundaries.</text>',
        )
    )

    body = f"""
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#475569"/></marker></defs>
<circle cx="{cx}" cy="{cy}" r="{radius - 10}" fill="#f8fafc"/>
{"".join(mesh_lines)}
{"".join(arcs)}
{"".join(knots)}
<path d="{arc_path(cx, cy, radius + 8, 150, 195)}" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>
<text x="118" y="92" class="label math">s increases around the loop</text>
{"".join(strip)}
"""
    return svg_page(1100, 560, body, "Four cyclic knots and two target arcs")


def robin_boundary_svg() -> str:
    body = """
<defs>
  <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#475569"/></marker>
</defs>
<text x="600" y="42" text-anchor="middle" class="title">同一个边界点：三种条件分别固定什么？</text>

<rect x="35" y="68" width="350" height="300" rx="18" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="210" y="105" text-anchor="middle" class="label">Dirichlet · 硬固定</text>
<line x1="80" y1="164" x2="340" y2="164" stroke="#f59e0b" stroke-width="3" stroke-dasharray="8 7"/>
<text x="92" y="151" class="small math">target c</text>
<path d="M 72 270 Q 185 238 278 164" fill="none" stroke="#0f9da8" stroke-width="14" stroke-linecap="round"/>
<circle cx="278" cy="164" r="13" fill="#0f9da8" stroke="#0f5f66" stroke-width="3"/>
<line x1="278" y1="135" x2="278" y2="193" stroke="#f97316" stroke-width="9"/>
<line x1="259" y1="135" x2="297" y2="135" stroke="#f97316" stroke-width="6"/>
<line x1="259" y1="193" x2="297" y2="193" stroke="#f97316" stroke-width="6"/>
<text x="210" y="310" text-anchor="middle" class="label formula">u = c</text>
<text x="210" y="340" text-anchor="middle" class="small">数值被钉死；法向通量由解决定</text>

<rect x="425" y="68" width="350" height="300" rx="18" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="600" y="105" text-anchor="middle" class="label">Neumann · 固定通量</text>
<path d="M 462 270 Q 575 235 668 208" fill="none" stroke="#0f9da8" stroke-width="14" stroke-linecap="round"/>
<circle cx="668" cy="208" r="13" fill="#0f9da8" stroke="#0f5f66" stroke-width="3"/>
<line x1="690" y1="208" x2="738" y2="208" stroke="#475569" stroke-width="3" marker-end="url(#arrow)"/>
<text x="715" y="194" text-anchor="middle" class="small math">normal flux g</text>
<text x="600" y="310" text-anchor="middle" class="label formula">∂ₙu = g</text>
<text x="600" y="340" text-anchor="middle" class="small">通量被规定；边界数值 u 仍由 PDE 求</text>

<rect x="815" y="68" width="350" height="300" rx="18" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="990" y="105" text-anchor="middle" class="label">Robin · 软弹簧</text>
<line x1="855" y1="154" x2="1130" y2="154" stroke="#f59e0b" stroke-width="5"/>
<text x="866" y="140" class="small math">target c</text>
<path d="M 852 270 Q 948 244 1054 220" fill="none" stroke="#0f9da8" stroke-width="14" stroke-linecap="round"/>
<circle cx="1054" cy="220" r="13" fill="#0f9da8" stroke="#0f5f66" stroke-width="3"/>
<path d="M 1054 166 l -13 5 l 26 8 l -26 8 l 26 8 l -26 8 l 13 5" fill="none" stroke="#f97316" stroke-width="5" stroke-linejoin="round"/>
<text x="1076" y="194" class="tiny math">spring κ</text>
<text x="990" y="310" text-anchor="middle" class="label formula">∂ₙu + κ(u−c) = 0</text>
<text x="990" y="340" text-anchor="middle" class="small">膜的通量与弹簧拉力平衡；u 没被钉死</text>

<line x1="35" y1="395" x2="1165" y2="395" stroke="#cbd5e1"/>
<rect x="35" y="420" width="390" height="255" rx="18" fill="#fff7ed" stroke="#fed7aa"/>
<text x="230" y="455" text-anchor="middle" class="label">一维可精确求解的小例子</text>
<text x="70" y="488" class="label formula">u″ = 0</text>
<text x="70" y="521" class="label formula">−u′(0)+κu(0) = 0</text>
<text x="70" y="554" class="label formula">u′(1)+κ[u(1)−1] = 0</text>
<text x="70" y="594" class="label formula">u(x) = [κx+1]/[κ+2]</text>
<text x="70" y="627" class="small">κ → 0⁺：趋向常数 1/2；κ=0 时不唯一</text>
<text x="70" y="657" class="small">κ → ∞：趋向 u=x，也就是 Dirichlet</text>

<text x="795" y="430" text-anchor="middle" class="label">弹簧越硬，两端越接近 target 0 与 1</text>
<line x1="500" y1="650" x2="1135" y2="650" stroke="#334155" stroke-width="3" marker-end="url(#arrow)"/>
<line x1="500" y1="650" x2="500" y2="447" stroke="#334155" stroke-width="3" marker-end="url(#arrow)"/>
<text x="1142" y="674" class="small math">x</text>
<text x="475" y="452" class="small math">u</text>
<text x="500" y="675" text-anchor="middle" class="tiny">0</text>
<text x="1120" y="675" text-anchor="middle" class="tiny">1</text>
<text x="480" y="654" text-anchor="end" class="tiny">0</text>
<text x="480" y="455" text-anchor="end" class="tiny">1</text>
<line x1="500" y1="550" x2="1120" y2="550" stroke="#e2e8f0" stroke-width="2"/>
<text x="480" y="555" text-anchor="end" class="tiny">0.5</text>
<line x1="500" y1="550" x2="1120" y2="550" stroke="#94a3b8" stroke-width="5"/>
<line x1="500" y1="583" x2="1120" y2="517" stroke="#2563eb" stroke-width="5"/>
<line x1="500" y1="633" x2="1120" y2="467" stroke="#f59e0b" stroke-width="5"/>
<line x1="500" y1="650" x2="1120" y2="450" stroke="#dc2626" stroke-width="4" stroke-dasharray="10 7"/>
<text x="1128" y="555" class="tiny">κ → 0⁺</text>
<text x="1128" y="522" class="tiny" fill="#2563eb">κ = 1</text>
<text x="1128" y="472" class="tiny" fill="#b45309">κ = 10</text>
<text x="1128" y="445" class="tiny" fill="#dc2626">κ → ∞</text>
"""
    return svg_page(
        1200,
        710,
        body,
        "Dirichlet, Neumann, and Robin boundary conditions with a one-dimensional example",
    )


def moving_arc_mass_svg() -> str:
    x0, x1, y = 100.0, 560.0, 355.0
    a, b = 235.0, 455.0
    basis_left = f"M {x0} {y} L {x0} 95 L {x1} {y}"
    basis_right = f"M {x0} {y} L {x1} 95 L {x1} {y}"
    body = f"""
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#475569"/></marker></defs>
<text x="90" y="48" class="label">Exact P1 integration on one boundary edge</text>
<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#334155" stroke-width="4"/>
<rect x="{a}" y="78" width="{b - a}" height="{y - 78}" fill="#fbbf24" opacity="0.18"/>
<path d="{basis_left}" fill="none" stroke="#2563eb" stroke-width="4"/>
<path d="{basis_right}" fill="none" stroke="#ef3e23" stroke-width="4"/>
<line x1="{a}" y1="72" x2="{a}" y2="{y + 12}" stroke="#b45309" stroke-width="3" stroke-dasharray="7 6"/>
<line x1="{b}" y1="72" x2="{b}" y2="{y + 12}" stroke="#b45309" stroke-width="3" stroke-dasharray="7 6"/>
<circle cx="{x0}" cy="{y}" r="7" fill="#111827"/><circle cx="{x1}" cy="{y}" r="7" fill="#111827"/>
<text x="{x0}" y="{y + 38}" text-anchor="middle" class="math label">i</text>
<text x="{x1}" y="{y + 38}" text-anchor="middle" class="math label">i+1</text>
<text x="{a}" y="63" text-anchor="middle" class="math label">a</text>
<text x="{b}" y="63" text-anchor="middle" class="math label">b</text>
<text x="145" y="120" class="math label" fill="#2563eb">φᵢ</text>
<text x="505" y="120" class="math label" fill="#ef3e23">φᵢ₊₁</text>
<text x="82" y="455" class="math label">Q(a,b) = ∫ₐᵇ φ(s) φ(s)ᵀ ds</text>

<line x1="650" y1="355" x2="1020" y2="355" stroke="#334155" stroke-width="4"/>
<path d="M 650 355 L 835 95 L 1020 355" fill="none" stroke="#7c3aed" stroke-width="4"/>
<circle cx="650" cy="355" r="7" fill="#111827"/><circle cx="835" cy="355" r="9" fill="#111827"/><circle cx="1020" cy="355" r="7" fill="#111827"/>
<path d="M 735 252 H 925" stroke="#475569" stroke-width="2" marker-end="url(#arrow)"/>
<circle cx="792" cy="155" r="9" fill="#f59e0b"/><circle cx="878" cy="155" r="9" fill="#f59e0b"/>
<text x="835" y="48" text-anchor="middle" class="label">Endpoint crosses a mesh vertex</text>
<text x="835" y="402" text-anchor="middle" class="math label">φ(s) stays continuous</text>
<text x="835" y="445" text-anchor="middle" class="math label">∂Q/∂a = −φ(a)φ(a)ᵀ</text>
<text x="835" y="482" text-anchor="middle" class="math label">∂Q/∂b = +φ(b)φ(b)ᵀ</text>
"""
    return svg_page(
        1100, 520, body, "Exact moving arc mass and continuous endpoint derivative"
    )


def loss_uniformity_svg() -> str:
    def contours(x: float, uneven: bool) -> str:
        ys = (100, 135, 190, 285, 335) if uneven else (105, 160, 215, 270, 325)
        paths = []
        for index, yy in enumerate(ys):
            bend = (index - 2) * 18 if uneven else 0
            paths.append(
                f'<path d="M {x} {yy} Q {x + 170} {yy + bend} {x + 340} {yy}" '
                'fill="none" stroke="#0f766e" stroke-width="5"/>'
            )
        return "".join(paths)

    body = f"""
<text x="255" y="48" text-anchor="middle" class="label">Uneven contour spacing</text>
<text x="845" y="48" text-anchor="middle" class="label">Uniform contour spacing</text>
<rect x="75" y="70" width="360" height="300" rx="18" fill="#f8fafc" stroke="#cbd5e1"/>
<rect x="665" y="70" width="360" height="300" rx="18" fill="#f8fafc" stroke="#cbd5e1"/>
{contours(85, True)}
{contours(675, False)}
<line x1="470" y1="120" x2="560" y2="120" stroke="#ef4444" stroke-width="4"/><line x1="470" y1="170" x2="620" y2="170" stroke="#ef4444" stroke-width="4"/><line x1="470" y1="220" x2="525" y2="220" stroke="#ef4444" stroke-width="4"/><line x1="470" y1="270" x2="640" y2="270" stroke="#ef4444" stroke-width="4"/>
<text x="552" y="315" text-anchor="middle" class="small">different |∇u|</text>
<line x1="1045" y1="120" x2="1090" y2="120" stroke="#2563eb" stroke-width="4"/><line x1="1045" y1="170" x2="1090" y2="170" stroke="#2563eb" stroke-width="4"/><line x1="1045" y1="220" x2="1090" y2="220" stroke="#2563eb" stroke-width="4"/><line x1="1045" y1="270" x2="1090" y2="270" stroke="#2563eb" stroke-width="4"/>
<text x="1068" y="315" text-anchor="middle" class="small">equal |∇u|</text>
<text x="550" y="425" text-anchor="middle" class="math label">L = Σf w_f (q_f / q̄ − 1)²,   q_f = |∇u_f|²,   q̄ = Σf w_f q_f</text>
<text x="550" y="470" text-anchor="middle" class="small">A smooth local-spacing surrogate: L = 0 exactly when every triangle has equal |∇u|.</text>
"""
    return svg_page(1120, 510, body, "Uniformity loss measures contour spacing")


def loss_history_svg() -> str:
    optimizer = HarmonicBoundaryOptimizer(load_obj(ROOT / "data" / "disk.obj"))
    histories = {
        seed: optimizer.optimize(
            random_knots(seed), max_iterations=100
        ).history.tolist()
        for seed in range(8)
    }
    return chart_svg(
        [("Disk · 8 random starts", histories)],
        title="Disk boundary optimization",
        themed=False,
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figures = {
        "disk-boundary-parameters.svg": boundary_parameterization_svg(),
        "robin-boundary-explained.svg": robin_boundary_svg(),
        "moving-arc-mass.svg": moving_arc_mass_svg(),
        "loss-uniformity.svg": loss_uniformity_svg(),
        "disk-loss-history.svg": loss_history_svg(),
    }
    for name, content in figures.items():
        (FIGURES / name).write_text(content, encoding="utf-8")
        print(f"wrote {FIGURES / name}")


if __name__ == "__main__":
    main()
