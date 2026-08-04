# Harmonic boundary optimization

Standalone four-parameter optimization for any connected triangle mesh with
exactly one manifold boundary loop. The surface may be planar or curved; the
implementation uses 3D boundary arc length, cotangent stiffness, and intrinsic
face gradients. It has no dependency on the adjacent knitting project.

The ordered cyclic knots define a P1-compatible boundary profile:

```text
theta0 -- 0 plateau -- theta1 -- linear rise -- theta2
theta2 -- 1 plateau -- theta3 -- linear fall -- theta0 + 1
```

`HarmonicBoundaryOptimizer` factorizes the fixed interior Laplacian once. Each
objective evaluation solves the harmonic extension, and each exact gradient
uses one adjoint backsolve. The field loss is the area-weighted coefficient of
variation of `|grad u|^2`, a scale-independent proxy for uniform physical
spacing of harmonic isolines.

This is an all-boundary Dirichlet model: the two plateaus are exactly 0 and 1,
while the other two boundary arcs vary linearly in normalized arc length. This
reproduces affine fields exactly on planar rectangles when the four knots are
the corners. The objective and analytic gradient are smooth while the knots
remain in the same boundary-edge cells and piecewise smooth when a knot crosses
a boundary vertex. Closed meshes and meshes with multiple boundary loops are
rejected explicitly.

The theoretical default has no target-width prior. A width prior remains
available through `--target-arc-width` and `--width-weight`, but it changes the
mathematical optimum and is therefore opt-in.

Run the checks and a deterministic multi-start scan:

```bash
uv run pytest
uv run python scan_mesh_seeds.py --mesh data/triple_peak.obj --seeds 16
uv run python plot_loss_curves.py \
  --history Plane output/plane_seed_scan_history.csv \
  --history Peak output/peak_seed_scan_history.csv \
  --history Peak2 output/peak2_seed_scan_history.csv \
  --history Triple-peak output/triple_peak_seed_scan_history.csv \
  --svg output/mesh_loss_curves.svg
uv run --extra visualization python visualize_mesh_optimization.py \
  --mesh data/triple_peak.obj
uv run --extra visualization python visualize_mesh_optimization.py \
  --mesh data/disk.obj --final-only --show
uv run --extra visualization python visualize_mesh_optimization.py \
  --mesh data/triple_peak.obj --animate --fps 8
```

The scanner writes summary and per-iteration CSV files and records setup and
per-seed optimization time. By default the Polyscope command writes a labeled
before/after PNG; `--final-only` instead shows the optimized mesh in a normal
Polyscope panel. Add `--show` to keep a static view open. `--animate` opens a
looping comparison of the initial state and every accepted L-BFGS iteration.
