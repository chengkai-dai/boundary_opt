# Differentiable harmonic boundary optimization

Standalone four-parameter optimization for any connected triangle mesh with
exactly one manifold boundary loop. The surface may be planar or curved; the
implementation uses 3D boundary arc length, cotangent stiffness, and intrinsic
face gradients. It has no dependency on the adjacent knitting project.

The ordered cyclic knots define a C2 boundary profile:

```text
theta0 -- 0 plateau -- theta1 -- smooth rise -- theta2
theta2 -- 1 plateau -- theta3 -- smooth fall -- theta0 + 1
```

`HarmonicBoundaryOptimizer` factorizes the fixed interior Laplacian once. Each
objective evaluation solves the harmonic extension, and each exact gradient
uses one adjoint backsolve. The field loss is the area-weighted coefficient of
variation of `|grad u|^2`, a smooth scale-independent proxy for uniform physical
spacing of harmonic isolines.

This is an all-boundary Dirichlet model: the two plateaus are exactly 0 and 1,
while the rest of the boundary receives smooth transition values. Closed meshes
and meshes with multiple boundary loops are rejected explicitly.

With spacing uniformity as the only objective, the optimizer tends to shrink
both constant plateaus to `minimum_gap`. The examples therefore use a target arc
width of `0.10` and weight `0.10`. Pass `--width-weight 0` to inspect the
unregularized solution.

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
per-seed optimization time. The Polyscope command writes a labeled before/after
PNG; add `--show` to keep the static comparison open. `--animate` opens a
looping comparison of the initial state and every accepted L-BFGS iteration.
