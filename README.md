# Harmonic boundary optimization

Standalone four-parameter optimization for a connected, nondegenerate triangle
mesh with exactly one manifold boundary loop and at least one interior vertex.
The surface may be planar or curved; the implementation uses 3D boundary arc
length, cotangent stiffness, and intrinsic face gradients. It has no dependency
on the adjacent knitting project.

For a from-scratch explanation of this linear all-boundary model, see
[`LINEAR_ALL_BOUNDARY_TUTORIAL.md`](LINEAR_ALL_BOUNDARY_TUTORIAL.md). The
separate continuous partial-Dirichlet and Wentzell model is documented in
[`ALGORITHM_TUTORIAL.md`](ALGORITHM_TUTORIAL.md).

The ordered cyclic knots define a P1-compatible boundary profile:

```text
theta0 -- 0 plateau -- theta1 -- linear rise -- theta2
theta2 -- 1 plateau -- theta3 -- linear fall -- theta0 + 1
```

`HarmonicBoundaryOptimizer` factorizes the fixed interior Laplacian once. Each
objective evaluation solves the harmonic extension, and each exact gradient
uses one adjoint backsolve. The field loss is the area-weighted squared
coefficient of variation of `|grad u|^2`, a scale-independent proxy for uniform
physical spacing of harmonic isolines.

This is an all-boundary Dirichlet model: the ideal profile has 0/1 plateaus and
two normalized-arc-length linear transitions. It is sampled at the original
boundary vertices, then interpolated by the mesh's P1 boundary edges; no knot
vertex is inserted inside an edge. This reproduces affine fields exactly on
planar rectangles when the four knots are corner vertices. The objective and
analytic gradient are smooth while the mean gradient energy is positive and the
knots remain in the same boundary-edge cells, and piecewise smooth when a knot
crosses a boundary vertex. Closed meshes and meshes with multiple boundary loops
are rejected explicitly.

The theoretical default has no target-width prior. A width prior remains
available to this all-boundary model through `--target-arc-width` and
`--width-weight`, but it changes the mathematical optimum and is therefore
opt-in.

The outer variables are the physical coordinates `(offset, g0, g1, g2)`, with
`g3 = 1 - g0 - g1 - g2`. Exact-gradient SLSQP enforces the closed simplex
`g_i >= minimum_gap`, so active gap constraints are attainable and are checked
with a projected KKT residual. Two complementary `u`/`1-u` charts are solved
to remove the coordinate bias introduced by choosing `g3` as the implicit gap.
Every nondegenerate feasible objective evaluation, including line-search
trials, can update a local incumbent; the two charts' incumbents are then
compared. If neither chart materially improves the original loss, the
canonicalized original start is returned. The public history keeps the selected
chart's nondegenerate feasible callback states in raw, non-monotone order. The
returned incumbent is appended when it is not already last, so plots and
animation finish at the returned field; this final record is not necessarily a
new SLSQP iteration.
If a very coarse boundary sampling produces a constant field, the CV objective
is undefined (`0/0`); only at that degenerate trial the solver uses a finite
directional recovery penalty. Every nonconstant feasible field still uses the
original loss and exact analytic gradient.
If the initial field itself is degenerate, `initial_loss` and `history[0]`
therefore report this recovery surrogate because the original CV loss does not
exist there.

## Hard partial-Dirichlet alternative

[`continuous_partial_opt.py`](continuous_partial_opt.py) is a separate reference
implementation with different PDE semantics. Its zero and one arcs are hard
Dirichlet constraints; all remaining vertices, including boundary vertices,
are variational unknowns. With the default `eta=0`, unconstrained boundary arcs
receive the natural zero-flux Neumann condition instead of a prescribed
transition trace.

The four endpoints remain continuous normalized arc-length coordinates. Each
affected triangle is integrated as a geometry-defined centroid fan. Endpoint
values are eliminated and the free centroid is Schur-condensed during local
assembly, so neither the original `V/F` arrays nor the global unknown vector is
ever enlarged. The construction is invariant to cyclic vertex renumbering.

```bash
uv run python continuous_partial_opt.py \
  --mesh data/disk.obj --seed 0 --iterations 60 --starts 2 \
  --output output/disk_partial.npz

uv run --extra visualization python visualize_continuous_partial.py \
  --mesh data/disk.obj --seed 0 --iterations 60 --starts 1 --show
```

This hard mixed Dirichlet/Neumann problem is smooth only away from mesh
vertices, snap thresholds, and other active-set changes. The reference outer
solve uses constrained SLSQP finite differences and multi-start initialization;
it is not an analytic-gradient pipeline and does not claim a global optimum.
Its loss values should not be compared as if they came from the all-boundary
linear-trace model: on the current Disk mesh they are about `0.693` versus about
`0.028`, chiefly because mixed-boundary junctions create strong local gradients.
A boundary-turning-angle candidate directly identifies Plane's theoretical
four corners and gives numerical zero loss.

### Wentzell boundary smoothing

Passing `--boundary-smoothing eta` adds the scale-invariant energy below to the
inner state solve:

```text
eta / 2 * integral_0^1 |du/dxi|^2 dxi
```

Here `xi` is normalized boundary arc length. If `s` is physical arc length and
`P` is the perimeter, the equivalent physical coefficient is `beta = eta P`,
and the free-boundary equation is
`normal_derivative(u) - beta * second_s_derivative(u) = 0`. The energy is zero
on the hard constant arcs. Intermediate boundary values are solved by the
coupled variational problem; no transition profile is supplied. `eta=0`
exactly recovers pure Neumann, while large `eta` approaches the automatically
selected piecewise-linear trace.

This boundary energy is not added to the reported outer objective. The outer
loss remains the area-weighted `CV^2(|grad u|^2)` of the solved bulk field, so
all loss numbers below keep the same meaning.

On the current meshes, `eta=3` gives Disk loss `0.030688` and keeps Plane at
numerical zero. Run and inspect that result with:

```bash
uv run python continuous_partial_opt.py \
  --mesh data/disk.obj --seed 0 --iterations 60 --starts 1 \
  --boundary-smoothing 3 --output output/wentzell_eta3_disk_seed0.npz

uv run --extra visualization python visualize_continuous_partial.py \
  --mesh data/disk.obj --seed 0 --iterations 60 --starts 1 \
  --boundary-smoothing 3 --show
```

Run the checks and a deterministic linear Disk scan:

```bash
uv run pytest
uv run python scan_mesh_seeds.py \
  --mesh data/disk.obj --seeds 16 --iterations 100 \
  --output output/closed_simplex_disk_seed_scan.csv \
  --history-output output/closed_simplex_disk_seed_scan_history.csv
uv run python plot_loss_curves.py \
  --history Disk output/closed_simplex_disk_seed_scan_history.csv \
  --svg output/closed_simplex_disk_loss_curves.svg
uv run --extra visualization python visualize_mesh_optimization.py \
  --mesh data/triple_peak.obj
uv run --extra visualization python visualize_mesh_optimization.py \
  --mesh data/disk.obj --final-only --show
uv run --extra visualization python visualize_mesh_optimization.py \
  --mesh data/triple_peak.obj --animate --fps 8
```

The scanner writes summary and per-history-record CSV files and records setup and
per-seed optimization time. By default the Polyscope command writes a labeled
before/after PNG; `--final-only` instead shows the optimized mesh in a normal
Polyscope panel. Add `--show` to keep a static view open. `--animate` plays the
public feasible-state record and ends at the returned best feasible field.
