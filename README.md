# Harmonic boundary optimization

Clean four-knot optimization for a connected triangle mesh with one manifold
boundary loop. The standalone package depends only on NumPy and SciPy and
contains only the complete full-boundary model described below.

The four ordered cyclic knots define a complete Dirichlet trace:

```text
theta0 -- 0 plateau -- theta1 -- linear rise -- theta2
theta2 -- 1 plateau -- theta3 -- linear fall -- theta0 + 1
```

The interior field is the cotangent-harmonic extension of this trace. The loss
is the area-weighted `CV²(|grad u|²)`, optionally plus a plateau-width prior.
One prefactorization, one forward solve, and one adjoint solve provide the exact
gradient for each objective evaluation.

## Architecture

```text
boundary_opt/
  mesh.py             mesh I/O, boundary topology, cotangent FEM geometry
  harmonic.py         prefactorized harmonic solve and adjoint
  boundary.py         knots, gaps, boundary profile, coordinate conversion
  simplex.py          feasibility, projection, projected-gradient residual
  loss.py             uniformity loss, width prior, field statistics
  slsqp_backend.py    one constrained SciPy SLSQP solve
  spg_backend.py      spectral projected gradient + nonmonotone Armijo
  optimizer.py        objective assembly, evaluation cache, backend dispatch
  __init__.py         public API only
```

The filenames follow the mathematical roles instead of collecting unrelated
helpers in a generic utility module. Both solver files know nothing about
meshes or harmonic fields: they receive only a `value_and_grad` callable.

## Shared optimization coordinates

Both backends use the same centered-phase full-gap state

```text
x = (center, g0, g1, g2, g3)
g_i >= minimum_gap
sum(g) = 1
```

Although five coordinates are stored, the equality leaves four degrees of
freedom. `center` is the mean of the four unwrapped knots. This removes the
special implicit `g3` chart and makes the complement transformation a phase
translation plus a permutation of the four gaps.

SLSQP uses four bounds and one linear equality constraint. Its objective has a
symmetric off-equality extension, so trial evaluations remain well-defined.
SPG projects all four gaps directly onto the lower-bounded simplex; its line
search therefore remains feasible throughout.

## High-level API

```python
from boundary_opt import BoundaryOptimizer, load_obj, random_knots

optimizer = BoundaryOptimizer(load_obj("data/disk.obj"))
initial = random_knots(seed=0)

slsqp = optimizer.optimize(initial, backend="slsqp", max_iterations=100)
spg = optimizer.optimize(initial, backend="spg", max_iterations=100)

# Or run both from exactly the same physical initial knots.
results = optimizer.optimize_backends(initial, max_iterations=100)
best = min(results.values(), key=lambda result: result.final_loss)
```

`slsqp` remains the default for compatibility:

```python
result = optimizer.optimize(initial)
```

The public result always contains the optimized `knots`, four physical `gaps`,
harmonic `field`, loss history, KKT residual, and constraint violation. Internal
`parameters` and `parameter_history` use centered phase plus four gaps.

Migration note: before this refactor, the low-level parameter helpers used a
four-entry reduced chart. Calls to `parameters_from_knots`,
`knots_from_parameters`, or `loss_and_gradient` must now use the five stored
coordinates `(center, g0, g1, g2, g3)`. The high-level four-knot `optimize`
call is unchanged. Low-level helpers live in `boundary_opt.boundary`,
`boundary_opt.simplex`, `boundary_opt.loss`, and `boundary_opt.mesh`; the
package root exposes only the high-level API.

`optimize` returns only a converged backend endpoint. A degenerate objective or
failed backend raises an exception; it never substitutes a recovery state or a
transient iterate. SciPy's SLSQP stopping test is not the same as this package's
projected KKT diagnostic, so inspect `kkt_residual` when stationarity matters.

## Commands

Run tests:

```bash
uv run pytest
```

Scan either backend:

```bash
uv run python scan_mesh_seeds.py \
  --mesh data/disk.obj --backend slsqp --seeds 16 --iterations 100

uv run python scan_mesh_seeds.py \
  --mesh data/disk.obj --backend spg --seeds 16 --iterations 100
```

Open the standard Polyscope panel:

```bash
uv run --extra visualization python visualize_mesh_optimization.py \
  --mesh data/disk.obj --backend spg --seed 0 --iterations 100 \
  --final-only --show
```

Animate the recorded feasible iterates:

```bash
uv run --extra visualization python visualize_mesh_optimization.py \
  --mesh data/disk.obj --backend slsqp --animate --fps 8
```

For the full derivation of the boundary profile, harmonic adjoint, centered
coordinates, SLSQP, and SPG, see
[`LINEAR_ALL_BOUNDARY_TUTORIAL.md`](LINEAR_ALL_BOUNDARY_TUTORIAL.md).

## Mathematical limits

- Both backends are local optimizers; neither proves a global minimum.
- The objective is smooth while knots remain in fixed boundary-edge cells and
  is only piecewise smooth when a knot crosses a boundary vertex.
- A discretely constant field makes the scale-free CV loss undefined and raises
  `DegenerateFieldError`.
- A width prior changes the mathematical optimum and is therefore opt-in.
