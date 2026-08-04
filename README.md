# 可微调和边界优化

这是一个干净、独立、通用的四端点优化器：在一张带单边界环的连通三角网格上，用四个循环有序端点定义两段目标弧，求解 latent Robin harmonic field $u$，并移动端点使场的梯度尽可能均匀。最终输出再校准为 canonical field

$$
\hat u=\frac{u-\mu_0}{\mu_1-\mu_0},
$$

使两条目标弧上的平均值精确为 0 和 1。它不依赖相邻的 knitting 工程。

![调和边界优化概念图](docs/figures/harmonic-boundary-intuition-ai.png)

*AI 概念示意，不是数值实验截图。*

蓝弧把 latent $u$ 拉向 0，红弧把它拉向 1，两段灰色边界采用自然 Neumann 条件。有限 Robin 强度并不保证 raw $u$ 精确达到 target；公开的 <code>result.field</code> 是校准后的 $\hat u$。灰色边界没有被人为插值赋值。

## 从这里开始

如果你希望从零理解网格、harmonic field、Robin 边界、有限元、Schur complement、伴随梯度和 simplex 约束优化，请直接阅读：

**[零基础算法与数学完整教程](docs/algorithm-guide-zh.md)**

教程同时包含：

- AI 生成的直觉图；
- 精确的边界积分和 loss 图；
- Robin、Dirichlet、Neumann 的逐项对照、弹簧直觉和一维精确解；
- 当前代码重新计算的 Disk、Plane Polyscope 对比；
- Plane 为什么四角构型是零 loss 的全局最优解；
- “完全可微”在当前 P1 离散里的准确含义；
- 数学公式到具体函数的逐项映射。

## 核心定义

四个端点按展开后的边界弧长排序：

~~~text
k0 -- target 0 -- k1 -- free -- k2 -- target 1 -- k3 -- free -- k0+1
~~~

移动弧使用精确 P1 boundary-mass integral：

$$
Q(a,b)=\int_a^b\phi(s)\phi(s)^T\,ds,
$$

其端点导数为：

$$
\frac{\partial Q}{\partial a}=-\phi(a)\phi(a)^T,
\qquad
\frac{\partial Q}{\partial b}=+\phi(b)\phi(b)^T.
$$

因此端点可以连续跨过网格顶点，不会出现离散顶点集合突然切换的 active-set jump。内部自由度在初始化时通过 Schur complement 消元，并预计算 harmonic lift $E$；之后 latent 场统一写成 $u=Eu_B$，反向传播统一写成 $E^T(\partial L/\partial u)$。每次 objective evaluation 只需求解当前边界系统的一次前向和一次伴随 backsolve。

令 $Q_0,Q_1$ 是两段目标弧的精确质量矩阵，$\ell_c=\mathbf1^TQ_c\mathbf1$，则

$$
\mu_c=\frac{\mathbf1^TQ_cu_B}{\ell_c},
\qquad
\Delta=\mu_1-\mu_0,
\qquad
\hat u=\frac{u-\mu_0}{\Delta}.
$$

$\hat u$ 仍满足内部 Laplace 方程和自由弧上的齐次 Neumann 条件；uniformity loss 对这次 affine calibration 严格不变。但它是原始 0/1 Robin 解的 affine 后处理，不再满足同一组 0/1 Robin 条件。代码同时报告 raw span $\Delta$、两侧 target RMS、梯度 CV 和真正的局部等值线间距 CV。

默认 loss 是 $|\nabla u|^2$ 的面积加权变异系数平方。它评价相对均匀性，不单独保证 raw span；0–1 语义由上面的 canonical calibration 明确负责。通用默认值为：

~~~text
minimum_gap     = 0.03
boundary_penalty = 100
width_weight     = 0
~~~

### v0.2 API 语义

- <code>result.field</code> 和 <code>field_from_knots()</code> 返回 canonical $\hat u$；原始解分别是 <code>result.raw_field</code> 和 <code>robin_field_from_knots()</code>。
- <code>field_and_boundary_statistics_from_knots()</code> 同时返回 canonical 场和明确区分 raw/canonical gauge 的边界诊断。
- <code>parameters</code> 现在直接表示 <code>(origin, gap0, gap1, gap2, gap3)</code>，四个 gap 满足 <code>sum(gaps)=1</code>；这是 5 个存储坐标、4 个可行自由度，不再使用 softmax logits。
- 收敛统计使用 <code>constraint_violation</code> 与 <code>kkt_residual</code>；场统计明确区分 <code>gradient_cv</code> 与 <code>spacing_cv</code>。

## 快速运行

测试：

~~~bash
uv run pytest
~~~

运行一个 Disk 优化：

~~~bash
uv run python - <<'PY'
from boundary_opt import HarmonicBoundaryOptimizer, load_obj, random_knots

optimizer = HarmonicBoundaryOptimizer(load_obj("data/disk.obj"))
result = optimizer.optimize(random_knots(seed=0), max_iterations=100)
print(f"{result.initial_loss:.6f} -> {result.final_loss:.6f}")
print("knots:", result.knots % 1.0)
print("gaps:", result.gaps)
print("raw target means:", result.boundary_statistics.raw_zero_mean,
      result.boundary_statistics.raw_one_mean)
print("raw span:", result.boundary_statistics.raw_span)
print("raw target RMS:", result.boundary_statistics.raw_zero_target_rms,
      result.boundary_statistics.raw_one_target_rms)
print("canonical target RMS:",
      result.boundary_statistics.canonical_zero_target_rms,
      result.boundary_statistics.canonical_one_target_rms)
print("canonical range:", result.field.min(), result.field.max())
PY
~~~

当前显式四-gap simplex 实现的 Disk seed 0 结果是 $15.684301\to0.252757$；16 个 seeds 的 best / median / max 分别是 $0.237119/0.245503/0.258962$。Plane seed 0 是 $11.090487\to 8.6\times10^{-14}$。

扫描多个随机种子：

~~~bash
uv run python scan_mesh_seeds.py \
  --mesh data/triple_peak.obj \
  --seeds 16 \
  --output output/triple_peak_seed_scan.csv \
  --history-output output/triple_peak_seed_scan_history.csv
~~~

生成 loss 曲线：

~~~bash
uv run python plot_loss_curves.py \
  --history Triple-Peak output/triple_peak_seed_scan_history.csv \
  --svg output/triple_peak_loss_curves.svg
~~~

生成 Polyscope 前后对比：

~~~bash
uv run --extra visualization python visualize_mesh_optimization.py \
  --mesh data/disk.obj \
  --seed 0 \
  --screenshot output/disk_before_after.png
~~~

播放优化动画：

~~~bash
uv run --extra visualization python visualize_mesh_optimization.py \
  --mesh data/disk.obj \
  --seed 0 \
  --animate \
  --fps 8
~~~

重新生成教程中的精确 SVG：

~~~bash
uv run python generate_tutorial_figures.py
~~~

## 文件地图

| 文件 | 用途 |
|---|---|
| <code>boundary_opt.py</code> | 网格、direct gaps、P1/FEM、Schur、harmonic lift、canonical calibration、loss、伴随梯度、SLSQP |
| <code>scan_mesh_seeds.py</code> | 对任意支持的 mesh 扫描随机 seed |
| <code>plot_loss_curves.py</code> | 从 history CSV 生成 SVG loss 图 |
| <code>visualize_mesh_optimization.py</code> | Polyscope 静态前后对比与优化动画 |
| <code>generate_tutorial_figures.py</code> | 生成教程使用的精确 SVG |
| <code>tests/test_boundary_opt.py</code> | 解析梯度、端点连续性、Plane sanity check 等测试 |
| <code>docs/algorithm-guide-zh.md</code> | 零基础完整技术教程 |

## 当前适用范围

- 连通三角网格；
- 恰好一个 manifold boundary loop；
- 两段目标弧和两段自由弧；
- P1 有限元与 Robin 边界惩罚；
- 一阶局部优化。

SLSQP 能寻找满足 simplex 约束的局部 stationary point，但不保证 global minimum；建议对新模型运行多个随机 seed。Robin 在有限 $\rho$ 下只逼近硬 Dirichlet；canonical $\hat u$ 的目标弧均值精确为 0/1，但可能轻微超出该范围。当前端点 pipeline 在正 span 区域是 $C^1$，一般不是 $C^\infty$。
