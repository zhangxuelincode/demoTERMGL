<p align="center">
  <a href="README.md">English</a> | <a href="README_zh-CN.md">简体中文</a>
</p>

<div align="center">

# TERMGL

**基于倾斜经验风险最小化的鲁棒变量结构发现**

一个双层优化框架：在重尾噪声、离群点、部分标注与特征污染等复杂场景下，
同时学习跨任务的共享组结构，并鲁棒地拟合多任务回归模型。

[![Paper](https://img.shields.io/badge/Paper-Applied%20Intelligence-blue)](https://doi.org/10.1007/s10489-023-04923-9)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-%3E%3D1.24-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](#许可证)

</div>

## 📰 新闻

- **[2026-09]** 发布 Python 参考实现：穿过 HQ-DFBB 下层求解器的精确展开式
  超梯度、SAGA 上层聚合、部分标注与噪声维度压力因子，以及完整的 pytest
  测试套件。
- **[2023-09]** 论文《Robust variable structure discovery based on tilted
  empirical risk minimization》发表于 Applied Intelligence 53(14)。

## ✨ 亮点

1. **倾斜经验风险（TERM）与闭式倾斜步。** 倾斜风险
   `(1/t)·log[(1/n)·Σᵢ exp(t·ℓᵢ(w))]` 具有变分形式
   `min_{q∈Δ} Σᵢ qᵢℓᵢ(w) + (1/t)·Σᵢ qᵢ log(n·qᵢ)`；最优倾斜权重
   `D = n·softmax(t·r²/2σ²)` 均值为 1，且在 t < 0 时将离群点权重以指数速度
   压缩至零。
2. **半二次（HQ）模迭代包裹对偶前向–后向（DFBB）格式。** 每个 W 步在
   对偶空间中以 Bregman 距离求解加权学习组 Lasso 问题；递推在 v 空间中进行，
   精确恒等式 `∇φ∘∇φ*(v) = v` 保证了对偶变量饱和到 η-球边界时的数值稳定性。
3. **包含指数倾斜权重链的精确展开式超梯度。** 对完整 HQ-DFBB 递推做反向模式
   微分，梯度会穿过权重 `D(w) = n·softmax(t·r²/2σ²)`——这一项在局部常数型
   鲁棒核（如 Epanechnikov）下消失，但对指数损失不可或缺，这正是 TERMGL
   双层问题及其优化方案与核加权双层基线存在本质差异的原因。同时提供核加权
   工程风格的快速不动点（Neumann）超梯度模式。
4. **上层 SAGA 方差缩减。** 逐任务超梯度以差分方式聚合
   （`aux_mean += (hg_t − aux_all_t)/|A|`），配合自动步长
   `10^{−(1+⌊log₁₀ max|hg|⌋)}` 以及结构矩阵 θ 逐行的单位单纯形欧氏投影
   （与 BiGLasso 工程完全一致）。
5. **超出论文的压力因子实验。** 原生支持部分标注（结构迁移到未标注任务）与
   噪声维度（trn/val 设计矩阵特征列污染、测试设计保持干净）。
6. **测试套件。** DFBB 与独立实现的 FISTA 组 Lasso 求解器交叉验证；超梯度
   与中心有限差分对照（tilt = 0 与 t < 0 两种情形）；覆盖 KKT 条件、TERM
   单调下降与 SAGA 聚合不变量。

## 🚀 快速开始

### 1. 环境

```bash
cd demo_TERMGL
pip install numpy scipy pandas pytest
```

### 2. 运行单元测试

```bash
python -m pytest tests -q
```

### 3. 运行第一个双层实验

```python
from termgl import TERMGLConfig
from termgl.bilevel import run_experiment

cfg = TERMGLConfig(
    N=50, P=50, T=200, L=5, G=2,        # 样本数 / 特征数 / 任务数 / 组数
    sigma=1.0, lam=0.5, mu=1e-3,
    tilt=-1e-3,                          # 鲁棒指数倾斜（t < 0）
    noise_distrib="normal",
    outlier_number=10, outlier_type=2,   # 20% 加性离群点
    modal_iter=2, inner_itermax=600,
    outer_itermax=300, batch_size=4,
    annotation_ratio=0.5,                # 部分标注
    noisy_dims_fraction=0.3,             # 30% 特征列被污染
    seed=0)

res = run_experiment(cfg)
print(res["eval"]["td_mean"])          # 相对无噪声 y 的总偏差
print(res["structure"]["assignment_accuracy"])
```

### 4. 复现论文风格的仿真网格

```bash
python run_simulation.py        # 4 种噪声 x 离群点 + 压力因子
# -> results/results.csv（TERMGL 对比 ridge 与 oracle 结构基线）
```

## 📊 对比方法

| 方法    | 结构 θ                        | 鲁棒损失     |
|---------|-------------------------------|--------------|
| TERMGL  | 双层学习（单纯形行）          | 倾斜指数风险 |
| ridge   | 均匀（无组结构）              | 无           |
| oracle  | 真实指示矩阵 θ*               | 倾斜指数风险 |

评估指标与 MATLAB `Evaluation.m` 一致：**ASE**（相对含噪测试响应）、
**TD**（相对无噪声响应）、α 分位数区间覆盖率，以及组分配准确率。

## 📁 仓库结构

```
├── demo.m, MAIN.m, ...          # 原始 MATLAB 实现
├── functions/                   # MATLAB 下层/上层求解器与数据合成
│   └── Dual_process.m           # DFBB 核心（w 在循环内，v 经 ∇φ∘∇φ*=id 累积）
└── demo_TERMGL/                 # Python 参考实现
    ├── termgl/
    │   ├── config.py            # 全部超参数（对应 demo.m）
    │   ├── data.py              # 数据合成、部分标注、噪声维度
    │   ├── dfbb.py              # HQ 模迭代 + 对偶前向–后向
    │   ├── hypergrad.py         # 精确展开式 / 不动点超梯度
    │   ├── projection.py        # Condat 单纯形投影
    │   ├── bilevel.py           # SAGA 上层、结构迁移
    │   └── evaluation.py        # ASE / TD / 覆盖率 / 结构指标
    ├── tests/                   # 测试套件（FISTA 与有限差分参照）
    ├── run_simulation.py        # 论文风格仿真网格 -> results.csv
    └── results/
```

## 🔍 双层设计说明

- **初始化很关键。** 精确均匀的结构矩阵是上层问题的对称临界点（由列置换
  对称性，超梯度恒为零），因此 θ 以 `proxl(1/L·1 + 0.01·randn)` 初始化——
  正是 BiGLasso 的技巧。
- **收敛预算。** 对偶变量只是代数式地逼近球边界，因此 W 步需要充裕的 DFBB
  预算（默认 `inner_itermax=1000`；FISTA 交叉验证测试使用 20 000）。
- **Dual_process.m。** MATLAB 核心将 `w` 保持在 DFBB 循环内，并通过恒等式
  `∇φ(∇φ*(v)) = v` 累积 `v`，与 BiGLasso 官方参考实现一致；已修复
  initialPoint 字段路径。

## 📖 引用

```bibtex
@article{zhang2023termgl,
  title   = {Robust variable structure discovery based on tilted empirical risk minimization},
  author  = {Zhang, Xuelin and others},
  journal = {Applied Intelligence},
  volume  = {53},
  number  = {14},
  pages   = {17865--17886},
  year    = {2023}
}
```

双层框架遵循 **BiGL**（Frecon, Salzo, Pontil, NeurIPS 2018）；SAGA / 自动
步长的上层工程设计遵循用于对比的核加权双层参考实现。

## 许可证

本项目基于 [MIT License](LICENSE) 发布。
