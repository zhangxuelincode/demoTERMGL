<p align="center">
  <a href="README.md">English</a> | <a href="README_zh-CN.md">简体中文</a>
</p>

<div align="center">

# TERMGL

**Robust Variable Structure Discovery via Tilted Empirical Risk Minimization**

A bilevel-optimization framework that simultaneously learns a shared group
structure across tasks and robustly fits multi-task regression models under
heavy-tailed noise, outliers, partial annotation and corrupted features.

[![Paper](https://img.shields.io/badge/Paper-Applied%20Intelligence-blue)](https://doi.org/10.1007/s10489-023-04923-9)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-%3E%3D1.24-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Tests-passing-brightgreen)](demo_TERMGL/tests)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)

</div>

## 📰 News

- **[2026-09]** Python reference implementation released: exact unrolled
  hypergradient through the HQ-DFBB lower solver, SAGA upper-level aggregation,
  partial-annotation and noisy-dimension stress factors, full pytest suite.
- **[2023-09]** Paper *Robust variable structure discovery based on tilted
  empirical risk minimization* published in Applied Intelligence 53(14).

## ✨ Highlights

1. **Tilted empirical risk (TERM) with a closed-form tilt step.** The tilted
   risk `(1/t)·log[(1/n)·Σᵢ exp(t·ℓᵢ(w))]` admits the variational form
   `min_{q∈Δ} Σᵢ qᵢℓᵢ(w) + (1/t)·Σᵢ qᵢ log(n·qᵢ)`; the optimal tilt weights
   `D = n·softmax(t·r²/2σ²)` have mean one and shrink outliers to zero weight
   exponentially fast (t < 0).
2. **Half-quadratic modal iterations wrap a dual forward–backward (DFBB)
   scheme.** Each W-step solves the weighted learned group-lasso problem in
   the dual with Bregman distances; the recursion is carried in v-space where
   the exact identity `∇φ∘∇φ*(v) = v` guarantees numerical stability as the
   dual saturates onto the η-ball boundary.
3. **Exact unrolled hypergradient including the exponential-tilt weight
   chain.** Reverse-mode differentiation of the full HQ-DFBB recursion
   propagates through the weights `D(w) = n·softmax(t·r²/2σ²)` — a term that
   vanishes for locally-constant robust kernels (e.g. Epanechnikov) but is
   essential for the exponential loss, which is what makes the TERMGL bilevel
   problem and its optimization scheme genuinely different from
   kernel-weighted bilevel baselines. A fast fixed-point (Neumann) mode in the
   style of kernel-weighted engineering is also provided.
4. **SAGA variance reduction on the upper level.** Per-task hypergradients are
   aggregated differentially (`aux_mean += (hg_t − aux_all_t)/|A|`) with an
   automatic step size `10^{−(1+⌊log₁₀ max|hg|⌋)}` and row-wise Euclidean
   projection of the structure matrix θ onto the unit simplex (exactly the
   BiGLasso engineering).
5. **Stress-factor experiments beyond the paper.** Partial annotation
   (structure transfer to unannotated tasks) and noisy dimensions (corrupted
   feature columns in trn/val with clean test designs) are supported out of
   the box.
6. **Verified correctness.** 33 unit tests: DFBB cross-validated against an
   independent FISTA group-lasso solver, hypergradient checked against
   central finite differences (tilt = 0 and t < 0), KKT conditions, TERM
   monotone decrease, SAGA aggregation invariants.

## 🚀 Quick Start

### 1. Environment

```bash
cd demo_TERMGL
pip install numpy scipy pandas pytest
```

### 2. Run the unit tests

```bash
python -m pytest tests -q
# 33 passed
```

### 3. Run a first bilevel experiment

```python
from termgl import TERMGLConfig
from termgl.bilevel import run_experiment

cfg = TERMGLConfig(
    N=50, P=50, T=200, L=5, G=2,        # samples / features / tasks / groups
    sigma=1.0, lam=0.5, mu=1e-3,
    tilt=-1e-3,                          # robust exponential tilt (t < 0)
    noise_distrib="normal",
    outlier_number=10, outlier_type=2,   # 20% additive outliers
    modal_iter=2, inner_itermax=600,
    outer_itermax=300, batch_size=4,
    annotation_ratio=0.5,                # partial annotation
    noisy_dims_fraction=0.3,             # 30% corrupted feature columns
    seed=0)

res = run_experiment(cfg)
print(res["eval"]["td_mean"])          # total deviation vs noiseless y
print(res["structure"]["assignment_accuracy"])
```

### 4. Reproduce the paper-style simulation grid

```bash
python run_simulation.py        # 4 noises x outliers + stress factors
# -> results/results.csv (TERMGL vs ridge vs oracle-structure baselines)
```

## 📊 Methods Compared

| Method   | Structure θ                     | Robust loss        |
|----------|---------------------------------|--------------------|
| TERMGL   | bilevel-learned (simplex rows)  | tilted exp. risk   |
| ridge    | uniform (no groups)             | none               |
| oracle   | true indicator θ*               | tilted exp. risk   |

Metrics follow the MATLAB `Evaluation.m`: **ASE** (vs noisy test responses),
**TD** (vs noiseless responses), α-level interval coverage, and
group-assignment accuracy.

## 📁 Repository Structure

```
├── demo.m, MAIN.m, ...          # original MATLAB implementation
├── functions/                   # MATLAB lower/upper solvers & data synthesis
│   └── Dual_process.m           # DFBB core (w inside the loop, v via ∇φ∘∇φ*=id)
└── demo_TERMGL/                 # Python reference implementation
    ├── termgl/
    │   ├── config.py            # all hyper-parameters (mirrors demo.m)
    │   ├── data.py              # synthesis, partial annotation, noisy dims
    │   ├── dfbb.py              # HQ modal iterations + dual forward-backward
    │   ├── hypergrad.py         # exact unrolled / fixed-point hypergradients
    │   ├── projection.py        # Condat simplex projection
    │   ├── bilevel.py           # SAGA upper level, structure transfer
    │   └── evaluation.py        # ASE / TD / coverage / structure metrics
    ├── tests/                   # 33 unit tests (FISTA & finite-difference refs)
    ├── run_simulation.py        # paper-style simulation grid -> results.csv
    └── results/
```

## 🔍 Notes on the Bilevel Design

- **Initialization matters.** The exact uniform structure matrix is a
  symmetric critical point of the upper problem (the hypergradient vanishes
  identically by column-permutation symmetry), so θ is initialized as
  `proxl(1/L·1 + 0.01·randn)` — exactly the BiGLasso trick.
- **Convergence budget.** The dual variable approaches the ball boundary only
  algebraically, so the W-step needs a generous DFBB budget (default
  `inner_itermax=1000`; the FISTA cross-validation test uses 20 000).
- **Dual_process.m.** The MATLAB core keeps `w` inside the DFBB loop and
  accumulates `v` through the identity `∇φ(∇φ*(v)) = v`, matching the official
  BiGLasso reference; the initialPoint field path is fixed.

## 📖 Reference

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

The bilevel framework follows **BiGL** (Frecon, Salzo, Pontil, NeurIPS 2018)
and the SAGA/automatic-step-size upper-level engineering follows the
kernel-weighted bilevel reference implementation used for comparison.

## License

This project is released under the [MIT License](LICENSE).
