# -*- coding: utf-8 -*-
"""TERMGL: Robust variable structure discovery via tilted empirical risk
minimization (TERM) on group-lasso structure learning (bilevel optimization).

Python re-implementation of the MATLAB demoTERMGL repository, matching the
paper "Robust variable structure discovery based on tilted empirical risk
minimization" (Zhang et al., Applied Intelligence 53(14):17865-17886, 2023)
and the underlying BiGL framework (Frecon, Salzo, Pontil, NeurIPS 2018).

Key design points (see termgl/dfbb.py and termgl/hypergrad.py docstrings):
* lower level  : tilted (exponential) empirical risk + learned group-lasso
                 penalty, solved by a dual forward-backward scheme with
                 Bregman distances (DFBB) inside half-quadratic (HQ) modal
                 iterations;
* upper level  : validation MSE, minimized over the structure matrix theta
                 (rows on the unit simplex) by a stochastic hypergradient
                 method with SAGA-style variance reduction;
* hypergradient: exact reverse-mode differentiation of the unrolled HQ-DFBB
                 recursion, including the exponential-tilt weight chain
                 D(w) -- the part that vanishes for locally-constant kernels
                 (e.g. Epanechnikov) but is essential for the exponential loss.
"""

from .config import TERMGLConfig
from .data import synthesize_dataset, apply_partial_annotation, add_noisy_dimensions
from .dfbb import term_hq_dfbb, tilted_weights, term_risk
from .hypergrad import compute_hypergradient
from .bilevel import run_termgl
from .evaluation import evaluate_tasks, structure_metrics

__all__ = [
    "TERMGLConfig",
    "synthesize_dataset",
    "apply_partial_annotation",
    "add_noisy_dimensions",
    "term_hq_dfbb",
    "tilted_weights",
    "term_risk",
    "compute_hypergradient",
    "run_termgl",
    "evaluate_tasks",
    "structure_metrics",
]
