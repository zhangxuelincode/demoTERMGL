# -*- coding: utf-8 -*-
"""Hypergradient correctness: the exact reverse-mode ('unroll') hypergradient
of the unrolled HQ-DFBB recursion -- including the exponential-tilt weight
chain that distinguishes TERMGL from the MAM_codes engineering -- is validated
against central finite differences of the true bilevel validation loss.
"""
import numpy as np
import pytest

from termgl.config import TERMGLConfig
from termgl.data import synthesize_dataset
from termgl.dfbb import term_hq_dfbb
from termgl.hypergrad import compute_hypergradient


def make_cfg(**kw):
    base = dict(N=40, P=6, T=2, L=3, G=2, sigma=1.0, lam=0.5, mu=1e-3,
                tilt=0.0, seed=0, modal_iter=2, inner_itermax=20000,
                inner_tol=1e-14)
    base.update(kw)
    return TERMGLConfig(**base)


def random_simplex_point(P, L, seed):
    """Interior point of the simplex product; the exact uniform point is a
    symmetric critical point of the bilevel problem (zero hypergradient by
    column-permutation symmetry), so tests must start from a generic point."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.full(L, 8.0), size=P)


def setup_task(cfg, seed=11):
    rng = np.random.default_rng(seed)
    data = synthesize_dataset(cfg.N, cfg.P, cfg.T, cfg.L, cfg.G, cfg, rng)
    return data["X"]["trn"][0], data["y"]["trn"][0], \
        data["X"]["val"][0], data["y"]["val"][0], data["thetastar"]


def upper_loss(X, y, Xv, yv, theta, cfg):
    """L(theta) = (1/2n_val)||Xv w(theta) - yv||^2 -- matches _val_gradient."""
    w, _, _ = term_hq_dfbb(X, y, theta, cfg, store=False)
    n_val = len(yv)
    return 0.5 * np.sum((Xv @ w - yv) ** 2) / n_val


def fd_directional_derivative(X, y, Xv, yv, theta0, direction, h, cfg):
    lp = upper_loss(X, y, Xv, yv, theta0 + h * direction, cfg)
    lm = upper_loss(X, y, Xv, yv, theta0 - h * direction, cfg)
    return (lp - lm) / (2 * h)


def analytic_directional(X, y, Xv, yv, theta0, direction, cfg, mode, K_fp=50):
    w, levels, _ = term_hq_dfbb(X, y, theta0, cfg, store=True)
    hg = compute_hypergradient(levels, Xv, yv, w, theta0, cfg,
                               mode=mode, K_fp=K_fp)
    return float(np.sum(hg * direction)), hg


@pytest.mark.parametrize("tilt", [0.0, -0.05])
def test_unroll_matches_finite_difference(tilt):
    cfg = make_cfg(tilt=tilt)
    X, y, Xv, yv, _ = setup_task(cfg)
    theta0 = random_simplex_point(cfg.P, cfg.L, seed=7)
    assert theta0.min() > 10 * 1e-3          # perturbations stay feasible
    rng = np.random.default_rng(42)
    d = rng.normal(size=(cfg.P, cfg.L))
    d -= d.mean(axis=1, keepdims=True)          # stay on the simplex rows
    d /= np.linalg.norm(d)

    h = 1e-3
    fd = fd_directional_derivative(X, y, Xv, yv, theta0, d, h, cfg)
    ana, hg = analytic_directional(X, y, Xv, yv, theta0, d, cfg, "unroll")
    # non-vacuous: the directional derivative must be clearly nonzero
    assert abs(ana) > 1e-5, f"gradient too small to test (d={ana:.2e})"
    denom = max(abs(fd), abs(ana), 1e-8)
    assert abs(fd - ana) / denom < 2e-2, f"fd={fd:.6e} analytic={ana:.6e}"
    assert np.isfinite(hg).all()


def test_fixedpoint_approximates_unroll():
    """MAM_codes-style fast mode: at a moderate inner budget the truncated
    adjoint must at least point in the same direction as the exact one."""
    cfg = make_cfg(tilt=-0.05, inner_itermax=300)
    X, y, Xv, yv, _ = setup_task(cfg)
    theta0 = random_simplex_point(cfg.P, cfg.L, seed=7)
    w, levels, _ = term_hq_dfbb(X, y, theta0, cfg, store=True)
    hg_u = compute_hypergradient(levels, Xv, yv, w, theta0, cfg, mode="unroll")
    hg_f = compute_hypergradient(levels, Xv, yv, w, theta0, cfg,
                                 mode="fixedpoint", K_fp=100)
    cos = float(hg_u.ravel() @ hg_f.ravel() /
                (np.linalg.norm(hg_u) * np.linalg.norm(hg_f) + 1e-30))
    assert cos > 0.5, f"cosine similarity {cos:.3f}"


def test_hypergradient_shape_and_finiteness():
    cfg = make_cfg()
    X, y, Xv, yv, _ = setup_task(cfg)
    theta0 = np.full((cfg.P, cfg.L), 1.0 / cfg.L)
    w, levels, _ = term_hq_dfbb(X, y, theta0, cfg, store=True)
    hg = compute_hypergradient(levels, Xv, yv, w, theta0, cfg, mode="unroll")
    assert hg.shape == (cfg.P, cfg.L)
    assert np.isfinite(hg).all()


def test_fixedpoint_works_without_full_storage():
    """The MAM_codes-style fast mode must run from a store=False solve."""
    cfg = make_cfg(inner_itermax=200)
    X, y, Xv, yv, _ = setup_task(cfg)
    theta0 = np.full((cfg.P, cfg.L), 1.0 / cfg.L)
    w, levels, _ = term_hq_dfbb(X, y, theta0, cfg, store=False)
    hg = compute_hypergradient(levels, Xv, yv, w, theta0, cfg,
                               mode="fixedpoint", K_fp=50)
    assert np.isfinite(hg).all()
