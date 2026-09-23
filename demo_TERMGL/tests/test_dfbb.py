# -*- coding: utf-8 -*-
"""Lower-level (HQ-DFBB) correctness tests.

* KKT conditions of the weighted group-lasso W-step (dual feasibility,
  complementarity, primal-dual residual);
* cross-validation against an independent FISTA solver on the plain
  (tilt=0) group-lasso primal;
* monotone decrease of the tilted risk across HQ modal sweeps;
* robustness: outlier samples receive exponentially vanishing tilt weights.
"""
import numpy as np
import pytest

from termgl.config import TERMGLConfig
from termgl.dfbb import term_hq_dfbb, tilted_weights, A_fwd, A_adj, grad_phi_star
from termgl.data import synthesize_dataset


def make_cfg(**kw):
    base = dict(N=40, P=12, T=4, L=3, G=2, sigma=1.0, lam=0.5, mu=1e-3,
                tilt=0.0, seed=0, inner_itermax=300, inner_tol=1e-12)
    base.update(kw)
    return TERMGLConfig(**base)


def make_task(cfg, seed=0):
    rng = np.random.default_rng(seed)
    data = synthesize_dataset(cfg.N, cfg.P, cfg.T, cfg.L, cfg.G, cfg, rng)
    return (data["X"]["trn"][0], data["y"]["trn"][0],
            data["thetastar"])


# ---------------------------------------------------------------------------
# FISTA reference solver (independent implementation of the same W-step)
# ---------------------------------------------------------------------------
def fista_group_lasso(X, y, theta, eta, eps, n_iter=5000, tol=1e-12):
    """min (1/2)||sqrt(D)? -- here D=1:  (1/2)w'Hw - c'w + eta sum_l ||theta_l (. w)||_2."""
    n, P = X.shape
    H = X.T @ X / n + eps * np.eye(P)
    c = X.T @ y / n
    lip = np.linalg.eigvalsh(H).max()
    step = 1.0 / lip
    w = np.zeros(P)
    z = w.copy()
    tk = 1.0
    L = theta.shape[1]

    def prox(v):
        out = v.copy()
        for l in range(L):
            m = theta[:, l] > 0
            zl = out[m]
            nz = np.linalg.norm(zl)
            if nz > 0:
                out[m] = zl * max(1.0 - step * eta / nz, 0.0)
        return out

    for _ in range(n_iter):
        grad = H @ z - c
        w_new = prox(z - step * grad)
        tk_new = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        z = w_new + (tk - 1) / tk_new * (w_new - w)
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w, tk = w_new, tk_new
    return w


# ---------------------------------------------------------------------------
def test_kkt_conditions():
    cfg = make_cfg(tilt=-0.05, inner_itermax=20000)
    X, y, theta = make_task(cfg)
    eta = cfg.lam / 2.0
    w, levels, _ = term_hq_dfbb(X, y, theta, cfg, store=False)

    rec = levels[-1]
    D, NSD, step = rec["D"], rec["NSD"], rec["step"]
    XwD = X * D[:, None]
    H = XwD.T @ X / cfg.N + (cfg.mu / 2.0) * np.eye(cfg.P)
    c = XwD.T @ y / cfg.N

    # consistency: final V (last stored iterate) -> u
    V = rec["V"][-1]
    u = grad_phi_star(V, eta)
    # dual feasibility
    assert np.all(np.sqrt(np.sum(u * u, axis=0)) <= eta + 1e-8)
    # primal-dual residual  H w + A'u - c = 0  (by construction, checked
    # numerically to catch sign/transpose slips in the operators)
    res = H @ w + A_adj(theta, u) - c
    assert np.linalg.norm(res) < 1e-8
    # complementarity: inactive groups (||u_l|| < eta) carry no signal
    norms = np.sqrt(np.sum(u * u, axis=0))
    for l in range(cfg.L):
        if norms[l] < eta * (1 - 1e-3):
            assert np.linalg.norm(theta[:, l] * w) < 1e-6


def test_matches_fista_reference():
    """DFBB W-step (tilt=0) must agree with an independent FISTA solver.

    The dual variable approaches the ball boundary only algebraically, so a
    generous DFBB budget is required for a tight match."""
    cfg = make_cfg(tilt=0.0, inner_itermax=20000)
    X, y, theta = make_task(cfg, seed=3)
    eta = cfg.lam / 2.0
    eps = cfg.mu / 2.0
    w, _, _ = term_hq_dfbb(X, y, theta, cfg, store=False)
    w_ref = fista_group_lasso(X, y, theta, eta, eps)
    rel = np.linalg.norm(w - w_ref) / (np.linalg.norm(w_ref) + 1e-12)
    assert rel < 1e-4, f"DFBB vs FISTA relative error {rel:.2e}"


def test_fista_reference_is_sparse():
    cfg = make_cfg(tilt=0.0)
    X, y, theta = make_task(cfg, seed=3)
    eta = cfg.lam / 2.0
    eps = cfg.mu / 2.0
    w_ref = fista_group_lasso(X, y, theta, eta, eps)
    # oracle has G=2 groups of size 4 -> at most 8 nonzeros
    assert np.linalg.norm(w_ref, 0) <= cfg.G * (cfg.P // cfg.L)


def test_dfbb_is_sparse_too():
    cfg = make_cfg(tilt=0.0, inner_itermax=20000)
    X, y, theta = make_task(cfg, seed=3)
    w, _, _ = term_hq_dfbb(X, y, theta, cfg, store=False)
    n_nonzero = int(np.sum(np.abs(w) > 1e-8))
    assert n_nonzero <= cfg.G * (cfg.P // cfg.L)


def test_term_risk_monotone_decrease():
    cfg = make_cfg(tilt=-0.05, modal_iter=5)
    X, y, theta = make_task(cfg, seed=7)
    _, _, info = term_hq_dfbb(X, y, theta, cfg, store=False)
    risks = info["term_risk"]
    for a, b in zip(risks[:-1], risks[1:]):
        assert b <= a + 1e-10, f"tilted risk increased: {risks}"


def test_tilt_downweights_outliers():
    r = np.ones(50)
    r[:5] = 50.0                      # five gross outliers
    D = tilted_weights(r, -0.05, 1.0)
    assert abs(D.mean() - 1.0) < 1e-9          # weights have mean one
    assert D[:5].max() < 1e-3                  # outliers essentially removed
    assert D[5:].min() > 0.9                   # clean samples ~ unit weight


def test_zero_tilt_gives_uniform_weights():
    r = np.array([1.0, 2.0, 100.0])
    assert np.allclose(tilted_weights(r, 0.0, 1.0), 1.0)
