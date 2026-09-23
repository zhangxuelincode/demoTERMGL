# -*- coding: utf-8 -*-
"""Hypergradient of the TERMGL bilevel problem.

Upper level loss (per task):  L(theta) = (1/2n_val) ||X_val w_M(theta) - y_val||^2
where w_M(theta) is the output of the M-sweep HQ-DFBB lower solver.

Two computation modes
---------------------
* 'unroll' (default, exact): reverse-mode differentiation of the unrolled
  HQ-DFBB recursion. For each modal sweep m = M..1 the adjoint is propagated
  backwards through the Q stored DFBB iterations,

      w_k = NSD (c - A' u_{k-1}),   v_k = v_{k-1} + s A w_k,   u_k = grad phi*(v_k)

  (NSD = H^{-1}), accumulating
    - the direct theta-dependence of every A and A' operator,
    - the adjoint of (H, c), which enters the *tilt weight chain*
      D_m = n softmax(t r^2 / 2 sigma^2) with r = y - X w_{m-1}: this is the
      exponential-loss contribution that is identically zero for locally
      constant kernels (e.g. Epanechnikov) and is what makes the TERMGL
      optimization scheme genuinely different from the MAM_codes one;

* 'fixedpoint' (fast, MAM_codes-style engineering): per sweep, a fixed number
  of adjoint iterations evaluated with the final-iterate Jacobians (Neumann
  approximation of the unrolled sum), matching the role of D_cal in
  Hq_Dual_Hypergradient.m.

The per-task hypergradients are aggregated in termgl.bilevel with the SAGA
variance-reduction scheme borrowed from BiGLasso / MAM_codes.
"""

import numpy as np

from .dfbb import A_fwd, A_adj, grad_phi_star, hess_phi, hess_phi_star

_EPS = np.finfo(float).eps


def _val_gradient(X_val, y_val, w):
    """dL/dw for L = (1/2n_val)||X_val w - y_val||^2."""
    n_val = len(y_val)
    return X_val.T @ (X_val @ w - y_val) / n_val


def _backward_unroll(rec, theta, eta, C_w):
    """Exact reverse pass over one sweep's stored DFBB iterates.

    C_w : adjoint of the sweep output w (P,) -- the adjoint of w_Q = the last
          computed w (which equals ws[Q]); it seeds the (H, c) adjoint and the
          u-adjoint through the final w-step  w_Q = NSD(c - A' u_{Q-1}).

    Returns (b, Gamma, gamma):
      b     P x L  theta-adjoint accumulated over the sweep
      Gamma P x P  adjoint of H  (dL/dH)
      gamma P      adjoint of c  (dL/dc)
    """
    NSD, step = rec["NSD"], rec["step"]
    V, U, ws = rec["V"], rec["U"], rec["w"]
    Q = rec["n_iter"]
    P, L = theta.shape

    a = np.zeros((P, L))
    b = np.zeros((P, L))
    Gamma = np.zeros((P, P))
    gamma = np.zeros(P)

    # injection of C_w through the final w-step  w_Q = NSD(c - A' u_{Q-1})
    Ng0 = NSD @ C_w
    b -= Ng0[:, None] * U[Q - 1]
    a = -A_fwd(theta, Ng0)
    gamma += Ng0
    Gamma -= np.outer(Ng0, ws[Q])

    for k in range(Q - 1, 0, -1):
        Vk, Ukm1, wk = V[k], U[k - 1], ws[k]
        vartmp = hess_phi_star(Vk, a, eta)             # adjoint of v_k
        NSDu = NSD @ A_adj(theta, vartmp)              # NSD (A' vartmp)
        Ng = step * NSDu                               # NSD (adjoint of w_k)

        # theta-contribution of iteration k:
        #   v_k = s A w_k               ->  s * vartmp (.) w_k   (columnwise w_p)
        #   w_k = NSD(c - A' u_{k-1})   ->  - (NSD g) (.) u_{k-1}
        b += step * vartmp * wk[:, None]
        b -= Ng[:, None] * Ukm1

        # (H, c)-adjoint of iteration k (tilt weight chain)
        gamma += Ng
        Gamma -= np.outer(Ng, wk)

        # adjoint of u_{k-1}
        a = hess_phi(V[k - 1], vartmp, eta) - step * theta * NSDu[:, None]

    return b, Gamma, gamma


def _backward_fixedpoint(rec, theta, eta, C_w, K_fp=20):
    """MAM_codes-style fast adjoint: final-iterate Jacobians, K_fp steps."""
    NSD, step = rec["NSD"], rec["step"]
    V, U, ws = rec["V"], rec["U"], rec["w"]
    P, L = theta.shape
    Vf, Uf, wf = V[-1], U[-1], ws[-1]

    b = np.zeros((P, L))
    Gamma = np.zeros((P, P))
    gamma = np.zeros(P)

    # injection of C_w through the final w-step  w = NSD(c - A'u)
    Ng0 = NSD @ C_w
    b -= Ng0[:, None] * Uf
    a = -A_fwd(theta, Ng0)
    gamma += Ng0
    Gamma -= np.outer(Ng0, wf)

    for _ in range(K_fp):
        vartmp = hess_phi_star(Vf, a, eta)
        NSDu = NSD @ A_adj(theta, vartmp)
        Ng = step * NSDu
        b += step * vartmp * wf[:, None]
        b -= Ng[:, None] * Uf
        gamma += Ng
        Gamma -= np.outer(Ng, wf)
        a = hess_phi(Vf, vartmp, eta) - step * theta * NSDu[:, None]

    return b, Gamma, gamma


def _tilt_chain(rec, Gamma, gamma, cfg):
    """Adjoint of the previous sweep output w_{m-1} through the tilt weights
    D = n softmax(t r^2 / 2 sigma^2), r = y - X w_{m-1}."""
    if cfg.tilt == 0.0:
        return np.zeros(rec["X"].shape[1])
    X, y, D, r = rec["X"], rec["y"], rec["D"], rec["r"]
    n = len(y)
    dLdD = (((X @ Gamma) * X).sum(axis=1) + y * (X @ gamma)) / n
    q = D / n
    mu = X.T @ (q * r)
    coef = dLdD * D
    return -(cfg.tilt / cfg.sigma ** 2) * (X.T @ (coef * r) - coef.sum() * mu)


def compute_hypergradient(levels, X_val, y_val, w_final, theta, cfg,
                          mode="unroll", K_fp=20):
    """Full hypergradient dL/dtheta of the validation loss.

    levels  : list of sweep records from term_hq_dfbb(..., store=True)
    w_final : final lower-level solution (P,)
    mode    : 'unroll' (exact reverse mode) | 'fixedpoint' (fast)
    """
    theta = np.asarray(theta, dtype=float)
    eta = cfg.lam / 2.0
    P, L = theta.shape

    C = _val_gradient(X_val, y_val, w_final)
    hg = np.zeros((P, L))
    C_cur = C

    for m in range(len(levels) - 1, -1, -1):
        rec = levels[m]
        if mode == "fixedpoint":
            b, Gamma, gamma = _backward_fixedpoint(rec, theta, eta, C_cur, K_fp)
        elif mode == "unroll":
            b, Gamma, gamma = _backward_unroll(rec, theta, eta, C_cur)
        else:
            raise ValueError(mode)
        hg += b
        if m > 0:
            C_cur = _tilt_chain(rec, Gamma, gamma, cfg)
        else:
            break  # w_0 (ridge reference) does not depend on theta
    return hg
