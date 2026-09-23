# -*- coding: utf-8 -*-
"""Lower level of TERMGL: tilted (exponential) empirical risk + learned
group-lasso penalty, solved by half-quadratic (HQ) modal iterations wrapping a
dual forward-backward scheme with Bregman distances (DFBB).

Problem (per task, given the structure matrix theta with simplex rows)
-----------------------------------------------------------------------
Lower level of the bilevel program:

    min_w  R_t(w) + Omega_theta(w) + (eps/2)||w||^2

with the tilted empirical risk (TERM, tilt t<0) of the scaled squared loss
    l_i(w) = (y_i - x_i'w)^2 / (2 sigma^2),
    R_t(w) = (1/t) log ( (1/n) sum_i exp(t l_i(w)) ),        t < 0,

and the learned-structure group-lasso penalty
    Omega_theta(w) = eta * sum_l || theta_l (.) w ||_2 ,   eta = lambda/2.

Fenchel conjugate of the tilted risk (t < 0, Donsker-Varadhan / variational
form) -- this is the "exponential loss" specialization requested for TERMGL:

    R_t(w) = min_{q in Delta_n} sum_i q_i l_i(w) + (1/t) sum_i q_i log(n q_i)

with the closed-form tilt step  q_i ∝ exp(t l_i(w))  (outliers, whose large
residuals enter exp(t l_i) with t<0, receive vanishing weight -- robustness).

HQ modal iterations alternate
  (M-step) q <- argmin_q ...            (closed form softmax, weights D = n q)
  (W-step) w <- argmin_w sum_i q_i l_i(w) + Omega_theta(w) + (eps/2)||w||^2
which decreases the joint objective monotonically (both steps are exact).

W-step via DFBB (dual forward-backward with Bregman distances; Frecon et al.
2018, Salzo's DFwB). With the weighted fidelity f_D(w) = (1/2n)||sqrt(D)(Xw-y)||^2
+ (eps/2)||w||^2 (D = n q has mean one, so f_D = sigma^2 * sum_i q_i l_i), the
dual of  min_w f_D(w) + Omega_theta(w)  is

    max_u  -f*_D(-A_theta' u) - sum_l phi*(u_l),   ||u_l|| <= eta,

where A_theta: w -> theta (.) w (broadcast) and the "Hellinger-like" reference
function for the Bregman backward step is
    phi(u)   = eta sqrt(1+||u||^2),           grad phi (u) = eta u/sqrt(1+||u||^2)
    phi*(v)  = eta - sqrt(eta^2 - ||v||^2),   grad phi*(v) = v/sqrt(eta^2-||v||^2)

One DFBB iteration (w inside the loop -- the official BiGLasso form; the
TERMGL MATLAB transcript Dual_process.m wrongly hoisted it outside):
    w_k = H^{-1} (c - A_theta' u_{k-1})          (H = X'DX/n + eps I, c = X'Dy/n)
    v_k = v_{k-1} + s * A_theta w_k              (exact: grad phi(grad phi*(v)) = v)
    u_k = grad phi*(v_k) = eta v_k / sqrt(1+||v_k||^2)

The identity grad phi(grad phi*(v)) = v makes the recursion on v exact and
numerically stable (no cancellation as ||u_l|| -> eta on active groups).
Step size s = 0.99 / Lip with Lip = lambda_max(sum_l diag(theta_l) H^{-1}
diag(theta_l)) (Lipschitz constant of the dual smooth part).
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

_EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------
# Tilted exponential loss pieces
# ---------------------------------------------------------------------------
def tilted_weights(r, tilt, sigma):
    """Exponential-tilt weights D = n * softmax(t * r^2 / (2 sigma^2)), t <= 0.

    D has mean one; t -> 0 recovers the plain (unweighted) group lasso. For
    t < 0 large residuals are exponentially down-weighted (robustness).
    """
    if tilt == 0.0:
        return np.ones_like(r)
    ell = 0.5 * (r / sigma) ** 2
    e = np.exp(tilt * (ell - ell.max()))          # stabilized softmax
    q = e / e.sum()
    return len(r) * q


def term_risk(r, tilt, sigma):
    """Tilted empirical risk (1/t) log mean exp(t l_i); the t->0 limit is the
    mean loss. Used to monitor monotone decrease of the HQ sweeps."""
    ell = 0.5 * (r / sigma) ** 2
    if tilt == 0.0:
        return float(ell.mean())
    m = ell.max()
    return float((m + np.log(np.exp(tilt * (ell - m)).mean())) / tilt)


# ---------------------------------------------------------------------------
# Legendre pieces of the Hellinger-like reference function (stable forms)
# ---------------------------------------------------------------------------
def grad_phi_star(V, eta):
    """grad phi*(v) = eta v / sqrt(1+||v||^2), column-wise (V is P x L)."""
    nrm = np.sqrt(1.0 + np.sum(V * V, axis=0))
    return eta * V / nrm


def hess_phi_star(V, A, eta):
    """Hessian-vector product of phi* at V applied to A (both P x L):
    eta [ a/sqrt(1+|v|^2) - v (v'a) / (1+|v|^2)^{3/2} ] (column-wise)."""
    n2 = 1.0 + np.sum(V * V, axis=0)                       # (L,)
    va = np.sum(V * A, axis=0)                             # (L,)
    return eta * (A / np.sqrt(n2) - V * va / n2 ** 1.5)


def hess_phi(V, A, eta):
    """Hessian-vector product of phi at u = grad phi*(V), applied to A.
    Using sqrt(eta^2-||u||^2) = eta/sqrt(1+||v||^2):
        sqrt(1+|v|^2)/eta * [ a + v (v'a)/(1+|v|^2) ]  (column-wise)."""
    n2 = 1.0 + np.sum(V * V, axis=0)
    va = np.sum(V * A, axis=0)
    return np.sqrt(n2) / eta * (A + V * va / n2)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
def A_fwd(theta, w):
    """A_theta w = theta (.) w  ->  P x L."""
    return theta * w[:, None]


def A_adj(theta, U):
    """A_theta' U = sum_l theta_l (.) u_l  ->  P."""
    return np.sum(theta * U, axis=1)


# ---------------------------------------------------------------------------
# HQ-DFBB solver
# ---------------------------------------------------------------------------
def _dual_lipschitz(Hf, theta, n_power=12):
    """Lipschitz constant of the dual smooth part gradient.

    The dual objective  g(u) = -f*_D(-A' u)  has Hessian operator
        du -> -A H^{-1} A' du,   (A du)_l = theta_l (.) du_l,
    whose spectral norm is estimated by a deterministic power iteration
    (the operator is symmetric PSD up to sign).
    """
    P, L = theta.shape
    delta = theta / (np.linalg.norm(theta) + _EPS)
    lam = 0.0
    for _ in range(n_power):
        z = A_adj(theta, delta)
        w = cho_solve(Hf, z)
        new = A_fwd(theta, w)
        denom = np.sum(delta * delta)
        lam = np.sum(new * delta) / max(denom, _EPS)
        delta = new / (np.linalg.norm(new) + _EPS)
    return max(lam, _EPS)


def _dfbb(X, y, theta, D, eta, eps, n_iter, tol, store):
    """One W-step: DFBB on the weighted learned group-lasso problem.

    Returns (w, record) where record carries everything the hypergradient
    needs: H factor, inverse, stepsize, residuals and per-iteration
    V/U/w iterates (if store=True, indices 0..Q with V_0 = U_0 = 0).
    """
    n, P = X.shape
    XwD = X * D[:, None]
    H = XwD.T @ X / n + eps * np.eye(P)
    c = XwD.T @ y / n
    Hf = cho_factor(H, lower=True)
    NSD = np.linalg.inv(H)

    step = 0.99 / _dual_lipschitz(Hf, theta)

    V = np.zeros((P, theta.shape[1]))
    U = np.zeros_like(V)
    w = cho_solve(Hf, c.copy())

    Vs, Us, ws = [V.copy()], [U.copy()], [w.copy()]
    prev_w = w.copy()
    executed = 0
    for _ in range(n_iter):
        w = cho_solve(Hf, c - A_adj(theta, U))
        V = V + step * A_fwd(theta, w)
        U = grad_phi_star(V, eta)
        executed += 1
        if store:
            Vs.append(V.copy()); Us.append(U.copy()); ws.append(w.copy())
        dw = np.max(np.abs(w - prev_w))
        prev_w = w.copy()
        # skip the check at the first executed iteration: there U is still 0,
        # so w equals the loop-entry ridge solution and dw == 0 trivially
        if executed > 1 and dw < tol:
            break
        if np.max(np.abs(V)) > 1e12:  # safety net, should not trigger
            break

    if not store:
        # keep the terminal iterates so the fixedpoint hypergradient
        # (which only reads V[-1], U[-1], w[-1]) stays valid
        Vs.append(V.copy()); Us.append(U.copy()); ws.append(w.copy())

    record = {
        "X": X, "y": y, "D": D, "Hf": Hf, "NSD": NSD, "step": step,
        "V": Vs, "U": Us, "w": ws, "n_iter": executed,
    }
    return w, record


def term_hq_dfbb(X, y, theta, cfg, store=False, w_init=None):
    """Full lower-level solve: M HQ modal sweeps of tilt-step + DFBB.

    Parameters
    ----------
    X, y   : task data (n, P), (n,)
    theta  : structure matrix (P, L), rows on the simplex
    cfg    : TERMGLConfig (uses tilt, sigma, lam, mu, modal_iter,
             inner_itermax, inner_tol)
    store  : store per-iteration records for the exact reverse-mode
             hypergradient
    w_init : optional warm-start w for the first tilt step (default: ridge)

    Returns
    -------
    w      : final primal solution (P,)
    levels : list of per-sweep records (for the hypergradient)
    info   : diagnostics (tilted risk per sweep, weights per sweep)
    """
    n, P = X.shape
    eta = cfg.lam / 2.0
    eps = cfg.mu / 2.0
    theta = np.asarray(theta, dtype=float)

    # w_0: (weighted-)ridge reference for the first tilt step
    if w_init is None:
        H0 = X.T @ X / n + eps * np.eye(P)
        w_prev = np.linalg.solve(H0, X.T @ y / n)
    else:
        w_prev = np.asarray(w_init, dtype=float).copy()

    levels, risks, Ds = [], [], []
    for _ in range(cfg.modal_iter):
        r = y - X @ w_prev
        D = tilted_weights(r, cfg.tilt, cfg.sigma)
        Ds.append(D)
        risks.append(term_risk(r, cfg.tilt, cfg.sigma))
        w, record = _dfbb(X, y, theta, D, eta, eps,
                          cfg.inner_itermax, cfg.inner_tol, store)
        record["r"] = r
        levels.append(record)
        w_prev = w

    info = {"term_risk": risks, "weights": Ds}
    return w, levels, info
