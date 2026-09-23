# -*- coding: utf-8 -*-
"""Bilevel driver of TERMGL: upper-level structure learning with SAGA
aggregation of per-task hypergradients.

Upper problem (structure theta, P x L, rows on the simplex):

    min_theta  (1/|A|) sum_{t in A} L_val( w_t(theta) )
    w_t(theta) = argmin_w R_tilt(w; task t) + eta sum_l ||theta_l (.) w||_2
                 + (eps/2)||w||^2

Engineering borrowed from MAM_codes / BiGLasso (hyper_modal.m /
optimBLGS_setting.m):

* SAGA variance reduction: a table aux_all[t] stores the last computed
  hypergradient of each annotated task; the aggregate aux_mean is updated
  differentially, aux_mean += (hg_t - aux_all_t) / |A|, which yields an
  unbiased gradient estimate with exponentially decaying variance;
* automatic step size 10^-(1 + floor(log10(max|hg|))) (adaptive, order-one
  updates without tuning);
* row-wise Euclidean projection of theta onto the simplex after each update
  (the learned structure is a soft group assignment).

Difference w.r.t. MAM_codes: the per-task hypergradient is the *exact
reverse-mode* gradient through the unrolled HQ-DFBB recursion including the
exponential-tilt weight chain (see termgl.hypergrad) -- in MAM_codes the
Epanechnikov kernel weights have (locally) vanishing Jacobian so the chain is
absent; here the tilt chain is the essential term that makes the bilevel
problem the one of the paper.
"""

import numpy as np

from .data import synthesize_dataset, apply_partial_annotation, add_noisy_dimensions
from .dfbb import term_hq_dfbb
from .hypergrad import compute_hypergradient
from .projection import proj_unit_simplex


def _auto_stepsize(hg):
    """MAM_codes-style step: 10^-(1+floor(log10(max|hg|)))."""
    m = np.max(np.abs(hg))
    if not np.isfinite(m) or m <= 0:
        return 1e-3
    return 10.0 ** (-(1 + np.floor(np.log10(m))))


def _task_hypergradient(t, data, theta, w_warm, cfg, mode, K_fp, store=True):
    """Lower solve + hypergradient for one task (exact or fixedpoint mode)."""
    w, levels, info = term_hq_dfbb(
        data["X"]["trn"][t], data["y"]["trn"][t], theta, cfg,
        store=store, w_init=w_warm)
    hg = compute_hypergradient(levels, data["X"]["val"][t],
                               data["y"]["val"][t], w, theta, cfg,
                               mode=mode, K_fp=K_fp)
    return w, hg, info


def _init_theta(P, L):
    """Uniform soft assignment plus a small random perturbation, projected
    back onto the simplex -- exactly the BiGLasso initialization
    (proxl(ones/L + 0.01*randn)). The perturbation is essential: at the exact
    uniform point the hypergradient vanishes identically by symmetry (every
    simplex-tangent direction leaves sum_l theta_{p,l}·(penalty weights)
    unchanged to first order), so the bilevel recursion could never start."""
    rng = np.random.default_rng(0)
    return proj_unit_simplex(np.full((P, L), 1.0 / L)
                             + 0.01 * rng.normal(size=(P, L)), axis=-1)


def run_termgl(data, cfg, annotated=None, mode="unroll", K_fp=20,
               theta_init=None, w_warm=None):
    """Run the full bilevel TERMGL procedure.

    Parameters
    ----------
    data      : dict from termgl.data.synthesize_dataset
    cfg       : TERMGLConfig
    annotated : indices of annotated tasks (default: all)
    mode      : 'unroll' (exact hypergradient) | 'fixedpoint' (fast)
    K_fp      : adjoint iterations for the fixedpoint mode

    Returns
    -------
    result : dict with keys
        theta   : learned structure (P, L)
        W       : final per-task lower solutions stacked (P, T)
        history : list of per-iteration diagnostics
    """
    rng = np.random.default_rng(cfg.seed)
    X, y = data["X"], data["y"]
    T = len(y["trn"])
    P = X["trn"][0].shape[1]
    L = cfg.L

    if annotated is None:
        annotated = np.arange(T)
    annotated = np.asarray(annotated)
    n_ann = len(annotated)

    theta = _init_theta(P, L) if theta_init is None else np.asarray(theta_init, float).copy()
    W = [None] * T if w_warm is None else [np.array(w_warm[:, t]) for t in range(T)]

    # ---- SAGA state (lazy full init over annotated tasks) ----
    aux_all = np.zeros((P, L, n_ann))
    aux_mean = np.zeros((P, L))
    aux_ready = np.zeros(n_ann, dtype=bool)

    history = []
    for it in range(cfg.outer_itermax):
        if not aux_ready.all():
            # first pass: fill the table over all annotated tasks
            tasks = annotated[~aux_ready]
            for j, t in enumerate(tasks):
                w_t, hg_t, _ = _task_hypergradient(
                    t, data, theta, W[t], cfg, mode, K_fp)
                W[t] = w_t
                aux_all[:, :, j] = hg_t
                aux_mean += hg_t / n_ann
                aux_ready[j] = True
        else:
            batch = rng.choice(annotated, size=min(cfg.batch_size, n_ann),
                               replace=False)
            for t in batch:
                j = np.searchsorted(annotated, t)
                w_t, hg_t, _ = _task_hypergradient(
                    t, data, theta, W[t], cfg, mode, K_fp)
                W[t] = w_t
                aux_mean += (hg_t - aux_all[:, :, j]) / n_ann
                aux_all[:, :, j] = hg_t

        step = cfg.outer_stepsize if cfg.outer_stepsize else _auto_stepsize(aux_mean)
        theta_new = proj_unit_simplex(theta - step * aux_mean, axis=-1)

        dtheta = np.max(np.abs(theta_new - theta))
        theta = theta_new

        rec = {"iter": it, "step": step, "dtheta": dtheta,
               "val_loss": _mean_val_loss(data, annotated, W, theta, cfg)}
        history.append(rec)
        if cfg.verbose and (it % 50 == 0 or it == cfg.outer_itermax - 1):
            print(f"[TERMGL] it={it:4d} val={rec['val_loss']:.4f} "
                  f"step={step:.1e} dtheta={dtheta:.2e}")
        if dtheta < 1e-9:
            break

    # structure transfer: solve the lower level for the unannotated tasks at
    # the final learned structure (their labels were never used above)
    missing = [t for t in range(T) if W[t] is None]
    for t in missing:
        w_t, _, _ = term_hq_dfbb(data["X"]["trn"][t], data["y"]["trn"][t],
                                 theta, cfg, store=False)
        W[t] = w_t

    return {"theta": theta, "W": np.stack(W, axis=1), "history": history}


def _mean_val_loss(data, annotated, W, theta, cfg):
    """Average upper-level validation loss at the *current* iterate."""
    eta = cfg.lam / 2.0
    tot, n = 0.0, len(annotated)
    for t in annotated:
        if W[t] is None:
            return float("nan")
        r = data["X"]["val"][t] @ W[t] - data["y"]["val"][t]
        # group-lasso penalty: sum over GROUPS l of ||theta_l (.) w||_2
        reg = eta * np.sum(np.sqrt(np.sum((theta * W[t][:, None]) ** 2, axis=0)))
        tot += (0.5 * r @ r / len(r)) + reg
    return tot / n


# ---------------------------------------------------------------------------
# Convenience end-to-end wrapper (data generation included)
# ---------------------------------------------------------------------------
def run_experiment(cfg, mode="unroll", K_fp=20):
    """synthesize -> stress factors -> bilevel -> evaluation dict."""
    rng = np.random.default_rng(cfg.seed)
    data = synthesize_dataset(cfg.N, cfg.P, cfg.T, cfg.L, cfg.G, cfg, rng)
    data = add_noisy_dimensions(data, cfg, rng)
    data, annotated = apply_partial_annotation(data, cfg, rng)
    res = run_termgl(data, cfg, annotated, mode=mode, K_fp=K_fp)
    from .evaluation import evaluate_tasks, structure_metrics
    res["eval"] = evaluate_tasks(data, res["W"], cfg)
    res["structure"] = structure_metrics(res["theta"], data["thetastar"])
    return res
