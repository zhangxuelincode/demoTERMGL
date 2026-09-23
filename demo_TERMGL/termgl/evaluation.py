# -*- coding: utf-8 -*-
"""Evaluation metrics mirroring Evaluation.m of the MATLAB repository.

* ASE  : average squared error of predictions on the clean test split
         (w.r.t. the noisy observations y.tst), /n;
* TD   : total deviation w.r.t. the noiseless responses y.true -- the
         quantity the paper's figures report;
* CI   : alpha-level quantile interval (normal-approximation replacement of
         the MATLAB ksdensity band) -- width and empirical coverage;
* structure metrics : group-assignment accuracy of the learned theta against
         the oracle indicator matrix (argmax over groups per feature).
"""

import numpy as np


def evaluate_tasks(data, W, cfg, alpha=None):
    """W: P x T stacked lower solutions. Returns dict of metric arrays."""
    if alpha is None:
        alpha = cfg.alpha
    T = W.shape[1]
    ase = np.zeros(T)
    td = np.zeros(T)
    ci_width = np.zeros(T)
    ci_cover = np.zeros(T)

    z = _quantile_normal(1.0 - (1.0 - alpha) / 2.0)
    for t in range(T):
        Xt, w = data["X"]["tst"][t], W[:, t]
        pred = Xt @ w
        # map the raw prediction into the standardized spaces of the noisy
        # test response and of the noiseless response, respectively
        m1, s1 = data["ystats"]["tst"][t]
        m0, s0 = data["ystats"]["true"][t]
        pred_noisy = (pred - m1) / s1
        pred_true = (pred - m0) / s0
        y_noisy = data["y"]["tst"][t]
        y_true = data["y"]["true"][t]

        ase[t] = np.mean((pred_noisy - y_noisy) ** 2)
        td[t] = np.mean((pred_true - y_true) ** 2)

        resid = pred_true - y_true
        s = resid.std(ddof=1) if len(resid) > 1 else 1.0
        hw = z * s / np.sqrt(len(resid))
        ci_width[t] = 2.0 * hw
        ci_cover[t] = np.mean(np.abs(resid) <= hw)

    return {"ase": ase, "td": td, "ci_width": ci_width, "ci_cover": ci_cover,
            "ase_mean": float(ase.mean()), "td_mean": float(td.mean()),
            "ci_cover_mean": float(ci_cover.mean())}


def _quantile_normal(p):
    """Acklam's inverse normal CDF (no scipy dependency at call sites)."""
    import math
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def structure_metrics(theta, thetastar):
    """Group-assignment accuracy: hard argmax assignment vs oracle indicator."""
    theta = np.asarray(theta, float)
    thetastar = np.asarray(thetastar, float)
    P, L = theta.shape
    assign = np.argmax(theta, axis=1)
    oracle = np.argmax(thetastar, axis=1)
    # resolve ties/oracle with multiple groups: match by indicator rows
    acc = float(np.mean(assign == oracle))
    # per-group recall of the oracle support
    recall = np.zeros(L)
    for l in range(L):
        mask = thetastar[:, l] > 0
        if mask.any():
            recall[l] = np.mean(assign[mask] == l)
    return {"assignment_accuracy": acc, "group_recall": recall.tolist(),
            "assign": assign.tolist(), "oracle": oracle.tolist()}
