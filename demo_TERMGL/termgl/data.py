# -*- coding: utf-8 -*-
"""Synthetic data generation following SynthesizeDataset.m of the paper.

Multi-task regression with a shared unknown group structure:
    task t:  y = X w_t + noise,   w_t supported on G random groups.

Beyond the paper's four noise distributions and outliers, this module adds the
two stress factors required by the experiments:

* partial annotation (`annotation_ratio`): only a fraction of tasks exposes
  labels to the bilevel procedure; the learned structure is transferred to the
  remaining tasks, whose labels are only used for final evaluation;
* noisy dimensions (`noisy_dims_fraction`): a random subset of feature columns
  of the training/validation designs is corrupted with extra Gaussian noise
  (test designs stay clean), so the robust tilted loss must still identify the
  informative groups.
"""

import numpy as np


def _group_assignment(P, L, rng, distrib="equal"):
    """Partition features into L groups; returns the P x L indicator matrix."""
    if distrib == "equal":
        assert P % L == 0, "'equal' grouping needs P multiple of L"
        idx = np.tile(np.arange(L), P // L)
    elif distrib == "rand":
        size = rng.normal(size=L)
        size = np.round(P * np.exp(size) / np.exp(size).sum()).astype(int)
        size[0] += P - size.sum()
        size[size < 0] = 0
        # repair rounding so the partition covers all P features
        diff = P - size.sum()
        order = np.argsort(-size)
        for i in range(abs(int(diff))):
            size[order[i % L]] += np.sign(diff)
        idx = np.repeat(np.arange(L), size)
        rng.shuffle(idx)
    else:
        raise ValueError(distrib)
    theta = np.zeros((P, L))
    theta[np.arange(P), idx] = 1.0
    return theta


def _sample_noise(distrib, degree, param, size, rng):
    if distrib == "normal":
        return rng.normal(param[0], np.sqrt(param[1]), size=size)
    if distrib == "t":
        return rng.standard_t(degree, size=size)
    if distrib == "chisq":
        return rng.chisquare(degree, size=size) - degree  # centered
    if distrib == "exp":
        return rng.exponential(2.0, size=size) - 2.0      # centered (MATLAB exprnd(2))
    raise ValueError(distrib)


def synthesize_dataset(N, P, T, L, G, cfg, rng):
    """Returns dict with X/y splits per task, oracle structure/weights.

    Per task: X.trn / X.val / X.tst ~ N(0,1) of shape (N, P);
    w_t has entries 1 on the features of G randomly drawn groups (oracle);
    outputs add noise with the paper's distributions and optional outliers.
    Outputs of each split are standardized as in SynthesizeDataset.m.
    """
    thetastar = _group_assignment(P, L, rng, "equal")

    X = {"trn": [], "val": [], "tst": []}
    y = {"trn": [], "val": [], "tst": [], "true": []}
    # standardization constants per split, needed to map raw predictions
    # X @ w back into the standardized response space at evaluation time
    ystats = {"trn": [], "val": [], "tst": [], "true": []}
    wstar = np.zeros((P, T))

    for t in range(T):
        groups = rng.choice(L, size=G, replace=False)
        w = np.zeros(P)
        for g in groups:
            w[thetastar[:, g] > 0] = 1.0
        wstar[:, t] = w

        Xt, Xv, Xs = (rng.normal(0.0, 1.0, size=(N, P)) for _ in range(3))
        noise_scale = cfg.noise_percentage
        yt = Xt @ w + noise_scale * _sample_noise(cfg.noise_distrib, cfg.noise_degree,
                                                  cfg.noise_param, N, rng)
        yv = Xv @ w + noise_scale * _sample_noise(cfg.noise_distrib, cfg.noise_degree,
                                                  cfg.noise_param, N, rng)
        ys = Xs @ w + noise_scale * _sample_noise(cfg.noise_distrib, cfg.noise_degree,
                                                  cfg.noise_param, N, rng)
        raw_true = Xs @ w

        # standardize outputs (as in the MATLAB generator), keeping stats
        stats = []
        for v in (yt, yv, ys, raw_true):
            m, s = v.mean(), v.std() + np.finfo(float).eps
            stats.append((float(m), float(s)))
        yt = (yt - stats[0][0]) / stats[0][1]
        yv = (yv - stats[1][0]) / stats[1][1]
        ys = (ys - stats[2][0]) / stats[2][1]
        ytrue = (raw_true - stats[3][0]) / stats[3][1]

        # outliers on the training responses (paper: percentage of corrupted pts)
        if cfg.outlier_number > 0:
            idx = rng.choice(N, size=cfg.outlier_number, replace=False)
            if cfg.outlier_type == 1:
                yt[idx] *= 100.0
                Xt[idx] *= 10.0
            else:
                yt[idx] += 100.0

        X["trn"].append(Xt); X["val"].append(Xv); X["tst"].append(Xs)
        y["trn"].append(yt); y["val"].append(yv); y["tst"].append(ys)
        y["true"].append(ytrue)
        for split, (m, s) in zip(("trn", "val", "tst", "true"), stats):
            ystats[split].append((m, s))

    return {"X": X, "y": y, "ystats": ystats,
            "thetastar": thetastar, "wstar": wstar}


def apply_partial_annotation(data, cfg, rng):
    """Mask a fraction of tasks as unannotated (labels hidden from training).

    Returns (data, annotated_idx). Unannotated tasks keep labels only for the
    final evaluation; the bilevel structure learning never sees them.
    """
    T = len(data["y"]["trn"])
    n_annotated = max(1, int(round(cfg.annotation_ratio * T)))
    annotated = np.sort(rng.choice(T, size=n_annotated, replace=False))
    return data, annotated


def add_noisy_dimensions(data, cfg, rng):
    """Corrupt a fraction of feature columns of trn/val designs with extra
    Gaussian noise (test designs untouched)."""
    frac = cfg.noisy_dims_fraction
    if frac <= 0:
        return data
    P = data["X"]["trn"][0].shape[1]
    n_noisy = max(1, int(round(frac * P)))
    dims = rng.choice(P, size=n_noisy, replace=False)
    for split in ("trn", "val"):
        for k, Xk in enumerate(data["X"][split]):
            Xk = Xk.copy()
            Xk[:, dims] += cfg.noisy_dims_level * rng.normal(0.0, 1.0, Xk[:, dims].shape)
            data["X"][split][k] = Xk
    return data
