# -*- coding: utf-8 -*-
"""Euclidean projection onto the unit simplex (Condat, 2015).

Used as the proximal operator of the upper-level simplex constraint on the
group-structure parameter theta (each task's column of theta lives on the
probability simplex over the P features x L groups).
"""

import numpy as np


def proj_unit_simplex(v, axis=-1):
    """Project each vector along `axis` onto the unit simplex.

    Row-wise exact projection: theta_i <- max(theta_i - tau, 0) with
    tau = (sum of the rho largest entries - 1) / rho, following
    Condat (2015), "Fast projection onto the simplex and the l1 ball".

    Supports 1-D input or 2-D input (projection applied row-wise with
    axis=-1, or column-wise with axis=0).
    """
    v = np.asarray(v, dtype=float)
    if axis == 0 and v.ndim == 2:
        return np.ascontiguousarray(proj_unit_simplex(v.T, axis=-1).T)
    u = np.sort(v, axis=-1)[..., ::-1]
    n = v.shape[-1]
    idx = np.arange(1, n + 1, dtype=u.dtype)
    css = np.cumsum(u, axis=-1)
    cond = u * idx > (css - 1.0)
    # last index where cond holds -> reverse-argmax trick
    rev = cond[..., ::-1]
    rho = (n - 1) - np.argmax(rev, axis=-1)
    css_rho = np.take_along_axis(css, rho[..., None], axis=-1)[..., 0]
    tau = (css_rho - 1.0) / (rho.astype(u.dtype) + 1.0)
    return np.maximum(v - tau[..., None], 0.0)
