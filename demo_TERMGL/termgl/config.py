# -*- coding: utf-8 -*-
"""Configuration mirroring demo.m of the original MATLAB TERMGL repository."""

from dataclasses import dataclass, field


@dataclass
class TERMGLConfig:
    # ----- synthesis parameters (demo.m) -----
    N: int = 50                  # samples per task (trn)
    P: int = 50                  # feature dimension
    T: int = 500                 # number of tasks
    L: int = 5                   # number of groups (1 <= L <= P)
    G: int = 2                   # non-zero groups per task
    sigma: float = 1.0           # bandwidth of the (tilted) loss
    lam: float = 0.5             # regularization parameter lambda
    mu: float = 1e-3             # smooth parameter of the (mu/2)||w||^2 ridge
    alpha: float = 0.9           # confidence level for interval coverage
    tilt: float = -1e-3          # TERM hyperparameter t (t < 0 robust tilt)

    # noise on outputs
    noise_distrib: str = "normal"    # 'normal' | 't' | 'chisq' | 'exp'
    noise_degree: int = 2            # dof for 't' / 'chisq'
    noise_param = (0.0, 1.0)         # mean / var for 'normal'
    noise_percentage: float = 1.0    # noise level multiplier
    outlier_number: int = 0          # number of corrupted samples per task
    outlier_type: int = 2            # 1: multiplicative, 2: additive

    # ----- lower level (HQ-DFBB) -----
    modal_iter: int = 2              # M: number of HQ (tilt <-> fit) sweeps
    inner_itermax: int = 1000        # Q: DFBB iterations per modal level
    inner_tol: float = 1e-10         # early-stopping tolerance on DFBB fixed point

    # ----- upper level -----
    outer_itermax: int = 2000        # Z: number of upper iterations
    batch_size: int = 1              # tasks sampled per upper iteration
    outer_stepsize: float = None     # None -> MAM-style automatic step size
    projection: str = "simplex"      # theta space ('simplex' supported)
    display_online: bool = False     # matplotlib live view (headless tests: off)
    verbose: bool = False

    # ----- experimental stress factors (this work) -----
    annotation_ratio: float = 1.0    # fraction of tasks whose labels are seen
                                     # during structure learning (partial labels)
    noisy_dims_fraction: float = 0.0 # fraction of feature columns corrupted in
                                     # trn/val design matrices (test stays clean)
    noisy_dims_level: float = 1.0    # std of the extra corruption noise

    seed: int = None

    def __post_init__(self):
        assert 1 <= self.L <= self.P, "1 <= L <= P required"
        assert self.tilt <= 0.0, "robust tilt requires t <= 0"
        assert 0.0 < self.annotation_ratio <= 1.0
        assert 0.0 <= self.noisy_dims_fraction < 1.0
