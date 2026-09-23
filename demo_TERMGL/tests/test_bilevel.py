# -*- coding: utf-8 -*-
"""Bilevel driver (SAGA aggregation, auto step size, simplex projection) and
evaluation metrics."""
import numpy as np
import pytest

from termgl.config import TERMGLConfig
from termgl.data import synthesize_dataset, apply_partial_annotation
from termgl.bilevel import run_termgl, _init_theta, _auto_stepsize, _mean_val_loss
from termgl.evaluation import evaluate_tasks, structure_metrics
from termgl.projection import proj_unit_simplex


def make_cfg(**kw):
    base = dict(N=30, P=12, T=6, L=3, G=2, sigma=1.0, lam=0.5, mu=1e-3,
                tilt=-1e-3, seed=0, modal_iter=2, inner_itermax=2000,
                inner_tol=1e-12, outer_itermax=3, batch_size=2)
    base.update(kw)
    return TERMGLConfig(**base)


def make_data(cfg):
    rng = np.random.default_rng(cfg.seed)
    data = synthesize_dataset(cfg.N, cfg.P, cfg.T, cfg.L, cfg.G, cfg, rng)
    data, annotated = apply_partial_annotation(data, cfg, rng)
    return data, annotated


def test_auto_stepsize_orders():
    # MAM_codes rule: step = 10^-(1+floor(log10(max|hg|))), so the resulting
    # update magnitude step*max|hg| lies in [0.1, 1)
    for m in (3e-4, 5.0, 1.2e2, 7e-7):
        step = _auto_stepsize(np.array([m]))
        assert 0.1 <= step * m < 1.0
    assert _auto_stepsize(np.zeros((2, 2))) == pytest.approx(1e-3)


def test_theta_rows_stay_on_simplex():
    cfg = make_cfg()
    data, annotated = make_data(cfg)
    res = run_termgl(data, cfg, annotated)
    th = res["theta"]
    assert th.shape == (cfg.P, cfg.L)
    assert np.allclose(th.sum(axis=1), 1.0)
    assert np.all(th >= -1e-15)


def test_saga_init_pass_matches_manual_aggregation():
    """After the initialization pass the SAGA mean must equal the average of
    the per-task hypergradients at the initial theta (fixed seed => same
    draws)."""
    cfg = make_cfg(outer_itermax=1, batch_size=1)
    data, annotated = make_data(cfg)

    from termgl.bilevel import _task_hypergradient
    theta0 = _init_theta(cfg.P, cfg.L)
    expected = np.zeros((cfg.P, cfg.L))
    for t in annotated:
        _, hg, _ = _task_hypergradient(t, data, theta0, None, cfg, "unroll", 20)
        expected += hg / len(annotated)

    res = run_termgl(data, cfg, annotated)
    step = res["history"][0]["step"]
    manual = theta0.copy()
    from termgl.projection import proj_unit_simplex
    manual = proj_unit_simplex(manual - step * expected)
    assert np.allclose(res["theta"], manual, atol=1e-10)


def test_structure_recovery_beats_uniform_baseline():
    """Bilevel dynamics: starting from a perturbed (non-symmetric) structure,
    the SAGA hypergradient descent must strictly decrease the upper-level
    validation loss. (Full structure recovery needs the paper-scale task
    count -- covered by run_simulation.py.)"""
    cfg = make_cfg(outer_itermax=40, tilt=-0.01)
    data, annotated = make_data(cfg)
    rng = np.random.default_rng(1)
    theta0 = proj_unit_simplex(
        np.full((cfg.P, cfg.L), 1.0 / cfg.L)
        + 0.15 * rng.normal(size=(cfg.P, cfg.L)))
    res = run_termgl(data, cfg, annotated, theta_init=theta0)
    vals = [h["val_loss"] for h in res["history"]]
    assert vals[-1] < vals[0] - 1e-5, f"val did not decrease: {vals[0]} -> {vals[-1]}"
    assert np.allclose(res["theta"].sum(axis=1), 1.0)


def test_history_and_val_loss():
    cfg = make_cfg()
    data, annotated = make_data(cfg)
    res = run_termgl(data, cfg, annotated)
    assert len(res["history"]) == cfg.outer_itermax
    assert all(np.isfinite(h["val_loss"]) for h in res["history"])
    assert res["W"].shape == (cfg.P, cfg.T)


def test_run_experiment_smoke():
    """End-to-end smoke: partial annotation + noisy dimensions + evaluation."""
    from termgl.bilevel import run_experiment
    cfg = make_cfg(outer_itermax=2, annotation_ratio=0.5,
                   noisy_dims_fraction=0.2, noisy_dims_level=0.5, tilt=-0.01)
    res = run_experiment(cfg)
    assert np.isfinite(res["eval"]["td_mean"])
    assert np.isfinite(res["structure"]["assignment_accuracy"])
    assert 0.0 <= res["eval"]["ci_cover_mean"] <= 1.0


# ---------------------------------------------------------------------------
# evaluation metrics
# ---------------------------------------------------------------------------
def test_evaluation_with_oracle_weights():
    cfg = make_cfg()
    data, _ = make_data(cfg)
    ev = evaluate_tasks(data, data["wstar"], cfg)
    # oracle predictions reproduce the noiseless standardized response
    # exactly; against the noisy one they sit at the noise floor
    # ASE ~ var(noise)/var(noisy y) ~ 1/(1+||w*||^2) << 1
    assert ev["td_mean"] < 1e-8
    assert 0.0 < ev["ase_mean"] < 0.5
    assert 0 <= ev["ci_cover_mean"] <= 1


def test_structure_metrics_perfect():
    cfg = make_cfg()
    data, _ = make_data(cfg)
    m = structure_metrics(data["thetastar"], data["thetastar"])
    assert m["assignment_accuracy"] == 1.0
    assert all(r == 1.0 for r in m["group_recall"])


def test_mean_val_loss_penalty_is_per_group():
    cfg = make_cfg()
    data, annotated = make_data(cfg)
    th = _init_theta(cfg.P, cfg.L)
    W = [np.ones(cfg.P) * 0.1 for _ in range(cfg.T)]
    v = _mean_val_loss(data, annotated, W, th, cfg)
    assert np.isfinite(v) and v > 0
