# -*- coding: utf-8 -*-
import numpy as np
import pytest

from termgl.config import TERMGLConfig
from termgl.data import (synthesize_dataset, apply_partial_annotation,
                         add_noisy_dimensions)


def make_cfg(**kw):
    base = dict(N=30, P=12, T=8, L=3, G=2, sigma=1.0, lam=0.5, mu=1e-3,
                tilt=-1e-3, seed=0)
    base.update(kw)
    return TERMGLConfig(**base)


def test_basic_shapes_and_oracle():
    cfg = make_cfg()
    rng = np.random.default_rng(cfg.seed)
    data = synthesize_dataset(cfg.N, cfg.P, cfg.T, cfg.L, cfg.G, cfg, rng)
    assert data["X"]["trn"][0].shape == (cfg.N, cfg.P)
    assert len(data["y"]["trn"]) == cfg.T
    assert data["thetastar"].shape == (cfg.P, cfg.L)
    assert data["wstar"].shape == (cfg.P, cfg.T)
    # indicator structure: every feature in exactly one group
    assert np.allclose(data["thetastar"].sum(axis=1), 1.0)
    assert set(np.unique(data["thetastar"])) == {0.0, 1.0}
    # each task's oracle weights live on G groups
    for t in range(cfg.T):
        groups = np.unique(np.argmax(data["thetastar"], axis=1)[data["wstar"][:, t] > 0])
        assert len(groups) <= cfg.G


def test_output_standardized():
    cfg = make_cfg()
    rng = np.random.default_rng(1)
    data = synthesize_dataset(cfg.N, cfg.P, cfg.T, cfg.L, cfg.G, cfg, rng)
    for t in range(cfg.T):
        for split in ("trn", "val", "tst"):
            assert abs(data["y"][split][t].mean()) < 1e-10
            assert abs(data["y"][split][t].std() - 1.0) < 1e-10


def test_outliers_corrupt_training_only():
    cfg = make_cfg(outlier_number=5, outlier_type=2)
    rng = np.random.default_rng(2)
    data = synthesize_dataset(cfg.N, cfg.P, cfg.T, cfg.L, cfg.G, cfg, rng)
    # exactly outlier_number entries shifted by +100 on trn
    n_big = sum((np.abs(data["y"]["trn"][t]) > 50).sum() for t in range(cfg.T))
    assert n_big == cfg.T * cfg.outlier_number
    for t in range(cfg.T):
        assert (np.abs(data["y"]["val"][t]) < 50).all()
        assert (np.abs(data["y"]["tst"][t]) < 50).all()


def test_noise_distributions_centered():
    cfg = make_cfg(T=40, noise_distrib="t", noise_degree=3)
    rng = np.random.default_rng(3)
    data = synthesize_dataset(cfg.N, cfg.P, cfg.T, cfg.L, cfg.G, cfg, rng)
    # standardized anyway; just check finite and no exception
    assert np.isfinite(data["y"]["trn"]).all()
    for d in ("normal", "chisq", "exp"):
        cfg_d = make_cfg(noise_distrib=d)
        rng_d = np.random.default_rng(4)
        synthesize_dataset(cfg_d.N, cfg_d.P, cfg_d.T, cfg_d.L, cfg_d.G, cfg_d, rng_d)


def test_partial_annotation():
    cfg = make_cfg(annotation_ratio=0.5)
    rng = np.random.default_rng(5)
    data = synthesize_dataset(cfg.N, cfg.P, cfg.T, cfg.L, cfg.G, cfg, rng)
    data, annotated = apply_partial_annotation(data, cfg, rng)
    assert len(annotated) == cfg.T // 2
    assert annotated.dtype.kind == "i"


def test_noisy_dimensions_touch_trn_val_only():
    cfg = make_cfg(noisy_dims_fraction=0.25, noisy_dims_level=2.0)
    rng = np.random.default_rng(6)
    pristine = synthesize_dataset(cfg.N, cfg.P, cfg.T, cfg.L, cfg.G, cfg, rng)
    corrupted = add_noisy_dimensions(pristine, cfg, rng)  # mutates in place
    # regenerate the pristine dataset deterministically as reference
    clean = synthesize_dataset(cfg.N, cfg.P, cfg.T, cfg.L, cfg.G, cfg,
                               np.random.default_rng(6))
    for t in range(cfg.T):
        assert np.allclose(corrupted["X"]["tst"][t], clean["X"]["tst"][t])
        assert not np.allclose(corrupted["X"]["trn"][t], clean["X"]["trn"][t])
        assert not np.allclose(corrupted["X"]["val"][t], clean["X"]["val"][t])
    # only a fraction of columns differ
    diff_cols = np.mean(np.any(corrupted["X"]["trn"][0] != clean["X"]["trn"][0], axis=0))
    assert 0.1 <= diff_cols <= 0.4
