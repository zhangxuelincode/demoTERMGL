# -*- coding: utf-8 -*-
import numpy as np
import pytest

from termgl.projection import proj_unit_simplex


def test_sums_to_one_and_nonneg():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(20, 5))
    p = proj_unit_simplex(v)
    assert p.shape == v.shape
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p >= 0)


def test_already_on_simplex_is_fixed_point():
    v = np.full((4, 3), 1.0 / 3)
    assert np.allclose(proj_unit_simplex(v), v)


def test_known_value():
    # Condat-style worked example: v=[0.6,0.5,0.4], rho=2, tau=(1.5-1)/3
    v = np.array([0.6, 0.5, 0.4])
    p = proj_unit_simplex(v)
    assert np.allclose(p, v - 0.5 / 3.0)
    assert np.allclose(p.sum(), 1.0)


def test_sparse_projection():
    v = np.array([2.0, 1.0, 0.5])
    p = proj_unit_simplex(v)
    # rho=1, tau=1 -> [1, 0, 0]
    assert np.allclose(p, [1.0, 0.0, 0.0])


def test_axis0_equivalent_to_transpose():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(6, 4))
    assert np.allclose(proj_unit_simplex(v, axis=0),
                       proj_unit_simplex(v.T, axis=-1).T)


def test_1d_input():
    p = proj_unit_simplex(np.array([0.2, 0.2, 0.2, 0.2]))
    assert np.allclose(p, 0.25)
    with pytest.raises(Exception):
        proj_unit_simplex("bad")
