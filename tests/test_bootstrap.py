"""bootstrap CI 的纯 CPU 单测。"""
import numpy as np

from metrics import bootstrap_ci


def test_bootstrap_constant_values():
    lo, hi = bootstrap_ci([1.0, 1.0, 1.0, 1.0], n_bootstrap=200, seed=0)
    assert np.isclose(lo, 1.0)
    assert np.isclose(hi, 1.0)


def test_bootstrap_empty_and_disabled():
    lo, hi = bootstrap_ci([], n_bootstrap=200, seed=0)
    assert np.isnan(lo) and np.isnan(hi)
    lo, hi = bootstrap_ci([0.0, 1.0], n_bootstrap=0, seed=0)
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_interval_contains_mean():
    values = np.array([0, 1] * 50, dtype=float)
    lo, hi = bootstrap_ci(values, n_bootstrap=500, seed=1)
    assert lo <= values.mean() <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_reproducible_with_seed():
    # 连续值避免 0/1 小样本的分位数落在同一格点
    values = np.random.default_rng(0).random(100)
    a = bootstrap_ci(values, n_bootstrap=300, seed=42)
    b = bootstrap_ci(values, n_bootstrap=300, seed=42)
    c = bootstrap_ci(values, n_bootstrap=300, seed=7)
    assert a == b
    assert a != c
