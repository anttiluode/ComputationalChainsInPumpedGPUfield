"""Synthetic checks for the capacity instruments (NumPy only, fast)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiments.memory_capacity import memory_curves, shift_null_pvalue

ARGS = argparse.Namespace(max_lag=8, train_fraction=0.7, discard=20, ridge=0.01)


def linear_reservoir(bits, n_feat=20, seed=0):
    """Features = random linear filters of the past bits plus input-independent clocks."""
    rng = np.random.default_rng(seed)
    n = len(bits)
    taps = rng.normal(size=(12, n_feat)) * np.exp(-np.arange(12) / 4)[:, None]
    X = np.zeros((n, n_feat))
    for k in range(12):
        X[k:] += bits[: n - k, None] * taps[k]
    t = np.arange(n)[:, None]
    X += np.cos(t * rng.uniform(0.05, 0.9, size=n_feat)) * 2.0
    return X


def test_linear_reservoir_has_linear_but_no_xor_capacity():
    bits = np.random.default_rng(1).choice([-1.0, 1.0], size=700)
    X = linear_reservoir(bits)
    m = memory_curves(X, bits, ARGS)
    assert m["MC_lin_sig"] > 3.0
    assert m["MC_nl_sig"] < 0.3
    _, _, _, p_lin = shift_null_pvalue(X, bits, ARGS, task="lin", draws=20)
    assert p_lin == 0.0


def test_planted_xor_is_detected_by_shift_null():
    bits = np.random.default_rng(2).choice([-1.0, 1.0], size=700)
    X = linear_reservoir(bits)
    n = len(bits)
    idx = np.arange(n)
    planted = 0.05 * X.std() * (bits[idx - 3] * bits[idx - 4])[:, None]
    obs, null_mean, null95, p = shift_null_pvalue(np.hstack([X, planted]), bits, ARGS, draws=20)
    assert obs > 0.9 and obs > null95 and p == 0.0


def test_gate_threshold_is_never_negative():
    bits = np.random.default_rng(3).choice([-1.0, 1.0], size=300)
    X = np.random.default_rng(4).normal(size=(300, 60))     # overfitting regime
    m = memory_curves(X, bits, ARGS)
    assert m["null_r2_99"] >= 0.0
    assert m["MC_nl_sig"] >= 0.0 and m["MC_lin_sig"] >= 0.0
