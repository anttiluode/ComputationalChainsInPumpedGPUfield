"""Small CPU and pure NumPy checks for the temporal-memory benchmark."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from experiments.temporal_memory import (
    acquire, fit_tasks, make_probes, make_worlds, score,
    standardized_ridge, validate_sequences,
)


def test_probes_are_distal_and_validate_matched_composition():
    probes = make_probes(96, 3.6, 8)
    assert len(probes) == 15
    assert min(x for x, _ in probes) > 60.6
    validate_sequences(["ACGTACGT", "AGCTAGCT"], 96, 3.6)
    with pytest.raises(ValueError, match="composition"):
        validate_sequences(["AAAA", "ACGT"], 96, 3.6)
    with pytest.raises(ValueError, match="too long"):
        validate_sequences(["ACGT" * 8], 96, 3.6)


def test_standardized_linear_readout_uses_training_only():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(100, 3))
    y = 0.5 + x @ np.array([1.0, -2.0, 3.0])
    pred = standardized_ridge(x[:70], x[70:], y[:70], alpha=0.0001)
    assert np.max(np.abs(pred - y[70:])) < 0.0001
    m = score(pred, y[70:], y[:70])
    assert m["test_r2"] > 0.999
    with pytest.raises(ValueError):
        validate_sequences([], 96, 3.6)


def test_delayed_bits_are_predictable_if_state_contains_them():
    n = 220
    bits = np.random.default_rng(1).choice([-1.0, 1.0], size=n)
    feature = np.zeros((n, 1))
    feature[2:, 0] = bits[:-2]
    series = {"bonded": feature, "bare": np.zeros_like(feature)}
    args = SimpleNamespace(discard=20, max_lag=8, train_fraction=0.7, ridge=0.01)
    result = fit_tasks(args, series, bits, seed=1)
    keys = {(r["condition"], r["task"]): r for r in result}
    assert keys[("bonded", "delay_2")]["test_r2"] > 0.99
    assert keys[("bare", "delay_2")]["test_r2"] < 0.2


def test_frozen_bond_cut_sham_null_for_identical_drive():
    args = SimpleNamespace(N=64, seed=0, device="cpu", warmup=5,
                           freeze=True, width=1.0, amplitude=0.05,
                           steps_per_symbol=3)
    seq = "ACGT"
    worlds = make_worlds(args, 0, seq)
    assert worlds["bonded"].pinned.all()
    input_bits = np.array([1, -1, 1, -1, -1], dtype=float)
    probes = make_probes(64, 3.6, 4)
    result = acquire(args, 0, seq, input_bits, probes)
    assert np.allclose(result["bonded"], result["cut"], atol=1e-6)
    assert np.allclose(result["cut"], result["sham"], atol=1e-6)
    assert np.max(np.abs(result["bare"])) > 0
