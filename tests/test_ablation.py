"""Tests for counterfactual topology ablation.

These tests require the actual World and run on CPU.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from chains import Config, World
from experiments.bond_ablation import build_counterfactuals, fork_world, measure
from experiments.sequence_response import layouts


def tiny_world():
    cfg = Config(N=48, seed=12, device="cpu", p_lig=0.0, kT=0.0,
                 phase_noise=0.0, chain_boost=0.0)
    w = World(cfg)
    w.add_chain("ACGT", (15.0, 24.0))
    for _ in range(6):
        w.step()
    return w


def test_fork_is_identical_but_not_aliased():
    original = tiny_world()
    copy = fork_world(original)
    assert copy.t == original.t
    for key in ("pos", "theta", "psi", "psi_prev", "nb", "held_by"):
        assert torch.equal(getattr(original, key), getattr(copy, key))
        assert getattr(original, key).data_ptr() != getattr(copy, key).data_ptr()
    assert torch.equal(original.gen.get_state(), copy.gen.get_state())
    assert original.rng.bit_generator.state == copy.rng.bit_generator.state
    copy.nb.fill_(-1)
    copy.cfg.k_bond = 0
    assert bool((original.nb >= 0).any()) and original.cfg.k_bond != 0
    # No intervention: two exact counterfactual forks must evolve identically.
    a, b = fork_world(original), fork_world(original)
    for _ in range(5):
        a.step()
        b.step()
    assert torch.equal(a.psi, b.psi)
    assert torch.equal(a.pos, b.pos)


def test_cut_matches_sham_without_bond_emission_boost():
    base = tiny_world()
    states = build_counterfactuals(base, freeze=False)
    cut = states[("cut", "driven")]
    sham = states[("sham", "driven")]
    assert (cut.nb == -1).all()
    assert sham.nb.ge(0).any()
    for _ in range(30):
        cut.step()
        sham.step()
    # Mechanical bonds are the only remaining difference: both are disabled.
    assert torch.allclose(cut.psi, sham.psi, atol=1e-6)
    assert torch.allclose(cut.pos, sham.pos, atol=1e-6)


def test_frozen_chains_have_zero_causal_bond_effect():
    seqs, positions, source, sensors = layouts("ACGT", "AGCT", 48, 3.6)
    args = SimpleNamespace(N=48, device="cpu", warmup=5, steps=25,
                           sample_every=5, freeze=True, kind="impulse",
                           amplitude=0.05, omega=0.11, start=0, duration=10,
                           width=1.0)
    stats, clock, drive, readings, interaction = measure(
        args, seqs[0], positions, source, sensors, seed=3)
    assert clock.tolist() == [5, 10, 15, 20, 25]
    assert np.array_equal(interaction, np.zeros_like(interaction))
    assert stats["sham_minus_cut_rms"] == 0.0
    assert stats["interaction_output_rms"] == 0.0
    assert stats["bond_cut_geometry_mean_distance"] == 0.0
    assert readings[("bonded", "driven")]["field"].shape == (5, 5)
