"""CPU regression checks for controlled injection and response measurements."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch
from chains import Config, World
from chains.probing import FieldProbes, Stimulus, record_response, source_kernel
from experiments.sequence_response import layouts


def test_stimulus_timing_and_validation():
    s = Stimulus((8, 8), kind="impulse", amplitude=2.5, start=3, duration=10)
    assert [s.value(t) for t in range(6)] == [0, 0, 0, 2.5, 0, 0]
    s = Stimulus((8, 8), kind="samples", amplitude=2, start=1,
                 duration=3, samples=(0.0, -1.0, 0.5))
    assert [s.value(t) for t in range(5)] == [0, 0, -2, 1, 0]
    assert Stimulus((1, 2), kind="sine", omega=math.pi, duration=2).value(1) == -0.05
    with pytest.raises(ValueError, match="samples"):
        Stimulus((0, 0), kind="samples", duration=2, samples=(1,))
    with pytest.raises(ValueError):
        Stimulus((0, 0), kind="other")


def test_kernel_is_periodic_and_normalized():
    world = World(Config(N=32, device="cpu"))
    a = source_kernel(world, (0.25, 31.75), 0.8)
    b = source_kernel(world, (32.25, -0.25), 0.8)
    assert torch.allclose(a, b, atol=2e-6)
    assert float(a.sum()) == pytest.approx(1.0, abs=1e-6)
    assert a[0, 0] > a[16, 16]
    spike = source_kernel(world, (31.8, -0.3), 0.0)
    assert spike[0, 0] == 1 and spike.sum() == 1


def test_bilinear_probe_wraps_edges():
    world = World(Config(N=32, device="cpu"))
    world.psi.zero_()
    world.psi[0, 31] = 10
    world.psi[0, 0] = 20
    world.psi[1, 31] = 30
    world.psi[1, 0] = 40
    p = FieldProbes(world, [(31.5, 0.5)])
    assert p.field(world).item() == pytest.approx(25.0)


def test_external_impulse_enters_field_once_and_old_path_stays_empty():
    w = World(Config(N=32, device="cpu"))
    ctrl = World(Config(N=32, device="cpu"))
    s = Stimulus((16, 16), kind="impulse", amplitude=0.5, duration=5, width=0)
    recorded = record_response(w, s, [(16, 16)], steps=3)
    reference = record_response(ctrl, None, [(16, 16)], steps=3)
    assert recorded["step"].tolist() == [1, 2, 3]
    assert recorded["drive"].tolist() == [0.5, 0.0, 0.0]
    assert recorded["field"][0, 0] == pytest.approx(0.5)
    assert np.array_equal(reference["field"], np.zeros((3, 1)))
    assert float(w.psi.abs().sum()) > 0


def test_matched_seed_and_null_bond_control_for_pinned_chains():
    cfg = Config(N=48, device="cpu", seed=4, chain_boost=0,
                 kT=0, phase_noise=0, p_lig=0)
    seq = "ACGT"
    positions = [(16 + i * cfg.d0, 24) for i in range(len(seq))]
    bonded = World(cfg)
    bonded.add_chain(seq, positions[0])
    bonded.pinned[:] = True
    unbonded = World(cfg)
    unbonded.add_monomers([unbonded.letters.index(ch) for ch in seq], pos=positions)
    unbonded.pinned[:] = True
    stim = Stimulus((11, 24), kind="sine", duration=35)
    sensors = [(32, 24), (35, 26)]
    r1 = record_response(bonded, stim, sensors, steps=45, warmup=10, sample_every=5)
    r2 = record_response(unbonded, stim, sensors, steps=45, warmup=10, sample_every=5)
    assert r1["field"].shape == (9, 2)
    assert np.allclose(r1["field"], r2["field"], atol=1e-5)
    assert np.max(np.abs(r1["field"])) > 0


def test_sequence_comparison_rejects_mismatched_counts():
    with pytest.raises(ValueError, match="letter counts"):
        layouts("AACC", "TTCC", 96, 3.6)
    sequences, positions, source, sensors = layouts("ACGT", "AGCT", 96, 3.6)
    assert sequences == ("ACGT", "AGCT") and len(positions) == 4 and len(sensors) == 5
