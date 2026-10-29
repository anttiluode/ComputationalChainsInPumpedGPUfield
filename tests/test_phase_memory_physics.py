"""The physical basis of the phase memory, as a regression test (~20 s on CPU).

A carrier at a pair's pump complement writes the pair's DIFFERENCE phase, which persists;
the SUM phase is pump-locked and relaxes; the bare field forgets the same write.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from chains import Config, World
from chains.probing import source_kernel
from experiments.bond_ablation import fork_world


def _wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def _world(beads, N=48):
    w = World(Config(N=N, seed=0, device="cpu", p_lig=0.0, kT=0.0, phase_noise=0.0, chain_boost=0.0))
    if beads:
        xs = [N / 2 - 5.4 + 3.6 * i for i in range(4)]
        w.add_monomers([w.letters.index(c) for c in "ACGT"], pos=[(x, N / 2) for x in xs])
        w.pinned[:] = True
    for _ in range(300):
        w.step()
    return w


def _write_pair(w, sign, steps_write=600, steps_wait=2000):
    v = fork_world(w)
    N = v.cfg.N
    k = source_kernel(v, (N / 2, N / 2 + 2.5), 1.0)
    for t in range(steps_write):
        v.step(external_source=k * float(sign * 0.2 * math.cos(0.09 * t)))
    for _ in range(steps_wait):
        v.step()
    return v


def test_complement_carrier_writes_persistent_difference_phase():
    w = _world(True)
    plus, minus = _write_pair(w, +1), _write_pair(w, -1)
    tp, tm = plus.theta, minus.theta            # order A, C, G, T
    d_diff = abs(_wrap(float((tp[1] - tp[2]) - (tm[1] - tm[2]))))
    d_sum = abs(_wrap(float((tp[1] + tp[2]) - (tm[1] + tm[2]))))
    assert d_diff > 1.0, d_diff                 # C/G difference phase remembers the bit
    assert d_sum < 0.5, d_sum                   # C/G sum phase is pump-locked and relaxed


def test_bare_field_forgets_the_same_write():
    w = _world(False)
    plus, minus = _write_pair(w, +1), _write_pair(w, -1)
    assert float((plus.psi - minus.psi).abs().max()) < 1e-3
