"""Small CPU checks. Run:  python -m pytest -q tests"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from chains import Config, World
from chains.analysis import complement_map


def test_complement_map_comes_from_frequencies():
    cm = complement_map(Config())
    assert cm == {"A": "T", "T": "A", "C": "G", "G": "C"}
    # move the pump: no pair sums to it any more
    assert all(v is None for v in complement_map(Config(Omega=0.076)).values())


def test_empty_field_stays_zero_and_field_is_stable():
    w = World(Config(N=32, device="cpu"))
    for _ in range(50):
        w.step()
    assert float(w.psi.abs().max()) == 0.0
    w = World(Config(N=32, device="cpu"))
    w.add_monomers([0, 1, 2, 3])
    for _ in range(3000):
        w.step()
    assert torch.isfinite(w.psi).all() and float(w.psi.abs().max()) < 10


def test_lone_monomer_feels_no_self_force():
    w = World(Config(N=32, device="cpu", kT=0.0))
    w.add_monomers([2], pos=[(16.3, 15.7)])
    w.pinned[:] = True
    acc = torch.zeros(2)
    for t in range(4000):
        w.step()
        if t > 2000:
            acc += w.field_force[0]
    assert float(acc.norm() / 2000) < 1e-4


def test_template_chain_is_bonded_in_order():
    w = World(Config(N=64, device="cpu"))
    ids = w.add_chain("GATTACA", (10, 30))
    paths = w.chains()
    assert len(paths) == 1 and len(paths[0]) == 7
    seq = w.sequence(paths[0])
    assert seq in ("GATTACA", "GATTACA"[::-1])


def test_ligation_is_type_blind_and_respects_two_ends():
    cfg = Config(N=32, device="cpu", kT=0.0, p_lig=1.0, lig_every=1, lig_score=4, pump_on=False, held_min=-1.0)
    w = World(cfg)
    # three identical letters in a row at the backbone spacing -> one chain of 3
    w.add_monomers([0, 0, 0], pos=[(10, 16), (13.6, 16), (17.2, 16)])
    w.mobility_off = True
    cfg.mobility = 0.0
    for _ in range(12):
        w.step()
    assert sorted(len(p) for p in w.chains()) == [3]
    assert int((w.nb >= 0).sum(1).max()) <= 2


def test_passing_encounter_does_not_ligate():
    cfg = Config(N=32, device="cpu", kT=0.0, p_lig=1.0, lig_every=1, lig_score=30, pump_on=False, held_min=-1.0)
    w = World(cfg)
    w.add_monomers([0, 1], pos=[(10, 16), (13.6, 16)])
    cfg.mobility = 0.0
    for _ in range(20):          # shorter than the hold time
        w.step()
    assert w.ligations == 0


def test_unheld_monomers_never_link():
    # pump off: nobody is held, so even a perfectly placed pair stays apart
    cfg = Config(N=32, device="cpu", kT=0.0, p_lig=1.0, lig_every=1, lig_score=2, pump_on=False)
    w = World(cfg)
    w.add_monomers([0, 3], pos=[(10, 16), (13.6, 16)])
    cfg.mobility = 0.0
    for _ in range(200):
        w.step()
    assert w.ligations == 0


def test_no_bond_after_the_pair_has_left():
    # a pair builds up a score at the backbone spacing, then is moved apart: it must not link
    cfg = Config(N=32, device="cpu", kT=0.0, p_lig=1.0, lig_every=1, lig_score=5,
                 pump_on=False, held_min=-1.0, mobility=0.0)
    w = World(cfg)
    w.add_monomers([0, 3], pos=[(10, 16), (13.6, 16)])
    cfg.p_lig = 0.0
    for _ in range(10):          # score climbs to ~10 but linking is switched off
        w.step()
    w.pos[1, 0] = 20.0           # pull them apart
    cfg.p_lig = 1.0
    for _ in range(3):
        w.step()
    assert w.ligations == 0
