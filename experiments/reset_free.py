"""Memory without a reset: saturation (injection-lock) writes from random initial phases.

phase_memory.py starts every trial from the same warmed state (a reset). Here every trial
starts from RANDOM bead phases (then 800 steps to relock the pump-locked pair sums), so
the neutral difference phases start anywhere.

  weak write   : amp 0.2, BPSK (+/- carrier), both registers at once (phase_memory default).
                 Measured: the direction a weak write pushes depends on the starting phase.
  strong write : injection lock. A carrier strong/long enough locks the addressed bead to
                 the carrier, erasing the old state, so the final difference phase depends on
                 the carrier phase only. A pi flip of the carrier moves T by pi and A by -pi
                 (the pump locks the sum), i.e. the difference by 2pi = nothing; so the bit is
                 a QUARTER turn of carrier phase (0 vs pi/2). The C/G carrier scrambles the
                 A/T register and not much vice versa, so C/G is written first, then A/T.

Readout as phase_memory.py (near probes, demod at letter frequencies and pump sidebands,
8-fold CV over trials, probe noise sigma). Bare field gets the same writes.

    python experiments/reset_free.py --trials 96
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
torch.set_num_threads(1)

from chains.probing import FieldProbes, source_kernel
from experiments.bond_ablation import fork_world
from experiments.phase_memory import build, probes_for, demod, cv_decode


def schedule(scheme, bits, args):
    """List of (n_steps, fn(t) -> source amplitude), t = world clock (pump-referenced)."""
    b1, b2 = bits
    if scheme == "weak":
        return [(args.weak_write, lambda t: args.weak_amp * (b1 * math.cos(0.03 * t) + b2 * math.cos(0.09 * t)))]
    p1, p2 = (math.pi / 2 if b1 > 0 else 0.0), (math.pi / 2 if b2 > 0 else 0.0)
    return [(args.strong_write, lambda t: args.strong_amp * math.cos(0.09 * t + p2)),
            (args.strong_write, lambda t: args.strong_amp * math.cos(0.03 * t + p1))]


def run(args, base, xs, beads, bits, i, scheme):
    w = fork_world(base)
    if beads:
        g = torch.Generator().manual_seed(40_000 + i)
        w.theta = torch.rand(w.P, generator=g, dtype=w.theta.dtype) * 2 * math.pi
    w.cfg.phase_noise = args.phase_noise
    w.gen.manual_seed(50_000 + i)
    for _ in range(args.relock):
        w.step()
    kernel = source_kernel(w, (args.N / 2, args.N / 2 + args.source_dy), 1.0)
    for n, fn in schedule(scheme, bits, args):
        for _ in range(n):
            w.step(external_source=kernel * float(fn(int(w.t))))
    near, _ = probes_for(args, xs)
    pn = FieldProbes(w, near)
    out, t = {}, 0
    for D in sorted(args.delays):
        while t < D:
            w.step(); t += 1
        t0 = int(w.t)
        rec = []
        for _ in range(args.read):
            w.step(); t += 1
            rec.append(pn.field(w).numpy().copy())
        out[D] = (np.array(rec), t0)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequence", default="ACGT")
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spacing", type=float, default=3.6)
    p.add_argument("--chain-boost", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--relock", type=int, default=800)
    p.add_argument("--source-dy", type=float, default=2.5)
    p.add_argument("--weak-amp", type=float, default=0.2)
    p.add_argument("--weak-write", type=int, default=600)
    p.add_argument("--strong-amp", type=float, default=1.0)
    p.add_argument("--strong-write", type=int, default=1200)
    p.add_argument("--delays", type=int, nargs="+", default=[600, 4000])
    p.add_argument("--read", type=int, default=400)
    p.add_argument("--trials", type=int, default=96)
    p.add_argument("--phase-noise", type=float, default=0.003)
    p.add_argument("--sigma", type=float, default=1e-3)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--out", default="runs/reset_free")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed + 7)
    pats = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]] * (a.trials // 4 + 1))[: a.trials]
    pats = pats[rng.permutation(len(pats))].astype(float)
    rows = []
    for world_name, beads in (("chain", True), ("bare", False)):
        base, xs = build(a, beads)
        for scheme in ("weak", "strong"):
            recs = [run(a, base, xs, beads, pats[i], i, scheme) for i in range(a.trials)]
            for D in a.delays:
                X = np.array([demod(recs[i][D][0], a.sigma, np.random.default_rng(60_000 + i), recs[i][D][1])
                              for i in range(a.trials)])
                acc = {"b1": cv_decode(X, pats[:, 0], a.ridge), "b2": cv_decode(X, pats[:, 1], a.ridge)}
                rows.append({"world": world_name, "scheme": scheme, "delay": D, **acc})
                print(f"{world_name:5s} {scheme:6s} write, random init, delay={D:5d}: "
                      f"b1 (A/T)={acc['b1']:.2f}  b2 (C/G)={acc['b2']:.2f}", flush=True)
            json.dump({"parameters": vars(a), "rows": rows}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
