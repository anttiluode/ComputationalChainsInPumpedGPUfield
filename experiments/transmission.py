"""Does a chain conduct a signal the gapped medium blocks? (sequence-programmed wire)

The field has a gap at w0 = arccos(1 - m2/2) ~ 0.1415 rad/step; anything slower is
evanescent (decay length ~4-6 cells). Through the pumped ear, a bead of frequency
w_i is driven resonantly by field at 2*Omega - w_i (its complement), which is also
sub-gap. So a chain whose NEIGHBOURS are complementary (ATATATAT) is a candidate
resonator waveguide; the same letters blocked (AAAATTTT) should not conduct.

Protocol (geometry frozen after warmup, so bonds/mechanics play no role):
  steady carrier u(t) = A cos(w t) injected `source_offset` cells before bead 0;
  response = driven - undriven twin at far probes (beyond the last bead) and at
  a near probe (input level); measured over the last `measure` steps.
Metrics: far RMS of the response, lock-in amplitude at w, and far/near ratio.

    python experiments/transmission.py --sequences ATATATAT,AAAATTTT --omegas 0.03 0.11
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from chains import Config, World
from chains.probing import FieldProbes, source_kernel
from experiments.bond_ablation import fork_world


def build(args, seed, sequence):
    cfg = Config(N=args.N, seed=seed, device=args.device, p_lig=0.0, kT=0.0,
                 phase_noise=0.0, chain_boost=0.0, Omega=args.Omega)
    w = World(cfg)
    L = len(sequence) if sequence else args.length
    x0 = args.N / 2 - (L - 1) * args.spacing / 2
    if sequence:
        types = [w.letters.index(ch) for ch in sequence]
        w.add_monomers(types, pos=[(x0 + i * args.spacing, args.N / 2) for i in range(L)])
        w.pinned[:] = True
    for _ in range(args.warmup):
        w.step()
    return w, x0, x0 + (L - 1) * args.spacing


def measure(args, seed, sequence, omega):
    base, x0, x1 = build(args, seed, sequence)
    y = args.N / 2
    far = [(x1 + d, y + o) for d in (3, 5, 8) for o in (-2, 0, 2)]
    near = [(x0 - args.source_offset + 1.5, y)]
    drv, und = fork_world(base), fork_world(base)
    kernel = source_kernel(base, (x0 - args.source_offset, y), args.width)
    pf, pn = FieldProbes(base, far), FieldProbes(base, near)
    T = args.drive_steps
    rf, rn, ts = [], [], []
    for t in range(T):
        drv.step(external_source=kernel * float(args.amplitude * np.cos(omega * t)))
        und.step()
        if t >= T - args.measure:
            rf.append((pf.field(drv) - pf.field(und)).cpu().numpy())
            rn.append((pn.field(drv) - pn.field(und)).cpu().numpy())
            ts.append(t)
    rf, rn, ts = np.array(rf), np.array(rn), np.array(ts)
    ref = np.exp(-1j * omega * ts)
    lock_far = np.abs((rf * ref[:, None]).mean(0)) * 2
    lock_near = np.abs((rn * ref[:, None]).mean(0)) * 2
    return {"sequence": sequence or "bare", "seed": seed, "omega": omega,
            "far_rms": float(np.sqrt((rf ** 2).mean())),
            "near_rms": float(np.sqrt((rn ** 2).mean())),
            "far_lock": float(lock_far.mean()), "near_lock": float(lock_near.mean()),
            "ratio_rms": float(np.sqrt((rf ** 2).mean()) / max(np.sqrt((rn ** 2).mean()), 1e-15))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequences", default="ATATATAT,AAAATTTT,CGCGCGCG,CCCCGGGG,ACGTACGT,AAAAAAAA,TTTTTTTT")
    p.add_argument("--omegas", type=float, nargs="+",
                   default=[0.02, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.16, 0.20, 0.25])
    p.add_argument("--N", type=int, default=96)
    p.add_argument("--Omega", type=float, default=0.07)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--drive-steps", type=int, default=2000)
    p.add_argument("--measure", type=int, default=1000)
    p.add_argument("--amplitude", type=float, default=0.05)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--spacing", type=float, default=3.6)
    p.add_argument("--length", type=int, default=8)
    p.add_argument("--source-offset", type=float, default=3.0)
    p.add_argument("--out", default="runs/transmission")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seqs = [s.strip().upper() for s in a.sequences.split(",")] + [""]
    rows = []
    for seed in range(a.seeds):
        for seq in seqs:
            for om in a.omegas:
                r = measure(a, seed, seq, om)
                rows.append(r)
                print(f"seed {seed} {r['sequence']:9s} w={om:.3f} far_rms={r['far_rms']:.2e} "
                      f"far_lock={r['far_lock']:.2e} near_rms={r['near_rms']:.2e}", flush=True)
            json.dump({"parameters": vars(a), "rows": rows},
                      open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
