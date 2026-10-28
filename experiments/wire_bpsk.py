"""Bits through a blocked medium: does a frozen chain carry a sub-gap BPSK signal?

Input: carrier at w_d near bead 0; each symbol of S steps multiplies it by a
random bit b in {-1,+1} (phase-continuous BPSK). The medium screens w_d < 0.1415,
so the bare field barely reaches the far probes.

Readout (identical for every condition): at each of 15 far probes, quadrature
demodulation over the second half of every symbol at w_d and at its pump
complement 2*Omega - w_d (the two frequencies a pumped chain can relay):
60 features per symbol. Gaussian probe noise of fixed absolute std sigma is
added to every raw sample BEFORE demodulation. Ridge readout, chronological
70/30 split, standardisation fitted on train only.

Geometry is pinned after warmup, so this is about sequence, not bonds.

    python experiments/wire_bpsk.py --sequences ATATATAT,AAAATTTT --carrier 0.11
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from chains import Config, World
from chains.probing import FieldProbes, source_kernel
from experiments.temporal_memory import make_probes, standardized_ridge, score


def run_one(args, seed, sequence, bits):
    cfg = Config(N=args.N, seed=seed, device=args.device, p_lig=0.0, kT=0.0,
                 phase_noise=0.0, chain_boost=0.0, Omega=args.Omega)
    w = World(cfg)
    L = args.length
    x0 = args.N / 2 - (L - 1) * args.spacing / 2
    if sequence:
        w.add_monomers([w.letters.index(c) for c in sequence],
                       pos=[(x0 + i * args.spacing, args.N / 2) for i in range(L)])
        w.pinned[:] = True
    for _ in range(args.warmup):
        w.step()
    probes = make_probes(args.N, args.spacing, L)
    sensor = FieldProbes(w, probes)
    kernel = source_kernel(w, (x0 - args.source_offset, args.N / 2), args.width)
    S = args.steps_per_symbol
    trace = np.zeros((len(bits) * S, len(probes)), dtype=np.float64)
    k = 0
    for b in bits:
        for s in range(S):
            w.step(external_source=kernel * float(args.amplitude * b * np.cos(args.carrier * k)))
            trace[k] = sensor.field(w).cpu().numpy()
            k += 1
    return trace


def demodulate(trace, nsym, args, sigma, rng):
    S = args.steps_per_symbol
    x = trace + (rng.normal(0, sigma, trace.shape) if sigma > 0 else 0)
    k = np.arange(trace.shape[0])
    feats = []
    for om in (args.carrier, 2 * args.Omega - args.carrier):
        for f in (np.cos, np.sin):
            ref = f(om * k)[:, None]
            prod = (x * ref).reshape(nsym, S, -1)[:, S // 2:, :].mean(1)
            feats.append(prod)
    return np.concatenate(feats, axis=1)


def decode(X, bits, args):
    n = len(bits)
    split = int(n * 0.7)
    L = args.max_lag
    train = np.arange(max(args.discard, L), split)
    test = np.arange(split + L, n)
    res = []
    for d in range(L + 1):
        y = bits[np.arange(n) - d]
        p = standardized_ridge(X[train], X[test], y[train], args.ridge)
        res.append(score(p, y[test], y[train]))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequences", default="ATATATAT,AAAATTTT,TTTTAAAA,ACGTACGT,AAAAAAAA,TTTTTTTT,CGCGCGCG")
    p.add_argument("--carrier", type=float, default=0.11)
    p.add_argument("--Omega", type=float, default=0.07)
    p.add_argument("--N", type=int, default=96)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--symbols", type=int, default=500)
    p.add_argument("--steps-per-symbol", type=int, default=120)
    p.add_argument("--amplitude", type=float, default=0.05)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--spacing", type=float, default=3.6)
    p.add_argument("--length", type=int, default=8)
    p.add_argument("--source-offset", type=float, default=3.0)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 1e-4, 1e-3, 3e-3])
    p.add_argument("--discard", type=int, default=20)
    p.add_argument("--max-lag", type=int, default=3)
    p.add_argument("--ridge", type=float, default=0.01)
    p.add_argument("--out", default="runs/wire_bpsk")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seqs = [s.strip().upper() for s in a.sequences.split(",") if s.strip()] + [""]
    rows = []
    for seed in range(a.seed0, a.seed0 + a.seeds):
        bits = np.random.default_rng(777 + seed).choice([-1.0, 1.0], size=a.symbols)
        for seq in seqs:
            tr = run_one(a, seed, seq, bits)
            np.save(os.path.join(a.out, f"trace_s{seed}_{seq or 'bare'}.npy"), tr.astype(np.float32))
            for sigma in a.sigmas:
                X = demodulate(tr, len(bits), a, sigma, np.random.default_rng(seed * 31 + 7))
                res = decode(X, bits, a)
                best = max(range(len(res)), key=lambda d: res[d]["test_r2"])
                rows.append({"seed": seed, "sequence": seq or "bare", "sigma": sigma,
                             "per_lag": res, "best_lag": best,
                             "best_r2": res[best]["test_r2"], "best_acc": res[best]["sign_accuracy"]})
                print(f"seed {seed} {seq or 'bare':9s} sigma={sigma:<7g} best lag {best}: "
                      f"R2={res[best]['test_r2']:+.3f} acc={res[best]['sign_accuracy']:.3f}", flush=True)
            json.dump({"parameters": vars(a), "rows": rows},
                      open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
