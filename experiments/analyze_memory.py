"""Re-score saved memory_capacity features under a probe noise floor.

A noise-free linear readout can decode arbitrarily faint echoes in a
deterministic simulation, so capacity without noise mostly measures how long
the periodic box reverberates. Real sensors have a floor. Here i.i.d. Gaussian
noise of FIXED ABSOLUTE std sigma is added to every probe sample, the same
draw for every condition at a given (seed, sigma, repeat).

    python experiments/analyze_memory.py runs/mc_hold --sigmas 0 1e-4 3e-4 1e-3
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiments.memory_capacity import CONDITIONS, memory_curves


def rescore(run_dir, sigmas, repeats=3, max_lag=None):
    summ = json.load(open(os.path.join(run_dir, "summary.json")))
    args = argparse.Namespace(**summ["parameters"])
    if max_lag is not None:
        args.max_lag = max_lag
    feats = np.load(os.path.join(run_dir, "features.npz"))
    seqs = [s.strip().upper() for s in args.sequences.split(",") if s.strip()]
    out = []
    for seed in range(args.seed, args.seed + args.seeds):
        bits = feats[f"s{seed}_bits"]
        for seq in seqs:
            for sigma in sigmas:
                for rep in range(repeats if sigma > 0 else 1):
                    rng = np.random.default_rng(1000 * seed + rep)
                    shape = feats[f"s{seed}_{seq}_bare_drv"].shape
                    noise = rng.normal(0.0, sigma, size=shape) if sigma > 0 else 0.0
                    for c in CONDITIONS:
                        X = feats[f"s{seed}_{seq}_{c}_drv"] + noise
                        m = memory_curves(X, bits, args)
                        out.append({"seed": seed, "sequence": seq, "sigma": sigma,
                                    "repeat": rep, "condition": c, **m})
    return out


def table(rows):
    keys = sorted({(r["sequence"], r["sigma"], r["condition"]) for r in rows},
                  key=lambda k: (k[0], k[1], k[2]))
    lines = []
    for seq, sigma, c in keys:
        sel = [r for r in rows if (r["sequence"], r["sigma"], r["condition"]) == (seq, sigma, c)]
        lin = np.array([r["MC_lin"] for r in sel]); nl = np.array([r["MC_nl"] for r in sel])
        lines.append(f"{seq:10s} sigma={sigma:<7g} {c:6s} MC_lin={lin.mean():6.2f}±{lin.std():.2f} "
                     f"MC_nl={nl.mean():5.3f}±{nl.std():.3f} (n={len(sel)})")
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0, 1e-4, 3e-4, 1e-3])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-lag", type=int, default=None)
    a = p.parse_args()
    rows = rescore(a.run_dir, a.sigmas, a.repeats, a.max_lag)
    json.dump(rows, open(os.path.join(a.run_dir, "noise_rescore.json"), "w"), indent=1)
    print(table(rows))
