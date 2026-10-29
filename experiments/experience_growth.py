"""Does experience select what assembles? Growth of a 4-letter soup under a carrier.

held_by (world.py) is a running average of c(t) * hear * cos(theta): large only for a
monomer that hears its pump complement 2*Omega - w in phase with itself, and ligation
needs held_by > held_min. So a carrier at 0.03 should hold T (and A through A/T locking),
a carrier at 0.09 should hold C (and G). If so, the frequencies a soup experiences decide
which letters end up in chains, i.e. experience writes the address map that the phase
memory (phase_memory.py) is later read and written through.

Arms: quiet | carrier 0.03 | carrier 0.09 | both. The carrier is a spatially uniform
external drive (sub-gap waves are evanescent, so a point source would reach only a few
cells). Default physics otherwise (noise, ligation, chain_boost on).

Reported per arm: fraction of each letter that is bonded, chain count/length, and the
time-averaged held_by per letter.

    python experiments/experience_growth.py --seeds 2 --grow-steps 20000
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

from chains import Config, World

ARMS = {"quiet": (), "c0.03": (0.03,), "c0.09": (0.09,), "both": (0.03, 0.09)}


def grow(args, seed, freqs):
    cfg = Config(N=args.N, seed=seed, device="cpu")
    w = World(cfg)
    rng = np.random.default_rng(seed)
    types = np.repeat(np.arange(4), args.n_each)
    rng.shuffle(types)
    w.add_monomers(types)
    ones = torch.ones(args.N, args.N)
    held_acc = np.zeros(len(types)); n_acc = 0
    for t in range(args.grow_steps):
        ext = None
        if freqs:
            ext = ones * float(args.amplitude * sum(math.cos(f * t) for f in freqs))
        w.step(external_source=ext)
        if t >= args.grow_steps // 2 and t % 50 == 0:
            held_acc += w.held_by.numpy(); n_acc += 1
    typ = w.typ.numpy()
    bonded = (w.nb >= 0).any(1).numpy()
    letters = w.letters
    lens = [len(p) for p in w.chains(min_len=2)]
    return {"bonded_fraction_by_letter": {letters[k]: float(bonded[typ == k].mean()) for k in range(4)},
            "held_by_by_letter": {letters[k]: float((held_acc / max(n_acc, 1))[typ == k].mean()) for k in range(4)},
            "ligations": int(w.ligations), "chains": len(lens), "longest": max(lens, default=0),
            "mean_len": float(np.mean(lens)) if lens else 0.0}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, default=96)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--n-each", type=int, default=80)
    p.add_argument("--grow-steps", type=int, default=20000)
    p.add_argument("--amplitude", type=float, default=0.01)
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--out", default="runs/experience_growth")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for seed in range(a.seed0, a.seed0 + a.seeds):
        for arm in a.arms:
            r = grow(a, seed, ARMS[arm])
            rows.append({"seed": seed, "arm": arm, **r})
            bf, hb = r["bonded_fraction_by_letter"], r["held_by_by_letter"]
            print(f"seed {seed} {arm:6s} bonded A={bf['A']:.2f} C={bf['C']:.2f} G={bf['G']:.2f} T={bf['T']:.2f} | "
                  f"held A={hb['A']:+.3f} C={hb['C']:+.3f} G={hb['G']:+.3f} T={hb['T']:+.3f} | "
                  f"chains {r['chains']} longest {r['longest']}", flush=True)
            json.dump({"parameters": vars(a), "rows": rows}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
